#!/usr/bin/env python3
"""
rename_images.py

Renames images based on their visual content using a vision-language model,
run either locally via MLX (Apple Silicon) or on a remote Ollama server.
With the local (default) backend, no data leaves your machine; with the
remote backend, images are sent to whatever host you point it at.

Setup (one time):
    uv tool install --editable .

This installs a `rename-images` command on your PATH (~/.local/bin) that
you can run from any directory. Being an editable install, it always
reflects the current state of this checkout.

Usage:
    # Dry run (default) — shows what WOULD happen, doesn't touch files.
    # Results (including the model's description and EXIF data of each
    # image) are cached, so a later run against the same folder skips
    # inference for any file whose contents haven't changed. `rename` is the
    # default subcommand, so it can be omitted, as below.
    rename-images /path/to/folder

    # Actually rename the files — reuses the dry-run cache, so this is fast
    # and doesn't reload the model if every file was already processed.
    rename-images /path/to/folder -a

    # Use a different/larger local model (better quality, slower)
    rename-images /path/to/folder -m mlx-community/Qwen2-VL-7B-Instruct-4bit

    # Offload inference to an Ollama server on another machine instead of
    # running MLX locally (see "Remote backend" in CLAUDE.md for setup)
    rename-images /path/to/folder -u http://192.168.1.50:11434

    # Send up to 4 images to the remote server concurrently instead of one
    # at a time (only helps if the server can serve requests in parallel)
    rename-images /path/to/folder -u http://192.168.1.50:11434 -w 4

    # Recurse into subfolders
    rename-images /path/to/folder -a -r

    # Print tokens generated per image, to help tune --max-tokens
    rename-images /path/to/folder -v

    # Ignore the cache and re-run inference on every image
    rename-images /path/to/folder -c

    # Print EXIF metadata for one image, or every image in a folder
    # (recursively with -r), as a table (default) or JSON
    rename-images exif /path/to/photo.jpg
    rename-images exif /path/to/folder -r -f json
"""

import base64
import hashlib
import io
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import click
from PIL import ExifTags, Image, ImageOps
from PIL.TiffImagePlugin import IFDRational
from pillow_heif import register_heif_opener

# Teach Pillow to open HEIC/HEIF (iPhone photos). Stock Pillow ships no HEIC
# codec, so without this every HEIC file silently fell back to file-creation
# time and an empty EXIF dict (Image.open() raised, swallowed by
# get_photo_metadata's fallback), and the local MLX path couldn't load them
# either. Registration is process-global, so mlx_vlm's own PIL loading
# benefits too.
register_heif_opener()

# Tags that are just byte-offset pointers to the Exif/GPS sub-IFDs, not real
# data — their actual contents are extracted separately and merged in under
# proper names, so the raw pointer values would just be noise.
_EXIF_POINTER_TAGS = {34665, 34853}  # ExifOffset, GPSInfo

# MakerNote is an opaque, manufacturer-proprietary binary blob (format
# undocumented and different per camera maker) that Pillow can't decode —
# it just comes back as raw bytes, which our cleanup then hex-encodes into
# a very long, not-actually-useful string. Excluded by default; pass
# include_maker_note=True (the exif command's -M/--maker-note) to keep it.
_MAKER_NOTE_TAG = 37500

# UserComment is often empty/null-padded in practice, and even when it has
# real text, Pillow returns it with its raw 8-byte charset-code prefix
# (e.g. b"ASCII\x00\x00\x00...") rather than a clean decoded string, so it's
# excluded by default too; pass include_user_comment=True (the exif command's
# -U/--user-comment) to keep it.
_USER_COMMENT_TAG = 37510

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tiff", ".gif"}

# Formats Ollama's server-side image loader decodes natively. Anything else
# (HEIC most notably) gets HTTP 400 "Failed to load image or audio file"
# back, so _remote_image_bytes() transcodes those to JPEG before upload.
_REMOTE_NATIVE_SUFFIXES = {".jpg", ".jpeg", ".png"}

CACHE_FILENAME = ".rename-images-cache.json"
CACHE_VERSION = 1

DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
DEFAULT_REMOTE_MODEL = "qwen2.5vl:7b"
REMOTE_TIMEOUT_SECS = 120
REMOTE_PREFLIGHT_TIMEOUT_SECS = 10

REMOTE_SETUP_HELP = """\
To set up a machine for offloaded inference:
  1. Install Ollama:      curl -fsSL https://ollama.com/install.sh | sh
  2. Pull a vision model: ollama pull {model}
  3. Make it reachable on the network (Ollama binds to localhost only by
     default), then (re)start it with that env var set:
       OLLAMA_HOST=0.0.0.0:11434 ollama serve
  4. Open port 11434 on that machine's firewall, if one is enabled.
Then point this tool at it with -u http://<remote-host>:11434"""

PROMPT = (
    "Look at this image and suggest a short, descriptive filename for it. "
    "Use 2 to 5 words, all lowercase, describing the main subject and setting. "
    "Do not include a file extension, punctuation, or quotes. "
    "Respond with ONLY the filename words separated by spaces, nothing else. "
    "Example good responses: 'golden retriever on beach', 'sunset over mountains', "
    "'birthday cake with candles'."
)


def slugify(text: str, max_words: int = 5) -> str:
    """Turn a model response into a clean, filesystem-safe slug."""
    text = text.strip().lower()
    # Drop anything after a newline (models sometimes add explanations)
    text = text.split("\n")[0]
    # Keep only letters, numbers, and spaces
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    words = re.split(r"[\s-]+", text)
    words = [w for w in words if w]
    words = words[:max_words]
    slug = "-".join(words)
    return slug or "image"


def _tag_name(enum_cls, tag_id: int) -> str:
    """Resolve a numeric EXIF tag id to its name via a Pillow tag enum, falling back to the id."""
    try:
        return enum_cls(tag_id).name
    except ValueError:
        return str(tag_id)


def _clean_exif_value(value):
    """Make a raw EXIF value JSON-safe and human-readable."""
    if isinstance(value, bytes):
        try:
            text = value.decode("ascii").rstrip("\x00")
        except UnicodeDecodeError:
            text = None
        return text if text and text.isprintable() else value.hex()
    if isinstance(value, tuple):
        return [_clean_exif_value(v) for v in value]
    if isinstance(value, IFDRational):
        return float(value)
    return value


def _parse_exif(exif, include_maker_note: bool = False, include_user_comment: bool = False) -> dict:
    """Flatten a PIL Exif object (base IFD + Exif and GPS sub-IFDs) into a JSON-safe dict."""
    data = {}
    for tag_id, value in exif.items():
        if tag_id in _EXIF_POINTER_TAGS:
            continue
        data[_tag_name(ExifTags.Base, tag_id)] = _clean_exif_value(value)

    for tag_id, value in exif.get_ifd(ExifTags.IFD.Exif).items():
        if tag_id == _MAKER_NOTE_TAG and not include_maker_note:
            continue
        if tag_id == _USER_COMMENT_TAG and not include_user_comment:
            continue
        data[_tag_name(ExifTags.Base, tag_id)] = _clean_exif_value(value)

    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if gps:
        data["GPSInfo"] = {
            _tag_name(ExifTags.GPS, tag_id): _clean_exif_value(value) for tag_id, value in gps.items()
        }

    return data


def get_photo_metadata(
    path: Path, include_maker_note: bool = False, include_user_comment: bool = False
) -> tuple[datetime, dict]:
    """Get (photo date, full EXIF dict) for an image in a single file open.

    The date falls back to the file's creation time when EXIF has no usable
    date tag (or the image has no/unreadable EXIF at all); the EXIF dict is
    empty in that case too.
    """
    exif_data = {}
    date = None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                exif_data = _parse_exif(
                    exif, include_maker_note=include_maker_note, include_user_comment=include_user_comment
                )
                exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                raw = exif_ifd.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
                if raw:
                    date = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass

    if date is None:
        stat = path.stat()
        date = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_ctime))

    return date, exif_data


def get_exif_data(path: Path, include_maker_note: bool = False, include_user_comment: bool = False) -> dict:
    """Get just the EXIF dict for one image (used by the standalone `exif` command)."""
    return get_photo_metadata(
        path, include_maker_note=include_maker_note, include_user_comment=include_user_comment
    )[1]


def unique_path(target: Path) -> Path:
    """Avoid overwriting existing files by appending -2, -3, etc."""
    if not target.exists():
        return target
    stem, suffix, parent = target.stem, target.suffix, target.parent
    i = 2
    while True:
        candidate = parent / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def find_images(folder: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    for p in sorted(folder.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def file_checksum(path: Path) -> str:
    """Stream a sha256 checksum so cache lookups detect content changes, not just renames."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(folder: Path) -> dict:
    """Load the dry-run description cache for a folder, discarding it if unreadable or stale."""
    cache_path = folder / CACHE_FILENAME
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if data.get("version") == CACHE_VERSION and isinstance(data.get("entries"), dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"version": CACHE_VERSION, "entries": {}}


def save_cache(folder: Path, cache: dict) -> None:
    (folder / CACHE_FILENAME).write_text(json.dumps(cache, indent=2, sort_keys=True))


class GenResult(NamedTuple):
    text: str
    tokens: int | None
    hit_limit: bool


def check_remote_backend(remote_url: str, model_name: str) -> None:
    """Fail fast with setup guidance if the remote Ollama server isn't reachable or ready.

    Run once, right before the first image that actually needs the remote
    backend (not up front) — a run where every image is a cache hit never
    needs to talk to the remote server at all.
    """
    tags_url = f"{remote_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=REMOTE_PREFLIGHT_TIMEOUT_SECS) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        click.echo(f"Could not reach Ollama at {remote_url} ({e}).\n", err=True)
        click.echo(REMOTE_SETUP_HELP.format(model=model_name), err=True)
        sys.exit(1)

    available = {m.get("name") or m.get("model") for m in data.get("models", [])}
    if model_name not in available:
        click.echo(f"Model '{model_name}' is not pulled on {remote_url}.", err=True)
        if available:
            click.echo(f"Models available there: {', '.join(sorted(available))}", err=True)
        click.echo(f"\nOn that machine, run:\n  ollama pull {model_name}", err=True)
        sys.exit(1)


def _remote_image_bytes(img_path: Path) -> bytes:
    """Get an image's bytes in a format the remote Ollama server can decode.

    JPEG/PNG pass through untouched (no re-encode loss, no work). Everything
    else — HEIC especially, which Ollama rejects with an HTTP 400 — is
    transcoded to JPEG in-memory. The transcoded copy carries no EXIF, so
    orientation is baked into the pixels first (exif_transpose); a side
    effect is that GPS and other metadata never leave the machine for these
    formats. The file on disk is never modified.
    """
    if img_path.suffix.lower() in _REMOTE_NATIVE_SUFFIXES:
        return img_path.read_bytes()
    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"could not convert {img_path.suffix} to JPEG for the remote backend ({e})") from e


def generate_remote(remote_url: str, model_name: str, img_path: Path, max_tokens: int) -> GenResult:
    """Caption an image via a remote Ollama server's /api/generate endpoint.

    HTTP is plain stdlib urllib — the only thing required on the remote
    machine is Ollama itself. Formats Ollama can't decode are transcoded to
    JPEG client-side first (see _remote_image_bytes).
    """
    image_b64 = base64.b64encode(_remote_image_bytes(img_path)).decode("ascii")
    payload = {
        "model": model_name,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    request = urllib.request.Request(
        f"{remote_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REMOTE_TIMEOUT_SECS) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace").strip()
        try:
            detail = json.loads(detail).get("error", detail)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {e.code} from Ollama: {detail or e.reason}") from e

    return GenResult(
        text=data.get("response", ""),
        tokens=data.get("eval_count"),
        hit_limit=data.get("done_reason") == "length",
    )


class DefaultGroup(click.Group):
    """A click.Group that falls back to a default subcommand when none is given.

    Lets `rename-images /path -a ...` keep working exactly as before now that
    the CLI has multiple subcommands — only an explicit, recognized
    subcommand name (or --help/-h) is treated as one; anything else is
    routed to `default_command`.
    """

    def __init__(self, *args, default_command: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_command = default_command

    def resolve_command(self, ctx, args):
        if args and (args[0] in self.commands or args[0] in ("--help", "-h")):
            return super().resolve_command(ctx, args)
        return super().resolve_command(ctx, [self.default_command, *args])


@click.group(cls=DefaultGroup, default_command="rename")
def cli():
    """Rename images by their content (default), or inspect their EXIF data."""


@cli.command("rename")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("-a", "--apply", is_flag=True, help="Actually rename files (default is dry-run)")
@click.option("-r", "--recursive", is_flag=True, help="Recurse into subfolders")
@click.option(
    "-m",
    "--model",
    "model_name",
    default=None,
    help=f"Model to use. Defaults to {DEFAULT_LOCAL_MODEL!r} locally, "
    f"or {DEFAULT_REMOTE_MODEL!r} when -u/--remote-url is set.",
)
@click.option("-t", "--max-tokens", type=int, default=30, show_default=True, help="Max tokens for the model's response")
@click.option("-v", "--verbose", is_flag=True, help="Print tokens generated per image (useful for tuning --max-tokens)")
@click.option("-c", "--no-cache", is_flag=True, help="Ignore/skip the description cache and re-run inference on every image")
@click.option(
    "-u",
    "--remote-url",
    default=None,
    help="Offload inference to an Ollama server at this base URL (e.g. http://192.168.1.50:11434) "
    "instead of running MLX locally",
)
@click.option(
    "-w",
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Concurrent requests to the remote backend (-u/--remote-url only; no effect on local "
    "MLX, which can't be parallelized)",
)
def rename_cmd(
    folder: Path,
    apply: bool,
    recursive: bool,
    model_name: str | None,
    max_tokens: int,
    verbose: bool,
    no_cache: bool,
    remote_url: str | None,
    workers: int,
):
    """Rename images based on their content using a vision model, local or remote."""
    if model_name is None:
        model_name = DEFAULT_REMOTE_MODEL if remote_url else DEFAULT_LOCAL_MODEL

    images = list(find_images(folder, recursive))
    if not images:
        click.echo("No images found.")
        return

    cache = {"version": CACHE_VERSION, "entries": {}} if no_cache else load_cache(folder)
    cache_dirty = False

    # EXIF/file-date lookups are I/O-bound and independent per image, so kick
    # them off on a thread pool now — they run alongside the (I/O-heavy)
    # model load below and are essentially free by the time we need them.
    date_pool = ThreadPoolExecutor()
    metadata_futures = {img_path: date_pool.submit(get_photo_metadata, img_path) for img_path in images}

    backend = f"remote ({remote_url})" if remote_url else "local MLX"
    mode = "APPLYING RENAMES" if apply else "DRY RUN (use --apply to actually rename)"
    click.echo(f"Found {len(images)} image(s). Backend: {backend}. Mode: {mode}\n")

    # The cache is namespaced by backend + model so switching between them
    # (or pointing -u at a different server) can't silently reuse a
    # description generated by a different model.
    cache_model_key = f"{'remote' if remote_url else 'local'}:{model_name}"

    # Checksums are computed once up front (rather than inline per image)
    # so the same values can be used both to find cache misses before the
    # main loop (for the remote pre-pass below) and inside it.
    checksums = {img_path: file_checksum(img_path) for img_path in images}

    def is_cache_hit(img_path: Path) -> bool:
        cached = cache["entries"].get(str(img_path.relative_to(folder)))
        return (
            not no_cache
            and cached is not None
            and cached.get("checksum") == checksums[img_path]
            and cached.get("model") == cache_model_key
        )

    # The local model is loaded lazily, on the first cache miss, and its
    # generate() calls stay strictly sequential: mlx-vlm funnels all
    # inference through one shared GPU/model instance, so threading those
    # calls wouldn't add real parallelism (Metal serializes the GPU work
    # anyway), only risk. See "Why the inference loop isn't threaded" in
    # CLAUDE.md.
    #
    # The remote backend has no such constraint — each call is an
    # independent HTTP request to (possibly) another machine — so cache
    # misses are farmed out to a thread pool up front, sized by
    # -w/--workers (default 1, i.e. today's sequential behavior). Results
    # are collected here and consumed by the main loop below in the
    # original image order, so output ordering and cache/rename logic stay
    # exactly as deterministic as the single-threaded path.
    model = processor = config = None
    remote_results: dict[Path, GenResult] = {}
    failed_images: set[Path] = set()

    if remote_url:
        misses = [img_path for img_path in images if not is_cache_hit(img_path)]
        if misses:
            check_remote_backend(remote_url, model_name)
            with ThreadPoolExecutor(max_workers=workers) as remote_pool:
                futures = {
                    img_path: remote_pool.submit(generate_remote, remote_url, model_name, img_path, max_tokens)
                    for img_path in misses
                }
                # Reported here (in submission order, blocking per image as
                # needed) rather than only after the whole pre-pass finishes —
                # with many images and a small worker count, waiting until
                # everything is done before printing anything would look
                # exactly like a hang.
                for i, (img_path, future) in enumerate(futures.items(), start=1):
                    prefix = f"  [{i}/{len(misses)}]"
                    try:
                        result = future.result()
                    except Exception as e:
                        click.echo(f"{prefix} [SKIP] {img_path.name}: remote error ({e})")
                        failed_images.add(img_path)
                        continue
                    remote_results[img_path] = result
                    progress = f"{prefix} {img_path.name}"
                    if verbose and result.tokens is not None:
                        hit_limit = " (hit --max-tokens limit)" if result.hit_limit else ""
                        progress += f": {result.tokens} tokens{hit_limit}"
                    click.echo(progress)

    used_names = set()

    for img_path in images:
        rel_key = str(img_path.relative_to(folder))
        checksum = checksums[img_path]
        date, exif_data = metadata_futures[img_path].result()
        cached = cache["entries"].get(rel_key)
        cache_hit = is_cache_hit(img_path)

        if cache_hit:
            desc = cached["desc"]
        else:
            if remote_url:
                if img_path in failed_images:
                    # Already reported during the pre-pass above.
                    continue
                result = remote_results[img_path]
            else:
                if model is None:
                    try:
                        from mlx_vlm import load, generate
                        from mlx_vlm.prompt_utils import apply_chat_template
                        from mlx_vlm.utils import load_config
                    except ImportError:
                        click.echo(
                            "mlx-vlm is not installed. Set it up with:\n  uv add mlx-vlm",
                            err=True,
                        )
                        sys.exit(1)

                    click.echo(f"Loading model {model_name} ...")
                    model, processor = load(model_name)
                    config = load_config(model_name)

                formatted_prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
                try:
                    response = generate(
                        model, processor, formatted_prompt, [str(img_path)],
                        max_tokens=max_tokens, verbose=False,
                    )
                    # mlx-vlm's generate() may return a string or an object with .text
                    text = response.text if hasattr(response, "text") else str(response)
                except Exception as e:
                    click.echo(f"  [SKIP] {img_path.name}: model error ({e})")
                    continue

                result = GenResult(
                    text=text,
                    tokens=getattr(response, "generation_tokens", None),
                    hit_limit=getattr(response, "finish_reason", None) == "length",
                )

            if not remote_url and verbose and result.tokens is not None:
                # For the remote backend this was already printed during the
                # concurrent pre-pass above, as each request completed.
                hit_limit = " (hit --max-tokens limit)" if result.hit_limit else ""
                click.echo(f"  [{img_path.name}] {result.tokens} tokens{hit_limit}")

            desc = slugify(result.text)
            if len(desc.split("-")) < 2:
                # Model produced no usable multi-word description (truncated
                # output, empty response, etc.) — fall back to the original
                # filename instead of a near-meaningless one-word slug.
                desc = img_path.stem

            cache["entries"][rel_key] = {"checksum": checksum, "model": cache_model_key, "desc": desc}
            cache_dirty = True

        # Keep the cached EXIF dict current regardless of whether desc was a
        # cache hit — it's independent of the model/backend and cheap to
        # have on hand, so every processed image ends up with it recorded,
        # including entries created before this field existed.
        entry = cache["entries"][rel_key]
        if entry.get("exif") != exif_data:
            entry["exif"] = exif_data
            cache_dirty = True

        date_str = date.strftime("%Y-%m-%d")
        slug = f"{date_str}-{desc}"
        # avoid collisions within this run before touching disk
        base_slug = slug
        n = 2
        while slug in used_names:
            slug = f"{base_slug}-{n}"
            n += 1
        used_names.add(slug)

        new_path = unique_path(img_path.with_name(f"{slug}{img_path.suffix.lower()}"))

        cache_note = " [cached]" if cache_hit else ""
        click.echo(f"  {img_path.name}  ->  {new_path.name}{cache_note}")

        if apply:
            img_path.rename(new_path)
            # The old path is gone and the new name is already descriptive,
            # so there's nothing useful left for the cache to remember it by.
            if cache["entries"].pop(rel_key, None) is not None:
                cache_dirty = True

    date_pool.shutdown()

    if cache_dirty and not no_cache:
        save_cache(folder, cache)

    if not apply:
        click.echo("\nNo files were changed (dry run). Re-run with --apply to rename them.")


def _table_rows(data: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a (possibly one-level-nested, e.g. GPSInfo) EXIF dict into (tag, value) rows for table display."""
    rows = []
    for tag, value in data.items():
        if isinstance(value, dict):
            rows.extend(_table_rows(value, prefix=f"{tag}."))
        else:
            rows.append((f"{prefix}{tag}", value))
    return rows


@cli.command("exif")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-r", "--recursive", is_flag=True, help="Recurse into subfolders when PATH is a directory")
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format",
)
@click.option(
    "-M",
    "--maker-note",
    "include_maker_note",
    is_flag=True,
    help="Include the MakerNote tag (opaque, manufacturer-proprietary binary data; omitted by default)",
)
@click.option(
    "-U",
    "--user-comment",
    "include_user_comment",
    is_flag=True,
    help="Include the UserComment tag (often empty/null-padded, and undecoded when present; omitted by default)",
)
def exif_cmd(
    path: Path,
    recursive: bool,
    output_format: str,
    include_maker_note: bool,
    include_user_comment: bool,
):
    """Print EXIF metadata for an image file, or every image in a folder."""
    if path.is_dir():
        images = list(find_images(path, recursive))
        root = path
    else:
        images = [path]
        root = path.parent

    if not images:
        click.echo("No images found.")
        return

    results = {
        str(img_path.relative_to(root)): get_exif_data(
            img_path, include_maker_note=include_maker_note, include_user_comment=include_user_comment
        )
        for img_path in images
    }

    if output_format == "json":
        click.echo(json.dumps(results, indent=2, sort_keys=True))
        return

    for i, (key, data) in enumerate(results.items()):
        if i:
            click.echo()
        click.echo(key)
        rows = _table_rows(data)
        if not rows:
            click.echo("  (no EXIF data)")
            continue
        width = max(len(tag) for tag, _ in rows)
        for tag, value in rows:
            click.echo(f"  {tag:<{width}}  {value}")


if __name__ == "__main__":
    cli()