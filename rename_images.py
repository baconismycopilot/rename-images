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
    # Results (including the model's description of each image) are cached,
    # so a later run against the same folder skips inference for any file
    # whose contents haven't changed.
    rename-images /path/to/folder

    # Actually rename the files — reuses the dry-run cache, so this is fast
    # and doesn't reload the model if every file was already processed.
    rename-images /path/to/folder -a

    # Use a different/larger local model (better quality, slower)
    rename-images /path/to/folder -m mlx-community/Qwen2-VL-7B-Instruct-4bit

    # Offload inference to an Ollama server on another machine instead of
    # running MLX locally (see "Remote backend" in CLAUDE.md for setup)
    rename-images /path/to/folder -u http://192.168.1.50:11434

    # Recurse into subfolders
    rename-images /path/to/folder -a -r

    # Print tokens generated per image, to help tune --max-tokens
    rename-images /path/to/folder -v

    # Ignore the cache and re-run inference on every image
    rename-images /path/to/folder -c
"""

import base64
import hashlib
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
from PIL import ExifTags, Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tiff", ".gif"}

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


def get_photo_date(path: Path) -> datetime:
    """Get the date a photo was taken from EXIF, falling back to file creation time."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
            raw = exif_ifd.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
            if raw:
                return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass

    stat = path.stat()
    return datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_ctime))


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


def generate_remote(remote_url: str, model_name: str, img_path: Path, max_tokens: int) -> GenResult:
    """Caption an image via a remote Ollama server's /api/generate endpoint.

    Uses only the stdlib (urllib) so the client side stays dependency-free —
    the only thing required on the remote machine is Ollama itself.
    """
    image_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
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


@click.command()
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
def main(
    folder: Path,
    apply: bool,
    recursive: bool,
    model_name: str | None,
    max_tokens: int,
    verbose: bool,
    no_cache: bool,
    remote_url: str | None,
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
    # The actual generation loop stays single-threaded on purpose: mlx-vlm
    # funnels all inference through one shared GPU/model instance, so
    # threading those calls wouldn't add real parallelism, only risk. The
    # remote/Ollama path is likewise kept sequential for now to match.
    date_pool = ThreadPoolExecutor()
    date_futures = {img_path: date_pool.submit(get_photo_date, img_path) for img_path in images}

    backend = f"remote ({remote_url})" if remote_url else "local MLX"
    mode = "APPLYING RENAMES" if apply else "DRY RUN (use --apply to actually rename)"
    click.echo(f"Found {len(images)} image(s). Backend: {backend}. Mode: {mode}\n")

    # The cache is namespaced by backend + model so switching between them
    # (or pointing -u at a different server) can't silently reuse a
    # description generated by a different model.
    cache_model_key = f"{'remote' if remote_url else 'local'}:{model_name}"

    # The local model is loaded lazily, on the first cache miss. If a prior
    # dry run already produced a description for every image (and none
    # changed since), an --apply pass can rename everything without ever
    # touching MLX. The remote backend's equivalent is a one-time preflight
    # check (reachability + model pulled) instead of a load step.
    model = processor = config = None
    remote_checked = False

    used_names = set()

    for img_path in images:
        rel_key = str(img_path.relative_to(folder))
        checksum = file_checksum(img_path)
        cached = cache["entries"].get(rel_key)
        cache_hit = (
            not no_cache
            and cached is not None
            and cached.get("checksum") == checksum
            and cached.get("model") == cache_model_key
        )

        if cache_hit:
            desc = cached["desc"]
        else:
            if remote_url:
                if not remote_checked:
                    check_remote_backend(remote_url, model_name)
                    remote_checked = True
                try:
                    result = generate_remote(remote_url, model_name, img_path, max_tokens)
                except Exception as e:
                    click.echo(f"  [SKIP] {img_path.name}: remote error ({e})")
                    continue
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

            if verbose and result.tokens is not None:
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

        date_str = date_futures[img_path].result().strftime("%Y-%m-%d")
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


if __name__ == "__main__":
    main()