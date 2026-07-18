# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file Python CLI (`rename_images.py`) that renames image files based on their visual content, using a vision-language model. By default this runs locally through MLX (Apple's array framework for Apple Silicon) and no image data or API calls leave the machine — the script targets the M5 Pro's GPU/Neural Engine specifically, so it should keep depending on `mlx-vlm`/MLX for this path rather than a cross-platform inference stack (e.g. `torch`/CUDA-oriented libraries), and any performance tuning of the local path should assume Apple Silicon, not generic CPU or CUDA. There's also an optional remote backend (`-u/--remote-url`) that offloads inference to an Ollama server on another machine — see "Remote backend" below. The two backends are mutually exclusive per run: MLX is never imported when `-u` is set, and Ollama is only ever talked to over plain HTTP via the stdlib, so neither path gains a dependency it doesn't need.

The CLI is a `click.Group` with two subcommands: `rename` (the behavior described above, and the default — see "CLI structure" below) and `exif`, which just prints an image's (or a folder's) EXIF metadata as a table or JSON without renaming anything.

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/) — use `uv`, not a manually-created venv or bare `pip`, for all dependency and environment management (adding deps, installing, running). Dependencies (`mlx-vlm`, `click`, `pillow`) are declared in `pyproject.toml`; add new ones with `uv add <package>` rather than pip-installing into the venv directly.

```bash
uv sync
```

The project also declares a console-script entry point (`[project.scripts]` in `pyproject.toml`, pointing `rename-images` at `rename_images:cli`, the click Group) and a `hatchling` build backend, so it can be installed as a standalone command:

```bash
uv tool install --editable .
```

`--editable` means the installed `rename-images` command always reflects the current state of this checkout — no reinstall needed after editing `rename_images.py`. The command lands in `~/.local/bin` (on `PATH` already via `uv tool install`) and works from any directory, not just this repo.

## Running

Once installed as a tool (see Setup), run it directly from anywhere:

```bash
# Dry run (default) — prints what would be renamed, touches nothing
rename-images /path/to/folder

# Actually rename files
rename-images /path/to/folder -a

# Recurse into subfolders
rename-images /path/to/folder -a -r

# Swap in a larger/better model (slower) — see "Model options" below
rename-images /path/to/folder -m mlx-community/Qwen2-VL-7B-Instruct-4bit

# Print tokens generated per image — use this to tune --max-tokens
rename-images /path/to/folder -v

# Ignore the description cache and re-run inference on every image
rename-images /path/to/folder -c

# Offload inference to an Ollama server on another machine (see "Remote
# backend" below for the one-time setup on that machine)
rename-images /path/to/folder -u http://192.168.1.50:11434

# Same, but send up to 4 images to the remote server concurrently instead
# of one at a time (only useful if the server can actually serve them in
# parallel — see "Why the local inference loop isn't threaded" below)
rename-images /path/to/folder -u http://192.168.1.50:11434 -w 4

# Print EXIF metadata for one image, or every image in a folder (-r to
# recurse), as a table (default) or JSON — doesn't rename anything
rename-images exif /path/to/photo.jpg
rename-images exif /path/to/folder -r -f json
```

`uv run rename_images.py ...` still works identically from inside this repo and is useful when you don't want to touch the globally-installed tool. `rename` is the default subcommand (see "CLI structure" below), so `rename-images /path/to/folder ...` and `rename-images rename /path/to/folder ...` are equivalent — the examples above all rely on the default. Every option has a single-letter form: `-a/--apply`, `-r/--recursive`, `-m/--model`, `-t/--max-tokens`, `-v/--verbose`, `-c/--no-cache`, `-u/--remote-url`, `-w/--workers` on `rename`; `-r/--recursive`, `-f/--format`, `-M/--maker-note`, `-U/--user-comment` on `exif`. Keep that pairing when adding new options — note `-t` was already taken by `--max-tokens`, which is why `--workers` uses `-w` rather than `-t/--threads`.

## Linting

Ruff is the standard for lint and style enforcement in this repo, run via uv (`uv add --dev ruff` to install it into the managed environment):

```bash
uv run ruff check .
uv run ruff format .
```

There's no `pyproject.toml`/`ruff.toml` checked in yet, so these currently run against Ruff's defaults — add config in `pyproject.toml` if project-specific rules are needed rather than relying on ad hoc per-invocation flags.

## Architecture

Everything lives in `rename_images.py`; the CLI is built with `click` (not `argparse` — `click.Path(exists=True, ...)` validates path arguments for us). See "CLI structure" below for how the two subcommands (`rename`, `exif`) are wired up. The flow through `rename_cmd()` (the `rename` subcommand's callback) is:

1. `find_images()` globs the target folder (optionally recursive) for known image extensions (`IMAGE_EXTS`).
2. `get_photo_metadata()` — one Pillow `Image.open()`/`getexif()` per image, returning `(photo date, full EXIF dict)` — is submitted to a `ThreadPoolExecutor` for *all* images up front, before the model loads. This is threading used purely for I/O overlap: these lookups are independent per image, so they run for free alongside the (I/O-heavy) model load. See "Why the inference loop isn't threaded" below for how this differs from the *generation* threading in step 4.
3. For each image, `file_checksum()` (streaming sha256, computed once up front into a `checksums` dict, reused everywhere) is checked against a per-folder JSON cache (`load_cache()`/`save_cache()`, stored as `.rename-images-cache.json` in the target folder — see "Description cache" below). A cache hit skips inference entirely and reuses the previously generated description.
4. On a cache miss, the backend selected by `-u/--remote-url` handles inference. Without `-u` (default), the MLX model/processor/config are loaded lazily on the *first* cache miss (`mlx_vlm.load` / `load_config`) — imports of `mlx_vlm` are deferred into `rename_cmd()` and guarded with a friendly install message, since it's an optional/manual dependency, not a pinned requirement. If every image is a cache hit, the model is never loaded at all. With `-u`, there's no load step; instead, *before* the main per-image loop, every cache-miss image is farmed out to a `ThreadPoolExecutor(max_workers=workers)` (`-w/--workers`, default 1) that calls `generate_remote()` concurrently. Results/failures are reported (`  [i/N] filename[: tokens|: [SKIP] reason]`) as each future's `.result()` resolves in submission order, **not** deferred until the whole pre-pass finishes — with many images and few workers, that first version of this looked exactly like a hang (see "Progress reporting on the remote pre-pass" below for that bug). The main loop then just looks up each image's already-fetched result from the collected dict, so all the per-image logic below (slugifying, collision avoidance, final `old -> new` printing, renaming) stays single-threaded and runs in the original, deterministic image order regardless of how the network calls completed. See "Why the inference loop isn't threaded" below.
5. Both backends funnel their result into a shared `GenResult(text, tokens, hit_limit)` before the rest of the loop runs, so downstream handling (verbose token printing, slugifying, caching) doesn't care which backend produced it. Locally, a fixed prompt (`PROMPT`) goes through `apply_chat_template` and `generate()`; remotely, `generate_remote()` sends the same `PROMPT` plus the base64-encoded image bytes to Ollama's `/api/generate`. With `-v/--verbose`, the token count (and whether the response was cut off) is printed — use this to tune `--max-tokens`. The result is written back into the cache keyed by the image's path (relative to the target folder) alongside its checksum and a `cache_model_key` of the form `local:<model>` or `remote:<model>`, so switching backends (or models) can't silently reuse a description from a different one.
6. `slugify()` turns the raw model response into a filesystem-safe slug (strips punctuation, caps word count, falls back to `"image"` if empty). The final name is `{YYYY-MM-DD}-{slug}`, using the date resolved in step 2.
7. Two layers of collision avoidance: `used_names` dedupes slugs within the current run (appending `-2`, `-3`, ...), and `unique_path()` separately guards against colliding with files already on disk.
8. Renames only happen when `-a/--apply` is passed; otherwise the script only prints the proposed `old -> new` mapping (dry-run is the default, by design — this is a destructive filesystem operation). When a rename does happen, its cache entry is removed immediately afterward, since the old path no longer exists and the new filename is already descriptive.

### CLI structure

`rename_images.py` exposes a `click.Group` (`cli`, the `rename-images` entry point) with two subcommands: `rename` (`rename_cmd()`, the behavior described above) and `exif` (`exif_cmd()`, see below). Since the tool predates having subcommands at all, `rename` is made the *default* via a small `DefaultGroup(click.Group)` subclass that overrides `resolve_command()`: if the first CLI token isn't a recognized subcommand name (or `--help`/`-h`), it prepends `"rename"` before delegating to Click's normal resolution. This is what lets `rename-images /path -a ...` keep working unchanged — `rename-images rename /path -a ...` is the explicit, equivalent form. `rename-images --help` (no subcommand token to redirect) shows the group-level help listing both subcommands, rather than `rename`'s specific options.

### `exif` command

`exif_cmd()` prints EXIF metadata for a single image or every image in a folder (`-r/--recursive`) as a table (default) or JSON (`-f/--format json`) — read-only, it never touches the filesystem. It reuses `find_images()` for directory input and `get_exif_data()` (a thin wrapper that discards the date half of `get_photo_metadata()`'s return value) for the actual extraction, so there's exactly one EXIF-parsing implementation shared with the `rename` flow (`_parse_exif()`, `_tag_name()`, `_clean_exif_value()`). Raw tag ids are resolved to names via Pillow's `ExifTags.Base` (base IFD + Exif sub-IFD tags share this one enum) and `ExifTags.GPS` (GPS sub-IFD) enums, falling back to the numeric id string for anything unmapped; the two internal pointer tags (`ExifOffset`, `GPSInfo`) that just hold sub-IFD byte offsets are dropped since their actual contents are already merged in under proper names. Values are cleaned for JSON/display: `IFDRational` → `float`, tuples → lists (recursively cleaned), and `bytes` → decoded ASCII text if printable, else a hex string. For table output, `_table_rows()` flattens the one level of nesting (`GPSInfo`) into `GPSInfo.<tag>` rows; JSON output keeps the nested structure as-is.

`MakerNote` (tag id `37500`, `_MAKER_NOTE_TAG`) is an opaque, manufacturer-proprietary binary blob with no public format spec — Pillow can't decode it, so it comes back as raw bytes that our value-cleaning turns into a very long hex string (hundreds to low-thousands of characters for a real camera). It's excluded by default from `_parse_exif()`/`get_photo_metadata()`/`get_exif_data()` (an `include_maker_note: bool = False` parameter threads through all three); the `exif` command exposes `-M/--maker-note` to opt back in. This also means the `rename` flow's cached `"exif"` field never carries this bloat, since it always calls with the default. Deliberately *not* solved by decoding the format properly — that would need either shelling out to the external `exiftool` CLI or the `pyexiv2` package (binds the C++ Exiv2 library), both heavier than this repo's dependency footprint warrants for informational output that isn't used by the renaming logic at all.

`UserComment` (tag id `37510`, `_USER_COMMENT_TAG`) gets the identical treatment (`include_user_comment` parameter, `-U/--user-comment`) for a different reason: it's frequently all null-padding (no real comment), and even when a camera/app *did* write one, Pillow returns it with its raw 8-byte charset-code prefix intact (e.g. `b"ASCII\x00\x00\x00actual text"`) rather than a clean decoded string — our generic bytes-cleanup (which expects trailing, not embedded, nulls) hex-encodes it same as MakerNote rather than stripping that prefix. Excluded by default for the same signal-to-noise reason, not because the format is undocumented like MakerNote's.

### Description cache

A dry run's whole point is to let you review descriptions before committing to them, so results are cached in `.rename-images-cache.json` (one per target folder, gitignored) keyed by each image's path relative to that folder. Each entry stores the file's sha256 checksum, a `cache_model_key` identifying the backend+model used, the generated description, and (independent of caption/model) its EXIF dict from `get_exif_data()`. A later run (dry or `--apply`) reuses the cached description instead of re-running inference, as long as both the checksum and the `cache_model_key` still match — editing/replacing an image invalidates its entry automatically (checksum changes), and switching `-m/--model` or toggling `-u/--remote-url` invalidates it too (comparing across backends/models would be meaningless), so the common `dry-run` → review → `--apply` workflow does inference exactly once per image. `-c/--no-cache` bypasses the cache entirely (no reads, no writes) for a one-off forced re-run. The `exif` field is kept current independently of the checksum/model match used for `desc` — cheap to (re)compute, so it's refreshed on every run regardless of cache hit/miss, which also backfills it onto cache entries written before this field existed.

### Remote backend

`-u/--remote-url <base-url>` sends each cache-miss image to an [Ollama](https://ollama.com) server instead of running MLX locally — useful for offloading to a machine with a real GPU (e.g. a CUDA box) when the Apple Silicon device is the bottleneck. Setup on the remote machine is deliberately minimal, which is why Ollama was chosen over vLLM/TGI/a hand-rolled server:

```bash
# On the remote (GPU) machine — install Ollama and pull a vision model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5vl:7b     # or another vision-capable tag; see ollama.com/library

# Ollama binds to localhost only by default — make it reachable from the
# Mac by setting this before/when starting the server, then restart it
# (e.g. `systemctl edit ollama` on Linux, or the app's Settings on macOS/Windows)
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Then point `rename-images` at it:

```bash
rename-images /path/to/folder -u http://<remote-host>:11434
```

`generate_remote()` (in `rename_images.py`) POSTs to `<remote-url>/api/generate` with the prompt, the image as base64, and `num_predict` set from `--max-tokens` — using only `urllib`/`base64` from the stdlib, so the client side needs no extra dependency for this path. `-m/--model` is repurposed as the Ollama model tag when `-u` is set (default `qwen2.5vl:7b`, defined as `DEFAULT_REMOTE_MODEL`); make sure whatever tag you pass has been `ollama pull`ed on the remote machine first. HTTP errors from Ollama (e.g. a 404 for an unpulled model) have their JSON error body surfaced in the skip message, not just the generic status text — so a per-image failure is still self-diagnosing even after the preflight check below has passed.

Before the first image that actually needs the remote backend (not up front — a fully-cached run never touches the network), `check_remote_backend()` GETs `<remote-url>/api/tags` once and exits with setup guidance (`REMOTE_SETUP_HELP`) if the server is unreachable, or with the list of models actually pulled there if the requested tag isn't one of them. This turns "58 identical skipped-image errors because of one misconfiguration" into a single actionable message before any per-image work starts. Runtime failures during the actual generate calls (a transient network blip, one bad image) are still caught per-image and skip just that file, same as local model errors.

Firewall note: if the remote machine has a firewall enabled, port 11434 needs to be open to the Mac's IP for both the preflight check and the generate calls to work.

### Why the local inference loop isn't threaded (and the remote one now is)

`mlx_vlm`'s own server (`mlx_vlm/server/generation.py`) funnels all generation through a single shared GPU thread and gets its throughput from *internal* continuous batching, not from client-side threads each calling `generate()`. Wrapping the per-image `generate()` call in a thread pool here wouldn't add real parallelism (Metal serializes the GPU work anyway) and risks touching shared model/KV-cache state unsafely. If per-image throughput ever needs to go up for real on the local path, the correct lever is a smaller/faster model (see below) or `--max-tokens`, not threading the model calls. **This constraint is specific to MLX's single shared local model instance and does not apply to the remote backend.**

The remote backend's `generate_remote()` calls are independent HTTP requests to a separate machine, with no shared GPU/model state on this side to protect — so `-w/--workers` (default `1`, i.e. the original sequential behavior) runs cache-miss images through a `ThreadPoolExecutor` concurrently (see step 4 above). Whether this actually speeds anything up depends on the *remote* Ollama server's ability to serve concurrent requests — governed by its `OLLAMA_NUM_PARALLEL` setting and whether the model + available VRAM support more than one loaded context — not on anything this script controls. Verified with a mock server that sleeps 1s per request: `-w 1` took ~4.2s for 4 images, `-w 4` took ~1.2s, with identical output ordering in both cases (results are collected into a dict and consumed by the single-threaded main loop in original image order, so concurrency only affects *when* the network calls complete, never the order things are printed, cached, or renamed in).

### Progress reporting on the remote pre-pass

The first version of `-w/--workers` collected every cache-miss image's result in the pre-pass loop *before* printing anything, on the reasoning that nothing needed to print until the main loop ran. In practice, a real run against 264 images with `-w 4` looked exactly like a hang: the tool printed "Found 264 image(s)... Mode: DRY RUN" and then went completely silent for as long as the whole batch took to finish (potentially many minutes), since the only thing the pre-pass printed at all was `[SKIP]` on failures.

The fix: the pre-pass loop itself now echoes `  [i/N] <filename>` (plus token count when `-v` is set, or `[SKIP] <filename>: remote error (...)` on failure) immediately as each future's `.result()` resolves, rather than only after the `with ThreadPoolExecutor(...)` block exits. It still iterates `futures.items()` in original submission order rather than `concurrent.futures.as_completed()` order — so an in-progress image can occasionally hold up reporting on a later one that's already finished in the background — but that trade-off keeps the progress lines in the same deterministic order as everything else this script prints, at the cost of not always reporting in strict "whichever finishes first" order. Given the point is "prove it's alive," not "show a perfectly real-time completion feed," that trade-off was judged worth it over the added complexity of reconciling `as_completed()` order with the deterministic-output guarantee described above. The main loop's own verbose token-count print is now skipped for the remote backend specifically (`if not remote_url and verbose and ...`), since the pre-pass already printed it — otherwise every remote image would show its token count twice.

## Model options

The task is short (2–5 word) captioning, not open-ended chat, so prefer the smallest model that still reliably obeys "respond with ONLY N words." All repo IDs below were confirmed to exist on Hugging Face; use `-v/--verbose` to check actual token counts per image before assuming a model is too slow.

| Model | Params | Pros | Cons |
|---|---|---|---|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` (default) | 2B, 4-bit | Good balance of caption quality and speed; the most battle-tested small model in the mlx-vlm ecosystem | Still a general chat VLM, not purpose-built for short captions |
| `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | 3B, 4-bit | Newer generation than Qwen2-VL, generally better instruction-following — more likely to strictly obey the "N words only" prompt | Slightly larger/slower than the 2B default; less widely used with mlx-vlm than Qwen2-VL |
| `mlx-community/SmolVLM-Instruct-4bit` | ~2.2B, 4-bit | Comparable size to the default | Reports of weaker instruction-following on structured/short-output prompts (see [mlx-vlm#188](https://github.com/Blaizzy/mlx-vlm/issues/188)) — riskier for this script's strict output format |
| `mlx-community/Qwen2-VL-7B-Instruct-4bit` | 7B, 4-bit | Noticeably better scene understanding for cluttered/ambiguous photos | Markedly slower per image and a much larger download (~4-5GB); overkill for a filename |

## Session History

A running log of meaningful changes and why they happened — kept here instead of relying solely on `git log`, since commit messages capture *what* changed but not always the reasoning or the sequence of decisions. Add a dated entry after any substantive change to this repo.

- **2026-07-17** — Added the checksum-keyed description cache (`.rename-images-cache.json`), so a `dry-run` → review → `--apply` workflow only runs inference once per image instead of twice.
- **2026-07-18** — Added the remote Ollama backend (`-u/--remote-url`) to offload inference to a machine with a real GPU. Includes a fail-fast preflight check (`check_remote_backend()`) that surfaces setup problems (unreachable server, model not pulled) with actionable guidance instead of per-image timeouts/404s, and per-backend cache namespacing so switching backends can't reuse a stale description.
- **2026-07-18** — Added the pytest test suite (`tests/`), a `Makefile` (lint/format/install/test targets), and `README.md`.
- **2026-07-18** — Set up the GitHub repo (`baconismycopilot/rename-images`). Made it public rather than private after confirming classic branch protection and rulesets both require GitHub Pro/Team for private repos on the free plan; applied basic protection to `main` (blocks force-push/branch deletion) via a scripted, reproducible `gh api` call (`.github/scripts/apply-branch-protection.sh`) rather than only through the UI.
- **2026-07-18** — Restructured the CLI from a single command into a `click.Group` (`rename`, made the default via a small `DefaultGroup` subclass, plus a new `exif` command). EXIF extraction was unified into one shared code path (`get_photo_metadata()`/`get_exif_data()`) so the `rename` flow's per-image EXIF read also populates the cache's new `"exif"` field, and the `exif` command reuses the same parsing for standalone table/JSON output.
- **2026-07-18** — Excluded the `MakerNote` EXIF tag by default (`_MAKER_NOTE_TAG`, `include_maker_note` parameter, `-M/--maker-note` on `exif`). It's an opaque, manufacturer-proprietary binary blob Pillow can't decode, so it was showing up as a multi-hundred/thousand-character hex string in both `exif` output and the cache; decoding it properly would need an external `exiftool` process or the C++-backed `pyexiv2`, judged not worth the dependency weight for informational-only data.
- **2026-07-18** — Gave `UserComment` the same excluded-by-default treatment (`_USER_COMMENT_TAG`, `include_user_comment` parameter, `-U/--user-comment` on `exif`), for a related but distinct reason: it's often all null-padding, and even with real content Pillow leaves its raw 8-byte charset-code prefix undecoded, so our generic bytes-cleanup hex-dumps it the same as MakerNote.
- **2026-07-18** — Parallelized the remote backend's cache-miss requests via a new `-w/--workers` option (default `1`, i.e. unchanged sequential behavior). Unlike local MLX (hard-blocked from threading — see "Why the local inference loop isn't threaded"), remote calls are independent HTTP requests with no shared state on this side, so they're farmed out to a `ThreadPoolExecutor` in a pre-pass before the (still single-threaded) rename/cache/collision logic runs. Verified with a mock server that sleeps 1s/request: `-w 1` took ~4.2s for 4 images, `-w 4` took ~1.2s. Real-world speedup depends on the remote Ollama server's own concurrency (`OLLAMA_NUM_PARALLEL`, available VRAM), not on this flag alone.
- **2026-07-18** — Fixed a bug in the above where a real run (264 images, `-w 4` against a real remote server) looked like a hang: the pre-pass silently collected every result before printing anything, so nothing appeared on screen until the *entire* batch finished. Now each pre-pass future's completion is reported immediately (`[i/N] filename`, with token count under `-v`, or `[SKIP] ...` on failure) as it resolves, instead of being buffered until the whole `ThreadPoolExecutor` block exits. See "Progress reporting on the remote pre-pass" below.
