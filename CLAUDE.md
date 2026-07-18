# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file Python CLI (`rename_images.py`) that renames image files based on their visual content, using a vision-language model. By default this runs locally through MLX (Apple's array framework for Apple Silicon) and no image data or API calls leave the machine — the script targets the M5 Pro's GPU/Neural Engine specifically, so it should keep depending on `mlx-vlm`/MLX for this path rather than a cross-platform inference stack (e.g. `torch`/CUDA-oriented libraries), and any performance tuning of the local path should assume Apple Silicon, not generic CPU or CUDA. There's also an optional remote backend (`-u/--remote-url`) that offloads inference to an Ollama server on another machine — see "Remote backend" below. The two backends are mutually exclusive per run: MLX is never imported when `-u` is set, and Ollama is only ever talked to over plain HTTP via the stdlib, so neither path gains a dependency it doesn't need.

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/) — use `uv`, not a manually-created venv or bare `pip`, for all dependency and environment management (adding deps, installing, running). Dependencies (`mlx-vlm`, `click`, `pillow`) are declared in `pyproject.toml`; add new ones with `uv add <package>` rather than pip-installing into the venv directly.

```bash
uv sync
```

The project also declares a console-script entry point (`[project.scripts]` in `pyproject.toml`, pointing `rename-images` at `rename_images:main`) and a `hatchling` build backend, so it can be installed as a standalone command:

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
```

`uv run rename_images.py ...` still works identically from inside this repo and is useful when you don't want to touch the globally-installed tool. Every option has a single-letter form: `-a/--apply`, `-r/--recursive`, `-m/--model`, `-t/--max-tokens`, `-v/--verbose`, `-c/--no-cache`, `-u/--remote-url`. Keep that pairing when adding new options.

## Linting

Ruff is the standard for lint and style enforcement in this repo, run via uv (`uv add --dev ruff` to install it into the managed environment):

```bash
uv run ruff check .
uv run ruff format .
```

There's no `pyproject.toml`/`ruff.toml` checked in yet, so these currently run against Ruff's defaults — add config in `pyproject.toml` if project-specific rules are needed rather than relying on ad hoc per-invocation flags.

## Architecture

Everything lives in `rename_images.py`; the CLI is built with `click` (not `argparse` — `click.Path(exists=True, ...)` validates the folder argument for us). The flow through `main()` is:

1. `find_images()` globs the target folder (optionally recursive) for known image extensions (`IMAGE_EXTS`).
2. `get_photo_date()` lookups (EXIF `DateTimeOriginal`/`DateTime` via Pillow, falling back to the file's `st_birthtime`) are submitted to a `ThreadPoolExecutor` for *all* images up front, before the model loads. This is the one place threading is used, deliberately: date lookups are pure I/O and independent per image, so they run for free alongside the (I/O-heavy) model load. See "Why the inference loop isn't threaded" below.
3. For each image, `file_checksum()` (streaming sha256) is checked against a per-folder JSON cache (`load_cache()`/`save_cache()`, stored as `.rename-images-cache.json` in the target folder — see "Description cache" below). A cache hit skips inference entirely and reuses the previously generated description.
4. On a cache miss, the backend selected by `-u/--remote-url` handles inference. Without `-u` (default), the MLX model/processor/config are loaded lazily on the *first* cache miss (`mlx_vlm.load` / `load_config`) — imports of `mlx_vlm` are deferred into `main()` and guarded with a friendly install message, since it's an optional/manual dependency, not a pinned requirement. If every image is a cache hit, the model is never loaded at all. With `-u`, there's no load step — `generate_remote()` POSTs directly to the given server.
5. Both backends funnel their result into a shared `GenResult(text, tokens, hit_limit)` before the rest of the loop runs, so downstream handling (verbose token printing, slugifying, caching) doesn't care which backend produced it. Locally, a fixed prompt (`PROMPT`) goes through `apply_chat_template` and `generate()`; remotely, `generate_remote()` sends the same `PROMPT` plus the base64-encoded image bytes to Ollama's `/api/generate`. With `-v/--verbose`, the token count (and whether the response was cut off) is printed — use this to tune `--max-tokens`. The result is written back into the cache keyed by the image's path (relative to the target folder) alongside its checksum and a `cache_model_key` of the form `local:<model>` or `remote:<model>`, so switching backends (or models) can't silently reuse a description from a different one.
6. `slugify()` turns the raw model response into a filesystem-safe slug (strips punctuation, caps word count, falls back to `"image"` if empty). The final name is `{YYYY-MM-DD}-{slug}`, using the date resolved in step 2.
7. Two layers of collision avoidance: `used_names` dedupes slugs within the current run (appending `-2`, `-3`, ...), and `unique_path()` separately guards against colliding with files already on disk.
8. Renames only happen when `-a/--apply` is passed; otherwise the script only prints the proposed `old -> new` mapping (dry-run is the default, by design — this is a destructive filesystem operation). When a rename does happen, its cache entry is removed immediately afterward, since the old path no longer exists and the new filename is already descriptive.

### Description cache

A dry run's whole point is to let you review descriptions before committing to them, so results are cached in `.rename-images-cache.json` (one per target folder, gitignored) keyed by each image's path relative to that folder. Each entry stores the file's sha256 checksum, a `cache_model_key` identifying the backend+model used, and the generated description — a later run (dry or `--apply`) reuses the cached description instead of re-running inference, as long as both the checksum and the `cache_model_key` still match. This means: editing/replacing an image invalidates its cache entry automatically (checksum changes), switching `-m/--model` or toggling `-u/--remote-url` invalidates it too (comparing across backends/models would be meaningless), and the common `dry-run` → review → `--apply` workflow does inference exactly once per image. `-c/--no-cache` bypasses the cache entirely (no reads, no writes) for a one-off forced re-run.

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

### Why the inference loop isn't threaded

`mlx_vlm`'s own server (`mlx_vlm/server/generation.py`) funnels all generation through a single shared GPU thread and gets its throughput from *internal* continuous batching, not from client-side threads each calling `generate()`. Wrapping the per-image `generate()` call in a thread pool here wouldn't add real parallelism (Metal serializes the GPU work anyway) and risks touching shared model/KV-cache state unsafely. If per-image throughput ever needs to go up for real, the correct lever is a smaller/faster model (see below) or `--max-tokens`, not threading the model calls. The remote/Ollama path is kept sequential for the same reason it started that way locally — simplicity — but unlike MLX it isn't inherently limited to one in-flight request; if remote throughput ever matters, adding a small thread pool around `generate_remote()` calls (Ollama can serve multiple requests concurrently, especially with `OLLAMA_NUM_PARALLEL`) is the natural next step, deliberately not done here to keep this change minimal.

## Model options

The task is short (2–5 word) captioning, not open-ended chat, so prefer the smallest model that still reliably obeys "respond with ONLY N words." All repo IDs below were confirmed to exist on Hugging Face; use `-v/--verbose` to check actual token counts per image before assuming a model is too slow.

| Model | Params | Pros | Cons |
|---|---|---|---|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` (default) | 2B, 4-bit | Good balance of caption quality and speed; the most battle-tested small model in the mlx-vlm ecosystem | Still a general chat VLM, not purpose-built for short captions |
| `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | 3B, 4-bit | Newer generation than Qwen2-VL, generally better instruction-following — more likely to strictly obey the "N words only" prompt | Slightly larger/slower than the 2B default; less widely used with mlx-vlm than Qwen2-VL |
| `mlx-community/SmolVLM-Instruct-4bit` | ~2.2B, 4-bit | Comparable size to the default | Reports of weaker instruction-following on structured/short-output prompts (see [mlx-vlm#188](https://github.com/Blaizzy/mlx-vlm/issues/188)) — riskier for this script's strict output format |
| `mlx-community/Qwen2-VL-7B-Instruct-4bit` | 7B, 4-bit | Noticeably better scene understanding for cluttered/ambiguous photos | Markedly slower per image and a much larger download (~4-5GB); overkill for a filename |
