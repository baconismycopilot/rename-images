# rename-images

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A CLI that renames image files based on what's actually *in* them — instead of `IMG_0842.JPG`, you get `2024-08-12-golden-retriever-on-beach.jpg`. A vision-language model looks at each image and generates a short description, which becomes the new filename, prefixed with the date the photo was taken.

By default, inference runs locally — via [MLX](https://github.com/ml-explore/mlx) on Apple Silicon, no image data or API calls leave your machine, or via a local [Ollama](https://ollama.com) server on any other platform (Linux, Windows, Intel Mac), auto-detected at startup. There's also an explicit remote mode (`-u/--remote-url`) that offloads the same workload to Ollama on a *different* machine (e.g. one with a CUDA GPU) over your network.

## Features

- **Content-aware renaming** — a VLM captions each image in 2–5 words; combined with the photo's EXIF/file date, that becomes the new filename.
- **Dry-run by default** — renaming is destructive, so the tool only *prints* what it would do until you pass `-a/--apply`.
- **Description cache** — every processed image's description (and EXIF data) is cached (keyed by a checksum of its contents), so a `dry-run → review → --apply` workflow only ever runs inference once per image, not twice. Editing an image or switching models automatically invalidates its cache entry.
- **Cross-platform local backend** — MLX on Apple Silicon; a local Ollama server everywhere else, picked automatically based on the detected platform, no flags needed.
- **Remote offload** — point the tool at an [Ollama](https://ollama.com) server on another machine (`-u/--remote-url`) to run inference on a more capable GPU instead of the local device. The tool checks that the server is reachable and has the requested model *before* processing anything, and prints setup guidance directly if it isn't. `-w/--workers` sends multiple images to the server concurrently instead of one at a time (real speedup depends on the server's own concurrency, not just this flag).
- **Model picker** — a `models` command lists popular vision-capable models for whichever backend/hardware it detects on this machine (MLX repo IDs on Apple Silicon, Ollama tags elsewhere), highlighting the best-fit recommendation in green and anything that clearly won't fit in red based on detected unified memory / GPU VRAM / system RAM. The default Ollama model (`llava:13b`) downgrades automatically to a lighter tag on the auto-selected local backend if this machine's hardware can't fit it.
- **EXIF inspection** — a separate `exif` command prints an image's (or a whole folder's) EXIF metadata as a table or JSON, no renaming involved.
- **HEIC support** — iPhone photos work everywhere (dates, EXIF, all backends) via [pillow-heif](https://pypi.org/project/pillow-heif/). Since Ollama can't decode HEIC server-side, the Ollama backends transparently convert HEIC (and other non-JPEG/PNG formats) to JPEG in memory before upload — the files on disk are never modified, and stripped metadata (GPS etc.) stays local.
- **Safe collision handling** — won't overwrite files; duplicate slugs within a run or on disk get `-2`, `-3`, ... suffixes.
- **Recursive mode**, **tunable output length** (`--max-tokens`), and a **verbose mode** for tuning token limits.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Optionally install it as a standalone command on your `PATH`:

```bash
uv tool install --editable .
```

`--editable` means the installed `rename-images` command always reflects the current state of this checkout — no reinstall needed after editing the script. A `Makefile` is also provided (`make install`, `make install-tool`, `make lint`, `make test`, `make help`).

## Usage

`rename-images` is a group with three subcommands, `rename`, `exif`, and `models`. `rename` is the default, so it can be omitted — `rename-images /path/to/folder` and `rename-images rename /path/to/folder` do the same thing.

```bash
# Dry run (default) — prints what would be renamed, touches nothing
rename-images /path/to/folder

# Actually rename the files
rename-images /path/to/folder -a

# Recurse into subfolders
rename-images /path/to/folder -a -r

# Swap in a larger/better local model (slower, better quality)
rename-images /path/to/folder -m mlx-community/Qwen2-VL-7B-Instruct-4bit

# Print tokens generated per image — useful for tuning --max-tokens
rename-images /path/to/folder -v

# Ignore the cache and re-run inference on every image
rename-images /path/to/folder -c

# Offload inference to a remote Ollama server instead of the local backend
# (local MLX on Apple Silicon, local Ollama everywhere else)
rename-images /path/to/folder -u http://192.168.1.50:11434

# Same, but send up to 4 images to the remote server concurrently
rename-images /path/to/folder -u http://192.168.1.50:11434 -w 4

# List popular vision-capable models for the backend/hardware detected on
# this machine, flagging which are likely to fit in its RAM/VRAM
rename-images models

# Print EXIF metadata for one image, or a whole folder (-r to recurse),
# as a table (default) or JSON — read-only, never renames anything
rename-images exif /path/to/photo.jpg
rename-images exif /path/to/folder -r -f json
```

`uv run rename_images.py ...` works identically from inside this repo if you don't want to install the tool globally.

### All options

| Flag | Description |
|---|---|
| `--version` | Show the installed version and exit |

**`rename`** (default subcommand)

| Flag | Description |
|---|---|
| `-a, --apply` | Actually rename files (default is dry-run) |
| `-r, --recursive` | Recurse into subfolders |
| `-m, --model TEXT` | Model to use — an MLX repo ID on Apple Silicon, or an Ollama tag otherwise (`rename-images models` lists options) |
| `-t, --max-tokens INTEGER` | Max tokens for the model's response (default: 30) |
| `-v, --verbose` | Print tokens generated per image |
| `-c, --no-cache` | Skip the description cache and re-run inference on every image |
| `-u, --remote-url TEXT` | Ollama server base URL to send inference to. Defaults to local MLX on Apple Silicon, or a local Ollama server (`http://localhost:11434`) elsewhere — pass this to point at a different machine instead |
| `-w, --workers INTEGER` | Concurrent requests to the Ollama backend (default: 1); no effect on local MLX |

**`models`**

| Flag | Description |
|---|---|
| `-f, --format [table\|json]` | Output format (default: table) |

**`exif`**

| Flag | Description |
|---|---|
| `-r, --recursive` | Recurse into subfolders when the given path is a directory |
| `-f, --format [table\|json]` | Output format (default: table) |
| `-M, --maker-note` | Include the `MakerNote` tag (opaque, manufacturer-proprietary binary data; omitted by default) |
| `-U, --user-comment` | Include the `UserComment` tag (often empty/null-padded, and undecoded when present; omitted by default) |

## Backend selection

`rename-images` picks a local backend automatically, no flags needed:

- **Apple Silicon** → [MLX](https://github.com/ml-explore/mlx), running fully on-device via the GPU/Neural Engine. No image data leaves the machine.
- **Everywhere else** (Linux, Windows, Intel Mac) → a local [Ollama](https://ollama.com) server at `http://localhost:11434`, since MLX isn't built for those platforms. Install Ollama and pull a vision model first (see below); if it's not reachable, the tool prints exactly what to do.

`-u/--remote-url` always overrides this to point at Ollama on a *different* machine instead — see "Remote offload setup" below. Run `rename-images models` to see which vision-capable models are recommended for whichever backend gets picked, sized against this machine's detected memory/VRAM.

## Remote offload setup

If your local machine is the bottleneck, you can offload the actual captioning to any other machine on your network that runs [Ollama](https://ollama.com) — for example a Linux/Windows box with a dedicated GPU. Setup on that machine is minimal:

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a vision-capable model (must match what you pass via -m,
#    or the default: llava:13b)
ollama pull llava:13b

# 3. Ollama binds to localhost only by default — make it reachable
#    on the network, then (re)start it with this env var set
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 4. If that machine has a firewall enabled, open port 11434 to your
#    other machine's IP.
```

Then point `rename-images` at it from your other machine:

```bash
rename-images /path/to/folder -u http://<remote-host>:11434
```

If the server isn't reachable, or the requested model hasn't been pulled there, the tool fails fast with a clear message (unreachable server → the checklist above; missing model → which models *are* available and the exact `ollama pull` command to fix it) instead of silently timing out or retrying per image. This check only happens if at least one image actually needs processing — a fully-cached run never touches the network at all. The same check (with simpler local-only guidance) also runs before the auto-selected local Ollama backend on non-Apple-Silicon platforms.

Each backend/model combination is cached separately, so switching between them (or changing `-m`) never reuses a description generated by a different one.

Ollama can only decode JPEG and PNG server-side — anything else it rejects with an HTTP 400 ("Failed to load image or audio file") regardless of which model is loaded. `rename-images` handles this for you: HEIC and other formats are converted to JPEG in memory just for the upload, so you never see those errors and nothing on disk is touched.

### Speeding it up with `-w/--workers`

By default, images are sent to the remote server one at a time (`-w 1`), same as local MLX. If your Ollama server can actually handle multiple requests at once — governed by its `OLLAMA_NUM_PARALLEL` setting and whether the model + available VRAM support more than one loaded context — raising `-w` sends that many images concurrently instead. This is a real speedup, not a placebo: with a mock server that takes ~1s per request, 4 images took ~4.2s at `-w 1` versus ~1.2s at `-w 4`. If the server can only serve one request at a time regardless, raising `-w` just queues requests server-side with no benefit — it won't make things worse, but it won't help either. Output ordering, caching, and renaming are unaffected either way: results are gathered before the (single-threaded) rename logic runs, in the original file order.

With a large folder and a small `-w`, this can take a while — you'll see a `[i/N] filename` line printed as each image's request completes (with a token count if `-v` is set, or `[SKIP] filename: ...` if it failed), so it's visibly making progress rather than appearing to hang. The final `old -> new` rename lines print afterward, in a second pass.

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # run tests
```

See [CLAUDE.md](CLAUDE.md) for a deeper architectural walkthrough.
