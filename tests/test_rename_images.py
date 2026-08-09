"""Basic functionality tests for rename_images.py.

Covers the pure helper functions plus the cache and remote (Ollama) backend,
using a mock HTTP server. The local MLX backend isn't exercised here — it
needs real Apple Silicon hardware and a multi-GB model download, so it's
out of scope for an automated suite; the remote backend covers the same
generation/cache/CLI code paths against a lightweight stand-in server.
"""

import base64
import io
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import ExifTags, Image

import rename_images as ri


def _make_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color="red").save(path)


# ---------- slugify ----------


def test_slugify_basic():
    assert ri.slugify("Golden Retriever on Beach") == "golden-retriever-on-beach"


def test_slugify_strips_punctuation_and_extra_lines():
    text = "sunset, over mountains!\nSome extra explanation the model added"
    assert ri.slugify(text) == "sunset-over-mountains"


def test_slugify_caps_word_count():
    assert ri.slugify("one two three four five six seven") == "one-two-three-four-five"


def test_slugify_empty_falls_back_to_image():
    assert ri.slugify("   ...   ") == "image"


# ---------- unique_path ----------


def test_unique_path_returns_target_when_free(tmp_path):
    target = tmp_path / "photo.jpg"
    assert ri.unique_path(target) == target


def test_unique_path_avoids_existing_files(tmp_path):
    (tmp_path / "photo.jpg").touch()
    (tmp_path / "photo-2.jpg").touch()
    assert ri.unique_path(tmp_path / "photo.jpg") == tmp_path / "photo-3.jpg"


# ---------- file_checksum ----------


def test_file_checksum_matches_identical_content(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    assert ri.file_checksum(a) == ri.file_checksum(b)


def test_file_checksum_differs_for_different_content(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"content one")
    b.write_bytes(b"content two")
    assert ri.file_checksum(a) != ri.file_checksum(b)


# ---------- find_images ----------


def test_find_images_filters_by_extension_and_sorts(tmp_path):
    _make_image(tmp_path / "b.jpg")
    _make_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("not an image")
    found = list(ri.find_images(tmp_path, recursive=False))
    assert [p.name for p in found] == ["a.png", "b.jpg"]


def test_find_images_recursive_flag(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_image(tmp_path / "top.jpg")
    _make_image(sub / "nested.jpg")
    assert len(list(ri.find_images(tmp_path, recursive=False))) == 1
    assert len(list(ri.find_images(tmp_path, recursive=True))) == 2


# ---------- cache load/save ----------


def test_cache_round_trip(tmp_path):
    cache = {
        "version": ri.CACHE_VERSION,
        "entries": {"a.jpg": {"checksum": "x", "model": "local:m", "desc": "d"}},
    }
    ri.save_cache(tmp_path, cache)
    assert ri.load_cache(tmp_path) == cache


def test_load_cache_missing_file_returns_empty_structure(tmp_path):
    assert ri.load_cache(tmp_path) == {"version": ri.CACHE_VERSION, "entries": {}}


def test_load_cache_ignores_corrupt_json(tmp_path):
    (tmp_path / ri.CACHE_FILENAME).write_text("{not valid json")
    assert ri.load_cache(tmp_path) == {"version": ri.CACHE_VERSION, "entries": {}}


def test_load_cache_ignores_version_mismatch(tmp_path):
    (tmp_path / ri.CACHE_FILENAME).write_text(
        json.dumps({"version": 999, "entries": {"a": {}}})
    )
    assert ri.load_cache(tmp_path) == {"version": ri.CACHE_VERSION, "entries": {}}


# ---------- get_photo_metadata / get_exif_data ----------


def test_get_photo_metadata_falls_back_to_file_time_when_no_exif(tmp_path):
    path = tmp_path / "plain.png"
    _make_image(path)
    date, exif_data = ri.get_photo_metadata(path)
    assert date.year >= 2020
    assert exif_data == {}


def test_get_exif_data_extracts_base_and_subifd_tags(tmp_path):
    path = tmp_path / "photo.jpg"
    img = Image.new("RGB", (4, 4), color="blue")
    exif = img.getexif()
    exif[271] = "Acme"  # Make
    exif[272] = "Camera 3000"  # Model
    img.save(path, exif=exif)

    data = ri.get_exif_data(path)

    assert data["Make"] == "Acme"
    assert data["Model"] == "Camera 3000"


def test_get_exif_data_returns_empty_dict_when_no_exif(tmp_path):
    path = tmp_path / "plain.png"
    _make_image(path)
    assert ri.get_exif_data(path) == {}


def _make_image_with_maker_note(path: Path) -> None:
    img = Image.new("RGB", (4, 4), color="blue")
    exif = img.getexif()
    exif.get_ifd(ExifTags.IFD.Exif)[ri._MAKER_NOTE_TAG] = b"PROPRIETARYDATA"
    img.save(path, exif=exif)


def test_get_exif_data_omits_maker_note_by_default(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image_with_maker_note(path)
    assert "MakerNote" not in ri.get_exif_data(path)


def test_get_exif_data_includes_maker_note_when_requested(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image_with_maker_note(path)
    data = ri.get_exif_data(path, include_maker_note=True)
    assert data["MakerNote"] == "PROPRIETARYDATA"


def _make_image_with_user_comment(path: Path) -> None:
    img = Image.new("RGB", (4, 4), color="blue")
    exif = img.getexif()
    exif.get_ifd(ExifTags.IFD.Exif)[ri._USER_COMMENT_TAG] = b"ASCII\x00\x00\x00hello"
    img.save(path, exif=exif)


def test_get_exif_data_omits_user_comment_by_default(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image_with_user_comment(path)
    assert "UserComment" not in ri.get_exif_data(path)


def test_get_exif_data_includes_user_comment_when_requested(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image_with_user_comment(path)
    data = ri.get_exif_data(path, include_user_comment=True)
    assert "UserComment" in data


# ---------- HEIC support / _remote_image_bytes ----------


def _make_heic_image(path: Path, exif=None) -> None:
    """Save a HEIC file — works because rename_images registers pillow-heif's opener on import."""
    img = Image.new("RGB", (8, 4), color="red")
    img.save(path, exif=exif if exif is not None else img.getexif())


def test_remote_image_bytes_passes_jpeg_and_png_through_untouched(tmp_path):
    for name in ("photo.jpg", "photo.png"):
        path = tmp_path / name
        _make_image(path)
        assert ri._remote_image_bytes(path) == path.read_bytes()


def test_remote_image_bytes_transcodes_heic_to_jpeg(tmp_path):
    path = tmp_path / "photo.heic"
    _make_heic_image(path)

    data = ri._remote_image_bytes(path)

    assert data[:3] == b"\xff\xd8\xff"  # JPEG magic bytes
    assert Image.open(io.BytesIO(data)).size == (8, 4)


def test_remote_image_bytes_bakes_orientation_into_transcoded_pixels(tmp_path):
    path = tmp_path / "photo.webp"
    img = Image.new("RGB", (8, 4), color="blue")
    exif = img.getexif()
    exif[ExifTags.Base.Orientation] = 6  # rotate 90° CW to display upright
    img.save(path, exif=exif)

    out = Image.open(io.BytesIO(ri._remote_image_bytes(path)))

    assert out.size == (4, 8)  # dimensions swapped: rotation applied to pixels


def test_remote_image_bytes_raises_clear_error_for_undecodable_file(tmp_path):
    path = tmp_path / "corrupt.heic"
    path.write_bytes(b"this is not an image")

    with pytest.raises(RuntimeError, match="could not convert .heic"):
        ri._remote_image_bytes(path)


def test_get_photo_metadata_reads_heic_exif_date(tmp_path):
    path = tmp_path / "photo.heic"
    img = Image.new("RGB", (8, 4), color="red")
    exif = img.getexif()
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] = (
        "2023:05:01 12:00:00"
    )
    _make_heic_image(path, exif=exif)

    date, exif_data = ri.get_photo_metadata(path)

    assert (date.year, date.month, date.day) == (2023, 5, 1)
    assert exif_data["DateTimeOriginal"] == "2023:05:01 12:00:00"


def test_generate_remote_sends_transcoded_jpeg_for_heic(tmp_path, mock_ollama):
    mock_ollama.set_generate_response(
        {"response": "a red rectangle", "eval_count": 3, "done_reason": "stop"}
    )
    path = tmp_path / "photo.heic"
    _make_heic_image(path)

    result = ri.generate_remote(mock_ollama.url, "qwen2.5vl:7b", path, max_tokens=30)

    assert result.text == "a red rectangle"
    sent = base64.b64decode(mock_ollama.server.state["generate_calls"][0]["images"][0])
    assert sent[:3] == b"\xff\xd8\xff"  # server received JPEG, not raw HEIC


def test_cli_renames_heic_via_remote_backend(tmp_path, mock_ollama):
    """End-to-end: a HEIC file must reach the remote backend as JPEG and get a date-prefixed slug name."""
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_generate_response(
        {"response": "a red rectangle", "eval_count": 3, "done_reason": "stop"}
    )
    img = Image.new("RGB", (8, 4), color="red")
    exif = img.getexif()
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] = (
        "2023:05:01 12:00:00"
    )
    _make_heic_image(tmp_path / "IMG_0173.heic", exif=exif)

    result = CliRunner().invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url, "-a"])

    assert result.exit_code == 0
    assert "[SKIP]" not in result.output
    assert (tmp_path / "2023-05-01-a-red-rectangle.heic").exists()


# ---------- remote backend (generate_remote / check_remote_backend) ----------


def test_generate_remote_success(tmp_path, mock_ollama):
    mock_ollama.set_generate_response(
        {"response": "a cat on a mat", "eval_count": 6, "done_reason": "stop"}
    )
    img = tmp_path / "cat.jpg"
    _make_image(img)

    result = ri.generate_remote(mock_ollama.url, "qwen2.5vl:7b", img, max_tokens=30)

    assert result == ri.GenResult(text="a cat on a mat", tokens=6, hit_limit=False)


def test_generate_remote_surfaces_ollama_error_body(tmp_path, mock_ollama):
    mock_ollama.set_generate_response(
        {"error": "model 'x' not found, try pulling it first"}, status=404
    )
    img = tmp_path / "cat.jpg"
    _make_image(img)

    with pytest.raises(RuntimeError, match="not found, try pulling it first"):
        ri.generate_remote(mock_ollama.url, "qwen2.5vl:7b", img, max_tokens=30)


def test_check_remote_backend_passes_when_model_available(mock_ollama):
    mock_ollama.set_models(["qwen2.5vl:7b"])
    ri.check_remote_backend(mock_ollama.url, "qwen2.5vl:7b")  # should not raise


def test_check_remote_backend_exits_when_model_missing(mock_ollama, capsys):
    mock_ollama.set_models(["llava:7b"])

    with pytest.raises(SystemExit) as exc_info:
        ri.check_remote_backend(mock_ollama.url, "qwen2.5vl:7b")

    assert exc_info.value.code == 1
    assert "ollama pull qwen2.5vl:7b" in capsys.readouterr().err


def test_check_remote_backend_exits_when_unreachable(capsys):
    with pytest.raises(SystemExit) as exc_info:
        ri.check_remote_backend("http://127.0.0.1:1", "qwen2.5vl:7b")

    assert exc_info.value.code == 1
    assert "Could not reach Ollama" in capsys.readouterr().err


@pytest.mark.parametrize(
    "is_local, expect_present, expect_absent",
    [
        (True, "defaults to a local Ollama server", "OLLAMA_HOST=0.0.0.0"),
        (False, "OLLAMA_HOST=0.0.0.0", None),
    ],
)
def test_check_remote_backend_help_when_unreachable(
    capsys, is_local, expect_present, expect_absent
):
    with pytest.raises(SystemExit):
        ri.check_remote_backend("http://127.0.0.1:1", "qwen2.5vl:7b", is_local=is_local)

    err = capsys.readouterr().err
    assert expect_present in err
    if expect_absent:
        assert expect_absent not in err


# ---------- platform-based backend selection ----------


@pytest.mark.parametrize(
    "system, machine, expected",
    [
        ("Darwin", "arm64", True),
        ("Linux", "x86_64", False),
        ("Darwin", "x86_64", False),
    ],
)
def test_is_apple_silicon(monkeypatch, system, machine, expected):
    monkeypatch.setattr(ri.platform, "system", lambda: system)
    monkeypatch.setattr(ri.platform, "machine", lambda: machine)
    assert ri.is_apple_silicon() is expected


def test_cli_defaults_to_local_ollama_on_non_apple_silicon(
    tmp_path, mock_ollama, monkeypatch
):
    """Without -u, a non-Apple-Silicon platform should hit local Ollama, never MLX."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "DEFAULT_OLLAMA_URL", mock_ollama.url)
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_generate_response(
        {"response": "a red square", "eval_count": 3, "done_reason": "stop"}
    )
    img = tmp_path / "photo.jpg"
    _make_image(img)

    result = CliRunner().invoke(ri.cli, [str(tmp_path)])

    assert result.exit_code == 0
    assert f"Backend: local Ollama ({mock_ollama.url})" in result.output
    assert mock_ollama.generate_call_count == 1


def test_cli_explicit_remote_url_overrides_auto_local_ollama_label(
    tmp_path, mock_ollama, monkeypatch
):
    """An explicit -u always wins, and is labeled 'remote', even on a non-Apple-Silicon platform."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_generate_response(
        {"response": "a red square", "eval_count": 3, "done_reason": "stop"}
    )
    img = tmp_path / "photo.jpg"
    _make_image(img)

    result = CliRunner().invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url])

    assert result.exit_code == 0
    assert f"Backend: remote Ollama ({mock_ollama.url})" in result.output


def test_cli_apple_silicon_defaults_to_local_mlx_label(tmp_path, monkeypatch):
    """A cache hit means MLX never actually has to load, so this only checks label/selection."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: True)
    img = tmp_path / "photo.jpg"
    _make_image(img)
    cache = {
        "version": ri.CACHE_VERSION,
        "entries": {
            "photo.jpg": {
                "checksum": ri.file_checksum(img),
                "model": f"local:{ri.DEFAULT_LOCAL_MODEL}",
                "desc": "cached-desc",
            }
        },
    }
    (tmp_path / ri.CACHE_FILENAME).write_text(json.dumps(cache))

    result = CliRunner().invoke(ri.cli, [str(tmp_path)])

    assert result.exit_code == 0
    assert "Backend: local MLX" in result.output


# ---------- default-model hardware fallback ----------


def test_choose_ollama_default_model_stays_default_when_it_fits(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 24.0)
    assert ri._choose_ollama_default_model(is_local=True) == ri.DEFAULT_REMOTE_MODEL


def test_choose_ollama_default_model_downgrades_when_it_does_not_fit(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 4.0)
    monkeypatch.setattr(ri, "_detect_system_ram_gb", lambda: None)
    assert ri._choose_ollama_default_model(is_local=True) == ri.FALLBACK_REMOTE_MODEL


def test_choose_ollama_default_model_ignores_local_hardware_for_explicit_remote(
    monkeypatch,
):
    """A genuinely remote -u host's capacity isn't this machine's to guess at."""
    monkeypatch.setattr(
        ri, "_detect_nvidia_vram_gb", lambda: 0.1
    )  # would downgrade if consulted
    assert ri._choose_ollama_default_model(is_local=False) == ri.DEFAULT_REMOTE_MODEL


def test_choose_ollama_default_model_survives_catalog_drift(monkeypatch):
    """If DEFAULT_REMOTE_MODEL and OLLAMA_MODELS ever drift apart, don't crash — keep the default."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "DEFAULT_REMOTE_MODEL", "some-model-not-in-catalog")
    monkeypatch.setattr(
        ri, "_detect_nvidia_vram_gb", lambda: 0.1
    )  # would downgrade if consulted and found
    assert ri._choose_ollama_default_model(is_local=True) == "some-model-not-in-catalog"


def test_cli_downgrades_default_model_when_local_hardware_cant_fit_it(
    tmp_path, mock_ollama, monkeypatch
):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "DEFAULT_OLLAMA_URL", mock_ollama.url)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 4.0)
    monkeypatch.setattr(ri, "_detect_system_ram_gb", lambda: None)
    mock_ollama.set_models([ri.FALLBACK_REMOTE_MODEL])
    mock_ollama.set_generate_response(
        {"response": "a red square", "eval_count": 3, "done_reason": "stop"}
    )
    img = tmp_path / "photo.jpg"
    _make_image(img)

    result = CliRunner().invoke(ri.cli, [str(tmp_path)])

    assert result.exit_code == 0
    assert f"using {ri.FALLBACK_REMOTE_MODEL!r} instead" in result.output
    assert (
        mock_ollama.server.state["generate_calls"][0]["model"]
        == ri.FALLBACK_REMOTE_MODEL
    )


def test_cli_keeps_default_model_when_local_hardware_fits_it(
    tmp_path, mock_ollama, monkeypatch
):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "DEFAULT_OLLAMA_URL", mock_ollama.url)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 24.0)
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_generate_response(
        {"response": "a red square", "eval_count": 3, "done_reason": "stop"}
    )
    img = tmp_path / "photo.jpg"
    _make_image(img)

    result = CliRunner().invoke(ri.cli, [str(tmp_path)])

    assert result.exit_code == 0
    assert "Note: default model" not in result.output
    assert (
        mock_ollama.server.state["generate_calls"][0]["model"]
        == ri.DEFAULT_REMOTE_MODEL
    )


def test_cli_does_not_downgrade_for_explicit_remote_host(
    tmp_path, mock_ollama, monkeypatch
):
    """This machine's own (tiny) hardware must not override an explicit, genuinely remote -u host."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 0.5)
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_generate_response(
        {"response": "a red square", "eval_count": 3, "done_reason": "stop"}
    )
    img = tmp_path / "photo.jpg"
    _make_image(img)

    result = CliRunner().invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url])

    assert result.exit_code == 0
    assert "Note: default model" not in result.output
    assert (
        mock_ollama.server.state["generate_calls"][0]["model"]
        == ri.DEFAULT_REMOTE_MODEL
    )


# ---------- CLI ----------


def test_cli_reports_no_images(tmp_path):
    result = CliRunner().invoke(ri.cli, [str(tmp_path)])
    assert result.exit_code == 0
    assert "No images found." in result.output


def test_cli_skips_hardware_detection_when_no_images_found(tmp_path, monkeypatch):
    """An empty folder must not pay for a hardware-detection subprocess call it'll never use."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)

    def _fail_if_called():
        raise AssertionError("hardware detection should not run on an empty folder")

    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: _fail_if_called())
    monkeypatch.setattr(ri, "_detect_amd_vram_gb", lambda: _fail_if_called())
    monkeypatch.setattr(ri, "_detect_system_ram_gb", lambda: _fail_if_called())

    result = CliRunner().invoke(ri.cli, [str(tmp_path)])

    assert result.exit_code == 0
    assert "No images found." in result.output


def test_cli_defaults_to_rename_subcommand(tmp_path):
    """A bare folder arg (no 'rename') must still resolve to the rename command."""
    result = CliRunner().invoke(ri.cli, [str(tmp_path)])
    assert result.exit_code == 0
    assert "No images found." in result.output


def test_cli_explicit_rename_subcommand_also_works(tmp_path):
    result = CliRunner().invoke(ri.cli, ["rename", str(tmp_path)])
    assert result.exit_code == 0
    assert "No images found." in result.output


def test_cli_version_flag_prints_version_and_exits(tmp_path):
    """--version must reach the group's own option, not get redirected to the default
    'rename' subcommand and treated as a (nonexistent) folder argument."""
    result = CliRunner().invoke(ri.cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_cli_help_lists_version_option(tmp_path):
    result = CliRunner().invoke(ri.cli, ["--help"])
    assert result.exit_code == 0
    assert "--version" in result.output


def test_cli_dry_run_then_apply_reuses_cache_over_remote_backend(tmp_path, mock_ollama):
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_generate_response(
        {"response": "a red square", "eval_count": 3, "done_reason": "stop"}
    )
    img = tmp_path / "photo.jpg"
    _make_image(img)
    runner = CliRunner()

    dry_run = runner.invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url])
    assert dry_run.exit_code == 0
    assert "a-red-square" in dry_run.output
    assert mock_ollama.generate_call_count == 1
    assert img.exists()  # dry run must not touch the filesystem
    assert (tmp_path / ri.CACHE_FILENAME).exists()

    apply_run = runner.invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url, "-a"])
    assert apply_run.exit_code == 0
    assert "[cached]" in apply_run.output
    assert mock_ollama.generate_call_count == 1  # no new network call — cache was used
    assert not img.exists()
    assert len(list(tmp_path.glob("*-a-red-square.jpg"))) == 1


def test_cli_workers_maps_concurrent_results_to_correct_images(tmp_path, mock_ollama):
    """With -w > 1, results must still land on the right image, not get mixed up across threads."""
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])

    # Distinguishable-by-size "images" (generate_remote() just reads+b64-encodes
    # raw bytes, so these don't need to be real images) — the mock server keys
    # its response off each request's payload size, standing in for "identity".
    contents = {
        "a.jpg": b"MARKER-A" * 100,
        "b.jpg": b"MARKER-B" * 200,
        "c.jpg": b"MARKER-C" * 300,
    }
    for name, content in contents.items():
        (tmp_path / name).write_bytes(content)

    expected_desc = {name: f"desc for {name}" for name in contents}
    b64len_to_desc = {
        len(base64.b64encode(content)): expected_desc[name]
        for name, content in contents.items()
    }

    def response_fn(payload):
        b64len = len(payload["images"][0])
        return {
            "response": b64len_to_desc[b64len],
            "eval_count": 1,
            "done_reason": "stop",
        }

    mock_ollama.set_generate_response_fn(response_fn)

    result = CliRunner().invoke(
        ri.cli, [str(tmp_path), "-u", mock_ollama.url, "-w", "3"]
    )

    assert result.exit_code == 0
    assert mock_ollama.generate_call_count == 3
    for name in contents:
        assert f"{name}  ->" in result.output
        expected_slug = ri.slugify(expected_desc[name])
        # each image's own line must contain its own description, not another's
        line = next(
            line_
            for line_ in result.output.splitlines()
            if line_.strip().startswith(name)
        )
        assert expected_slug in line


def test_cli_workers_reports_progress_as_requests_complete(tmp_path, mock_ollama):
    """Regression test: results must be reported as they complete, not only after the whole batch finishes."""
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    mock_ollama.set_generate_response(
        {"response": "a scene", "eval_count": 2, "done_reason": "stop"}
    )

    result = CliRunner().invoke(
        ri.cli, [str(tmp_path), "-u", mock_ollama.url, "-w", "2"]
    )

    assert result.exit_code == 0
    for i in range(1, 4):
        assert f"[{i}/3]" in result.output


def test_cli_workers_partial_failure_skips_only_the_failing_image(
    tmp_path, mock_ollama
):
    """One failing image among several concurrent requests must not affect the others."""
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    good_content = b"GOOD" * 100
    bad_content = b"BAD" * 200
    (tmp_path / "good.jpg").write_bytes(good_content)
    (tmp_path / "bad.jpg").write_bytes(bad_content)

    bad_b64len = len(base64.b64encode(bad_content))

    def response_fn(payload):
        if len(payload["images"][0]) == bad_b64len:
            return 500, {"error": "model error"}
        return {"response": "a good image", "eval_count": 1, "done_reason": "stop"}

    mock_ollama.set_generate_response_fn(response_fn)

    result = CliRunner().invoke(
        ri.cli, [str(tmp_path), "-u", mock_ollama.url, "-w", "2"]
    )

    assert result.exit_code == 0
    assert "[SKIP] bad.jpg" in result.output
    assert "good.jpg  ->  " in result.output
    assert "a-good-image" in result.output


def test_cli_rename_caches_exif_data(tmp_path, mock_ollama):
    """The rename flow must also populate the cache's "exif" field for every image."""
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_generate_response(
        {"response": "a red square", "eval_count": 3, "done_reason": "stop"}
    )
    img = tmp_path / "photo.jpg"
    _make_image(img)

    result = CliRunner().invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url])
    assert result.exit_code == 0

    cache = ri.load_cache(tmp_path)
    entry = cache["entries"]["photo.jpg"]
    assert entry["exif"] == ri.get_exif_data(img)


# ---------- exif command ----------


def test_exif_cmd_on_single_file_table(tmp_path):
    img = tmp_path / "photo.jpg"
    _make_image(img)

    result = CliRunner().invoke(ri.cli, ["exif", str(img)])

    assert result.exit_code == 0
    assert "photo.jpg" in result.output


def test_exif_cmd_on_single_file_json(tmp_path):
    img = tmp_path / "photo.jpg"
    _make_image(img)

    result = CliRunner().invoke(ri.cli, ["exif", str(img), "-f", "json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert list(data.keys()) == ["photo.jpg"]


def test_exif_cmd_on_directory_recursive_json(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_image(tmp_path / "top.jpg")
    _make_image(sub / "nested.jpg")

    non_recursive = CliRunner().invoke(ri.cli, ["exif", str(tmp_path), "-f", "json"])
    assert json.loads(non_recursive.output).keys() == {"top.jpg"}

    recursive = CliRunner().invoke(ri.cli, ["exif", str(tmp_path), "-r", "-f", "json"])
    keys = json.loads(recursive.output).keys()
    assert keys == {"top.jpg", str(Path("sub") / "nested.jpg")}


def test_exif_cmd_reports_no_images(tmp_path):
    result = CliRunner().invoke(ri.cli, ["exif", str(tmp_path)])
    assert result.exit_code == 0
    assert "No images found." in result.output


def test_exif_cmd_omits_maker_note_unless_flag_passed(tmp_path):
    img = tmp_path / "photo.jpg"
    _make_image_with_maker_note(img)

    default_run = CliRunner().invoke(ri.cli, ["exif", str(img), "-f", "json"])
    assert "MakerNote" not in json.loads(default_run.output)["photo.jpg"]

    with_flag = CliRunner().invoke(ri.cli, ["exif", str(img), "-M", "-f", "json"])
    assert json.loads(with_flag.output)["photo.jpg"]["MakerNote"] == "PROPRIETARYDATA"


def test_exif_cmd_omits_user_comment_unless_flag_passed(tmp_path):
    img = tmp_path / "photo.jpg"
    _make_image_with_user_comment(img)

    default_run = CliRunner().invoke(ri.cli, ["exif", str(img), "-f", "json"])
    assert "UserComment" not in json.loads(default_run.output)["photo.jpg"]

    with_flag = CliRunner().invoke(ri.cli, ["exif", str(img), "-U", "-f", "json"])
    assert "UserComment" in json.loads(with_flag.output)["photo.jpg"]


# ---------- hardware detection ----------


def test_detect_nvidia_vram_gb_picks_largest_of_multiple_gpus(monkeypatch):
    """Ollama runs a model on a single GPU, so the largest card is the right capacity estimate
    — not the first line of nvidia-smi's output, which is arbitrary device-enumeration order."""
    monkeypatch.setattr(ri.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(ri, "_run_command_stdout", lambda cmd: "4096\n24576\n8192")

    assert ri._detect_nvidia_vram_gb() == 24576 / 1024


def test_detect_nvidia_vram_gb_none_when_no_gpu_lines(monkeypatch):
    monkeypatch.setattr(ri.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(ri, "_run_command_stdout", lambda cmd: "")

    assert ri._detect_nvidia_vram_gb() is None


def _make_drm_card(root, name, vram_bytes=None, vendor="0x1002"):
    """Build one fake /sys/class/drm/<name>/device entry; vram_bytes=None omits the VRAM file."""
    device = root / name / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(f"{vendor}\n")
    if vram_bytes is not None:
        (device / "mem_info_vram_total").write_text(f"{vram_bytes}\n")
    return device


def test_detect_amd_vram_gb_picks_dgpu_over_integrated(tmp_path, monkeypatch):
    """The real numbers from an APU + dGPU machine: the 0.5GB iGPU carve-out must not win."""
    _make_drm_card(tmp_path, "card0", 536870912)  # Raphael integrated graphics
    _make_drm_card(tmp_path, "card1", 17095983104)  # Radeon RX 9070
    monkeypatch.setattr(ri, "_DRM_SYSFS_ROOT", tmp_path)

    assert ri._detect_amd_vram_gb() == 17095983104 / (1024**3)


def test_detect_amd_vram_gb_none_when_only_integrated_graphics(tmp_path, monkeypatch):
    """An APU works out of system RAM, so its tiny carve-out must fall through to the RAM path."""
    _make_drm_card(tmp_path, "card0", 536870912)
    monkeypatch.setattr(ri, "_DRM_SYSFS_ROOT", tmp_path)

    assert ri._detect_amd_vram_gb() is None


def test_detect_amd_vram_gb_ignores_other_vendors(tmp_path, monkeypatch):
    _make_drm_card(tmp_path, "card0", 17095983104, vendor="0x10de")
    monkeypatch.setattr(ri, "_DRM_SYSFS_ROOT", tmp_path)

    assert ri._detect_amd_vram_gb() is None


def test_detect_amd_vram_gb_none_on_missing_or_unreadable_sysfs(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "_DRM_SYSFS_ROOT", tmp_path / "does-not-exist")
    assert ri._detect_amd_vram_gb() is None

    # A card directory with no VRAM attribute at all (non-amdgpu driver), and
    # one whose attribute isn't a number — neither may raise.
    _make_drm_card(tmp_path, "card0")
    device = _make_drm_card(tmp_path, "card1", 17095983104)
    (device / "mem_info_vram_total").write_text("not-a-number\n")
    monkeypatch.setattr(ri, "_DRM_SYSFS_ROOT", tmp_path)
    assert ri._detect_amd_vram_gb() is None


def test_detect_hardware_reports_amd_gpu_when_no_nvidia(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_amd_vram_gb", lambda: 16.0)

    hw = ri._detect_hardware()

    assert hw.capacity == 16.0
    assert hw.gpu_vendor == "amd"
    assert "AMD GPU" in hw.summary
    assert "VRAM: 16 GB" in hw.summary


def test_detect_hardware_prefers_nvidia_when_both_present(monkeypatch):
    """NVIDIA is probed first so machines that already worked keep their existing behavior."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 24.0)
    monkeypatch.setattr(ri, "_detect_amd_vram_gb", lambda: 16.0)

    hw = ri._detect_hardware()

    assert (hw.capacity, hw.gpu_vendor) == (24.0, "nvidia")


# ---------- Ollama CPU-fallback warning ----------


@pytest.fixture
def amd_gpu(monkeypatch):
    """Pretend this machine has a 16GB AMD GPU, whatever it actually has."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_amd_vram_gb", lambda: 16.0)


def test_cpu_fallback_warning_when_model_is_not_in_vram(mock_ollama, amd_gpu):
    mock_ollama.set_loaded("llava:13b", size=8_000_000_000, size_vram=0)

    warning = ri._ollama_cpu_fallback_warning(mock_ollama.url, "llava:13b")

    assert warning is not None
    assert "entirely on the CPU" in warning
    assert "AMD GPU (16 GB)" in warning
    assert "ollama-rocm" in warning  # the actual fix, not just the diagnosis


def test_no_cpu_fallback_warning_when_model_is_on_the_gpu(mock_ollama, amd_gpu):
    mock_ollama.set_loaded("llava:13b", size=8_000_000_000, size_vram=8_000_000_000)

    assert ri._ollama_cpu_fallback_warning(mock_ollama.url, "llava:13b") is None


def test_no_cpu_fallback_warning_when_model_not_loaded(mock_ollama, amd_gpu):
    """Nothing loaded (or a different model loaded) is ambiguous, not evidence of CPU inference."""
    assert ri._ollama_cpu_fallback_warning(mock_ollama.url, "llava:13b") is None

    mock_ollama.set_loaded("some-other-model", size=8_000_000_000, size_vram=0)
    assert ri._ollama_cpu_fallback_warning(mock_ollama.url, "llava:13b") is None


def test_no_cpu_fallback_warning_without_a_gpu_to_use(mock_ollama, monkeypatch):
    """CPU inference on a machine with no GPU is expected — warning about it would be noise."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_amd_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_system_ram_gb", lambda: 32.0)
    mock_ollama.set_loaded("llava:13b", size=8_000_000_000, size_vram=0)

    assert ri._ollama_cpu_fallback_warning(mock_ollama.url, "llava:13b") is None


def test_cpu_fallback_warning_blames_model_size_when_it_cannot_fit(
    mock_ollama, amd_gpu
):
    """A GPU-enabled Ollama reports size_vram == 0 too when not one layer of the model fits."""
    mock_ollama.set_loaded(
        "llava:34b", size=20_000_000_000, size_vram=0
    )  # ~26GB on a 16GB card

    warning = ri._ollama_cpu_fallback_warning(mock_ollama.url, "llava:34b")

    assert warning is not None
    assert "more than this GPU has" in warning
    assert "rename-images models" in warning


def test_no_cpu_fallback_warning_on_a_malformed_ps_response(mock_ollama, amd_gpu):
    """Go marshals a nil slice as null — parsing must not raise out and take the batch with it."""
    for payload in ({"models": None}, {"models": ["not-a-dict"]}, {}, [], "nonsense"):
        mock_ollama.set_ps(payload)
        assert ri._ollama_cpu_fallback_warning(mock_ollama.url, "llava:13b") is None


def test_no_cpu_fallback_warning_when_server_cannot_be_reached(amd_gpu):
    """A diagnostic must never be the thing that breaks a run."""
    assert ri._ollama_cpu_fallback_warning("http://127.0.0.1:1", "llava:13b") is None


def test_cli_warns_once_about_cpu_fallback_on_local_backend(
    tmp_path, mock_ollama, monkeypatch, amd_gpu
):
    monkeypatch.setattr(ri, "DEFAULT_OLLAMA_URL", mock_ollama.url)
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_loaded(ri.DEFAULT_REMOTE_MODEL, size=8_000_000_000, size_vram=0)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _make_image(tmp_path / name)

    result = CliRunner().invoke(ri.cli, [str(tmp_path)])

    assert result.exit_code == 0
    assert result.stderr.count("entirely on the CPU") == 1


def test_cli_does_not_warn_about_cpu_fallback_for_explicit_remote_host(
    tmp_path, mock_ollama, amd_gpu
):
    """What this machine's GPU is doing says nothing about an explicitly-targeted host."""
    mock_ollama.set_models([ri.DEFAULT_REMOTE_MODEL])
    mock_ollama.set_loaded(ri.DEFAULT_REMOTE_MODEL, size=8_000_000_000, size_vram=0)
    _make_image(tmp_path / "photo.jpg")

    result = CliRunner().invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url])

    assert result.exit_code == 0
    assert "entirely on the CPU" not in result.output


# ---------- fit / best-match helpers ----------


def test_fit_yes_tight_no_unknown_bands():
    assert ri._fit(None, 10) == "?"
    assert ri._fit(10, 10) == "yes"
    assert ri._fit(8, 10) == "tight"  # 8 >= 10 * 0.75
    assert ri._fit(5, 10) == "no"


def test_fit_color_green_for_best_match_regardless_of_fit_band():
    assert ri._fit_color("tight", is_best_match=True) == "green"
    assert ri._fit_color("yes", is_best_match=True) == "green"


def test_fit_color_red_for_unfit_non_best_match():
    assert ri._fit_color("no", is_best_match=False) == "red"


def test_fit_color_none_for_fitting_or_unknown_non_best_match():
    assert ri._fit_color("yes", is_best_match=False) is None
    assert ri._fit_color("tight", is_best_match=False) is None
    assert ri._fit_color("?", is_best_match=False) is None


def test_best_match_name_picks_largest_fitting_model():
    rows = [
        {"name": "small", "fits": "yes", "approx_gb_needed": 2},
        {"name": "medium", "fits": "yes", "approx_gb_needed": 6},
        {"name": "big", "fits": "no", "approx_gb_needed": 24},
    ]
    assert ri._best_match_name(rows) == "medium"


def test_best_match_name_none_when_nothing_fits():
    rows = [{"name": "big", "fits": "no", "approx_gb_needed": 24}]
    assert ri._best_match_name(rows) is None


# ---------- models command ----------


def test_models_cmd_lists_mlx_models_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(ri, "_detect_apple_memory_gb", lambda: 16.0)

    result = CliRunner().invoke(ri.cli, ["models"])

    assert result.exit_code == 0
    assert "Apple Silicon Mac" in result.output
    assert "unified memory: 16 GB" in result.output
    assert ri.DEFAULT_LOCAL_MODEL in result.output
    assert "qwen2.5vl:7b" not in result.output  # an Ollama-only tag


def test_models_cmd_lists_ollama_models_with_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 8.0)

    result = CliRunner().invoke(ri.cli, ["models"])

    assert result.exit_code == 0
    assert "NVIDIA GPU" in result.output
    assert "qwen2.5vl:7b" in result.output
    assert "fits:yes" in result.output  # 8GB covers the 6GB-rated 7b tag
    assert "fits:no" in result.output  # 8GB doesn't cover the 24GB-rated 32b tag


def test_models_cmd_sizes_against_amd_vram_not_system_ram(monkeypatch, amd_gpu):
    """The bug this exists to prevent: a 16GB card being handed a 34B recommendation
    because the only capacity figure available was 30GB of system RAM."""
    monkeypatch.setattr(ri, "_detect_system_ram_gb", lambda: 30.0)

    result = CliRunner().invoke(ri.cli, ["models", "-f", "json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "AMD GPU" in data["hardware"]
    fits = {m["name"]: m["fits"] for m in data["models"]}
    assert fits["llava:13b"] == "yes"  # 16GB comfortably covers the ~10GB default
    assert fits["llava:34b"] == "no"  # ...and clearly doesn't cover a 34B
    assert [m["name"] for m in data["models"] if m["best_match"]] == ["llava:13b"]


def test_models_cmd_falls_back_to_cpu_ram_when_no_gpu_detected(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_amd_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_system_ram_gb", lambda: 32.0)

    result = CliRunner().invoke(ri.cli, ["models"])

    assert result.exit_code == 0
    assert "no GPU detected" in result.output
    assert "system RAM: 32 GB" in result.output


def test_models_cmd_unknown_hardware_marks_every_model_fit_unknown(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_amd_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_system_ram_gb", lambda: None)

    result = CliRunner().invoke(ri.cli, ["models"])

    assert result.exit_code == 0
    assert "RAM unknown" in result.output
    assert "fits:?" in result.output
    assert "fits:yes" not in result.output


def test_models_cmd_json_output_structure(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_amd_vram_gb", lambda: None)
    monkeypatch.setattr(ri, "_detect_system_ram_gb", lambda: None)

    result = CliRunner().invoke(ri.cli, ["models", "-f", "json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["ecosystem"] == "Ollama"
    assert data["models"]
    assert all(m["fits"] == "?" for m in data["models"])
    assert {"name", "params", "approx_gb_needed", "fits", "notes"} <= data["models"][
        0
    ].keys()


def test_models_cmd_json_flags_best_match(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 8.0)

    result = CliRunner().invoke(ri.cli, ["models", "-f", "json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    # With 8GB detected: everything up to minicpm-v (7GB) fits; it's the largest that does.
    best = [m["name"] for m in data["models"] if m["best_match"]]
    assert best == ["minicpm-v"]


def test_models_cmd_table_colors_best_match_green_and_unfit_red(monkeypatch):
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 8.0)

    result = CliRunner().invoke(ri.cli, ["models"], color=True)

    assert result.exit_code == 0
    green_lines = [line for line in result.output.splitlines() if "\x1b[32m" in line]
    assert any("minicpm-v" in line for line in green_lines)
    red_lines = [line for line in result.output.splitlines() if "\x1b[31m" in line]
    assert any("llava:34b" in line for line in red_lines)
    assert not any("minicpm-v" in line for line in red_lines)


def test_models_cmd_strips_color_by_default_for_non_tty_output(monkeypatch):
    """Without an explicit tty, output must stay plain text — no leaked ANSI escapes."""
    monkeypatch.setattr(ri, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ri, "_detect_nvidia_vram_gb", lambda: 8.0)

    result = CliRunner().invoke(ri.cli, ["models"])

    assert result.exit_code == 0
    assert "\x1b[" not in result.output
