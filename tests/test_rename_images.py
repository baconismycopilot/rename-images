"""Basic functionality tests for rename_images.py.

Covers the pure helper functions plus the cache and remote (Ollama) backend,
using a mock HTTP server. The local MLX backend isn't exercised here — it
needs real Apple Silicon hardware and a multi-GB model download, so it's
out of scope for an automated suite; the remote backend covers the same
generation/cache/CLI code paths against a lightweight stand-in server.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

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
    (tmp_path / ri.CACHE_FILENAME).write_text(json.dumps({"version": 999, "entries": {"a": {}}}))
    assert ri.load_cache(tmp_path) == {"version": ri.CACHE_VERSION, "entries": {}}


# ---------- get_photo_date ----------


def test_get_photo_date_falls_back_to_file_time_when_no_exif(tmp_path):
    path = tmp_path / "plain.png"
    _make_image(path)
    date = ri.get_photo_date(path)
    assert date.year >= 2020


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


# ---------- CLI ----------


def test_cli_reports_no_images(tmp_path):
    result = CliRunner().invoke(ri.main, [str(tmp_path)])
    assert result.exit_code == 0
    assert "No images found." in result.output


def test_cli_dry_run_then_apply_reuses_cache_over_remote_backend(tmp_path, mock_ollama):
    mock_ollama.set_models(["qwen2.5vl:7b"])
    mock_ollama.set_generate_response(
        {"response": "a red square", "eval_count": 3, "done_reason": "stop"}
    )
    img = tmp_path / "photo.jpg"
    _make_image(img)
    runner = CliRunner()

    dry_run = runner.invoke(ri.main, [str(tmp_path), "-u", mock_ollama.url])
    assert dry_run.exit_code == 0
    assert "a-red-square" in dry_run.output
    assert mock_ollama.generate_call_count == 1
    assert img.exists()  # dry run must not touch the filesystem
    assert (tmp_path / ri.CACHE_FILENAME).exists()

    apply_run = runner.invoke(ri.main, [str(tmp_path), "-u", mock_ollama.url, "-a"])
    assert apply_run.exit_code == 0
    assert "[cached]" in apply_run.output
    assert mock_ollama.generate_call_count == 1  # no new network call — cache was used
    assert not img.exists()
    assert len(list(tmp_path.glob("*-a-red-square.jpg"))) == 1
