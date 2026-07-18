"""Basic functionality tests for rename_images.py.

Covers the pure helper functions plus the cache and remote (Ollama) backend,
using a mock HTTP server. The local MLX backend isn't exercised here — it
needs real Apple Silicon hardware and a multi-GB model download, so it's
out of scope for an automated suite; the remote backend covers the same
generation/cache/CLI code paths against a lightweight stand-in server.
"""

import base64
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
    from PIL import ExifTags

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
    from PIL import ExifTags

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


def test_cli_dry_run_then_apply_reuses_cache_over_remote_backend(tmp_path, mock_ollama):
    mock_ollama.set_models(["qwen2.5vl:7b"])
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
    mock_ollama.set_models(["qwen2.5vl:7b"])

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
        len(base64.b64encode(content)): expected_desc[name] for name, content in contents.items()
    }

    def response_fn(payload):
        b64len = len(payload["images"][0])
        return {"response": b64len_to_desc[b64len], "eval_count": 1, "done_reason": "stop"}

    mock_ollama.set_generate_response_fn(response_fn)

    result = CliRunner().invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url, "-w", "3"])

    assert result.exit_code == 0
    assert mock_ollama.generate_call_count == 3
    for name in contents:
        assert f"{name}  ->" in result.output
        expected_slug = ri.slugify(expected_desc[name])
        # each image's own line must contain its own description, not another's
        line = next(line_ for line_ in result.output.splitlines() if line_.strip().startswith(name))
        assert expected_slug in line


def test_cli_workers_reports_progress_as_requests_complete(tmp_path, mock_ollama):
    """Regression test: results must be reported as they complete, not only after the whole batch finishes."""
    mock_ollama.set_models(["qwen2.5vl:7b"])
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    mock_ollama.set_generate_response({"response": "a scene", "eval_count": 2, "done_reason": "stop"})

    result = CliRunner().invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url, "-w", "2"])

    assert result.exit_code == 0
    for i in range(1, 4):
        assert f"[{i}/3]" in result.output


def test_cli_workers_partial_failure_skips_only_the_failing_image(tmp_path, mock_ollama):
    """One failing image among several concurrent requests must not affect the others."""
    mock_ollama.set_models(["qwen2.5vl:7b"])
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

    result = CliRunner().invoke(ri.cli, [str(tmp_path), "-u", mock_ollama.url, "-w", "2"])

    assert result.exit_code == 0
    assert "[SKIP] bad.jpg" in result.output
    assert "good.jpg  ->  " in result.output
    assert "a-good-image" in result.output


def test_cli_rename_caches_exif_data(tmp_path, mock_ollama):
    """The rename flow must also populate the cache's "exif" field for every image."""
    mock_ollama.set_models(["qwen2.5vl:7b"])
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
