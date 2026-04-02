"""Unit tests for reproducibility utility helpers."""

from pathlib import Path

from repro_utils import capture_env_metadata, write_json, write_markdown


def test_capture_env_metadata_has_core_fields():
    meta = capture_env_metadata()
    assert "timestamp_utc" in meta
    assert "git_commit" in meta
    assert "torch_version" in meta
    assert "cuda_available" in meta


def test_write_json_and_markdown_create_parent_dirs(tmp_path: Path):
    json_path = tmp_path / "nested" / "report.json"
    md_path = tmp_path / "nested" / "report.md"

    write_json(str(json_path), {"ok": True})
    write_markdown(str(md_path), "# title\n")

    assert json_path.exists()
    assert md_path.exists()
    assert '"ok": true' in json_path.read_text(encoding="utf-8")
    assert md_path.read_text(encoding="utf-8").startswith("# title")
