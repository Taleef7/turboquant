"""Binding and source-coverage contracts for qualified evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import pytest

import scripts.repro_utils as repro_utils


ROOT = Path(__file__).resolve().parents[1]


def test_binding_uses_pinned_hub_identity_not_local_manifest_as_model_hash() -> None:
    binding = repro_utils.benchmark_binding(ROOT)

    assert binding["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert binding["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert binding["model_revision_sha256"] == hashlib.sha256(
        b"Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28"
    ).hexdigest()
    assert "model_sha256" not in binding


def test_binding_code_hash_covers_direct_codec_and_adapter_surface() -> None:
    required = {
        "core/codec.py",
        "core/packing.py",
        "core/reference_attention.py",
        "core/profiles.py",
        "core/packed_store.py",
        "core/packed_cache.py",
        "core/qwen_adapter.py",
        "core/turboquant_simple.py",
        "kernels/packed_attention.py",
        "utils/math_utils.py",
        "configs/profiles/qwen25-7b-direct-v1.json",
        "configs/models/qwen25-7b.json",
    }

    assert required <= set(repro_utils.DIRECT_EVIDENCE_SOURCE_PATHS)


@pytest.mark.parametrize(
    "relative_path", ("core/turboquant_simple.py", "utils/math_utils.py")
)
def test_binding_code_hash_changes_when_a_direct_source_changes(
    tmp_path: Path, relative_path: str
) -> None:
    for relative_path in repro_utils.DIRECT_EVIDENCE_SOURCE_PATHS:
        source = ROOT / relative_path
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for codebook_name in ("codebook_d128_b8.json", "codebook_d128_b6.json"):
        destination = tmp_path / "codebooks" / codebook_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "codebooks" / codebook_name, destination)

    before = repro_utils.benchmark_binding(tmp_path)
    source = tmp_path / relative_path
    source.write_text(source.read_text(encoding="utf-8") + "\n# binding mutation\n", encoding="utf-8")
    after = repro_utils.benchmark_binding(tmp_path)

    assert after["code_sha256"] != before["code_sha256"]
