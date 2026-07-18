"""Binding and source-coverage contracts for qualified evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

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
        "kernels/packed_attention.py",
        "configs/profiles/qwen25-7b-direct-v1.json",
        "configs/models/qwen25-7b.json",
    }

    assert required <= set(repro_utils.DIRECT_EVIDENCE_SOURCE_PATHS)
