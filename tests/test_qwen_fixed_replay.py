"""Contract tests for the validator-first fixed-token replay artifact."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.replay_qwen25_fixed_tokens import (
    _direct_memory,
    build_unavailable_artifact,
    expected_row_ids,
    load_replay_config,
    validate_replay_artifact,
)
from core.packed_cache import PackedDynamicCache
from core.profiles import load_profile


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmarks" / "qwen25-fixed-replay-v1.json"


def test_config_declares_exact_replay_matrix() -> None:
    config = load_replay_config(CONFIG)
    assert config["contexts"] == [1024, 2048]
    assert config["backends"] == [
        "baseline",
        "packed_reference",
        "serial_direct",
        "parallel_direct",
    ]
    assert expected_row_ids(config) == [
        "1024-baseline",
        "1024-packed_reference",
        "1024-serial_direct",
        "1024-parallel_direct",
        "2048-baseline",
        "2048-packed_reference",
        "2048-serial_direct",
        "2048-parallel_direct",
    ]


def test_unavailable_fixture_is_complete_and_explicit() -> None:
    config = load_replay_config(CONFIG)
    artifact = build_unavailable_artifact(
        config,
        config_path=CONFIG,
        reason="local pinned model fixture is unavailable",
    )
    result = validate_replay_artifact(artifact, config)
    assert result.valid, result.errors
    assert artifact["status"] == "unavailable"
    assert artifact["diagnostic_only"] is True
    assert artifact["claims"] == []
    assert len(artifact["rows"]) == 8
    assert all(row["status"] == "unavailable" for row in artifact["rows"])


def test_validator_rejects_missing_row_and_claims() -> None:
    config = load_replay_config(CONFIG)
    artifact = build_unavailable_artifact(config, config_path=CONFIG, reason="missing")
    artifact["rows"].pop()
    artifact["claims"] = ["speedup"]
    result = validate_replay_artifact(artifact, config)
    assert not result.valid
    assert any("rows must exactly" in error for error in result.errors)
    assert any("claims" in error for error in result.errors)


def test_validator_requires_metrics_for_completed_rows() -> None:
    config = load_replay_config(CONFIG)
    artifact = build_unavailable_artifact(config, config_path=CONFIG, reason="fixture")
    row = artifact["rows"][0]
    row.update({"status": "completed"})
    artifact["status"] = "partial"
    result = validate_replay_artifact(artifact, config)
    assert not result.valid
    assert any("completed quality" in error for error in result.errors)
    assert any("completed memory" in error for error in result.errors)


def test_fixed_stream_and_quality_metrics_are_deterministic() -> None:
    from scripts.replay_qwen25_fixed_tokens import _fixed_stream, _quality

    class FakeTokenizer:
        vocab_size = 101

        def __call__(self, text: str, *, return_tensors: str):
            _ = text, return_tensors
            return type(
                "Encoded", (), {"input_ids": torch.arange(1000).reshape(1, 1000)}
            )()

    tokenizer = FakeTokenizer()
    first = _fixed_stream(tokenizer, 32, 26009)
    second = _fixed_stream(tokenizer, 32, 26009)
    assert torch.equal(first, second)
    logits = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    metrics = _quality(logits, logits)
    assert metrics == {
        "max_abs_logit": 0.0,
        "mean_abs_logit": 0.0,
        "kl_divergence": 0.0,
        "top1_agreement": 1.0,
        "greedy_agreement": True,
    }


def test_direct_memory_uses_one_logical_formula_for_arena_store() -> None:
    profile = load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")
    cache = PackedDynamicCache(profile, layer_count=1, maximum_sequence_length=640)
    keys = torch.randn((1, 4, 640, 128), dtype=torch.float16)
    cache.update(keys, keys.clone(), 0)
    session = type(
        "Session",
        (),
        {
            "capabilities": type("Capabilities", (), {"layer_count": 1})(),
            "cache": cache,
            "_parallel_workspace": None,
        },
    )()
    memory = _direct_memory(session, parallel=True)
    arena = cache.store_for(0).arena
    assert arena is not None
    expected_logical = (
        arena.byte_breakdown().logical_payload + arena.byte_breakdown().tail_logical
    )
    assert memory["logical_payload_bytes"] == expected_logical


@pytest.mark.parametrize(
    "field", ["artifact_type", "schema_version", "diagnostic_only"]
)
def test_validator_rejects_identity_contract_breaks(field: str) -> None:
    config = load_replay_config(CONFIG)
    artifact = build_unavailable_artifact(config, config_path=CONFIG, reason="fixture")
    artifact[field] = {
        "artifact_type": "wrong",
        "schema_version": 0,
        "diagnostic_only": False,
    }[field]
    result = validate_replay_artifact(artifact, config)
    assert not result.valid
