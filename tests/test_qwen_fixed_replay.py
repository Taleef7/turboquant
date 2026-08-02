"""Contract tests for the validator-first fixed-token replay artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.replay_qwen25_fixed_tokens import (
    _direct_memory,
    _fixture_identity,
    build_unavailable_artifact,
    expected_row_ids,
    load_replay_config,
    validate_replay_artifact,
)
from core.packed_cache import PackedDynamicCache
from core.profiles import load_profile


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmarks" / "qwen25-fixed-replay-v1.json"


def _write_snapshot_fixture(root: Path, *, revision: str | None = None) -> Path:
    config = load_replay_config(CONFIG)
    snapshot = (
        root
        / "models--Qwen--Qwen2.5-7B-Instruct"
        / "snapshots"
        / (revision or config["model_revision"])
    )
    snapshot.mkdir(parents=True)
    model_config = {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
    }
    (snapshot / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"fixture")
    return snapshot


def _completed_fixture(config: dict) -> dict:
    """Build a small, internally consistent completed artifact for validator tests."""
    artifact = build_unavailable_artifact(config, config_path=CONFIG, reason="fixture")
    artifact["status"] = "diagnostic"
    layer_calls = int(config["decode_tokens"]) * 28
    for row in artifact["rows"]:
        backend = row["backend"]
        dispatch = {
            "standard_prefill_calls": 0,
            "packed_decode_calls": 0,
            "packed_reference_decode_calls": 0,
            "parallel_decode_calls": 0,
            "parallel_page_launches": 0,
            "parallel_reduction_launches": 0,
            "parallel_tail_launches": 0,
            "fallback": False,
        }
        if backend == "baseline":
            dispatch["standard_attention_calls"] = layer_calls + 28
        elif backend == "packed_reference":
            dispatch["packed_reference_decode_calls"] = layer_calls
        elif backend == "serial_direct":
            dispatch["packed_decode_calls"] = layer_calls
        else:
            dispatch.update(
                {
                    "packed_decode_calls": layer_calls,
                    "parallel_decode_calls": layer_calls,
                    "parallel_page_launches": layer_calls,
                    "parallel_reduction_launches": layer_calls,
                    "parallel_tail_launches": layer_calls,
                }
            )
        row.update(
            {
                "status": "completed",
                "quality": {
                    "max_abs_logit": 0.0,
                    "mean_abs_logit": 0.0,
                    "kl_divergence": 0.0,
                    "top1_agreement": 1.0,
                    "greedy_agreement": True,
                },
                "quality_gate_pass": True,
                "dispatch": dispatch,
                "memory": {
                    "logical_payload_bytes": 1,
                    "physical_cache_bytes": 1,
                    "arena_capacity_bytes": 1 if backend == "parallel_direct" else 0,
                    "tail_capacity_bytes": 0,
                    "workspace_bytes": 1 if backend == "parallel_direct" else 0,
                    "unique_resident_bytes": 2 if backend == "parallel_direct" else 1,
                },
                "timing": {
                    "prefill_ms": 1.0,
                    "decode_token_ms": [1.0] * int(config["decode_tokens"]),
                    "decode_p50_ms": 1.0,
                    "decode_mean_ms": 1.0,
                },
            }
        )
    return artifact


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


def test_quality_clamps_tiny_negative_kl_roundoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.nn.functional as F

    from scripts.replay_qwen25_fixed_tokens import _quality

    monkeypatch.setattr(
        F,
        "kl_div",
        lambda *args, **kwargs: torch.tensor(-8e-9),
    )
    logits = [torch.tensor([[1.0, 0.0]])]
    metrics = _quality(logits, logits)
    assert metrics["kl_divergence"] == 0.0


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
    expected_physical = (
        arena._unique_storage_bytes() + cache.store_for(0).physical_nbytes().workspace
    )
    assert memory["physical_cache_bytes"] == expected_physical
    assert memory["unique_resident_bytes"] == expected_physical


def test_validator_accepts_consistent_completed_fixture() -> None:
    config = load_replay_config(CONFIG)
    artifact = _completed_fixture(config)
    result = validate_replay_artifact(artifact, config)
    assert result.valid, result.errors


def test_validator_rejects_quality_gate_mutation() -> None:
    config = load_replay_config(CONFIG)
    artifact = _completed_fixture(config)
    artifact["rows"][1]["quality"]["max_abs_logit"] = 0.02
    result = validate_replay_artifact(artifact, config)
    assert not result.valid
    assert any("quality_gate_pass" in error for error in result.errors)


def test_validator_rejects_parallel_dispatch_mutation() -> None:
    config = load_replay_config(CONFIG)
    artifact = _completed_fixture(config)
    artifact["rows"][3]["dispatch"]["parallel_tail_launches"] = 0
    result = validate_replay_artifact(artifact, config)
    assert not result.valid
    assert any("parallel parallel_tail_launches" in error for error in result.errors)


def test_validator_rejects_memory_undercount_mutation() -> None:
    config = load_replay_config(CONFIG)
    artifact = _completed_fixture(config)
    artifact["rows"][3]["memory"]["unique_resident_bytes"] = 0
    result = validate_replay_artifact(artifact, config)
    assert not result.valid
    assert any("undercount physical" in error for error in result.errors)


def test_validator_returns_errors_for_malformed_completed_rows() -> None:
    config = load_replay_config(CONFIG)
    artifact = build_unavailable_artifact(config, config_path=CONFIG, reason="fixture")
    artifact["status"] = "diagnostic"
    for row in artifact["rows"]:
        row.update(
            {
                "status": "completed",
                "quality": {},
                "quality_gate_pass": False,
                "dispatch": {},
                "memory": {},
                "timing": {},
            }
        )
    result = validate_replay_artifact(artifact, config)
    assert not result.valid
    assert result.errors


def test_fixture_identity_rejects_unpinned_local_path(tmp_path: Path) -> None:
    config = load_replay_config(CONFIG)
    snapshot = _write_snapshot_fixture(tmp_path, revision="wrong-revision")
    identity, reason = _fixture_identity(snapshot, config)
    assert identity is None
    assert reason is not None
    assert "pinned HF snapshot layout" in reason


def test_fixture_identity_rejects_architecture_mismatch(tmp_path: Path) -> None:
    config = load_replay_config(CONFIG)
    snapshot = _write_snapshot_fixture(tmp_path)
    config_payload = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config_payload["num_hidden_layers"] = 32
    (snapshot / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
    identity, reason = _fixture_identity(snapshot, config)
    assert identity is None
    assert reason is not None
    assert "architecture mismatch" in reason


def test_fixture_identity_records_manifest_for_pinned_snapshot(tmp_path: Path) -> None:
    config = load_replay_config(CONFIG)
    snapshot = _write_snapshot_fixture(tmp_path)
    # The synthetic fixture has its own digest; disable the production digest
    # only to exercise manifest construction and recording in isolation.
    identity, reason = _fixture_identity(
        snapshot, {**config, "fixture_manifest_sha256": None}
    )
    assert reason is None
    assert identity is not None
    assert identity["snapshot_id"] == config["model_revision"]
    assert identity["resolved_path"] == str(snapshot.resolve())
    assert identity["manifest_sha256"]
    assert {entry["path"] for entry in identity["files"]} == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
    }


def test_fixture_identity_rejects_manifest_mismatch(tmp_path: Path) -> None:
    config = load_replay_config(CONFIG)
    snapshot = _write_snapshot_fixture(tmp_path)
    identity, reason = _fixture_identity(snapshot, config)
    assert identity is None
    assert reason is not None
    assert "manifest SHA-256" in reason


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
