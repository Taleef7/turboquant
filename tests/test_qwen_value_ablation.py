"""Contract tests for the opt-in Qwen value-bit ablation artifact."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_qwen25_value_ablation import (
    CONFIG_PATH,
    build_unavailable_artifact,
    expected_row_ids,
    load_ablation_config,
    run_ablation,
    validate_ablation_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _completed_fixture(config: dict) -> dict:
    artifact = build_unavailable_artifact(
        config, config_path=CONFIG_PATH, reason="fixture"
    )
    artifact["status"] = "diagnostic"
    layer_calls = (
        (config["warmup_tokens"] + config["measured_decode_tokens"])
        * config["repeats"]
        * 28
    )
    standard_calls = (
        (1 + config["warmup_tokens"] + config["measured_decode_tokens"])
        * config["repeats"]
        * 28
    )
    sample_count = config["repeats"] * config["measured_decode_tokens"]
    steady_count = config["repeats"] * (config["measured_decode_tokens"] - 1)
    for row in artifact["rows"]:
        baseline = row["profile_id"] == "baseline"
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
                "dispatch": {
                    "standard_attention_calls": standard_calls if baseline else 0,
                    "standard_prefill_calls": config["repeats"] * 28
                    if baseline
                    else 0,
                    "packed_decode_calls": 0 if baseline else layer_calls,
                    "packed_reference_decode_calls": 0,
                    "parallel_decode_calls": 0,
                    "parallel_page_launches": 0,
                    "parallel_reduction_launches": 0,
                    "parallel_tail_launches": 0,
                    "fallback": False,
                },
                "memory": {
                    "logical_payload_bytes": 1,
                    "physical_cache_bytes": 1,
                    "arena_capacity_bytes": 0,
                    "tail_capacity_bytes": 0,
                    "workspace_bytes": 0,
                    "unique_resident_bytes": 1,
                },
                "timing": {
                    "prefill_ms": [1.0] * config["repeats"],
                    "warmup_ms": [1.0] * config["repeats"],
                    "first_run_decode_ms": [1.0] * config["repeats"],
                    "steady_state_decode_ms": [1.0] * steady_count,
                    "decode_token_ms": [1.0] * sample_count,
                    "decode_p50_ms": 1.0,
                    "decode_p95_ms": 1.0,
                    "decode_mean_ms": 1.0,
                    "first_run_p50_ms": 1.0,
                    "first_run_p95_ms": 1.0,
                    "steady_state_p50_ms": 1.0,
                    "steady_state_p95_ms": 1.0,
                },
            }
        )
    return artifact


def test_config_binds_v6_and_v8_profiles_on_long_context_matrix() -> None:
    config = load_ablation_config()
    assert config["contexts"] == [4096, 8192]
    assert [entry["id"] for entry in config["profiles"]] == ["v6", "v8"]
    assert [entry["value_bits"] for entry in config["profiles"]] == [6, 8]
    assert expected_row_ids(config) == [
        "4096-baseline",
        "4096-v6-serial_direct",
        "4096-v8-serial_direct",
        "8192-baseline",
        "8192-v6-serial_direct",
        "8192-v8-serial_direct",
    ]


def test_unavailable_artifact_is_complete_and_valid() -> None:
    config = load_ablation_config()
    artifact = build_unavailable_artifact(
        config, config_path=CONFIG_PATH, reason="missing fixture"
    )
    result = validate_ablation_artifact(artifact, config)
    assert result.valid, result.errors
    assert artifact["claims"] == []
    assert len(artifact["rows"]) == 6


def test_validator_accepts_consistent_completed_fixture() -> None:
    config = load_ablation_config()
    result = validate_ablation_artifact(_completed_fixture(config), config)
    assert result.valid, result.errors


def test_validator_rejects_profile_and_parallel_dispatch_mutations() -> None:
    config = load_ablation_config()
    artifact = _completed_fixture(config)
    artifact["rows"][1]["profile"]["value_bits"] = 8
    result = validate_ablation_artifact(artifact, config)
    assert not result.valid
    assert any("profile binding" in error for error in result.errors)

    artifact = _completed_fixture(config)
    artifact["rows"][2]["dispatch"]["parallel_decode_calls"] = 1
    result = validate_ablation_artifact(artifact, config)
    assert not result.valid
    assert any("parallel dispatch" in error for error in result.errors)


def test_validator_rejects_model_fixture_provenance_mutation() -> None:
    config = load_ablation_config()
    artifact = _completed_fixture(config)
    source = artifact["source_identity"]
    source["model_fixture"] = {
        "model_id": config["model_id"],
        "snapshot_id": "wrong-revision",
        "resolved_path": "/tmp/wrong-snapshot",
        "manifest_sha256": "0" * 64,
    }
    for row in artifact["rows"]:
        row["source_identity"] = source
    result = validate_ablation_artifact(artifact, config)
    assert not result.valid
    assert any("model_fixture snapshot_id" in error for error in result.errors)
    assert any("model_fixture manifest" in error for error in result.errors)

    artifact = _completed_fixture(config)
    artifact["profile_bindings"][1]["value_bits"] = 6
    result = validate_ablation_artifact(artifact, config)
    assert not result.valid
    assert any("profile_bindings" in error for error in result.errors)

    artifact = _completed_fixture(config)
    artifact["source_identity"]["runner_sha256"] = "0" * 64
    result = validate_ablation_artifact(artifact, config)
    assert not result.valid
    assert any("runner_sha256" in error for error in result.errors)


def test_config_rejects_profile_file_hash_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["profiles"][0]["file_sha256"] = "0" * 64
    candidate = tmp_path / "ablation.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_ablation_config(candidate)
    except ValueError as exc:
        assert "profile file hash mismatch" in str(exc)
    else:
        raise AssertionError("accepted a mutated profile file hash")


def test_runner_fail_closes_mismatched_fixture_path(tmp_path: Path) -> None:
    output = tmp_path / "ablation.json"
    artifact = run_ablation(CONFIG_PATH, output, model_path=tmp_path)
    config = load_ablation_config()
    result = validate_ablation_artifact(artifact, config)
    assert result.valid, result.errors
    assert artifact["status"] == "unavailable"
    assert all(row["status"] == "unavailable" for row in artifact["rows"])
