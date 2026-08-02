"""Contract tests for the fixed-token warmup/repeat benchmark."""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark_qwen25_fixed_warmup import (
    _stats,
    build_unavailable_artifact,
    expected_row_ids,
    load_warmup_config,
    run_warmup,
    validate_warmup_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmarks" / "qwen25-fixed-warmup-v1.json"


def _completed_fixture(config: dict) -> dict:
    artifact = build_unavailable_artifact(config, config_path=CONFIG, reason="fixture")
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
        backend = row["backend"]
        dispatch = {
            "standard_attention_calls": standard_calls if backend == "baseline" else 0,
            "standard_prefill_calls": config["repeats"] * 28
            if backend != "baseline"
            else 0,
            "packed_decode_calls": layer_calls
            if backend in {"serial_direct", "parallel_direct"}
            else 0,
            "packed_reference_decode_calls": layer_calls
            if backend == "packed_reference"
            else 0,
            "parallel_decode_calls": layer_calls if backend == "parallel_direct" else 0,
            "parallel_page_launches": layer_calls
            if backend == "parallel_direct"
            else 0,
            "parallel_reduction_launches": layer_calls
            if backend == "parallel_direct"
            else 0,
            "parallel_tail_launches": layer_calls
            if backend == "parallel_direct"
            else 0,
            "fallback": False,
        }
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
                    "tail_capacity_bytes": 1 if backend == "parallel_direct" else 0,
                    "workspace_bytes": 1 if backend == "parallel_direct" else 0,
                    "unique_resident_bytes": 2 if backend == "parallel_direct" else 1,
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


def test_config_declares_fixed_warmup_matrix() -> None:
    config = load_warmup_config(CONFIG)
    assert config["contexts"] == [1024, 2048]
    assert config["stream_seed"] == 26009
    assert config["warmup_tokens"] == 1
    assert config["measured_decode_tokens"] == 3
    assert config["repeats"] == 3
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


def test_unavailable_artifact_is_explicit_and_valid() -> None:
    config = load_warmup_config(CONFIG)
    artifact = build_unavailable_artifact(
        config, config_path=CONFIG, reason="missing fixture"
    )
    result = validate_warmup_artifact(artifact, config)
    assert result.valid, result.errors
    assert artifact["status"] == "unavailable"
    assert artifact["claims"] == []
    assert len(artifact["rows"]) == 8


def test_stats_reports_p50_p95_and_mean() -> None:
    stats = _stats([1.0, 2.0, 3.0, 4.0])
    assert stats["p50_ms"] == 2.5
    assert stats["p95_ms"] == 3.85
    assert stats["mean_ms"] == 2.5


def test_validator_accepts_consistent_completed_fixture() -> None:
    config = load_warmup_config(CONFIG)
    artifact = _completed_fixture(config)
    result = validate_warmup_artifact(artifact, config)
    assert result.valid, result.errors


def test_validator_rejects_gate_and_dispatch_mutations() -> None:
    config = load_warmup_config(CONFIG)
    artifact = _completed_fixture(config)
    artifact["rows"][1]["quality"]["max_abs_logit"] = 0.02
    result = validate_warmup_artifact(artifact, config)
    assert not result.valid
    assert any("quality_gate_pass" in error for error in result.errors)

    artifact = _completed_fixture(config)
    artifact["rows"][3]["dispatch"]["parallel_tail_launches"] = 0
    result = validate_warmup_artifact(artifact, config)
    assert not result.valid
    assert any("parallel parallel_tail_launches" in error for error in result.errors)

    artifact = _completed_fixture(config)
    artifact["rows"][0]["timing"]["decode_p95_ms"] = 0.0
    result = validate_warmup_artifact(artifact, config)
    assert not result.valid
    assert any("decode_p95_ms disagrees" in error for error in result.errors)

    artifact = _completed_fixture(config)
    artifact["source_identity"]["git_commit"] = "unknown"
    result = validate_warmup_artifact(artifact, config)
    assert not result.valid
    assert any("git_commit" in error for error in result.errors)


def test_validator_returns_errors_for_malformed_completed_rows() -> None:
    config = load_warmup_config(CONFIG)
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
    result = validate_warmup_artifact(artifact, config)
    assert not result.valid
    assert result.errors


def test_runner_fail_closes_mismatched_fixture_path(tmp_path: Path) -> None:
    output = tmp_path / "warmup.json"
    artifact = run_warmup(CONFIG, output, model_path=tmp_path)
    result = validate_warmup_artifact(artifact)
    assert result.valid, result.errors
    assert artifact["status"] == "unavailable"
    assert all(row["status"] == "unavailable" for row in artifact["rows"])
    assert all("fixture rejected" in row["reason"] for row in artifact["rows"])
