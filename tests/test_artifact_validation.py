"""Validation tests for quality, systems, and release evidence artifacts."""

from __future__ import annotations

from scripts.benchmark_schema import (
    validate_artifact,
    validate_qualified_run,
    validate_systems_artifact,
)
from tests.test_benchmark_schema import _binding, complete_matrix


def _qualified_report() -> dict:
    quality = complete_matrix()
    quality["status"] = "qualified"
    systems = _systems_artifact()
    decision = {
        "preflight_valid": True, "eight_k_retrieval_safe": True,
        "eight_k_systems_valid": True, "eight_k_capacity_admitted": True,
        "status": "approved", "attempt_32k": True,
        "reason": "preflight_and_8k_gate_passed",
    }
    return {
        "artifact_type": "qualified_run", "schema_version": 1, "binding": _binding(),
        "metadata": {"timestamp_utc": "2026-07-18T00:00:00+00:00", "git_commit": "a" * 40},
        "mode": "qualified", "status": "qualified",
        "quality": {"status": "qualified", "artifact": quality},
        "systems": {"status": "completed", "artifact": systems},
        "qualified_32k_decision": decision,
        "legs": {"32k": {"quality": {"status": "qualified", "artifact": quality}, "systems": {"structurally_valid": True, "artifact": systems}}},
    }


def _systems_artifact() -> dict:
    return {
        "artifact_type": "systems_benchmark",
        "schema_version": 1,
        "binding": _binding(),
        "status": "completed",
        "metadata": {
            "timestamp_utc": "2026-07-18T00:00:00+00:00",
            "git_commit": "a" * 40,
        },
        "expected_backends": ["baseline", "packed_reference", "triton_direct"],
        "trials": [
            {
                "backend": backend,
                "status": "completed",
                "oom": False,
                "exception": None,
                "actual_store_bytes": 6 if backend != "baseline" else 0,
                "physical_bytes": {
                    "keys": 1 if backend != "baseline" else 0,
                    "values": 2 if backend != "baseline" else 0,
                    "metadata": 3 if backend != "baseline" else 0,
                    "fp16_window": 0,
                    "padding": 0,
                },
                "max_memory_allocated": 4,
                "max_memory_reserved": 5,
                "prefill_ms": 6.0,
                "ttft_ms": 7.0,
                "decode_token_ms": [8.0, 9.0],
                "decode_p50_ms": 8.5,
                "decode_p95_ms": 8.95,
                "decode_tokens_per_second": 120.0,
                "dispatch_counts": (
                    {"standard_attention": 1}
                    if backend == "baseline"
                    else {
                        "triton_direct": 1 if backend == "triton_direct" else 0,
                        "packed_reference": 1 if backend == "packed_reference" else 0,
                        "packed_blocks_consumed": 1 if backend == "triton_direct" else 0,
                    }
                ),
            }
            for backend in ["baseline", "packed_reference", "triton_direct"]
        ],
        "warmups": [
            {"backend": backend, "status": "completed", "oom": False, "exception": None}
            for backend in ["baseline", "packed_reference", "triton_direct"]
        ],
        "oom_boundary": {"attempted_contexts": [1024, 4096, 8192], "status": "not_reached_within_configured_probe_range", "probes": [{"context_tokens": c, "status": "completed", "oom": False} for c in [1024, 4096, 8192]]},
    }


def test_systems_artifact_requires_every_backend_and_physical_measurement():
    artifact = _systems_artifact()
    artifact["trials"].pop()
    artifact["trials"][0]["actual_store_bytes"] = None

    result = validate_systems_artifact(artifact)

    assert result.valid is False
    assert any("expected backend" in error for error in result.errors)
    assert any("actual_store_bytes" in error for error in result.errors)


def test_systems_artifact_rejects_oom_and_exception():
    artifact = _systems_artifact()
    artifact["trials"][0]["oom"] = True
    artifact["trials"][1]["exception"] = "boom"

    result = validate_artifact(artifact)

    assert result.valid is False
    assert any("OOM" in error for error in result.errors)
    assert any("exception" in error for error in result.errors)


def test_validator_rejects_changed_binding_hashes():
    matrix = complete_matrix()
    matrix["binding"]["codebook_sha256"] = "wrong"

    result = validate_artifact(matrix, expected_binding=_binding())

    assert result.valid is False
    assert any("codebook_sha256" in error for error in result.errors)


def test_systems_validator_rejects_runner_source_drift_and_nonfinite_metrics():
    artifact = _systems_artifact()
    expected = _binding()
    artifact["binding"]["runner_sha256"] = "changed"
    artifact["trials"][0]["prefill_ms"] = float("nan")
    artifact["trials"][1]["decode_token_ms"] = []

    result = validate_systems_artifact(artifact, expected)

    assert result.valid is False
    assert any("runner_sha256" in error for error in result.errors)
    assert any("finite" in error for error in result.errors)
    assert any("non-empty" in error for error in result.errors)


def test_systems_validator_rejects_missing_or_failed_warmup():
    artifact = _systems_artifact()
    artifact["warmups"][0]["status"] = "error"

    result = validate_systems_artifact(artifact)

    assert result.valid is False
    assert any("warmup" in error for error in result.errors)


def test_systems_validator_requires_two_point_oom_probe_or_explicit_range_result():
    artifact = _systems_artifact()
    artifact["oom_boundary"] = {"attempted_contexts": [1024], "status": "not_reached_within_configured_probe_range"}

    result = validate_systems_artifact(artifact)

    assert result.valid is False
    assert any("at least two" in error for error in result.errors)


def test_systems_validator_rejects_oom_boundary_without_probe_outcomes():
    artifact = _systems_artifact()
    artifact["oom_boundary"].pop("probes")
    result = validate_systems_artifact(artifact)
    assert result.valid is False
    assert any("probes" in error for error in result.errors)


def test_systems_validator_rejects_fake_direct_dispatch_and_byte_mismatch():
    artifact = _systems_artifact()
    direct = next(trial for trial in artifact["trials"] if trial["backend"] == "triton_direct")
    reference = next(
        trial for trial in artifact["trials"] if trial["backend"] == "packed_reference"
    )
    direct["dispatch_counts"]["triton_direct"] = 0
    direct["dispatch_counts"]["packed_reference"] = 1
    direct["dispatch_counts"]["packed_blocks_consumed"] = 0
    direct["actual_store_bytes"] = 7
    reference["dispatch_counts"]["triton_direct"] = 1

    result = validate_systems_artifact(artifact)

    assert result.valid is False
    assert any("triton_direct dispatches" in error for error in result.errors)
    assert any("packed_reference dispatches" in error for error in result.errors)
    assert any("packed_blocks_consumed" in error for error in result.errors)
    assert any("actual_store_bytes" in error for error in result.errors)


def test_systems_validator_requires_every_physical_byte_category_and_full_total():
    for category in ("fp16_window", "padding"):
        artifact = _systems_artifact()
        direct = next(
            trial for trial in artifact["trials"] if trial["backend"] == "triton_direct"
        )
        direct["physical_bytes"].pop(category)

        result = validate_systems_artifact(artifact)

        assert result.valid is False
        assert any(category in error for error in result.errors)

    artifact = _systems_artifact()
    direct = next(
        trial for trial in artifact["trials"] if trial["backend"] == "triton_direct"
    )
    direct["physical_bytes"]["fp16_window"] = 4

    result = validate_systems_artifact(artifact)

    assert result.valid is False
    assert any("actual_store_bytes" in error for error in result.errors)


def test_validator_rejects_unknown_git_commit_for_evidence():
    matrix = complete_matrix()
    matrix["metadata"]["git_commit"] = "unknown"

    result = validate_artifact(matrix)

    assert result.valid is False
    assert any("git_commit" in error for error in result.errors)


def test_qualified_run_cannot_pass_when_child_is_missing_or_invalid():
    report = {
        "artifact_type": "qualified_run",
        "schema_version": 1,
        "binding": _binding(),
        "mode": "qualified",
        "status": "qualified",
        "quality": {"status": "valid", "artifact": complete_matrix()},
        "systems": {"status": "missing", "artifact": None},
    }

    result = validate_qualified_run(report)

    assert result.valid is False
    assert any("systems" in error for error in result.errors)


def test_qualified_run_rejects_unknown_top_level_git_commit():
    report = {
        "artifact_type": "qualified_run",
        "schema_version": 1,
        "binding": _binding(),
        "metadata": {
            "timestamp_utc": "2026-07-18T00:00:00+00:00",
            "git_commit": "unknown",
        },
        "mode": "qualified",
        "status": "qualified",
        "quality": {"status": "qualified", "artifact": complete_matrix()},
        "systems": {"status": "completed", "artifact": _systems_artifact()},
    }

    result = validate_qualified_run(report)

    assert result.valid is False
    assert any("release metadata.git_commit" in error for error in result.errors)


def test_qualified_run_rejects_skipped_32k_bypass_and_bad_decision_reason():
    report = _qualified_report()
    report["qualified_32k_decision"].update({
        "eight_k_capacity_admitted": False, "status": "skipped",
        "attempt_32k": False, "reason": "8k_systems_invalid_or_failed",
    })
    report["legs"] = {}

    result = validate_qualified_run(report)

    assert result.valid is False
    assert any("approved 32K decision" in error for error in result.errors)
    assert any("reason does not match" in error for error in result.errors)
    assert any("32K child legs" in error for error in result.errors)


def test_qualified_run_accepts_a_valid_approved_32k_ladder():
    assert validate_qualified_run(_qualified_report()).valid is True
