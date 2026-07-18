"""Validation tests for quality, systems, and release evidence artifacts."""

from __future__ import annotations

from scripts.benchmark_schema import (
    validate_artifact,
    validate_qualified_run,
    validate_systems_artifact,
)
from tests.test_benchmark_schema import _binding, complete_matrix


def _systems_artifact() -> dict:
    return {
        "artifact_type": "systems_benchmark",
        "schema_version": 1,
        "binding": _binding(),
        "status": "completed",
        "metadata": {"timestamp_utc": "2026-07-18T00:00:00+00:00", "git_commit": "test"},
        "expected_backends": ["baseline", "packed_reference", "triton_direct"],
        "trials": [
            {
                "backend": backend,
                "status": "completed",
                "oom": False,
                "exception": None,
                "actual_store_bytes": 123,
                "physical_bytes": {"keys": 1, "values": 2, "metadata": 3},
                "max_memory_allocated": 4,
                "max_memory_reserved": 5,
                "prefill_ms": 6.0,
                "ttft_ms": 7.0,
                "decode_token_ms": [8.0, 9.0],
                "decode_p50_ms": 8.5,
                "decode_p95_ms": 8.95,
                "decode_tokens_per_second": 120.0,
                "dispatch_counts": {"attention": 1},
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
