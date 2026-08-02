"""Contracts for the serial page-attention diagnostic benchmark."""

from __future__ import annotations

import copy
from scripts.benchmark_parallel_page_attention import (
    DEFAULT_PAGE_COUNTS,
    MEASURED_TRIALS,
    WARMUP_RUNS,
    validate_parallel_page_artifact,
)


def _metrics(page_count: int) -> dict[str, object]:
    samples = [float(page_count + index) for index in range(MEASURED_TRIALS)]
    return {
        "page_count": page_count,
        "launch_counts": {
            "rotation": 1,
            "page": page_count,
            "reduction": 0,
            "tail": 1,
            "fallback": 0,
            "allocation": 4 + 3 * page_count,
        },
        "timing": {
            "cuda_synchronized": True,
            "cold_jit_runs": 1,
            "warmup_runs": WARMUP_RUNS,
            "measured_trials": MEASURED_TRIALS,
            "cuda_event_ms": samples,
            "p50_ms": 4.5 + page_count,
            "p95_ms": 8.55 + page_count,
            "p99_ms": 8.91 + page_count,
        },
        "memory": {
            "logical_payload_bytes": page_count * 100,
            "capacity_bytes": page_count * 100,
            "runtime_bytes": 1024,
            "workspace_bytes": 1024,
            "unique_resident_bytes": page_count * 100 + 1024,
        },
        "parity": {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "top1_agreement": 1.0,
            "repeat_bit_identical": True,
        },
        "diagnostic_only": True,
    }


def _artifact() -> dict[str, object]:
    rows = [
        {
            "row_id": f"serial-pages-{page_count}",
            "status": "passed",
            "reason": None,
            "identity": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "weights_sha256": "a" * 64,
                "profile_id": "qwen25-7b-direct-v1",
                "profile_sha256": "b" * 64,
                "kv_dtype": "float16",
            },
            "metrics": _metrics(page_count),
            "sidecars": [],
        }
        for page_count in DEFAULT_PAGE_COUNTS
    ]
    return {
        "artifact_type": "parallel_page_attention_diagnostic",
        "schema_version": 2,
        "artifact_id": "parallel-page-baseline",
        "status": "partial",
        "diagnostic_only": True,
        "paper": {
            "paper_id": "arxiv:2504.19874v1",
            "source_url": "https://arxiv.org/pdf/2504.19874v1",
            "retrieved_at": "2026-07-31T06:23:40Z",
            "sha256": "c" * 64,
        },
        "source": {"git_commit": "d" * 40, "dirty": True},
        "execution_class": {
            "name": "local_rtx5070ti_12gb",
            "gpu_name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
            "total_vram_bytes": 12_820_480_000,
            "driver_version": "610.47",
            "cuda_runtime": "12.8",
        },
        "identity": rows[0]["identity"],
        "statistics": {
            "method": "CUDA events; percentile interpolation; diagnostic only",
            "confidence_level": 0.95,
            "seed": 17,
            "iterations": MEASURED_TRIALS,
        },
        "expected_row_ids": [f"serial-pages-{count}" for count in DEFAULT_PAGE_COUNTS],
        "rows": rows,
        "pass_state_counts": {
            "passed": len(rows),
            "failed": 0,
            "invalid": 0,
            "oom": 0,
            "unsupported": 0,
            "unavailable": 0,
        },
        "claims": [],
        "extensions": {"implementation": "serial"},
    }


def test_serial_contract_accepts_complete_diagnostic_artifact() -> None:
    result = validate_parallel_page_artifact(_artifact())
    assert result.valid, result.errors


def test_contract_rejects_missing_page_row() -> None:
    artifact = _artifact()
    artifact["rows"] = artifact["rows"][:-1]  # type: ignore[index]
    result = validate_parallel_page_artifact(artifact)
    assert not result.valid
    assert any("missing expected row IDs" in error for error in result.errors)


def test_contract_rejects_fallback_and_unsynchronized_or_short_timing() -> None:
    artifact = _artifact()
    row = copy.deepcopy(artifact["rows"][0])  # type: ignore[index]
    row["metrics"]["launch_counts"]["fallback"] = 1
    row["metrics"]["timing"]["cuda_synchronized"] = False
    row["metrics"]["timing"]["cuda_event_ms"] = [1.0]
    artifact["rows"][0] = row  # type: ignore[index]
    result = validate_parallel_page_artifact(artifact)
    assert not result.valid
    assert any("fallback" in error for error in result.errors)
    assert any("cuda_synchronized" in error for error in result.errors)
    assert any("measured_trials" in error for error in result.errors)


def test_contract_rejects_non_diagnostic_win_claim() -> None:
    artifact = _artifact()
    artifact["claims"] = [
        {
            "claim_id": "speed-win",
            "artifact_id": artifact["artifact_id"],
            "row_id": "serial-pages-31",
            "query": "metrics.timing.p50_ms",
        }
    ]
    result = validate_parallel_page_artifact(artifact)
    assert not result.valid
    assert any("diagnostic" in error for error in result.errors)
