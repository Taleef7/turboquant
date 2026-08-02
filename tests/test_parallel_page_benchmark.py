"""Contracts for the serial page-attention diagnostic benchmark."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from scripts.benchmark_parallel_page_attention import (
    DEFAULT_PAGE_COUNTS,
    MEASURED_TRIALS,
    _parallel_arena_tail_views,
    _parallel_page_workspace_nbytes,
    _synthetic_arena,
    WARMUP_RUNS,
    validate_parallel_page_artifact,
)
from scripts.benchmark_schema import validate_artifact


def _metrics(page_count: int) -> dict[str, object]:
    samples = [float(page_count + index) for index in range(MEASURED_TRIALS)]
    return {
        "page_count": page_count,
        "launch_counts": {
            "rotation": 1,
            "page": page_count,
            "reduction": 0,
            "tail": 1,
            "reduce_tail": 0,
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
            "tail_logical_bytes": 0,
            "tail_capacity_bytes": 0,
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


def test_page_state_contract_accepts_one_grid_launch_per_page_set() -> None:
    artifact = _artifact()
    artifact["artifact_id"] = "parallel-page-state"
    artifact["extensions"] = {
        "implementation": "page-state",
        "implementations": ["page-state"],
    }
    artifact["expected_row_ids"] = [
        f"page-state-pages-{page_count}" for page_count in DEFAULT_PAGE_COUNTS
    ]
    rows = []
    for row in artifact["rows"]:  # type: ignore[index]
        row = copy.deepcopy(row)
        row["row_id"] = row["row_id"].replace("serial-", "page-state-")
        row["metrics"]["launch_counts"].update(  # type: ignore[index]
            {"page": 1, "tail": 0}
        )
        rows.append(row)
    artifact["rows"] = rows
    result = validate_parallel_page_artifact(artifact)
    assert result.valid, result.errors


def test_parallel_contract_requires_reduction_and_tail_launches() -> None:
    artifact = _artifact()
    artifact["artifact_id"] = "parallel-page-complete"
    artifact["extensions"] = {
        "implementation": "parallel",
        "implementations": ["parallel"],
    }
    artifact["expected_row_ids"] = [
        f"parallel-pages-{page_count}" for page_count in DEFAULT_PAGE_COUNTS
    ]
    rows = []
    for row in artifact["rows"]:  # type: ignore[index]
        row = copy.deepcopy(row)
        row["row_id"] = row["row_id"].replace("serial-", "parallel-")
        row["metrics"]["launch_counts"].update(  # type: ignore[index]
            {
                "page": 1,
                "reduction": 0,
                "tail": 0,
                "reduce_tail": 1,
                "allocation": 0,
            }
        )
        rows.append(row)
    artifact["rows"] = rows
    result = validate_parallel_page_artifact(artifact)
    assert result.valid, result.errors
    rows[0]["metrics"]["launch_counts"]["reduce_tail"] = 0  # type: ignore[index]
    result = validate_parallel_page_artifact(artifact)
    assert not result.valid
    assert any("reduce_tail" in error for error in result.errors)


def test_mixed_contract_rejects_inconsistent_serial_parallel_tail_accounting() -> None:
    artifact = _artifact()
    parallel_rows = []
    for row in artifact["rows"]:  # type: ignore[index]
        row = copy.deepcopy(row)
        row["row_id"] = row["row_id"].replace("serial-", "parallel-")
        row["metrics"]["launch_counts"].update(  # type: ignore[index]
            {
                "page": 1,
                "reduction": 0,
                "tail": 0,
                "reduce_tail": 1,
                "allocation": 0,
            }
        )
        row["metrics"]["memory"]["tail_logical_bytes"] = 256  # type: ignore[index]
        row["metrics"]["memory"]["tail_capacity_bytes"] = 256  # type: ignore[index]
        parallel_rows.append(row)
    artifact["rows"] = artifact["rows"] + parallel_rows  # type: ignore[operator]
    artifact["extensions"] = {
        "implementation": "mixed",
        "implementations": ["serial", "parallel"],
    }
    artifact["expected_row_ids"] = [
        f"{implementation}-pages-{page_count}"
        for implementation in ("serial", "parallel")
        for page_count in DEFAULT_PAGE_COUNTS
    ]
    artifact["pass_state_counts"]["passed"] = len(artifact["expected_row_ids"])  # type: ignore[index]
    result = validate_parallel_page_artifact(artifact)
    assert not result.valid
    assert any("tail logical-byte accounting" in error for error in result.errors)


@pytest.mark.gpu
def test_parallel_benchmark_tail_views_are_arena_owned() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    arena, _blocks_cpu, _prepared = _synthetic_arena(
        1, seed=2222, device=torch.device("cuda")
    )
    keys = torch.randn((1, 4, 8, 128), device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    arena.append_tail(keys, values)
    view_keys, view_values = _parallel_arena_tail_views(arena)
    assert view_keys.data_ptr() == arena.tail_keys.data_ptr()
    assert view_values.data_ptr() == arena.tail_values.data_ptr()
    assert view_keys.stride(1) == arena.tail_keys.stride(1)
    assert view_keys.stride(2) == arena.tail_keys.stride(2)
    assert view_values.stride(1) == arena.tail_values.stride(1)
    assert view_values.stride(2) == arena.tail_values.stride(2)


def test_page_state_workspace_bytes_include_all_reduction_buffers() -> None:
    workspace = type(
        "Workspace",
        (),
        {
            "rotated_query": torch.empty((28, 128), dtype=torch.float32),
            "page_max": torch.empty((3, 28), dtype=torch.float32),
            "page_denominator": torch.empty((3, 28), dtype=torch.float32),
            "page_numerator": torch.empty((3, 28, 128), dtype=torch.float32),
            "reduced_max": torch.empty((28,), dtype=torch.float32),
            "reduced_denominator": torch.empty((28,), dtype=torch.float32),
            "reduced_numerator": torch.empty((28, 128), dtype=torch.float32),
        },
    )()
    assert _parallel_page_workspace_nbytes(workspace) == 72_576


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


def test_documented_generic_validator_enforces_diagnostic_contract() -> None:
    artifact = _artifact()
    result = validate_artifact(artifact, repo_root=Path.cwd())
    assert result.valid, result.errors
    artifact["rows"][0]["metrics"]["launch_counts"]["fallback"] = 1  # type: ignore[index]
    result = validate_artifact(artifact, repo_root=Path.cwd())
    assert not result.valid
    assert any("launch_counts.fallback" in error for error in result.errors)


def test_documented_validate_artifact_cli_enforces_diagnostic_contract(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts" / "validate_artifact.py"),
        str(path),
        "--no-binding-check",
    ]
    valid = subprocess.run(command, capture_output=True, text=True, check=False)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    artifact["rows"][0]["metrics"]["launch_counts"]["fallback"] = 1  # type: ignore[index]
    path.write_text(json.dumps(artifact), encoding="utf-8")
    invalid = subprocess.run(command, capture_output=True, text=True, check=False)
    assert invalid.returncode != 0
    assert "launch_counts.fallback" in invalid.stdout


def test_synthetic_identity_uses_manifest_digest_without_weight_claim() -> None:
    artifact = _artifact()
    artifact["identity"]["model_manifest_sha256"] = artifact["identity"].pop(  # type: ignore[index]
        "weights_sha256"
    )
    for row in artifact["rows"]:  # type: ignore[index]
        row["identity"] = dict(artifact["identity"])  # type: ignore[index]
    result = validate_artifact(artifact, repo_root=Path.cwd())
    assert result.valid, result.errors


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_abs_error", float("inf")),
        ("max_abs_error", 1.0),
        ("mean_abs_error", float("inf")),
        ("mean_abs_error", 1.0),
        ("top1_agreement", 0.0),
    ),
)
def test_contract_rejects_bad_parity(field: str, value: float) -> None:
    artifact = _artifact()
    artifact["rows"][0]["metrics"]["parity"][field] = value  # type: ignore[index]
    result = validate_parallel_page_artifact(artifact)
    assert not result.valid
    assert any("parity" in error for error in result.errors)
