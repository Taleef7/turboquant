#!/usr/bin/env python3
"""Measure the current serial packed-page decode path.

This is deliberately a diagnostic harness.  It uses deterministic synthetic
Qwen2.5-shaped pages to isolate page-count overhead; it does not establish a
model-quality, throughput, or memory claim.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    import torch
    from kernels.packed_attention import PackedAttentionBlock


DEFAULT_PAGE_COUNTS = (3, 7, 15, 31, 63)
COLD_JIT_RUNS = 1
WARMUP_RUNS = 2
MEASURED_TRIALS = 10
PARITY_MAX_ABS_TOLERANCE = 1e-2
PARITY_MEAN_ABS_TOLERANCE = 1e-4
HEAD_DIM = 128
KV_HEADS = 4
QUERY_HEADS = 28
PAGE_TOKENS = 256
TAIL_TOKENS = 128
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return subprocess.check_output(
            ["git.exe", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()


def _source_metadata() -> dict[str, object]:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain").strip())
    if not _COMMIT.fullmatch(commit):
        raise RuntimeError(f"git did not return a resolved commit: {commit!r}")
    return {"git_commit": commit, "dirty": dirty}


def _driver_version() -> str:
    try:
        value = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .splitlines()[0]
            .strip()
        )
        return value or "unavailable"
    except (OSError, IndexError, subprocess.CalledProcessError):
        return "unavailable"


def _execution_class() -> dict[str, object]:
    import torch

    properties = torch.cuda.get_device_properties()
    name = str(properties.name)
    lowered = name.lower()
    if "5070" in lowered and "ti" in lowered:
        class_name = "local_rtx5070ti_12gb"
    elif "a100" in lowered and properties.total_memory >= 70 * 1024**3:
        class_name = "gilbreth_a100_80gb"
    elif "a100" in lowered:
        class_name = "gilbreth_a100_40gb"
    elif "a30" in lowered:
        class_name = "gilbreth_a30"
    else:
        class_name = "gilbreth_a10"
    return {
        "name": class_name,
        "gpu_name": name,
        "total_vram_bytes": int(properties.total_memory),
        "driver_version": _driver_version(),
        "cuda_runtime": str(torch.version.cuda or "unavailable"),
    }


def _identity() -> dict[str, str]:
    profile_path = ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json"
    manifest_path = ROOT / "configs" / "models" / "qwen25-7b.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        # This harness never loads model weights.  Bind the synthetic fixture
        # to the architecture manifest without mislabelling it as a weight
        # digest.
        "model_manifest_sha256": _sha256(manifest_path),
        "profile_id": str(profile["name"]),
        "profile_sha256": _sha256(profile_path),
        "kv_dtype": "float16",
    }


def _synthetic_blocks(
    page_count: int, *, seed: int, device: "torch.device"
) -> tuple[tuple["PackedAttentionBlock", ...], tuple["PackedAttentionBlock", ...]]:
    import torch
    from kernels.packed_attention import PackedAttentionBlock

    if (
        not isinstance(page_count, int)
        or isinstance(page_count, bool)
        or page_count < 1
    ):
        raise ValueError("page_count must be a positive built-in int")
    generator = torch.Generator(device="cpu").manual_seed(seed + page_count)
    blocks_cpu: list[PackedAttentionBlock] = []
    for _ in range(page_count):
        packed_keys = torch.randint(
            0,
            256,
            (KV_HEADS, PAGE_TOKENS, HEAD_DIM),
            generator=generator,
            dtype=torch.uint8,
        )
        key_norms = (
            torch.rand((KV_HEADS, PAGE_TOKENS), generator=generator)
            .clamp_min(0.1)
            .half()
        )
        packed_values = torch.randint(
            0,
            64,
            (KV_HEADS, PAGE_TOKENS, HEAD_DIM * 6 // 8),
            generator=generator,
            dtype=torch.uint8,
        )
        value_mins = torch.randn(
            (KV_HEADS, PAGE_TOKENS, HEAD_DIM // 32),
            generator=generator,
            dtype=torch.float32,
        ).half()
        value_scales = (
            torch.rand(
                (KV_HEADS, PAGE_TOKENS, HEAD_DIM // 32),
                generator=generator,
                dtype=torch.float32,
            )
            .clamp_min(0.01)
            .half()
        )
        blocks_cpu.append(
            PackedAttentionBlock(
                packed_keys=packed_keys,
                key_norms=key_norms,
                key_codebook=torch.linspace(-1.0, 1.0, 256),
                key_bits=8,
                packed_values=packed_values,
                value_mins=value_mins,
                value_scales=value_scales,
                value_bits=6,
                value_group_size=32,
            )
        )
    blocks_device = tuple(
        PackedAttentionBlock(
            packed_keys=block.packed_keys.to(device),
            key_norms=block.key_norms.to(device),
            key_codebook=block.key_codebook.to(device),
            key_bits=block.key_bits,
            packed_values=block.packed_values.to(device),
            value_mins=block.value_mins.to(device),
            value_scales=block.value_scales.to(device),
            value_bits=block.value_bits,
            value_group_size=block.value_group_size,
        )
        for block in blocks_cpu
    )
    return blocks_device, tuple(blocks_cpu)


def _storage_bytes(blocks: tuple["PackedAttentionBlock", ...]) -> int:
    from core.packing import tensor_nbytes

    total = 0
    for block in blocks:
        total += sum(
            tensor_nbytes(tensor)
            for tensor in (
                block.packed_keys,
                block.key_norms,
                block.key_codebook,
                block.packed_values,
                block.value_mins,
                block.value_scales,
            )
        )
    return total


def _allocation_probe(
    prepared: object, query: "torch.Tensor", tail: tuple["torch.Tensor", "torch.Tensor"]
) -> int:
    import torch

    """Count constructor calls on one un-timed launch without perturbing timings."""
    count = 0
    with ExitStack() as stack:
        for name in ("empty", "empty_like", "full", "zeros"):
            original = getattr(torch, name)

            def wrapped(*args: object, _original=original, **kwargs: object):
                nonlocal count
                count += 1
                return _original(*args, **kwargs)

            stack.enter_context(_patch_attribute(torch, name, wrapped))
        prepared.launch(query, fp16_window=tail)
    torch.cuda.synchronize()
    return count


class _patch_attribute:
    def __init__(self, target: object, name: str, value: object) -> None:
        self.target, self.name, self.value = target, name, value
        self.original: object | None = None

    def __enter__(self) -> object:
        self.original = getattr(self.target, self.name)
        setattr(self.target, self.name, self.value)
        return self.value

    def __exit__(self, *_: object) -> None:
        assert self.original is not None
        setattr(self.target, self.name, self.original)


def _parity(
    actual: "torch.Tensor",
    query_cpu: "torch.Tensor",
    blocks_cpu: tuple["PackedAttentionBlock", ...],
    rotation_cpu: "torch.Tensor",
    mapping_cpu: "torch.Tensor",
    tail_cpu: tuple["torch.Tensor", "torch.Tensor"],
) -> dict[str, object]:
    from kernels.packed_attention import packed_attention_decode

    oracle = packed_attention_decode(
        query_cpu,
        blocks_cpu,
        rotation_cpu,
        mapping_cpu,
        scale=1 / math.sqrt(HEAD_DIM),
        fp16_window=tail_cpu,
        allow_torch_fallback=True,
    )
    actual_cpu = actual.detach().cpu()
    error = (actual_cpu.float() - oracle.float()).abs()
    return {
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "top1_agreement": float(
            (actual_cpu.argmax(dim=-1) == oracle.argmax(dim=-1)).float().mean().item()
        ),
        "repeat_bit_identical": True,
    }


def _run_serial_row(page_count: int, *, seed: int) -> dict[str, object]:
    import torch
    from kernels.packed_attention import (
        PackedAttentionHotPathDebugState,
        prepare_packed_attention_decode,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the serial page diagnostic")
    device = torch.device("cuda")
    blocks_device, blocks_cpu = _synthetic_blocks(page_count, seed=seed, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 + page_count)
    query_cpu = torch.randn(
        (QUERY_HEADS, 1, HEAD_DIM), generator=generator, dtype=torch.float16
    )
    tail_cpu = (
        torch.randn(
            (KV_HEADS, TAIL_TOKENS, HEAD_DIM), generator=generator, dtype=torch.float16
        ),
        torch.randn(
            (KV_HEADS, TAIL_TOKENS, HEAD_DIM), generator=generator, dtype=torch.float16
        ),
    )
    query = query_cpu.to(device)
    tail = tuple(tensor.to(device) for tensor in tail_cpu)
    rotation_cpu = torch.eye(HEAD_DIM, dtype=torch.float32)
    rotation = rotation_cpu.to(device)
    mapping_cpu = torch.arange(QUERY_HEADS, dtype=torch.int32) // (
        QUERY_HEADS // KV_HEADS
    )
    mapping = mapping_cpu.to(device)
    debug = PackedAttentionHotPathDebugState()
    prepared = prepare_packed_attention_decode(
        blocks_device,
        rotation,
        mapping,
        num_kv_heads=KV_HEADS,
        scale=1 / math.sqrt(HEAD_DIM),
        debug=debug,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = int(torch.cuda.memory_allocated())
    # Exactly one cold launch is retained separately from warmups and measured trials.
    cold_output = prepared.launch(query, fp16_window=tail)
    torch.cuda.synchronize()
    launch_counts = {
        "rotation": debug.rotation_launches,
        "page": debug.fused_packed_page_launches,
        "reduction": 0,
        "tail": debug.tail_kernel_dispatches,
        "fallback": 0,
        "allocation": _allocation_probe(prepared, query, tail),
    }
    for _ in range(WARMUP_RUNS):
        prepared.launch(query, fp16_window=tail)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(MEASURED_TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prepared.launch(query, fp16_window=tail)
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    repeated = prepared.launch(query, fp16_window=tail)
    torch.cuda.synchronize()
    parity = _parity(
        cold_output,
        query_cpu,
        blocks_cpu,
        rotation_cpu,
        mapping_cpu,
        tail_cpu,
    )
    parity["repeat_bit_identical"] = bool(
        torch.equal(cold_output.cpu(), repeated.cpu())
    )
    cache_bytes = _storage_bytes(blocks_device)
    peak_allocated = int(torch.cuda.max_memory_allocated())
    runtime_bytes = max(0, peak_allocated - baseline_allocated)
    memory = {
        "logical_payload_bytes": cache_bytes,
        "capacity_bytes": cache_bytes,
        "runtime_bytes": runtime_bytes,
        "workspace_bytes": runtime_bytes,
        "unique_resident_bytes": cache_bytes + runtime_bytes,
    }
    return {
        "row_id": f"serial-pages-{page_count}",
        "status": "passed",
        "reason": None,
        "identity": _identity(),
        "metrics": {
            "page_count": page_count,
            "launch_counts": launch_counts,
            "timing": {
                "cuda_synchronized": True,
                "cold_jit_runs": COLD_JIT_RUNS,
                "warmup_runs": WARMUP_RUNS,
                "measured_trials": MEASURED_TRIALS,
                "cuda_event_ms": samples,
                "p50_ms": _percentile(samples, 0.50),
                "p95_ms": _percentile(samples, 0.95),
                "p99_ms": _percentile(samples, 0.99),
            },
            "memory": memory,
            "parity": parity,
            "diagnostic_only": True,
        },
        "sidecars": [],
    }


def validate_parallel_page_artifact(artifact: Mapping[str, Any]) -> ValidationResult:
    """Strictly validate the serial diagnostic contract before publication."""
    errors: list[str] = []
    if artifact.get("artifact_type") != "parallel_page_attention_diagnostic":
        errors.append("artifact_type must be parallel_page_attention_diagnostic")
    if artifact.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if artifact.get("diagnostic_only") is not True:
        errors.append("diagnostic_only must be true")
    if artifact.get("status") not in {"partial", "qualified"}:
        errors.append("status must be partial or qualified")
    expected_ids = [f"serial-pages-{count}" for count in DEFAULT_PAGE_COUNTS]
    if artifact.get("expected_row_ids") != expected_ids:
        errors.append("expected_row_ids must exactly declare serial page rows")
    if artifact.get("claims") not in ([], None):
        errors.append("diagnostic artifact cannot publish speed or memory claims")
    identity = artifact.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("identity is required")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        return ValidationResult(False, tuple(errors + ["rows must be a list"]))
    rows_by_id = {row.get("row_id"): row for row in rows if isinstance(row, Mapping)}
    missing_ids = sorted(set(expected_ids) - set(rows_by_id))
    if missing_ids:
        errors.append(f"missing expected row IDs: {missing_ids}")
    if set(rows_by_id) != set(expected_ids) or len(rows_by_id) != len(rows):
        errors.append("rows must contain each expected row ID exactly once")
    declared_counts = artifact.get("pass_state_counts")
    if declared_counts != {
        "passed": len(expected_ids),
        "failed": 0,
        "invalid": 0,
        "oom": 0,
        "unsupported": 0,
        "unavailable": 0,
    }:
        errors.append("pass_state_counts must describe five passed rows")
    for row_id in expected_ids:
        row = rows_by_id.get(row_id)
        if not isinstance(row, Mapping):
            continue
        if row.get("status") != "passed" or row.get("reason") not in (None, ""):
            errors.append(f"row {row_id} must be passed without a reason")
        if row.get("identity") != identity:
            errors.append(f"row {row_id} identity must match artifact identity")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            errors.append(f"row {row_id} metrics are required")
            continue
        if metrics.get("diagnostic_only") is not True or metrics.get(
            "page_count"
        ) != int(row_id.rsplit("-", 1)[1]):
            errors.append(
                f"row {row_id} must declare its page count and diagnostic label"
            )
        counts = metrics.get("launch_counts")
        if not isinstance(counts, Mapping):
            errors.append(f"row {row_id} launch_counts are required")
        else:
            page_count = metrics["page_count"]
            expected_counts = {
                "rotation": 1,
                "page": page_count,
                "reduction": 0,
                "tail": 1,
                "fallback": 0,
            }
            for key, expected in expected_counts.items():
                if counts.get(key) != expected:
                    errors.append(
                        f"row {row_id} launch_counts.{key} must be {expected}"
                    )
            if (
                not isinstance(counts.get("allocation"), int)
                or isinstance(counts.get("allocation"), bool)
                or counts["allocation"] < 1
            ):
                errors.append(f"row {row_id} launch_counts.allocation must be positive")
        timing = metrics.get("timing")
        if not isinstance(timing, Mapping):
            errors.append(f"row {row_id} timing is required")
        else:
            if timing.get("cuda_synchronized") is not True:
                errors.append(f"row {row_id} timing.cuda_synchronized must be true")
            for key, expected in (
                ("cold_jit_runs", COLD_JIT_RUNS),
                ("warmup_runs", WARMUP_RUNS),
                ("measured_trials", MEASURED_TRIALS),
            ):
                if timing.get(key) != expected:
                    errors.append(f"row {row_id} timing.{key} must be {expected}")
            samples = timing.get("cuda_event_ms")
            if (
                not isinstance(samples, list)
                or len(samples) != MEASURED_TRIALS
                or any(
                    not isinstance(value, (float, int))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value <= 0
                    for value in samples
                )
            ):
                errors.append(
                    f"row {row_id} must contain ten positive finite CUDA-event samples"
                )
                if not isinstance(samples, list) or len(samples) != MEASURED_TRIALS:
                    errors.append(
                        f"row {row_id} timing.measured_trials does not match its event samples"
                    )
            else:
                for key, percentile in (
                    ("p50_ms", 0.50),
                    ("p95_ms", 0.95),
                    ("p99_ms", 0.99),
                ):
                    value = timing.get(key)
                    if (
                        not isinstance(value, (float, int))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        or not math.isclose(
                            value,
                            _percentile([float(x) for x in samples], percentile),
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                    ):
                        errors.append(
                            f"row {row_id} timing.{key} must match its CUDA-event samples"
                        )
        memory = metrics.get("memory")
        if not isinstance(memory, Mapping):
            errors.append(f"row {row_id} memory accounting is required")
        else:
            fields = (
                "logical_payload_bytes",
                "capacity_bytes",
                "runtime_bytes",
                "workspace_bytes",
                "unique_resident_bytes",
            )
            if any(
                not isinstance(memory.get(field), int)
                or isinstance(memory.get(field), bool)
                or memory[field] < 0
                for field in fields
            ):
                errors.append(
                    f"row {row_id} memory fields must be non-negative integers"
                )
            elif (
                memory["capacity_bytes"] < memory["logical_payload_bytes"]
                or memory["unique_resident_bytes"] < memory["capacity_bytes"]
            ):
                errors.append(
                    f"row {row_id} memory capacity/residency must cover logical payload"
                )
        parity = metrics.get("parity")
        max_error = parity.get("max_abs_error") if isinstance(parity, Mapping) else None
        mean_error = (
            parity.get("mean_abs_error") if isinstance(parity, Mapping) else None
        )
        top1 = parity.get("top1_agreement") if isinstance(parity, Mapping) else None
        if (
            not isinstance(parity, Mapping)
            or parity.get("repeat_bit_identical") is not True
            or not isinstance(max_error, (float, int))
            or isinstance(max_error, bool)
            or not math.isfinite(max_error)
            or max_error < 0
            or max_error > PARITY_MAX_ABS_TOLERANCE
            or not isinstance(mean_error, (float, int))
            or isinstance(mean_error, bool)
            or not math.isfinite(mean_error)
            or mean_error < 0
            or mean_error > PARITY_MEAN_ABS_TOLERANCE
            or not isinstance(top1, (float, int))
            or isinstance(top1, bool)
            or top1 != 1.0
        ):
            errors.append(
                f"row {row_id} parity must be deterministic with max error <= "
                f"{PARITY_MAX_ABS_TOLERANCE}, mean error <= {PARITY_MEAN_ABS_TOLERANCE}, "
                "and exact top-1 agreement"
            )
    return ValidationResult(not errors, tuple(errors))


def build_artifact(*, page_counts: tuple[int, ...], seed: int) -> dict[str, object]:
    if page_counts != DEFAULT_PAGE_COUNTS:
        raise ValueError(f"Task 1 requires page counts exactly {DEFAULT_PAGE_COUNTS}")
    rows = [_run_serial_row(page_count, seed=seed) for page_count in page_counts]
    source = _source_metadata()
    paper_reference = json.loads(
        (ROOT / "docs" / "references" / "turboquant-2504.19874v1.json").read_text(
            encoding="utf-8"
        )
    )
    paper = {
        key: paper_reference[key]
        for key in ("paper_id", "source_url", "retrieved_at", "sha256")
    }
    artifact = {
        "artifact_type": "parallel_page_attention_diagnostic",
        "schema_version": 2,
        "artifact_id": "parallel-page-baseline",
        "status": "partial" if source["dirty"] else "qualified",
        "diagnostic_only": True,
        "paper": paper,
        "source": source,
        "execution_class": _execution_class(),
        "identity": _identity(),
        "statistics": {
            "method": "CUDA events; linear percentile interpolation; one cold/JIT, two warmups, ten measured trials",
            "confidence_level": 0.95,
            "seed": seed,
            "iterations": MEASURED_TRIALS,
        },
        "expected_row_ids": [f"serial-pages-{count}" for count in page_counts],
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
        "extensions": {
            "implementation": "serial",
            "page_tokens": PAGE_TOKENS,
            "tail_tokens": TAIL_TOKENS,
        },
    }
    result = validate_parallel_page_artifact(artifact)
    if not result.valid:
        raise RuntimeError(
            "generated artifact failed its own contract: " + "; ".join(result.errors)
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementations", nargs="+", default=["serial"], choices=("serial",)
    )
    parser.add_argument(
        "--page-counts", nargs="+", type=int, default=list(DEFAULT_PAGE_COUNTS)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.implementations != ["serial"]:
        raise SystemExit("Task 1 only implements --implementations serial")
    artifact = build_artifact(page_counts=tuple(args.page_counts), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "valid": True,
                "artifact": str(args.output),
                "rows": len(artifact["rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
