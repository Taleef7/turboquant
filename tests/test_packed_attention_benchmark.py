"""Separately labelled microbenchmark for packed Triton kernel mechanics."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess

import pytest
import torch

from core.packing import pack_indices
from kernels.packed_attention import (
    PACKED_ATTENTION_COMPUTE_SANITIZER_COMMAND,
    PackedAttentionBenchmarkDebug,
    prepare_packed_key_scores_rotated,
    prepare_packed_value_mix,
    prepare_rotate_query,
)


@pytest.mark.gpu
def test_packed_kernel_microbenchmark_emits_qualified_json(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; packed kernel benchmark requires a CUDA GPU.")
    triton = pytest.importorskip("triton", reason="Triton is unavailable for benchmark")
    torch.manual_seed(91)
    key_bits, value_bits, head_dim, kv_heads, gqa_ratio, tokens = 6, 6, 128, 2, 4, 256
    heads, query_tokens, group_size = kv_heads * gqa_ratio, 1, 32
    device = torch.device("cuda")
    query = torch.randn((heads, query_tokens, head_dim), device=device, dtype=torch.float16)
    rotation = torch.eye(head_dim, device=device)
    codebook = torch.linspace(-1, 1, 1 << key_bits, device=device)
    packed_keys = pack_indices(torch.randint(0, 1 << key_bits, (kv_heads * tokens, head_dim), device=device, dtype=torch.uint8), key_bits).reshape(kv_heads, tokens, -1)
    norms = torch.ones((kv_heads, tokens), device=device, dtype=torch.float16)
    mapping = torch.arange(heads, device=device, dtype=torch.int32) // gqa_ratio
    packed_values = pack_indices(torch.randint(0, 1 << value_bits, (kv_heads * tokens, head_dim), device=device, dtype=torch.uint8), value_bits).reshape(kv_heads, tokens, -1)
    mins = torch.zeros((kv_heads, tokens, head_dim // group_size), device=device, dtype=torch.float16)
    scales = torch.ones_like(mins) / 63
    debug = PackedAttentionBenchmarkDebug()
    prepared_rotation = prepare_rotate_query(query, rotation, debug=debug)
    for _ in range(3):
        rotated_query = prepared_rotation.launch()
    assert rotated_query.is_cuda
    prepared_scores = prepare_packed_key_scores_rotated(
        rotated_query, packed_keys, codebook, norms, mapping,
        key_bits=key_bits, scale=1 / math.sqrt(head_dim), debug=debug,
    )
    for _ in range(3):
        scores = prepared_scores.launch()
    assert scores.is_cuda
    prior = (
        torch.full_like(scores[..., 0], -torch.inf),
        torch.zeros_like(scores[..., 0]),
        torch.zeros((heads, query_tokens, head_dim), device=device),
    )
    prepared_values = prepare_packed_value_mix(
        scores, packed_values, mins, scales, mapping, value_bits=value_bits,
        group_size=group_size, prior_state=prior, debug=debug,
    )
    for _ in range(3):
        prepared_values.launch()
    do_bench = triton.testing.do_bench

    def measure(operation: object) -> dict[str, float]:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        median_ms, p95_ms = do_bench(operation, quantiles=[0.5, 0.95])
        return {
            "cuda_event_ms": start.elapsed_time(end),
            "median_ms": median_ms,
            "p95_ms": p95_ms,
        }

    validation_reductions_before_timing = debug.validation_reductions
    validation_passes_before_timing = debug.validation_passes
    metrics = {
        "rotate_query": measure(prepared_rotation.launch),
        "packed_key_scores": measure(prepared_scores.launch),
        "packed_value_mix": measure(prepared_values.launch),
    }
    assert debug.validation_reductions == validation_reductions_before_timing
    assert debug.validation_passes == validation_passes_before_timing
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        # This linked Windows worktree is exercised from native WSL Python.
        source_commit = subprocess.check_output(
            ["git.exe", "rev-parse", "HEAD"], text=True
        ).strip()
    result = {
        "label": "kernel_microbenchmark",
        "kernels": metrics,
        "shapes": {"key_bits": key_bits, "value_bits": value_bits, "head_dim": head_dim, "kv_heads": kv_heads, "gqa_ratio": gqa_ratio, "tokens": tokens, "query_tokens": query_tokens},
        "dtype": str(query.dtype).removeprefix("torch."),
        "gpu": {"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())},
        "source_commit": source_commit,
        "reference_tolerance": {"rtol": 3e-3, "atol": 3e-3},
        "used_torch_fallback": False,
        "benchmark_validation": {
            "prepared_validation_passes": debug.validation_passes,
            "prepared_validation_reductions": debug.validation_reductions,
            "validation_reductions_during_timing": 0,
            "timed_launches": debug.timed_launches,
        },
    }
    output = Path(
        os.environ.get(
            "TURBOQUANT_BENCHMARK_OUTPUT",
            str(tmp_path / "packed_attention_kernel_microbenchmark.json"),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["label"] == "kernel_microbenchmark"
    assert saved["used_torch_fallback"] is False
    assert set(saved["kernels"]) == {
        "rotate_query", "packed_key_scores", "packed_value_mix"
    }
    for metric in saved["kernels"].values():
        assert metric["p95_ms"] >= metric["median_ms"] >= 0
    assert saved["benchmark_validation"]["validation_reductions_during_timing"] == 0
    assert saved["benchmark_validation"]["prepared_validation_reductions"] == 4
    assert "compute-sanitizer" in PACKED_ATTENTION_COMPUTE_SANITIZER_COMMAND
