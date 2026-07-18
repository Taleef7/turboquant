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
    packed_key_scores,
    packed_value_mix,
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
    for _ in range(3):
        scores, execution = packed_key_scores(query, packed_keys, codebook, norms, rotation, mapping, key_bits=key_bits, scale=1 / math.sqrt(head_dim), return_execution=True)
    assert execution.kernel == "triton" and not execution.used_torch_fallback
    prior = (
        torch.full_like(scores[..., 0], -torch.inf),
        torch.zeros_like(scores[..., 0]),
        torch.zeros((heads, query_tokens, head_dim), device=device),
    )
    for _ in range(3):
        _, execution = packed_value_mix(scores, packed_values, mins, scales, mapping, value_bits=value_bits, group_size=group_size, prior_state=prior, return_execution=True)
    assert execution.kernel == "triton" and not execution.used_torch_fallback
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    packed_key_scores(query, packed_keys, codebook, norms, rotation, mapping, key_bits=key_bits, scale=1 / math.sqrt(head_dim))
    end.record()
    torch.cuda.synchronize()
    cuda_event_ms = start.elapsed_time(end)
    do_bench = triton.testing.do_bench
    median_ms, p95_ms = do_bench(
        lambda: packed_key_scores(query, packed_keys, codebook, norms, rotation, mapping, key_bits=key_bits, scale=1 / math.sqrt(head_dim)),
        quantiles=[0.5, 0.95],
    )
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
        "kernel": "packed_key_scores",
        "shapes": {"key_bits": key_bits, "value_bits": value_bits, "head_dim": head_dim, "kv_heads": kv_heads, "gqa_ratio": gqa_ratio, "tokens": tokens, "query_tokens": query_tokens},
        "dtype": str(query.dtype).removeprefix("torch."),
        "gpu": {"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())},
        "source_commit": source_commit,
        "cuda_event_ms": cuda_event_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "reference_tolerance": {"rtol": 3e-3, "atol": 3e-3},
        "used_torch_fallback": False,
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
    assert saved["p95_ms"] >= saved["median_ms"] >= 0
    assert "compute-sanitizer" in PACKED_ATTENTION_COMPUTE_SANITIZER_COMMAND
