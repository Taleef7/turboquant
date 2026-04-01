#!/usr/bin/env python3
"""Detailed profiling of TurboQuantCacheV2 operations."""

import torch
import time
import sys

sys.path.insert(0, "/home/taleef/projects/turboquant")

from core.turboquant_simple import TurboQuantMSE, TurboQuantValueMSE


def profile_ops():
    """Profile individual operations."""
    device = torch.device("cuda")
    head_dim = 128
    bits = 6

    # Create quantizers
    key_quant = TurboQuantMSE(head_dim, bits, seed=42, device="cuda")
    val_quant = TurboQuantValueMSE(head_dim, bits, device="cuda")

    # Test data: batch=1, heads=28, seq=1, head_dim=128 (single token)
    batch = 1
    heads = 28
    seq = 1

    x = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)

    # Warmup
    print("Warmup...")
    for _ in range(10):
        c = key_quant.compress(x)
        d = key_quant.decompress(c)
    torch.cuda.synchronize()

    # Profile compress
    n_runs = 100

    print(f"\nProfiling {n_runs} runs for single token (batch={batch}, heads={heads}):")
    print("=" * 60)

    # Key compress
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        c = key_quant.compress(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Key compress:     {elapsed * 1000 / n_runs:.3f}ms per token")

    # Key decompress
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        d = key_quant.decompress(c)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Key decompress:   {elapsed * 1000 / n_runs:.3f}ms per token")

    # Value compress
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        c = val_quant.compress(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Value compress:   {elapsed * 1000 / n_runs:.3f}ms per token")

    # Value decompress
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        d = val_quant.decompress(c)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Value decompress: {elapsed * 1000 / n_runs:.3f}ms per token")

    # Profile with larger sequence (prefill)
    print(f"\nProfiling prefill (seq=100):")
    print("=" * 60)
    seq = 100
    x_prefill = torch.randn(
        batch, heads, seq, head_dim, device=device, dtype=torch.float16
    )

    # Warmup
    for _ in range(5):
        c = key_quant.compress(x_prefill)
        d = key_quant.decompress(c)
    torch.cuda.synchronize()

    n_runs = 50

    # Key compress
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        c = key_quant.compress(x_prefill)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"Key compress (100 tok):     {elapsed * 1000 / n_runs:.3f}ms ({elapsed * 1000 / n_runs / 100:.4f}ms per token)"
    )

    # Key decompress
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        d = key_quant.decompress(c)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"Key decompress (100 tok):   {elapsed * 1000 / n_runs:.3f}ms ({elapsed * 1000 / n_runs / 100:.4f}ms per token)"
    )

    # Per-layer overhead simulation
    print(f"\nSimulating full model (28 layers, single token):")
    print("=" * 60)

    num_layers = 28
    x_single = torch.randn(
        batch, heads, 1, head_dim, device=device, dtype=torch.float16
    )

    # Create per-layer quantizers (different seeds)
    key_quants = [
        TurboQuantMSE(head_dim, bits, seed=42 + i * 7, device="cuda")
        for i in range(num_layers)
    ]
    val_quants = [
        TurboQuantValueMSE(head_dim, bits, device="cuda") for i in range(num_layers)
    ]

    # Warmup all layers
    for i in range(num_layers):
        _ = key_quants[i].compress(x_single)
        _ = val_quants[i].compress(x_single)
    torch.cuda.synchronize()

    # Time one full decode step (all layers)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        for i in range(num_layers):
            k_c = key_quants[i].compress(x_single)
            v_c = val_quants[i].compress(x_single)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Compress K+V all layers: {elapsed * 1000 / n_runs:.3f}ms per decode step")

    # Decompress
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        for i in range(num_layers):
            _ = key_quants[i].decompress(k_c)
            _ = val_quants[i].decompress(v_c)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Decompress K+V all layers: {elapsed * 1000 / n_runs:.3f}ms per decode step")

    # Total overhead estimate
    compress_ms = elapsed * 1000 / n_runs  # reuse last measurement for estimate
    total_overhead = compress_ms * 2  # compress + decompress
    print(f"\nEstimated total cache overhead: {total_overhead:.1f}ms per decode step")
    print(f"At 20 tok/s target, overhead budget: {1000 / 20:.0f}ms per decode")
    print(f"Cache overhead percentage: {total_overhead / 50 * 100:.1f}%")


if __name__ == "__main__":
    profile_ops()
