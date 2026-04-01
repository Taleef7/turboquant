#!/usr/bin/env python3
"""Micro-profile value compress to find bottleneck."""

import torch
import time
import sys

sys.path.insert(0, "/home/taleef/projects/turboquant")


def profile_value_compress():
    device = torch.device("cuda")
    head_dim = 128
    bits = 6
    n_levels = 2**bits
    group_size = 32
    n_groups = head_dim // group_size

    # Test data
    batch = 1
    heads = 28
    seq = 1
    x = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(10):
        flat = x.reshape(-1, head_dim)
        grouped = flat.view(-1, n_groups, group_size)
        mins = grouped.amin(dim=-1)
        maxs = grouped.amax(dim=-1)
        scales = (maxs - mins) / (n_levels - 1)
    torch.cuda.synchronize()

    n_runs = 100
    print(f"Profiling value compress micro-ops ({n_runs} runs):")
    print("=" * 60)

    # reshape
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        flat = x.reshape(-1, head_dim)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"reshape:     {elapsed * 1000 / n_runs:.4f}ms")

    # view
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        grouped = flat.view(-1, n_groups, group_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"view:        {elapsed * 1000 / n_runs:.4f}ms")

    # amin
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        mins = grouped.amin(dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"amin:        {elapsed * 1000 / n_runs:.4f}ms")

    # amax
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        maxs = grouped.amax(dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"amax:        {elapsed * 1000 / n_runs:.4f}ms")

    # scales calc
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        scales = (maxs - mins) / (n_levels - 1)
        scales = torch.clamp(scales, min=1e-8)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"scales:      {elapsed * 1000 / n_runs:.4f}ms")

    # normalize
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        normalized = (grouped - mins.unsqueeze(-1)) / scales.unsqueeze(-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"normalize:   {elapsed * 1000 / n_runs:.4f}ms")

    # round + clamp + cast
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        indices = torch.round(normalized).clamp_(0, n_levels - 1).to(torch.uint8)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"round/cast:  {elapsed * 1000 / n_runs:.4f}ms")

    # Full operation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        flat = x.reshape(-1, head_dim)
        grouped = flat.view(-1, n_groups, group_size)
        mins = grouped.amin(dim=-1)
        maxs = grouped.amax(dim=-1)
        scales = (maxs - mins) / (n_levels - 1)
        scales = torch.clamp(scales, min=1e-8)
        normalized = (grouped - mins.unsqueeze(-1)) / scales.unsqueeze(-1)
        indices = torch.round(normalized).clamp_(0, n_levels - 1).to(torch.uint8)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"\nFull op:     {elapsed * 1000 / n_runs:.4f}ms")

    # Compare: fused min/max
    print("\n--- Testing alternatives ---")

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        mins, _ = grouped.min(dim=-1)
        maxs, _ = grouped.max(dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"min+max:     {elapsed * 1000 / n_runs:.4f}ms")

    # aminmax (fused)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        mins, maxs = torch.aminmax(grouped, dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"aminmax:     {elapsed * 1000 / n_runs:.4f}ms")


if __name__ == "__main__":
    profile_value_compress()
