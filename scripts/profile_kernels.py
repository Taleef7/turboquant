"""
Micro-benchmark suite for TurboQuant kernels.

Measures individual kernel performance with varying parameters to identify
bottlenecks and optimization opportunities.

Usage:
    python scripts/profile_kernels.py
    python scripts/profile_kernels.py --kernel compress --seq-lengths 1,8,32,128
    python scripts/profile_kernels.py --kernel all --block-sizes 16,32,64,128
"""

import argparse
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.math_utils import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    get_centroids_2bit,
    get_centroids_3bit,
)
from kernels.compress_kv import compress_kv_python, compress_kv_hybrid, build_outlier_mask
from kernels.decompress_kv import decompress_kv_python, decompress_kv_hybrid


def benchmark_with_cuda_events(fn, *args, warmup=10, iters=100, sync_before=True):
    """
    Accurate GPU timing using CUDA events.
    
    Args:
        fn: Function to benchmark
        *args: Arguments to pass to function
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        sync_before: Whether to sync before starting timer
        
    Returns:
        Mean time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    if sync_before:
        torch.cuda.synchronize()
    
    start.record()
    for _ in range(iters):
        _ = fn(*args)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iters  # ms per call


def benchmark_compress(seq_lengths, head_dim=128, n_outliers=32, batch=1, heads=1):
    """Benchmark compress_kv_hybrid with varying sequence lengths."""
    print(f"\n{'='*70}")
    print(f"COMPRESS KERNEL BENCHMARK")
    print(f"{'='*70}")
    print(f"Configuration: head_dim={head_dim}, n_outliers={n_outliers}, batch={batch}, heads={heads}")
    print(f"\n{'Seq Len':<10} {'Python (ms)':<15} {'Hybrid (ms)':<15} {'Speedup':<10} {'Tokens/ms':<12}")
    print(f"{'-'*70}")
    
    device = torch.device("cuda")
    Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=device)
    S = generate_qjl_matrix(head_dim, 128, dtype=torch.float32, device=device)
    c2 = get_centroids_2bit(head_dim, device=device)
    c3 = get_centroids_3bit(head_dim, device=device)
    
    results = []
    for seq_len in seq_lengths:
        # Create input data
        n_slices = batch * heads
        x = torch.randn(n_slices * seq_len, head_dim, dtype=torch.float32, device=device)
        mask = build_outlier_mask(x, n_outliers)
        
        # Benchmark Python implementation
        time_python = benchmark_with_cuda_events(
            compress_kv_python, x, Pi, S, c2, c3, mask, warmup=5, iters=50
        )
        
        # Benchmark Hybrid implementation
        time_hybrid = benchmark_with_cuda_events(
            compress_kv_hybrid, x, Pi, S, c2, c3, mask, warmup=5, iters=50
        )
        
        speedup = time_python / time_hybrid
        tokens_per_ms = seq_len / time_hybrid
        
        print(f"{seq_len:<10} {time_python:<15.3f} {time_hybrid:<15.3f} {speedup:<10.2f}x {tokens_per_ms:<12.1f}")
        
        results.append({
            'seq_len': seq_len,
            'time_python_ms': time_python,
            'time_hybrid_ms': time_hybrid,
            'speedup': speedup,
            'tokens_per_ms': tokens_per_ms
        })
    
    return results


def benchmark_decompress(seq_lengths, head_dim=128, n_outliers=32, batch=1, heads=1):
    """Benchmark decompress_kv_hybrid with varying sequence lengths."""
    print(f"\n{'='*70}")
    print(f"DECOMPRESS KERNEL BENCHMARK")
    print(f"{'='*70}")
    print(f"Configuration: head_dim={head_dim}, n_outliers={n_outliers}, batch={batch}, heads={heads}")
    print(f"\n{'Seq Len':<10} {'Python (ms)':<15} {'Hybrid (ms)':<15} {'Speedup':<10} {'Tokens/ms':<12}")
    print(f"{'-'*70}")
    
    device = torch.device("cuda")
    Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=device)
    S = generate_qjl_matrix(head_dim, 128, dtype=torch.float32, device=device)
    c2 = get_centroids_2bit(head_dim, device=device)
    c3 = get_centroids_3bit(head_dim, device=device)
    
    results = []
    for seq_len in seq_lengths:
        # Create compressed data
        n_slices = batch * heads
        x = torch.randn(n_slices * seq_len, head_dim, dtype=torch.float32, device=device)
        mask = build_outlier_mask(x, n_outliers)
        
        # Compress first to get decompress inputs
        idx_all, qjl_bits, gamma = compress_kv_hybrid(x, Pi, S, c2, c3, mask)
        
        # Benchmark Python implementation
        time_python = benchmark_with_cuda_events(
            decompress_kv_python, idx_all, qjl_bits, gamma, Pi, S, c2, c3, mask,
            warmup=5, iters=50
        )
        
        # Benchmark Hybrid implementation
        time_hybrid = benchmark_with_cuda_events(
            decompress_kv_hybrid, idx_all, qjl_bits, gamma, Pi, S, c2, c3, mask,
            warmup=5, iters=50
        )
        
        speedup = time_python / time_hybrid
        tokens_per_ms = seq_len / time_hybrid
        
        print(f"{seq_len:<10} {time_python:<15.3f} {time_hybrid:<15.3f} {speedup:<10.2f}x {tokens_per_ms:<12.1f}")
        
        results.append({
            'seq_len': seq_len,
            'time_python_ms': time_python,
            'time_hybrid_ms': time_hybrid,
            'speedup': speedup,
            'tokens_per_ms': tokens_per_ms
        })
    
    return results


def benchmark_memory_transfer(seq_lengths, head_dim=128, batch=1, heads=1):
    """Benchmark memory transfer overhead (.to() calls)."""
    print(f"\n{'='*70}")
    print(f"MEMORY TRANSFER BENCHMARK")
    print(f"{'='*70}")
    print(f"Configuration: head_dim={head_dim}, batch={batch}, heads={heads}")
    print(f"\n{'Seq Len':<10} {'Reshape (ms)':<15} {'To FP32 (ms)':<15} {'To Device (ms)':<15} {'Total (ms)':<15}")
    print(f"{'-'*70}")
    
    device = torch.device("cuda")
    
    results = []
    for seq_len in seq_lengths:
        # Create input as it would come from attention (FP16 on GPU)
        x_input = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
        n_slices = batch * heads
        
        # Benchmark reshape
        def reshape_only():
            return x_input.reshape(n_slices * seq_len, head_dim)
        time_reshape = benchmark_with_cuda_events(reshape_only, warmup=10, iters=100)
        
        # Benchmark to(float32)
        x_reshaped = x_input.reshape(n_slices * seq_len, head_dim)
        def to_float32():
            return x_reshaped.to(torch.float32)
        time_to_float32 = benchmark_with_cuda_events(to_float32, warmup=10, iters=100)
        
        # Benchmark to(device) when already on device (should be no-op)
        x_float = x_reshaped.to(torch.float32)
        def to_device():
            return x_float.to(device)
        time_to_device = benchmark_with_cuda_events(to_device, warmup=10, iters=100)
        
        # Total (chained like in current code)
        def chained():
            return x_input.reshape(n_slices * seq_len, head_dim).to(torch.float32).to(device)
        time_total = benchmark_with_cuda_events(chained, warmup=10, iters=100)
        
        print(f"{seq_len:<10} {time_reshape:<15.3f} {time_to_float32:<15.3f} {time_to_device:<15.3f} {time_total:<15.3f}")
        
        results.append({
            'seq_len': seq_len,
            'time_reshape_ms': time_reshape,
            'time_to_float32_ms': time_to_float32,
            'time_to_device_ms': time_to_device,
            'time_total_ms': time_total
        })
    
    return results


def benchmark_kernel_launch_overhead():
    """Measure empty kernel launch latency."""
    print(f"\n{'='*70}")
    print(f"KERNEL LAUNCH OVERHEAD")
    print(f"{'='*70}")
    
    device = torch.device("cuda")
    
    # Create minimal operation (empty no-op)
    def minimal_kernel():
        torch.cuda.synchronize()
    
    # Measure launch overhead
    time_launch = benchmark_with_cuda_events(minimal_kernel, warmup=50, iters=1000, sync_before=False)
    
    print(f"Empty kernel launch + sync: {time_launch:.3f} ms")
    print(f"Estimated launches per second: {1000.0/time_launch:.0f}")
    print(f"\nNote: With 28 layers × 2 (K+V) × 2 (compress+decompress) = 112 kernel calls")
    print(f"      Launch overhead per token: {112 * time_launch:.3f} ms")
    
    return {'launch_overhead_ms': time_launch}


def main():
    parser = argparse.ArgumentParser(description="TurboQuant kernel micro-benchmarks")
    parser.add_argument(
        '--kernel', 
        type=str, 
        default='all',
        choices=['compress', 'decompress', 'memory', 'launch', 'all'],
        help='Which kernel to benchmark'
    )
    parser.add_argument(
        '--seq-lengths',
        type=str,
        default='1,8,32,128,256',
        help='Comma-separated sequence lengths to test'
    )
    parser.add_argument(
        '--head-dim',
        type=int,
        default=128,
        help='Head dimension (default: 128 for Qwen2.5)'
    )
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return
    
    seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
    
    print(f"\nTurboQuant Kernel Profiling Suite")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    results = {}
    
    if args.kernel in ['compress', 'all']:
        results['compress'] = benchmark_compress(seq_lengths, head_dim=args.head_dim)
    
    if args.kernel in ['decompress', 'all']:
        results['decompress'] = benchmark_decompress(seq_lengths, head_dim=args.head_dim)
    
    if args.kernel in ['memory', 'all']:
        results['memory'] = benchmark_memory_transfer(seq_lengths, head_dim=args.head_dim)
    
    if args.kernel in ['launch', 'all']:
        results['launch'] = benchmark_kernel_launch_overhead()
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    main()
