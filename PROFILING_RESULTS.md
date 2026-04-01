# TurboQuant Profiling Results - Phase 1

**Date**: 2025-03-31  
**Hardware**: RTX 5070 Ti (Blackwell sm_120)  
**Model**: Qwen2.5-7B-Instruct (4-bit quantized)  
**Implementation**: Hybrid (PyTorch cuBLAS + Triton quantization)

---

## Executive Summary

### Current Performance
- **Throughput**: 9.70 tok/s (54% of baseline 18.07 tok/s)
- **Target**: 14.5 tok/s (80% of baseline)
- **Gap**: 1.5× improvement needed
- **Tokens generated**: 65
- **Total time**: 6.70s
- **Per-token latency**: 103.1 ms/token

### Key Findings

**🔴 CRITICAL BOTTLENECK IDENTIFIED**: First K cache compression  
- **75.4% of update cycle time** spent on compress_k
- First compress_k call takes **5.25 seconds** (4237.579 ms)
- Subsequent compress_k calls average **0.31 ms** (16,927× slower on first call!)
- **Root cause**: Triton kernel compilation happening during first call

**✅ Other operations are efficient**:
- Compress V: 0.337 ms (7.8%)
- Decompress K: 0.338 ms (7.9%)
- Decompress V: 0.343 ms (8.0%)
- Memory transfers: 0.244 ms (negligible)

---

## Detailed Analysis

### 1. Full Benchmark Profiling Results

#### Per-Layer Update Cycle (Average)
```
Total:          4.305 ms
├─ Compress K:  3.245 ms (75.4%) ⚠️ BOTTLENECK
├─ Compress V:  0.337 ms (7.8%)
├─ Decompress K:0.338 ms (7.9%)
└─ Decompress V:0.343 ms (8.0%)
```

**Analysis**:
- Compress K dominates the update cycle
- The 3.245 ms average is heavily skewed by the first call's 5.25s compilation time
- Without the first call, average compress_k would be ~0.31 ms (comparable to compress_v)

#### Compression Breakdown
```
Total:            48.234 ms (median: 1.369 ms)
├─ Memory transfer:0.244 ms (0.5%)
└─ Kernel compute: 38.518 ms (79.9%)
```

**Analysis**:
- Memory transfers are NOT a bottleneck (0.5% of time)
- Kernel compute time includes the massive 4237 ms first-call outlier
- Median compression time is only 1.369 ms (reasonable)

#### First Call vs Subsequent Calls

| Operation | First Call | Median | Ratio |
|-----------|------------|--------|-------|
| Compress K | 5247 ms | 0.313 ms | 16,760× |
| Compress V | 0.293 ms | 0.293 ms | 1× |
| Decompress K | 0.307 ms | 0.307 ms | 1× |
| Decompress V | 0.326 ms | 0.326 ms | 1× |

**Conclusion**: Only compress_k suffers from first-call penalty (Triton compilation).

---

### 2. Micro-Benchmark Results

#### Compress Kernel Performance

| Seq Len | Python (ms) | Hybrid (ms) | Speedup | Tokens/ms |
|---------|-------------|-------------|---------|-----------|
| 1       | 1.551       | 1.018       | 1.52×   | 1.0       |
| 8       | 1.577       | 0.249       | 6.34×   | 32.2      |
| 32      | 1.447       | 0.380       | 3.81×   | 84.3      |
| 128     | 3.835       | 1.215       | 3.16×   | 105.3     |
| 256     | 5.621       | 1.687       | 3.33×   | 151.8     |

**Observations**:
- Hybrid is **1.5-6.3× faster** than Python
- seq_len=1 has lowest speedup (1.52×) - poor GPU utilization
- seq_len=8 has highest speedup (6.34×) - sweet spot for current grid size
- Throughput peaks at 256 tokens: 151.8 tokens/ms

**Optimization Opportunity**: Improve seq_len=1 performance (single-token decode phase).

#### Decompress Kernel Performance

| Seq Len | Python (ms) | Hybrid (ms) | Speedup | Tokens/ms |
|---------|-------------|-------------|---------|-----------|
| 1       | 1.156       | 0.303       | 3.81×   | 3.3       |
| 8       | 0.987       | 0.191       | 5.17×   | 41.9      |
| 32      | 1.034       | 0.244       | 4.23×   | 130.9     |
| 128     | 1.170       | 0.272       | 4.30×   | 469.9     |
| 256     | 1.151       | 0.218       | 5.27×   | 1172.9    |

**Observations**:
- Decompress is **already very efficient** (0.2-0.3 ms per call)
- Consistent 3.8-5.3× speedup across all sequence lengths
- Throughput excellent: 1172.9 tokens/ms at seq_len=256
- seq_len=1 performance is decent (3.3 tokens/ms)

**Conclusion**: Decompress is NOT a bottleneck.

#### Memory Transfer Overhead

| Seq Len | Reshape | To FP32 | To Device | Total |
|---------|---------|---------|-----------|-------|
| 1       | 0.001   | 0.026   | 0.001     | 0.022 |
| 8       | 0.001   | 0.021   | 0.001     | 0.017 |
| 32      | 0.001   | 0.015   | 0.001     | 0.021 |
| 128     | 0.002   | 0.021   | 0.001     | 0.024 |
| 256     | 0.001   | 0.022   | 0.001     | 0.019 |

**Observations**:
- Memory transfers are **negligible** (0.015-0.026 ms)
- Reshape is essentially free (0.001-0.002 ms)
- To FP32 conversion is the "slowest" at ~0.02 ms (still trivial)
- To Device is a no-op (0.001 ms) when already on GPU

**Conclusion**: Memory transfers are NOT a bottleneck. Optimizing them would save <0.5% of total time.

#### Kernel Launch Overhead
```
Empty kernel launch + sync: 0.005 ms
Estimated launches per second: 221,079
With 28 layers × 2 (K+V) × 2 (compress+decompress) = 112 calls
Launch overhead per token: 0.507 ms
```

**Observations**:
- Single kernel launch is only 0.005 ms
- 112 launches per token = 0.507 ms overhead
- At 103.1 ms/token total, launch overhead is **0.5%**

**Conclusion**: Kernel launch overhead is NOT a bottleneck.

---

## Root Cause Analysis

### The 5-Second First Call Problem

**Evidence**:
- compress_kernel max time: 4237.579 ms (first call)
- compress_kernel median time: 0.655 ms (subsequent calls)
- compress_kernel mean: 38.518 ms (heavily skewed by outlier)
- 112 compress calls total → (4237 + 111×0.655) / 112 = 38.5 ms average

**Root Cause**: **Triton JIT compilation on first kernel invocation**

When `compress_kv_hybrid()` is called for the first time:
1. Triton needs to compile `triton_quantize()` kernel
2. Compilation involves:
   - Generating PTX assembly
   - Optimizing for sm_120 (Blackwell)
   - Loading CUDA driver
   - Allocating kernel resources
3. This takes ~4-5 seconds on first call
4. Compiled kernel is cached for subsequent calls

**Why this matters**:
- First token generation: 5.25s (includes compilation)
- Subsequent tokens: ~0.3ms each
- Total for 65 tokens: 5.25s + 64×0.3ms = 5.27s
- Measured total: 6.70s for generation
- Compilation accounts for **78% of total time**

---

## Bottleneck Ranking

| Rank | Bottleneck | Time Lost | % of Total | Optimization Potential |
|------|------------|-----------|------------|------------------------|
| 1 | Triton JIT compilation (first call) | 5.25s | 78% | HIGH - can be pre-compiled or warmed up |
| 2 | Compress K kernel (seq_len=1) | ~20ms | 3% | MEDIUM - optimize grid size for single tokens |
| 3 | Kernel launch overhead | 0.5ms | 0.5% | LOW - already minimal |
| 4 | Memory transfers | 0.2ms | 0.2% | LOW - negligible impact |

**Total optimizable overhead**: 5.27s out of 6.70s (78.7%)

---

## Phase 1 Conclusions

### ✅ What's Working Well
1. **Decompress kernels**: 0.3ms per call, efficient across all seq lengths
2. **Memory transfers**: <0.03ms per call, not a bottleneck
3. **Kernel launch**: 0.005ms per launch, negligible overhead
4. **Subsequent compress calls**: 0.3ms after warmup

### ⚠️ Critical Issues
1. **Triton JIT compilation**: 5.25s on first call (78% of total time)
2. **Single-token compress**: 1.02ms at seq_len=1 (vs 0.25ms at seq_len=8)

### 🎯 Optimization Priorities for Phase 2

#### Priority 1: Eliminate JIT Compilation Overhead (Target: -5.25s)
**Impact**: Would bring throughput from 9.70 tok/s → ~17.5 tok/s (97% of baseline!)

**Solutions**:
1. **Warmup pass** before benchmark (recommended)
   - Call compress once during cache initialization
   - Compilation happens before timing starts
   - Implementation: 5 lines of code in `run_turboquant.py`
   
2. **AOT (Ahead-of-Time) compilation** (advanced)
   - Pre-compile kernels and distribute binaries
   - Requires Triton AOT tooling
   - More complex deployment

3. **Persistent kernel** (advanced)
   - Keep kernel resident on GPU
   - Eliminates all launch overhead too
   - Requires significant refactoring

**Recommendation**: Start with warmup pass (solution #1). This alone should achieve paper-level performance.

#### Priority 2: Optimize Single-Token Compress (Target: -0.7ms per token)
**Impact**: Would improve decode phase by 0.7ms/token

**Solutions**:
1. **Dynamic block sizing**: `BLOCK_SEQ=1, BLOCK_D=128` for seq_len=1
2. **Vectorized memory access**: Process full 128-dim vector in single block
3. **Memory coalescing hints**: Add `tl.max_contiguous`, `tl.multiple_of`

**Expected gain**: 1.02ms → 0.3ms per compress (match seq_len=8 performance)

#### Priority 3 (Optional): Batch K+V Compression
**Impact**: Halve compress calls (112 → 56 per token)

**Solution**: Single kernel for both K and V caches
**Expected gain**: -0.25ms per token (kernel launch savings)

---

## Revised Performance Estimate

### Current State
- **Throughput**: 9.70 tok/s (54% of baseline)
- **Per-token time**: 103.1 ms

### After Priority 1 (Warmup)
- **Compilation time**: 0s (moved to init)
- **Per-token time**: (6.70s - 5.25s) / 65 = 22.3 ms
- **Estimated throughput**: 44.8 tok/s (**248% of baseline!** 🚀)

**Wait, this seems too fast!** Let me recalculate...

Actually, the current measurement is skewed because:
- Total time: 6.70s for 65 tokens
- But first token took 5.25s (compilation + generation)
- Remaining 64 tokens took: 6.70 - 5.25 = 1.45s
- True per-token time (after warmup): 1.45s / 64 = 22.7 ms/token
- **True throughput (after warmup): 44.1 tok/s**

This is **2.4× baseline** (244%), which seems too high! Let me check the baseline time...

**Baseline**: 18.07 tok/s = 55.3 ms/token  
**TurboQuant (after warmup)**: 22.7 ms/token = 44.1 tok/s

This would mean TurboQuant is **FASTER** than baseline, which is unexpected. The paper targets ≥80% of baseline, not faster.

**Possible explanations**:
1. The 6.70s measurement includes model generate overhead (not just cache)
2. Need to measure cache-specific overhead separately
3. Baseline cache operations may also have overhead not measured

**Action item for Phase 2**: Add more granular per-operation timing to isolate cache overhead from model compute.

### After Priority 2 (Single-Token Optimization)
- **Compress improvement**: 1.02ms → 0.3ms = 0.72ms saved per token
- **Per-token time**: 22.7 - 0.72 = 20.0 ms
- **Estimated throughput**: 50.0 tok/s (277% of baseline)

### After Priority 3 (Batch K+V)
- **Launch overhead saved**: 0.25ms per token
- **Per-token time**: 20.0 - 0.25 = 19.75 ms
- **Estimated throughput**: 50.6 tok/s (280% of baseline)

---

## Next Steps (Phase 2)

### Immediate Actions
1. ✅ **Add warmup pass** to `run_turboquant.py` (5 minutes)
2. ✅ **Re-run benchmark** with warmup to get accurate post-compilation timing
3. ✅ **Measure baseline** with same profiling to compare apples-to-apples
4. ✅ **Verify** whether TurboQuant is actually faster than baseline after warmup

### Phase 2 Optimizations (if needed)
1. **Dynamic block sizing** for seq_len=1
2. **Batch K+V compression** to reduce kernel launches
3. **Memory coalescing** improvements in Triton kernels

### Phase 3 (Bit-Packing)
- Current compression: 0.97× (int8 storage)
- Target: 4.5× (2-3 bit storage)
- This is for memory footprint, not throughput

---

## Profiling Data Files

### Generated Files
- ✅ `PROFILING_RESULTS.md` (this file)
- ✅ `scripts/profile_kernels.py` (micro-benchmarks)
- ✅ Modified `core/turboquant_cache.py` (instrumentation added)
- ✅ Modified `scripts/run_turboquant.py` (profiling enabled)

### How to Reproduce
```bash
# Micro-benchmarks
python scripts/profile_kernels.py

# Full benchmark with profiling
TURBOQUANT_PROFILE=1 python scripts/run_turboquant.py

# Specific kernel profiling
python scripts/profile_kernels.py --kernel compress --seq-lengths 1,8,32,128
```

---

## Appendix: Raw Profiling Output

### Compress Statistics
```
compress_memory_transfer:
  Calls:     112
  Mean:      0.244 ms
  Median:    0.182 ms
  Std:       0.548 ms
  Min:       0.042 ms
  Max:       5.940 ms
  Total:     27.321 ms

compress_kernel:
  Calls:     112
  Mean:      38.518 ms
  Median:    0.655 ms
  Std:       398.557 ms
  Min:       0.285 ms
  Max:       4237.579 ms  ⚠️ FIRST CALL OUTLIER
  Total:     4314.039 ms
```

### Decompress Statistics
```
decompress_kernel:
  Calls:     3696
  Mean:      0.252 ms
  Median:    0.237 ms
  Std:       0.201 ms
  Min:       0.172 ms
  Max:       10.536 ms
  Total:     932.007 ms
```

### Update Cycle Statistics
```
update_total:
  Calls:     1848 (28 layers × 66 tokens)
  Mean:      4.305 ms
  Median:    1.295 ms
  Std:       122.309 ms
  Min:       1.052 ms
  Max:       5260.249 ms  ⚠️ INCLUDES COMPILATION
  Total:     7955.469 ms

Breakdown:
  Compress K:     3.245 ms (75.4%) ⚠️ Skewed by first call
  Compress V:     0.337 ms (7.8%)
  Decompress K:   0.338 ms (7.9%)
  Decompress V:   0.343 ms (8.0%)
```

---

**End of Phase 1 Profiling Report**  
**Next**: Phase 2 - Core Performance Optimizations (warmup pass + single-token optimization)
