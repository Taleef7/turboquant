# TurboQuant Paper Replication - Implementation Plan

**Goal**: Achieve full TurboQuant paper replication with ≥80% baseline throughput, 4.5× compression, and easily replicable codebase.

**Paper Reference**: arXiv:2504.19874 - "TurboQuant: Efficient Near-Lossless KV Cache Compression"

**Hardware**: RTX 5070 Ti (Blackwell sm_120, 12.8GB VRAM)

---

## Current Status

### ✅ Completed
- [x] Basic TurboQuant implementation (Python, Triton, Hybrid)
- [x] Rotation matrix generation (QR decomposition with sign correction)
- [x] Lloyd-Max optimal quantization (2-bit: 4 levels, 3-bit: 8 levels)
- [x] QJL correction formula implementation
- [x] Hybrid approach: PyTorch cuBLAS + Triton quantization
- [x] All 25 unit tests passing
- [x] Qwen2.5-7B-Instruct benchmarking infrastructure

### 📊 Current Performance (as of 2025-03-31)
- **Baseline**: 14.61 tok/s (200 tokens from 3800 token prompt)
- **TurboQuant (Hybrid)**: ~10 tok/s BUT only generates 65 tokens before EOS
- **Current throughput**: ~68% of baseline (when generating)
- **Current compression**: 0.97× (int8 storage, no bit-packing)

### 🔴 CRITICAL BUG: No-buffer decode quality still degraded (2026-04-01)

**Symptom**: TurboQuant in strict no-buffer mode still produces degraded output.
- With prompt "The capital of France is", baseline predicts " Paris" ✓
- TurboQuant WITH buffer (128): " Paris" ✓ (tokens stay in buffer, no compression)
- TurboQuant WITHOUT buffer: Degraded/repetitive output ❌

**Root Cause (Phase 1.5 update)**:
1. ✅ **Fixed**: Value compression path was wrong (K/V treated identically).
   - Implemented separate V path: per-group min/max quantization (no rotation, no QJL)
   - Unit tests pass and V roundtrip quality is high.
2. 🔴 **Remaining**: Key path quality in no-buffer decode remains unstable under current settings.
    - Buffer mode behaves correctly.
    - No-buffer mode still drifts from baseline despite V fix.
3. ✅ **Confirmed isolation**: Disabling key compression restores no-buffer output quality.
   - Added optional raw-key storage path (`compress_keys=False` or `TURBOQUANT_COMPRESS_KEYS=0`).
   - With raw keys + compressed values, no-buffer generation matches baseline-style output.

**Per-Layer Cosine Similarity Analysis**:
| Layer | K cosine sim | V cosine sim | Status |
|-------|--------------|--------------|--------|
| 0 | 98.5% ✓ | 98.7% ✓ | OK |
| 1 | 98.0% ✓ | **18.7%** ❌ | BROKEN |
| 2 | 95.0% ✓ | **24.9%** ❌ | BROKEN |
| 4-26 | 72-94% | **24-55%** ❌ | BROKEN |
| 27 | 98.9% ✓ | 74.3% | Marginal |

**Why V Fails While K Succeeds**:
1. **Value vectors have much smaller ranges** than Keys:
   - K at layer 1: `[-39.3, 62.7]` (large dynamic range)
   - V at layer 1: `[-1.0, 0.9]` (tiny dynamic range)
2. After unit-norm normalization + rotation, V coordinates may not match the Lloyd-Max codebook range `[-0.24, 0.24]`
3. The quantization designed for N(0, 1/d) distribution may not fit Value distributions
4. Current implementation treats K and V identically in `_compress()` (lines 243-344 of `turboquant_cache.py`)

**Blocking Phase 2**: Cannot proceed with performance optimization until quality is fixed.

**ROOT CAUSE CONFIRMED** (from 0xSero/turboquant research 2026-04-01):
The TurboQuant paper's Algorithm 2 (TurboQuantProd) is designed **specifically for Keys** which participate in Q·K^T attention scoring. Values only need MSE-accurate reconstruction.

**The 0xSero reference implementation treats K and V COMPLETELY DIFFERENTLY:**

| Aspect | Keys (K) | Values (V) |
|--------|----------|------------|
| **Method** | TurboQuantProd (Algorithm 2) | Simple Group Quantization |
| **Rotation** | Yes (Haar-uniform random rotation) | **NO ROTATION** |
| **QJL** | Yes (sign bits on residual) | **NO QJL** |
| **Quantization** | Lloyd-Max optimal codebook | Per-group min-max (asymmetric) |
| **Why** | Keys need unbiased inner product for attention | Values just need reconstruction |

**Industry Research Consensus (KIVI, KVQuant papers)**:
- Keys: Benefit from **per-channel** quantization (outliers concentrate in specific channels)
- Values: Benefit from **per-token/per-group** quantization (no channel outlier pattern)

**Implemented Solution (Step 1 complete)**:
- Keys: TurboQuant key path retained
- Values: New group quantization (per-group min-max, NO rotation)
- Added optional bit-packed storage path for 4-bit indices + 1-bit QJL signs

### 🎯 Target Performance (from paper)
- **Throughput**: ≥14.5 tok/s (≥80% of baseline)
- **Compression**: 4.5× (with bit-packing at 3.5 bits/channel)
- **Quality**: Pass NIAH tests, perplexity within paper bounds
- **Models**: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B, Gemma-2-9B
- **Context lengths**: 4K, 16K, 32K, 64K, 128K

### ⚠️ Performance Gap
- Need **4× throughput improvement** (~3.56-3.80 → 14.5 tok/s in latest no-buffer runs)
- Need **5.5× overhead reduction** (83ms/token → <15ms/token)
- Need **4.7× compression improvement** (0.97× → 4.5×)

---

## Research Findings

### Paper Methodology (from arXiv:2504.19874)
1. **Random rotation**: Haar-uniform orthogonal matrix (QR with sign correction)
2. **Lloyd-Max quantization**: Optimal scalar quantizers for N(0, 1/d)
   - 2-bit: ±0.453/√d, ±1.510/√d
   - 3-bit: {±0.245, ±0.756, ±1.344, ±2.152} × 1/√d
3. **QJL correction**: x̃ = Π^T·ŷ + (√(π/2)/k)·γ·S^T·qjl
4. **Batched processing**: All batch*heads*seq in single kernel
5. **Expected results**: ≥80% throughput, 4.5× compression, quality neutral at 3.5 bits/channel

### Identified Bottlenecks (from profiling research)
1. **Memory transfers**: 20-30ms/token - redundant `.to()` calls
2. **Kernel launch overhead**: 15-25ms/token - small grids (1×4 blocks) for seq_len=1
3. **Synchronization points**: 8-15ms/token - PyTorch ↔ Triton handoffs
4. **Memory access patterns**: 10-15ms/token - poor coalescing for single-token decode
5. **Full sequence decompression**: 10-15ms/token - reading entire cache every token

### Optimization Strategies (from GPU research)
1. **Eliminate redundant memory transfers**: In-place ops, pre-allocation, pinned memory
2. **Dynamic block sizes**: BLOCK_SEQ=1 for seq_len<8, BLOCK_D=128 for full vectorization
3. **Kernel fusion**: Batch K+V compression, fuse pack/unpack operations
4. **Memory coalescing**: `tl.max_contiguous`, `tl.multiple_of` hints, aligned access
5. **Persistent kernels**: Keep kernel resident to eliminate launch overhead
6. **CUDA graphs**: Capture decode iteration for 5-10× faster replay
7. **FP8 on Blackwell**: Native `tl.float8e4nv` for 2× throughput

---

## Implementation Phases

### Phase 1: Profiling & Measurement ✅ COMPLETED
**Duration**: 1 day (completed 2025-03-31)
**Status**: 🟢 Complete  
**Priority**: CRITICAL

#### 1.1 Add Detailed Instrumentation ✅
**Files**:
- `/home/taleef/projects/turboquant/core/turboquant_cache.py` ✅
- `/home/taleef/projects/turboquant/kernels/compress_kv.py` (not needed - instrumented at cache level)
- `/home/taleef/projects/turboquant/kernels/decompress_kv.py` (not needed - instrumented at cache level)

**Tasks**:
- [x] Add CUDA event timing around memory transfers
- [x] Add timing for each kernel call (compress, decompress)
- [x] Add timing for PyTorch matmuls (rotation, QJL)
- [x] Add timing for full `update()` cycle
- [x] Environment variable: `TURBOQUANT_PROFILE=1` to enable

**Completed**: Per-component breakdown of 103.1ms/token overhead collected

#### 1.2 Create Micro-benchmark Suite ✅
**New file**: `/home/taleef/projects/turboquant/scripts/profile_kernels.py` ✅

**Tasks**:
- [x] Benchmark compress with varying seq_len (1, 8, 32, 128, 256)
- [x] Benchmark decompress with varying seq_len
- [x] Isolate memory transfer overhead
- [x] Measure kernel launch latency
- [x] Test different BLOCK_SIZE values (via seq_len variations)
- [x] Profile grid size sensitivity

**Completed**: Comprehensive micro-benchmarks showing 1.5-6.3× hybrid speedup

#### 1.3 Analyze Results ✅
**Tasks**:
- [x] Run `TURBOQUANT_PROFILE=1 python scripts/run_turboquant.py`
- [x] Run `python scripts/profile_kernels.py`
- [x] Create bottleneck report with specific line numbers
- [x] Validate that identified bottlenecks sum to ~103ms/token
- [x] Prioritize optimizations by impact (ms saved per optimization)

**Deliverable**: `PROFILING_RESULTS.md` with detailed analysis ✅

**Key Findings**:
1. 🔴 **CRITICAL**: Triton JIT compilation takes 5.25s on first call (78% of total time)
2. ⚠️ Single-token compress: 1.02ms (vs 0.25ms for seq_len=8)
3. ✅ Decompress is efficient: 0.3ms per call
4. ✅ Memory transfers negligible: 0.02ms
5. ✅ Kernel launch overhead minimal: 0.005ms

**Major Discovery**: After warmup, TurboQuant may actually be **faster** than baseline!
- Current: 9.70 tok/s (includes 5.25s compilation)
- After warmup: ~44 tok/s estimated (244% of baseline!)
- Need to verify with proper warmup pass

---

### Phase 1.5: Quality Bug Investigation 🔴 IN PROGRESS
**Duration**: 1-2 days
**Status**: 🟡 In Progress (started 2026-03-31)
**Priority**: CRITICAL (blocking all other phases)

**Goal**: Diagnose why Value cache compression fails and implement fix.

#### 1.5.1 Create Fast Diagnostic Script ✅
**New file**: `/home/taleef/projects/turboquant/scripts/diagnose_kv_quality.py`

**Tasks**:
- [x] Load model ONCE (avoid slow reloading)
- [x] Run prefill with baseline DynamicCache to capture real K/V tensors
- [x] Test compression/decompression roundtrip SEPARATELY for K and V
- [x] Compute per-layer cosine similarities
- [x] Analyze statistical properties:
  - Mean, std, min, max per layer
  - Distribution shape (histogram)
  - Outlier percentage
- [x] Compare K vs V distributions to understand why V fails
- [x] Output structured report

**Expected deliverable**: Clear diagnosis of K vs V distribution differences

#### 1.5.2 Verify K vs V Quality Gap ✅
**Tasks**:
- [x] Run diagnostic script
- [x] Confirm V has much worse cosine similarity than K
- [x] Identify which layers are most affected
- [x] Measure correlation between range size and quality

#### 1.5.3 Implement and Test Fix 🟡 PARTIAL
**Based on diagnosis, implement ONE of**:
- [ ] **Option A**: Separate rotation matrices for K vs V
- [ ] **Option B**: Different bit widths (4-bit V, 2-bit K)
- [x] **Option C**: Separate codebooks/paths for K vs V behavior
- [ ] **Option D**: Don't compress V (check if paper does this)
- [ ] **Option E**: Adaptive scaling per layer

**Verification**:
- [x] Re-run quality test with fix
- [x] Confirm V quality restored (>99% cosine in tests)
- [x] Verify generation produces correct output with buffer mode
- [x] Run full benchmark with and without bit-packing
- [ ] Confirm no-buffer generation quality matches baseline

#### 1.5.4 Document Findings 🟡 IN PROGRESS
- [ ] Update DEBUG_LOG.md with root cause analysis
- [x] Update PLAN.md with fix details
- [x] Add unit tests for K vs V quality

#### 1.5.5 Bit-packing milestone (new) ✅
- [x] Added `kernels/bitpack_triton.py` (vectorized pack/unpack helpers)
- [x] Added 4-bit index pack/unpack + 1-bit sign pack/unpack
- [x] Integrated optional bit-packing storage in `TurboQuantCache` via `TURBOQUANT_USE_BITPACKING=1`
- [x] Added tests for bitpacking roundtrip and cache compression ratio with bitpacking
- [x] Verified compressed KV size dropped significantly in benchmark output
- [x] Added packed append/decode-safe handling for incremental cache growth
- [ ] Optimize packed-path runtime (currently throughput regressed)

#### 1.5.6 Latest measured status (2026-04-01)
- `scripts/test_kernels.py`: **23/23 passing**
- `scripts/debug_incremental.py`:
  - Baseline: correct
  - TurboQuant + buffer: correct
  - TurboQuant + no buffer: still degraded (`pérdida ... TheTheThe...`)
- `scripts/run_turboquant.py` (latest):
  - unpacked mode: ~7.80 tok/s, compressed KV ~188 MB
  - packed mode: ~3.56 tok/s, compressed KV ~86.8 MB
  - Compression improved significantly, throughput currently regressed
- No-buffer isolation run (`TURBOQUANT_FORCE_PYTHON=1`, 20-token decode from "The capital of France is"):
  - `compress_keys=1`: degraded/repetitive (`pérdida ... TheThe...`)
  - `compress_keys=0`: coherent output (`Paris. It is located...`)
- Quick timing/size comparison (same short prompt, no-buffer):
  - `compress_keys=1`: ~40.6s, ~1127 KB cached
  - `compress_keys=0`: ~1.0s, ~1102 KB cached
  - Indicates current key compression path is both quality and latency bottleneck under strict no-buffer decode.

**Expected outcome**: Quality bug fixed, Phase 2 unblocked

---

### Phase 2: Core Performance Optimizations ⏸️ PENDING
**Duration**: 3-5 days  
**Status**: 🔴 Blocked (waiting for Phase 1.5 quality fix)  
**Priority**: CRITICAL

#### 2.1 Eliminate Memory Transfer Overhead
**Target**: -20-30ms/token  
**Files**: `core/turboquant_cache.py`

**Tasks**:
- [ ] Remove redundant `.to()` calls (line 152)
- [ ] Ensure inputs are already on correct device/dtype
- [ ] Use in-place dtype conversion where possible
- [ ] Pre-allocate output buffers and reuse
- [ ] Use pinned memory for rotation matrices (Pi, S)
- [ ] Add `out=` parameter for in-place operations

**Verification**: Re-run profiling, confirm 20-30ms reduction

#### 2.2 Optimize Triton Kernel Grid for Single-Token Decode
**Target**: -15-25ms/token  
**Files**: `kernels/quantize_triton.py`

**Tasks**:
- [ ] Implement dynamic block sizing: `BLOCK_SEQ = 1 if seq_len < 8 else 32`
- [ ] Increase `BLOCK_D` to 128 for full vectorization
- [ ] Add memory coalescing hints: `tl.max_contiguous`, `tl.multiple_of`
- [ ] Ensure 128-byte aligned access patterns
- [ ] Use vectorized loads where supported
- [ ] Test grid configurations: (1,1) vs (1,4) vs (32,4)

**Verification**: Profile seq_len=1 case specifically

#### 2.3 Fuse Compress+Decompress Operations
**Target**: -10-15ms/token  
**Files**: `core/turboquant_cache.py`

**Tasks**:
- [ ] Implement uncompressed buffer for recent tokens
- [ ] Add `uncompressed_buffer_size=32` parameter
- [ ] Keep last N tokens uncompressed in cache
- [ ] Only compress when buffer full or on eviction
- [ ] Avoid compress+decompress roundtrip for recent tokens

**Verification**: Measure cache hit rate, confirm overhead reduction

#### 2.4 Batch K+V Compression
**Target**: -5-10ms/token  
**Files**: `core/turboquant_cache.py`, `kernels/compress_kv.py`

**Tasks**:
- [ ] Create `compress_kv_batched()` function
- [ ] Concatenate K and V caches: `torch.cat([k_flat, v_flat], dim=0)`
- [ ] Single kernel launch for both K and V
- [ ] Split outputs after compression
- [ ] Update `_compress()` to use batched version

**Verification**: Halve kernel launch count, measure timing improvement

#### 2.5 Expected Results
- **Throughput**: 12-14 tok/s (67-77% of baseline)
- **Cache overhead**: 20-35ms/token (down from 83ms)
- **Status**: Still below target, but major progress

---

### Phase 3: Bit-Packing Implementation ⏸️ PENDING
**Duration**: 2-3 days  
**Status**: 🔴 Blocked (waiting for Phase 1.5 + Phase 2)  
**Priority**: CRITICAL (for paper replication)

#### 3.1 Design Bit-Packed Storage Format
**New file**: `/home/taleef/projects/turboquant/core/packed_cache.py`

**Storage layout** (per token):
- 2-bit channels (96 dims): 192 bits = 24 bytes
- 3-bit outliers (32 dims): 96 bits = 12 bytes
- QJL signs: 128 bits = 16 bytes
- Gamma: 4 bytes (float32)
- **Total**: 56 bytes (vs 256 bytes FP16 = **4.57× compression**)

**Tasks**:
- [ ] Define `CompressedKVCachePacked` class
- [ ] Implement storage allocation: `idx_2bit_packed`, `idx_3bit_packed`, `qjl_packed`, `gamma`
- [ ] Design memory layout for efficient GPU access
- [ ] Add metadata: sequence length, head count, layer info

#### 3.2 Implement Pack/Unpack Triton Kernels
**New file**: `/home/taleef/projects/turboquant/kernels/bitpack_triton.py`

**Kernels**:
- [ ] `pack_2bit_kernel`: Pack 4 indices (0-3) into 1 byte
- [ ] `pack_3bit_kernel`: Pack 2-3 indices (0-7) efficiently
- [ ] `unpack_2bit_kernel`: Extract indices from packed bytes
- [ ] `unpack_3bit_kernel`: Extract indices from packed bytes
- [ ] `pack_qjl_kernel`: Pack ±1 signs into bits

**Implementation notes**:
```python
@triton.jit
def pack_2bit(idx0, idx1, idx2, idx3):
    # Pack 4 × 2-bit values into 1 byte
    return (idx0 & 0x3) | ((idx1 & 0x3) << 2) | ((idx2 & 0x3) << 4) | ((idx3 & 0x3) << 6)
```

**Tasks**:
- [ ] Implement bit manipulation with masking and shifting
- [ ] Handle edge cases (non-multiple-of-4 sequences)
- [ ] Optimize for coalesced memory access
- [ ] Add unit tests for pack/unpack roundtrip

#### 3.3 Integrate with Compress/Decompress Pipeline
**Files**: `kernels/compress_kv.py`, `kernels/decompress_kv.py`

**Tasks**:
- [ ] Modify `compress_kv_hybrid()` to call pack kernel after quantization
- [ ] Modify `decompress_kv_hybrid()` to call unpack kernel before dequantization
- [ ] Update `TurboQuantCache` to use packed storage
- [ ] Add environment variable: `TURBOQUANT_USE_BITPACKING=1`
- [ ] Maintain backward compatibility with int8 storage

#### 3.4 Verification
**Tasks**:
- [ ] Add unit tests: `scripts/test_bitpacking.py`
- [ ] Verify roundtrip correctness: compress → pack → unpack → decompress
- [ ] Measure compression ratio: confirm 4.5× achieved
- [ ] Measure throughput impact: should be neutral or positive (less memory bandwidth)
- [ ] Run full benchmark: `python scripts/run_turboquant.py`

**Expected results**:
- **Compression**: 4.5× (56 bytes vs 256 bytes)
- **Throughput**: 14-16 tok/s (bit-packing may improve memory bandwidth)
- **Quality**: No degradation (lossless packing)

---

### Phase 4: Advanced Optimizations ⏸️ PENDING
**Duration**: 2-3 days  
**Status**: 🔴 Blocked (waiting for Phase 3)  
**Priority**: MEDIUM (if needed to reach ≥80% target)

#### 4.1 Persistent Kernels (If needed for latency)
**Target**: Eliminate kernel launch overhead entirely  
**New file**: `/home/taleef/projects/turboquant/kernels/persistent_kernels.py`

**Tasks**:
- [ ] Implement persistent kernel pattern with loop
- [ ] Add signal mechanism for incoming tokens
- [ ] Keep kernel resident on GPU between calls
- [ ] Measure launch overhead reduction

**Note**: Only implement if Phase 2+3 don't reach 80% target

#### 4.2 CUDA Graphs (For decode phase)
**Target**: 5-10× faster kernel launch  
**Files**: `scripts/run_turboquant.py`

**Tasks**:
- [ ] Capture entire decode iteration as CUDA graph
- [ ] Replay graph for subsequent tokens
- [ ] Handle dynamic shapes (if needed)
- [ ] Measure latency reduction

**Note**: Only implement if Phase 2+3 don't reach 80% target

#### 4.3 FP8 Quantization (Blackwell-specific)
**Target**: 2× throughput vs FP16  
**Files**: `kernels/quantize_triton.py`

**Tasks**:
- [ ] Use `tl.float8e4nv` for tensor operations
- [ ] Leverage native FP8 tensor cores on sm_120
- [ ] Measure quality impact (should be minimal)
- [ ] Compare throughput vs current FP32/FP16 pipeline

**Note**: Optional - extends beyond paper scope

---

### Phase 5: Multi-Model Testing & Validation ⏸️ PENDING
**Duration**: 2-3 days  
**Status**: 🔴 Blocked (waiting for Phase 3 or 4)  
**Priority**: HIGH (for paper replication)

#### 5.1 Test on Paper's Benchmark Models
**Models** (from paper):
- [ ] Qwen2.5-7B-Instruct (already done)
- [ ] Llama-3.1-8B
- [ ] Mistral-7B-v0.3
- [ ] Gemma-2-9B
- [ ] (Optional) Phi-3-medium

**New scripts**:
- [ ] `/home/taleef/projects/turboquant/scripts/run_llama.py`
- [ ] `/home/taleef/projects/turboquant/scripts/run_mistral.py`
- [ ] `/home/taleef/projects/turboquant/scripts/run_gemma.py`

**Tasks**:
- [ ] Download and test each model
- [ ] Run baseline benchmarks
- [ ] Run TurboQuant benchmarks
- [ ] Verify ≥80% throughput on all models
- [ ] Record results in comparison table

#### 5.2 Long-Context Testing
**Sequence lengths** (from paper):
- [x] 4K (current)
- [ ] 16K
- [ ] 32K
- [ ] 64K
- [ ] 128K (paper's max)

**Tasks**:
- [ ] Generate or download long-context prompts
- [ ] Modify benchmark scripts to support longer sequences
- [ ] Verify compression ratio remains stable
- [ ] Verify throughput scales linearly
- [ ] Measure VRAM usage at each length

#### 5.3 Quality Benchmarks
**Metrics** (from paper):
- [ ] Needle-in-a-Haystack @ 10% depth
- [ ] Needle-in-a-Haystack @ 50% depth
- [ ] Needle-in-a-Haystack @ 90% depth
- [ ] Perplexity on WikiText-2
- [ ] Inner product distortion measurements
- [ ] MSE distortion measurements

**New scripts**:
- [ ] `/home/taleef/projects/turboquant/scripts/test_niah.py`
- [ ] `/home/taleef/projects/turboquant/scripts/test_perplexity.py`
- [ ] `/home/taleef/projects/turboquant/scripts/test_distortion.py`

**Tasks**:
- [ ] Implement each quality metric
- [ ] Run on baseline cache
- [ ] Run on TurboQuant cache
- [ ] Compare against paper's reported values
- [ ] Verify quality neutrality at 3.5 bits/channel

#### 5.4 Expected Results
Create comprehensive results table in `TESTING_RESULTS.md`:

| Model | Baseline TPS | TurboQuant TPS | % of Baseline | Compression | NIAH Pass |
|-------|--------------|----------------|---------------|-------------|-----------|
| Qwen2.5-7B | 18.07 | 14.5+ | ≥80% | 4.5× | ✓ |
| Llama-3.1-8B | TBD | TBD | ≥80% | 4.5× | ✓ |
| Mistral-7B | TBD | TBD | ≥80% | 4.5× | ✓ |
| Gemma-2-9B | TBD | TBD | ≥80% | 4.5× | ✓ |

---

### Phase 6: Documentation & Replicability ⏸️ PENDING
**Duration**: 1-2 days  
**Status**: 🔴 Blocked (waiting for Phase 5)  
**Priority**: HIGH (for public release)

#### 6.1 Update Documentation
**Files to update**:
- [ ] `README.md`: Add bit-packing explanation, multi-model results, usage examples
- [ ] `TESTING_RESULTS.md`: Comprehensive benchmark table with all models and metrics
- [ ] `OPTIMIZATION_GUIDE.md`: Document all optimizations, tradeoffs, and architecture decisions
- [ ] `PROFILING_RESULTS.md`: Detailed profiling analysis and bottleneck breakdown
- [ ] `API.md`: Public API documentation for TurboQuantCache

#### 6.2 Create Reproduction Scripts
**New files**:
- [ ] `/home/taleef/projects/turboquant/scripts/reproduce_paper_results.sh`
  - One-click script to run all benchmarks
  - Compare against paper's tables
  - Generate performance plots
  - Output: `paper_comparison_report.md`

- [ ] `/home/taleef/projects/turboquant/scripts/setup_environment.sh`
  - Automated setup for CUDA, PyTorch nightly, Triton
  - Check for RTX 5070 Ti compatibility (sm_120)
  - Verify driver versions
  - Download test models

- [ ] `/home/taleef/projects/turboquant/scripts/download_models.sh`
  - Download all benchmark models
  - Verify checksums
  - Set up Hugging Face cache

#### 6.3 Add Configuration Presets
**New file**: `/home/taleef/projects/turboquant/configs/presets.py`

```python
PRESETS = {
    "quality": {
        "n_bits_normal": 3,
        "n_bits_outlier": 4,
        "n_outliers": 32,
        "use_bitpacking": True,
        "description": "Quality neutral (3.5 bits/channel avg)"
    },
    "balanced": {
        "n_bits_normal": 2,
        "n_bits_outlier": 3,
        "n_outliers": 24,
        "use_bitpacking": True,
        "description": "Marginal degradation (2.5 bits/channel avg)"
    },
    "aggressive": {
        "n_bits_normal": 2,
        "n_bits_outlier": 2,
        "n_outliers": 16,
        "use_bitpacking": True,
        "description": "High compression (2.0 bits/channel avg)"
    },
}
```

**Tasks**:
- [ ] Implement preset system
- [ ] Add CLI flag: `--preset quality|balanced|aggressive`
- [ ] Document each preset's quality/compression tradeoff
- [ ] Add preset examples to README

#### 6.4 Create Tutorial Notebooks
**New files**:
- [ ] `notebooks/01_quickstart.ipynb`: Basic usage example
- [ ] `notebooks/02_configuration.ipynb`: Advanced configuration options
- [ ] `notebooks/03_profiling.ipynb`: How to profile and optimize
- [ ] `notebooks/04_multi_model.ipynb`: Using with different models

#### 6.5 Add CI/CD
**New file**: `.github/workflows/test.yml`

**Tasks**:
- [ ] Set up GitHub Actions for automated testing
- [ ] Run unit tests on every commit
- [ ] Run benchmark suite on tagged releases
- [ ] Check for CUDA/Triton compatibility
- [ ] Generate performance reports

#### 6.6 Final Checklist
- [ ] All code is documented with docstrings
- [ ] All scripts have `--help` flags with clear descriptions
- [ ] README includes installation, quickstart, and examples
- [ ] Performance results match or exceed paper's claims
- [ ] Repository is ready for public release

---

## Success Criteria

### ✅ Must-Have (Paper Replication)
- [ ] **Throughput**: ≥14.5 tok/s (80% of 18.07 baseline) on Qwen2.5-7B
- [ ] **Compression**: 4.5× with bit-packing (56 bytes vs 256 bytes FP16)
- [ ] **Quality**: Pass NIAH tests at 10%, 50%, 90% depth
- [ ] **Correctness**: All unit tests pass (25/25 + new bit-packing tests)
- [ ] **Multi-model**: Test on ≥3 models (Qwen, Llama, Mistral)

### ✅ Should-Have (Robustness)
- [ ] **Long-context**: Verify up to 64K tokens
- [ ] **Perplexity**: Within paper's reported bounds
- [ ] **Documentation**: Clear setup and reproduction instructions
- [ ] **Presets**: Easy-to-use quality/balanced/aggressive modes

### ⚪ Nice-to-Have (Beyond Paper)
- [ ] **128K context**: Full paper replication (if hardware allows)
- [ ] **FP8 optimization**: Leverage Blackwell architecture
- [ ] **CUDA graphs**: Further latency reduction
- [ ] **CI/CD**: Automated testing and benchmarking
- [ ] **Tutorial notebooks**: Interactive examples

---

## Timeline & Milestones

| Phase | Duration | Start Date | End Date | Status | Milestone |
|-------|----------|------------|----------|--------|-----------|
| Phase 1 | 1 day | 2025-03-31 | 2025-03-31 | 🟢 Complete | Bottleneck analysis complete |
| Phase 1.5 | 1-2 days | 2026-03-31 | TBD | 🟡 In Progress | Quality bug fixed |
| Phase 2 | 3-5 days | TBD | TBD | 🔴 Blocked | 12-14 tok/s achieved |
| Phase 3 | 2-3 days | TBD | TBD | 🔴 Blocked | 4.5× compression achieved |
| Phase 4 | 2-3 days | TBD | TBD | 🔴 Blocked | ≥80% throughput confirmed |
| Phase 5 | 2-3 days | TBD | TBD | 🔴 Blocked | Multi-model validation complete |
| Phase 6 | 1-2 days | TBD | TBD | 🔴 Blocked | Public release ready |
| **Total** | **12-19 days** | 2025-03-31 | TBD | ⏳ In Progress | Full paper replication |

---

## Knowledge Base

### Key Implementation Decisions

#### Why Hybrid Approach?
- **cuBLAS for matmuls**: Already highly optimized, hard to beat
- **Triton for quantization**: Custom logic, element-wise operations benefit from Triton
- **Result**: Best of both worlds, sm_120 compatible

#### Why Dynamic Block Sizing?
- **Single-token decode** (seq_len=1): Small grids waste GPU capacity
- **Prompt processing** (seq_len=128+): Large grids maximize parallelism
- **Solution**: `BLOCK_SEQ = 1 if seq_len < 8 else 32`

#### Why Uncompressed Buffer?
- **Observation**: Recent tokens are accessed frequently during generation
- **Compress+decompress roundtrip**: Expensive for just-generated tokens
- **Solution**: Keep last 32 tokens uncompressed, only compress on eviction

#### Why Bit-Packing Matters?
- **int8 storage**: 1 byte per element = 128 bytes for indices + 128 bytes for QJL = 256 bytes (same as FP16!)
- **Bit-packing**: 2-3 bits per element = 24+12+16 bytes = 52 bytes (4.9× compression)
- **Memory bandwidth**: Fewer bytes to transfer = faster decompression

### Common Pitfalls

#### Triton on sm_120
- ❌ **Don't use**: `tl.static_range(0, 128)` - causes compilation hang
- ✅ **Use instead**: Tiled operations with `tl.dot()`, vectorized gather

#### Memory Transfers
- ❌ **Don't do**: `x.to(device).to(dtype)` - redundant copies
- ✅ **Do instead**: `x.float()` if already on correct device

#### Kernel Launch Overhead
- ❌ **Don't do**: Launch many small kernels in loop
- ✅ **Do instead**: Batch operations, use vectorized single kernel

### Useful Commands

```bash
# Activate virtual environment (REQUIRED before all Python commands)
source venv312/bin/activate

# Enable profiling
TURBOQUANT_PROFILE=1 python scripts/run_turboquant.py

# Force Python fallback (CPU/GPU debugging, or non-2-bit configs)
# CRITICAL: Use this for non-2-bit configs due to Triton kernel bug
TURBOQUANT_FORCE_PYTHON=1 python scripts/run_turboquant.py

# Enable bit-packing (Phase 3+)
TURBOQUANT_USE_BITPACKING=1 python scripts/run_turboquant.py

# Run specific benchmark
python scripts/run_baseline.py
python scripts/run_turboquant.py

# Run all tests
pytest scripts/test_math.py scripts/test_kernels.py -v

# Profile specific kernel
python scripts/profile_kernels.py --kernel compress --seq-len 1

# Debug incremental compression (with buffer)
python scripts/debug_incremental.py

# Reproduce paper results (Phase 6)
bash scripts/reproduce_paper_results.sh
```

### Key Files Reference

| File | Purpose |
|------|---------|
| `core/turboquant_cache.py` | Main cache class, `_compress()` at lines 243-344 handles K and V identically |
| `utils/math_utils.py` | Lloyd-Max codebooks, rotation matrices, quantization functions |
| `kernels/compress_kv.py` | Compression kernels (Python lines 40-114, Hybrid lines 120-173) |
| `kernels/decompress_kv.py` | Decompression kernels |
| `kernels/quantize_triton.py` | Triton quantization (⚠️ hardcoded 2-bit bug at lines 68-79) |
| `scripts/debug_incremental.py` | Debug script comparing baseline vs TurboQuant |
| `scripts/capture_real_kv.py` | Capture real KV tensors (⚠️ bug at line 66) |
| `scripts/test_math.py` | Unit tests for math utilities |
| `scripts/test_kernels.py` | Unit tests for kernels |

### Known Bugs

1. **Triton kernel hardcoded 2-bit** (`kernels/quantize_triton.py` lines 68-79):
   - Centroid counts are hardcoded for 2-bit (4 centroids)
   - Non-2-bit configs fail silently or produce wrong results
   - **Workaround**: Use `TURBOQUANT_FORCE_PYTHON=1` for other bit widths

2. **capture_real_kv.py attribute error** (line 66):
   - Uses `cache_layer.key_states` but `DynamicLayer` doesn't have that
   - **Fix**: Use `cache.key_cache[layer_idx]` and `cache.value_cache[layer_idx]`

3. **No-buffer decode quality remains degraded** (Phase 1.5):
    - V-path catastrophic issue fixed, but key-path no-buffer decode still drifts
    - Buffer mode works; no-buffer mode still repetitive/off-distribution
    - **Status**: Active investigation
    - **Temporary mitigation**: disable key compression with `TURBOQUANT_COMPRESS_KEYS=0`
      to preserve decode quality while key quantization path is being fixed.

4. **Bit-packed runtime overhead currently high** (new):
   - Compression ratio improves, but throughput regresses (~3.56 tok/s in latest run)
   - Cause: extra pack/unpack overhead in hot path and repeated unpack on decode
   - **Status**: Needs optimization

### Performance Targets Reference

| Metric | Current | Phase 2 Target | Phase 3 Target | Final Target |
|--------|---------|----------------|----------------|--------------|
| Throughput | 7.24 tok/s | 12-14 tok/s | 14-16 tok/s | ≥14.5 tok/s |
| % of Baseline | 40% | 67-77% | 77-89% | ≥80% |
| Cache Overhead | 83 ms/token | 20-35 ms/token | 10-20 ms/token | <15 ms/token |
| Compression | 0.97× | 0.97× | 4.5× | 4.5× |
| Memory Used | 256 bytes/tok | 256 bytes/tok | 56 bytes/tok | 56 bytes/tok |

---

## Change Log

### 2026-03-31 (Evening) - Quality Bug Deep Dive
- 🔴 **CRITICAL BUG ROOT CAUSE IDENTIFIED**: Value cache compression catastrophically broken
- Ran `scripts/debug_incremental.py`:
  - Baseline: " Paris" ✓ (correct)
  - TurboQuant WITH buffer (128): " Paris" ✓ (all tokens in buffer, compression bypassed)
  - TurboQuant WITHOUT buffer: Timed out during model reload (120s limit)
- **Key Discovery**: Per-layer analysis shows V cosine similarity 18-55% while K is 72-98%
- **Root Cause**: V vectors have tiny ranges ([-1, 0.9]) vs K ranges ([-39, 62])
  - Quantization designed for N(0, 1/d) doesn't fit V distribution
  - Current `_compress()` treats K and V identically (lines 243-344)
- Created Phase 1.5: Quality Bug Investigation (blocking Phase 2)
- Identified fix hypotheses:
  1. Separate rotation matrices for K vs V
  2. Different bit widths (4-bit V, 2-bit K)
  3. Separate codebooks for K vs V distributions
  4. Don't compress V at all (check paper)
  5. Adaptive scaling per layer
- **Next**: Create `diagnose_kv_quality.py` to analyze K vs V distributions

### 2026-04-01 - V-path fix + bit-packing integration
- ✅ Implemented separate K/V compression behavior in `TurboQuantCache`
  - K path keeps TurboQuant key logic
  - V path uses group min/max quantization (no rotation, no QJL)
- ✅ Added optional bit-packed storage (`TURBOQUANT_USE_BITPACKING=1`)
  - 4-bit index packing and 1-bit QJL sign packing
- ✅ Added and passed new tests (21/21 in `scripts/test_kernels.py`)
- ✅ Benchmarked packed mode:
  - Compressed KV size dropped to ~86.8 MB (from ~188 MB unpacked in same benchmark)
  - Throughput regressed to ~3.56 tok/s (needs optimization)
- 🔴 Remaining blocker: no-buffer generation quality still degraded

### 2026-04-01 - No-buffer key-path isolation + mitigation
- ✅ Added optional raw key storage mode in `TurboQuantCache`:
  - constructor flag: `compress_keys=False`
  - env toggle: `TURBOQUANT_COMPRESS_KEYS=0`
- ✅ Added regression tests:
  - `test_cache_no_buffer_raw_keys_preserve_exactly`
  - `test_cache_raw_keys_storage_path_is_accounted`
- ✅ Verified isolation: no-buffer quality issue disappears when key compression is bypassed.
- 🔴 Conclusion: key compression path is remaining root cause for no-buffer degradation.

### Known Issues with Existing Scripts
- `scripts/capture_real_kv.py` has bug at line 66: uses `cache_layer.key_states` but `DynamicLayer` doesn't have that attribute
  - Should use `cache.key_cache[layer_idx]` and `cache.value_cache[layer_idx]` instead
- `scripts/debug_incremental.py` times out (120s) when testing without buffer due to model reload
  - Need faster diagnostic that loads model once

### 2025-03-31 (Evening)
- ✅ **PHASE 1 COMPLETED**: Profiling & Measurement
- Added CUDA event timing to `turboquant_cache.py`
- Created `scripts/profile_kernels.py` micro-benchmark suite
- Ran comprehensive profiling on full benchmark
- Created `PROFILING_RESULTS.md` with detailed analysis
- **Key Discovery**: Triton JIT compilation (5.25s) is the bottleneck, not kernel performance
- **Major Finding**: After warmup, TurboQuant is estimated at ~44 tok/s (244% of baseline!)
- Updated `PLAN.md` with Phase 1 completion status

### 2025-03-31 (Morning)
- Created comprehensive implementation plan
- Identified 6 phases with detailed tasks
- Documented current status: 7.24 tok/s (40% of baseline)
- Established success criteria and timeline (11-18 days)
- Ready to begin Phase 1: Profiling & Measurement

---

## Next Actions

1. **Immediate**: Stabilize no-buffer decode quality (key-path focused)
   - Add targeted diagnostics for per-token drift in key cache
   - Validate effect of key bit-width and outlier settings under no-buffer decode

2. **Immediate**: Optimize packed-path performance
   - Reduce repeated unpack operations in hot decode path
   - Profile pack/unpack cost and move work out of per-token critical path

3. **After quality/perf stabilization**: Continue Phase 2 optimization tasks
   - kernel/grid tuning, memory transfers, and decode-path overhead reduction

---

*This document is a living knowledge base and will be updated as we progress through each phase.*
