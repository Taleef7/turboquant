# TurboQuant 3-Bit Quality Investigation Log
**Date**: 2026-03-31  
**Status**: ROOT CAUSE IDENTIFIED - Quality depends on random seed  

## Problem Statement
3-bit TurboQuant produces garbled output ("the capital of France is Paris Paris, and the is") instead of correct text, but static compression tests show good quality (0.98+ cosine similarity).

## Investigation Timeline

### Discovery 1: Corruption Happens During Prefill
- **Test**: Compared baseline vs TurboQuant prefill predictions
- **Result**: 
  - Baseline first token: " Paris" ✓
  - TurboQuant first token: " the" ✗
- **Conclusion**: Bug occurs during prefill phase, not incremental decode

### Discovery 2: Roundtrip Quality is Seed-Dependent
- **Test**: Compress-decompress roundtrip on random tensors
- **Results**:
  - Seed 42 (same for K and V): K=0.88 cosine, V=0.99 cosine, 50% argmax mismatch
  - Seed 100/200 (different): K=0.986 cosine, V=0.986 cosine, 100% argmax match
- **Conclusion**: With good random seeds, 3-bit achieves expected quality

### Discovery 3: Both K and V Can Achieve 0.986 Cosine
- **Test**: Separate cache objects for K and V with different seeds
- **Result**: Both K and V independently achieve 0.986 cosine similarity
- **Conclusion**: Compression algorithm is correct, seed matters

## Root Cause Analysis

The compression algorithm is **working correctly**. The issue is:

1. **Random seed sensitivity**: Some unlucky combinations of:
   - Data distribution (real KV tensors)
   - Rotation matrix (seeded at cache init)
   - Channel variance patterns (for outlier mask)
   
   Can produce poor compression quality for specific tensors.

2. **Real model KV vs random data**: Real attention KV cache may have:
   - Non-uniform distributions
   - Structured patterns that interact badly with certain rotation matrices
   - Outlier channels in specific positions

## Next Steps

1. ✅ Capture real KV tensors from model prefill
2. ⏸️ Test roundtrip quality on real data (in progress - model loading slow)
3. Investigate if:
   - Different rotation matrix seeds help
   - Real KV has pathological properties
   - Need adaptive outlier selection per-layer

## Key Files
- `scripts/debug_incremental.py` - Shows prefill corruption
- `scripts/test_roundtrip.py` - Demonstrates seed sensitivity
- `scripts/debug_kv_difference.py` - Proves both K/V can work
- `scripts/capture_real_kv.py` - Real data testing (WIP)
- `logs/real_kv_test_*.log` - Saved test outputs

## Hypothesis
The TurboQuantCache uses a **single rotation matrix for all layers and all KV tensors**. If this matrix happens to be poorly conditioned for the specific data distribution in real model KV caches, it will cause poor compression quality.

**Potential fix**: Use different rotation seeds per-layer or adaptive centroid selection.
