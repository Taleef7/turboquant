# Local fused page-attention results

This diagnostic isolates the GPU page-attention schedule from model loading and
tokenization. It compares the serial packed path, the page-state path, and the
new fused parallel path on the RTX 5070 Ti Laptop GPU.

## Reproducibility

| Check | Value |
| --- | --- |
| Raw artifact | `artifacts/local-page-fused-clean-eba1c5b.json` |
| Bound commit | `eba1c5b0239971ab2b78e1acceadda0885b6cb28` |
| Source tree | clean detached worktree (`dirty: false`) |
| Page counts | `3, 7, 15, 31, 63` |
| Tail | fixed 256-token synthetic tail |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |

Run it with:

```bash
python scripts/benchmark_parallel_page_attention.py \
  --implementations serial page-state parallel \
  --output artifacts/local-page-fused.json
```

The generated artifact validates itself and contains no public claims.

## Fused schedule

The parallel lane now performs three GPU launches per decode step:

1. query rotation;
2. page-state construction;
3. one Triton kernel that chronologically reduces page states and merges the
   raw tail.

The old reduction and tail launches remain available for kernel diagnostics,
but are not used by the parallel adapter. Every measured parallel row reports
`rotation=1`, `page=1`, `reduction=0`, `tail=0`, `reduce_tail=1`, and
`allocation=0`.

## Timing and parity

| Pages | Serial p50 (ms) | Page-state p50 (ms) | Fused parallel p50 (ms) | Parallel vs serial |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.106816 | 0.028944 | 0.079888 | 25.2% faster |
| 7 | 0.186976 | 0.055552 | 0.106832 | 42.9% faster |
| 15 | 0.486192 | 0.100496 | 0.155344 | 68.0% faster |
| 31 | 0.965632 | 0.184448 | 0.235552 | 75.6% faster |
| 63 | 2.046992 | 0.365136 | 0.419488 | 79.5% faster |

All 15 rows are `passed`. The fused rows have exact repeatability,
top-1 agreement `1.0`, and maximum absolute error at or below
`1.05e-05` against the float32 synthetic oracle. The dedicated kernel suite
also covers five page counts and seven tail lengths (35 exact fused-vs-staged
parity cases).

## Interpretation

The fused path removes one launch and is substantially faster than serial
page-by-page decoding, especially as page count grows. It is still slower than
the page-state-only stage because that stage intentionally omits tail attention;
that is not an apples-to-apples end-to-end comparison. The separate Qwen
4K/8K warmup artifact therefore remains the authority for model-level quality,
VRAM, and latency conclusions.
