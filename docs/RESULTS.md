# Forge Evidence Results

## Current result: not qualified

**Raw artifact ID:** `qualified-v4/20260718T053419Z-unknown`
**Raw local path:** `artifacts/qualified-v4/20260718T053419Z-unknown`
**Release status:** `failed`

Raw artifacts are ignored by Git. Reproduce them with:

```bash
python scripts/reproduce_release.py --mode qualified --output-dir artifacts/repro_qualified
```

The command exits nonzero when the profile is not qualified, while preserving
the JSON, CSV, Markdown, and child-process logs needed to inspect the result.

## Quality outcome

The quality artifact is structurally complete: contexts 4K/8K × depths
0.1/0.5/0.9 × seeds 0/1, all completed and tokenization-preserving. It reports:

> Complete 4K-8K matrix: baseline 12/12; direct profile 0/12
>
> Exact 8K subset: baseline 6/6; direct profile 0/6

The paired delta and the deterministic upper 95% bootstrap bound are both 100
percentage points. The retrieval-safety threshold is an upper bound no greater
than two percentage points, so this direct profile is **not retrieval-safe**.

## Systems outcome

The systems artifact completed a baseline, packed-reference, and Triton-direct
trial at 8K. The direct entry recorded physical packed-store bytes, allocator
peaks, timing distributions, and dispatch counters. It demonstrates that the
direct path runs and consumes packed blocks; it does not establish a performance
or memory win. Exact values and category breakdowns are in the raw artifact
under `8k-systems.json`.

The 8K-calibrated capacity decision estimates 12,809,961,472 bytes for 32K
against a 10,405,439,078-byte safety budget. Capacity admission therefore fails.

## 32K outcome

**32K: skipped.** The persisted decision reason is
`8k_retrieval_gate_failed`; the calibrated capacity admission also fails. No
32K child benchmark was launched and no 32K success claim exists.

## Model smoke and kernel correctness

The real-model Qwen2.5 smoke test covered a 1K prefill plus 64-token decode and
passed its direct/reference token agreement, finite-logit, packed-block, and
historical-concatenation checks. The maximum emitted fp16-logit drift was
`16.935546875`. That smoke is a dispatch/correctness diagnostic—not a substitute
for the failed retrieval matrix.

The GPU parity suite compares direct packed kernel outputs with the packed
PyTorch reference. It validates kernel mechanics, not model-level quality.

## How to use this result

Treat this artifact as a baseline for profile improvement and regression tests.
Do not use it to claim retrieval safety, a speedup, reduced end-to-end memory,
or portability to another model family. The release becomes qualified only when
the protocol's quality, systems, capacity, and validator gates all pass.
