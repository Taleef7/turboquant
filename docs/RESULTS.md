# Forge Evidence Results

## Current result: invalid, not qualified

**Raw artifact ID:** `qualified-v4/20260718T053419Z-unknown`
**Raw local path:** `artifacts/qualified-v4/20260718T053419Z-unknown`
**Raw report status:** `failed`
**Current validation status: `invalid`**

Raw artifacts are ignored by Git. Reproduce a fresh evidence ladder with:

```bash
python scripts/reproduce_release.py --mode qualified --output-dir artifacts/repro_qualified
```

The command exits nonzero whenever any context, provenance, quality, systems,
or capacity gate fails, while preserving the JSON, CSV, Markdown, and child
process logs needed to inspect the result.

## Why the retained artifact is invalid

The raw quality rows requested 4K/8K contexts, but their actual model input
lengths were 3,656/3,659 and **7751/7754** tokens. The current contract defines
`context_target` as requested model-input tokens and `context_actual` as the
post-tokenization input length. It requires at least 99% fidelity: 4,056 tokens
for a 4K row and 8,111 tokens for an 8K row. Every retained row is below that
floor.

The raw artifact also records `git_commit: unknown` and lacks the current Hub
identity binding fields. It therefore cannot validate against the profile,
model revision, and source set that produced a qualified result.

Its recorded diagnostic outcomes—baseline 12/12 and direct profile 0/12 across
the underfilled rows—are **not a qualified 8K result** and cannot support a
retrieval, memory, or throughput claim. The retained evidence is **not retrieval-safe**.

## Systems observation

The raw systems JSON recorded baseline, packed-reference, and Triton-direct
trials with physical-store, allocator, timing, and dispatch fields. It remains
useful for debugging, but it is invalid under the corrected commit/binding
contract. Do not quote its bytes, peak memory, latency, or decode rate as a
result claim.

## 32K outcome

**32K: skipped.** The raw report persisted
`8k_retrieval_gate_failed`; the old calibrated capacity admission also failed.
No 32K child benchmark was launched. An invalid predecessor cannot turn that
skip into a 32K qualification.

## Correctness evidence that remains separate

The real-model Qwen2.5 smoke test covers a 1K prefill plus 64-token decode and
checks direct/reference token agreement, finite logits, packed-block
consumption, and absence of historical concatenation. The GPU parity suites
compare direct packed kernels with the packed PyTorch reference, including the
exact Qwen profile shape. These are implementation-correctness diagnostics, not
substitutes for a qualified retrieval artifact.

## What a future qualified result requires

A future artifact must use the pinned Hub revision, resolved Git commit, full
source/profile/codebook binding, context-fidelity-compliant rows, direct
Triton dispatch provenance, physical-byte consistency, and every declared
quality/systems/capacity gate. See [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).
