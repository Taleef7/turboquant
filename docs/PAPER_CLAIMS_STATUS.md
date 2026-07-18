# Paper and Blog Claims Status

TurboQuant Forge is a clean-room engineering project inspired by the TurboQuant
paper. This table distinguishes an implemented mechanism from a reproduced
paper result. It is not a claim of equivalence with the paper or any other
implementation.

| Paper-area claim | Forge status | Evidence or gap |
| --- | --- | --- |
| Low-bit KV representation can be physically stored compactly | Implemented | Fixed packed blocks, byte accounting, and CPU store tests exist. Physical-size conclusions belong only to a named systems artifact. |
| Compressed attention can be evaluated without full-history reconstruction | Implemented for pinned Qwen2.5 decode | Blockwise reference and direct Triton kernels have parity tests; the legacy V2 cache is excluded. |
| Compressed KV preserves long-context retrieval quality | Not reproduced | The latest complete 4K-8K Forge NIAH matrix fails: baseline 12/12 and direct profile 0/12 overall; its exact 8K subset is baseline 6/6 and direct profile 0/6. |
| End-to-end speed or memory advantage | Not established | The current direct systems artifact is slower than its baseline and is not a performance or memory win claim. |
| QJL residual path delivers the paper operating point | Not reproduced | Forge has no qualified QJL profile or evidence artifact. |
| LongBench, RULER, or ANN/vector-search comparisons | Not implemented | No runner or result artifact is shipped for these benchmark families. |
| Multi-model adapter portability | Not established | Qwen3, Qwen3.5, and Gemma 4 are manifest-only/fail-closed entries. |

## Safe release statements

- Forge has a tested direct packed-decode implementation only for the pinned
  Qwen2.5-7B-Instruct text configuration.
- Its evidence machinery records physical bytes, allocator peaks, timing,
  profile/model/codebook/code hashes, logs, and quality-gate outcomes.
- Failed and capacity-rejected runs are retained so the project can report
  regressions instead of converting them into marketing claims.

## Statements this repository does not make

- It does not claim a retrieval-safe profile, paper-level compression result,
  or an end-to-end throughput/memory improvement.
- It does not claim direct adapter support for Qwen3, Qwen3.5, or Gemma 4.
- It does not attribute model weights, vision processing state, or other
  non-KV allocations to KV-cache compression.

See [RESULTS.md](RESULTS.md) for the current artifact-backed result and
[PORTABILITY.md](PORTABILITY.md) for promotion requirements.
