# Paper and Blog Claims Status

TurboQuant Forge is a clean-room engineering project inspired by the TurboQuant
paper. This table distinguishes an implemented mechanism from a reproduced
paper result. It is not a claim of equivalence with the paper or any other
implementation.

| Paper-area claim | Forge status | Evidence or gap |
| --- | --- | --- |
| Low-bit KV representation can be physically stored compactly | Implemented | Fixed packed blocks, byte accounting, and CPU store tests exist. Physical-size conclusions belong only to a named systems artifact. |
| Rotation, Lloyd-Max, and Product-style codec families can be compared on controlled data | Locally evidenced as a v1 partial reproduction | `codec-local-v1` contains 1,408 passed rows, 16 paired seeds, four synthetic distributions, and a pinned GloVe subset. See [REPRODUCTION.md](REPRODUCTION.md) and [CLAIMS.md](CLAIMS.md). |
| Compressed attention can be evaluated without full-history reconstruction | Implemented for pinned Qwen2.5 decode | Blockwise reference and direct Triton kernels have parity tests; the legacy V2 cache is excluded. |
| Compressed KV preserves long-context retrieval quality | Partially evidenced at 8K only | Bound artifact `qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0` has an exact 12/12 4K/8K NIAH matrix with output-token-ID agreement. Its 32K leg was not launched after capacity admission failed. |
| End-to-end speed or memory advantage | Not established as an advantage | The same 8K comparison reports direct 1.29 tok/s versus baseline 33.99 tok/s and higher peak reserved/allocated memory, so it is negative systems evidence rather than a win. |
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
- The codec artifact is a clean, local v1 partial reproduction with explicit
  seed, dataset, timing, and retrieval policies; its claim ledger resolves
  every published numeric value to an artifact query.

## Statements this repository does not make

- It does not claim 32K retrieval safety, a paper-level compression result, or
  an end-to-end throughput/memory improvement.
- It does not claim direct adapter support for Qwen3, Qwen3.5, or Gemma 4.
- It does not attribute model weights, vision processing state, or other
  non-KV allocations to KV-cache compression.

See [RESULTS.md](RESULTS.md) for the current artifact-backed result and
[PORTABILITY.md](PORTABILITY.md) for promotion requirements.
