# Paper/Blog Claims Status (Qwen-Focused Repo Scope)

This matrix makes claim status explicit so users can distinguish what is replicated here vs deferred.

| Claim | Source | Repo Status | How to Verify | Expected Result |
|---|---|---|---|---|
| TurboQuant can preserve long-context retrieval quality with strong compression | Paper + blog | **Partially replicated (Qwen scope)** | `scripts/test_long_context.py --mode paired` through 16K | Delta `<= 2.0pp` in this scope |
| TurboQuant supports strong completion quality | Repo validation target | **Replicated** | `scripts/test_multimodel.py --model qwen2.5-7b` | TurboQuant output quality matches baseline trend |
| Practical KV memory savings are substantial | Paper + blog + repo | **Replicated (project metrics)** | `scripts/benchmark_throughput.py` | Compression summary reported (project target exceeded) |
| Near-lossless 3-bit operating point from paper/blog | Paper + blog | **Not yet replicated in this repo scope** | N/A | Deferred (current robust settings are higher bit-width) |
| LongBench parity table | Paper + blog | **Not yet replicated** | N/A | Deferred |
| RULER parity | Blog | **Not yet replicated** | N/A | Deferred |
| Vector-search (PQ/RabitQ recall@k) parity | Paper + blog | **Not yet replicated** | N/A | Deferred |

## Current Claim Policy

Safe to claim now:
- Qwen-focused paired NIAH replication path is implemented and reproducible in this repo.
- Throughput/compression and quality comparisons are reproducible with provided scripts.

Do not claim yet:
- Full paper benchmark parity (LongBench, RULER, vector-search comparisons).
- Generalized near-lossless low-bit parity across all models/contexts.
