# TurboQuant Forge Issue Tracker

**Last reconciled:** 2026-07-18
**Evidence baseline:** `artifacts/qualified-v4/20260718T053419Z-unknown`

This tracker separates implementation closure from evidence promotion. A code
path is marked resolved only when it has implementation and automated tests; a
model/profile result remains open until the evidence ladder qualifies it.

## Resolved implementation work

| Item | Status | Evidence in repository |
| --- | --- | --- |
| Reproducible CPU/GPU runtime contract | Resolved | `tests/test_environment.py`, lock inputs, runtime lock, CI CPU lane. |
| Immutable manifests, profiles, and fail-closed capability gates | Resolved | `tests/test_capabilities.py`, `tests/test_profiles.py`. |
| Physically packed KV blocks and exact byte accounting | Resolved | `tests/test_packing.py`, `tests/test_codec.py`, `tests/test_packed_store.py`. |
| Blockwise packed reference attention | Resolved | `tests/test_reference_attention.py`. |
| Direct Triton score/value kernels | Resolved | `tests/test_packed_attention_kernels.py` and GPU benchmark test. |
| Pinned Qwen2.5 direct adapter boundary | Resolved for its narrow contract | `tests/test_qwen_adapter.py`, opt-in `tests/test_qwen_adapter_gpu.py`. |
| Validator-first quality/systems artifacts | Resolved | `tests/test_benchmark_schema.py`, `tests/test_artifact_validation.py`, `tests/test_release_ladder.py`. |
| Public-claim regression guard | Resolved | `tests/test_documentation_claims.py`. |

`core/turboquant_cache_v2.py` remains intentionally **legacy**. New profiles
cannot select it and it is not a resolved direct-backend route.

## Open issue: Qwen2.5 profile quality promotion

**Priority:** critical
**Status:** open — current profile is not retrieval-safe

The complete current 4K-8K quality matrix records baseline 12/12 versus direct
profile 0/12; its exact 8K subset records baseline 6/6 versus direct profile
0/6. Its release report is `failed`, and it explicitly skips 32K after the 8K
quality gate. Do not use kernel parity or the short model smoke to close this
issue.

### Promotion criteria

- A complete, binding-consistent quality matrix with baseline at least 95%.
- Paired bootstrap upper 95% delta no greater than two percentage points.
- A structurally valid systems artifact with direct dispatch and no fallback.
- Calibrated capacity admission before any 32K leg.
- Updated [RESULTS.md](docs/RESULTS.md) naming the resulting raw artifact.

## Open issue: Qwen3 dense adapter

**Priority:** high
**Status:** open — manifest-only, no adapter

`configs/models/qwen3.json` permits capability discovery only. It does not
verify a Transformers attention layout, RoPE/cache behavior, model smoke,
quality matrix, or systems benchmark. Implement a dedicated adapter and all
promotion steps in [PORTABILITY.md](docs/PORTABILITY.md).

## Open issue: Qwen3.5 hybrid adapter

**Priority:** high
**Status:** open — manifest-only, no native linear-state runtime

The manifest identifies full-attention layers as potentially compressible and
linear-attention layers as native-only. Forge currently rejects that hybrid
layout for direct decode. The adapter must preserve recurrent linear state and
compress only full-attention KV blocks before it can be tested or promoted.

## Open issue: Gemma 4 E4B adapter and capacity scope

**Priority:** high
**Status:** open — manifest-only, multimodal boundary unimplemented

The manifest records a required capacity preflight and excludes vision state
from KV claims. It does not prove model loading, processor integration, cache
layout, or a workable GPU footprint. A dedicated adapter and explicit
text/vision cache tests are required.

## Open issue: broader research comparison

**Priority:** medium
**Status:** open

Forge does not yet run LongBench, RULER, or the paper's ANN/vector-search
evaluation families. A QJL residual profile is also unqualified. Add each only
with a fixed protocol, a reference comparator, and raw evidence artifacts.

## Deferred by design

Serving integrations, batching, beam search, sampling, speculative/MTP decode,
distributed execution, offload, and multimodal cache compression are outside
this milestone. They must remain fail-closed until an explicit capability,
adapter, and evidence plan is added.
