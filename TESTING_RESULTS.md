# TurboQuant Forge — Testing Results

**Evidence status:** the latest qualified release attempt failed its quality
gate. This document supersedes historical prototype summaries; it does not
promote a failed artifact into a successful result.

## Evidence sources

Raw release artifacts are intentionally ignored from version control because
they contain machine-specific logs and measurements. Recreate them with
`scripts/reproduce_release.py`; the source for the observations below is:

```text
artifacts/qualified-v4/20260718T053419Z-unknown
```

The artifact was captured on the local RTX 5070 Ti Laptop GPU (12,820,480,000
bytes reported total VRAM), WSL2, Python 3.12.3, PyTorch 2.7.0+cu128,
Transformers 4.57.3, and Triton 3.3.0. Its generated `git_commit` field is
`unknown`, so the directory name and binding hashes—not a claimed revision—are
the available identifiers for this run.

## Current release outcome

`repro_report.json` in the raw artifact reports:

| Field | Value |
| --- | --- |
| Release status | `failed` |
| Quality status | `failed` |
| Systems status | `completed` and structurally valid |
| Retrieval-safe | `false` |
| 32K decision | `skipped` (`8k_retrieval_gate_failed`) |

### Quality gate

The complete 4K-8K quality matrix has 12 completed, tokenization-preserving
cases: contexts 4K and 8K; depths 0.1, 0.5, and 0.9; seeds 0 and 1. Across that
complete matrix, the result is **baseline 12/12, direct profile 0/12**. The
exact 8K subset is **baseline 6/6, direct profile 0/6**. The paired delta and
deterministic bootstrap upper 95% bound are both 100 percentage points. The
profile is therefore **not retrieval-safe** under the protocol's
two-percentage-point upper-bound gate.

This is useful negative evidence: the harness caught a model-level quality
regression that kernel parity and a short decode smoke cannot detect.

### Systems observation, not a win claim

The 8K systems artifact completed one interleaved trial for baseline,
packed-reference, and Triton-direct backends. Its direct trial reported:

| Measurement | Artifact value |
| --- | ---: |
| Physical packed store bytes | 233,447,424 |
| Peak allocated bytes | 9,278,377,472 |
| Peak reserved bytes | 11,876,171,776 |
| Decode p50 / p95 | 719.165 / 1,088.576 ms |
| Decode rate | 1.240 tok/s |
| Direct dispatches | 1,792 |
| Packed blocks consumed | 55,552 |

These are measurements from the named raw artifact, not an assertion of a
speed or memory improvement. In the same artifact, the baseline decode rate is
26.039 tok/s, so this profile has no end-to-end performance claim.

The calibrated 8K-to-32K reservation estimate is 12,809,961,472 bytes against
a 10,405,439,078-byte safety budget. It fails capacity admission independently
of the failed quality gate. No 32K quality or systems child process was run.

## Correctness evidence that did pass

- CPU tests cover bit packing, codec/store accounting, capability/profile
  validation, blockwise reference attention, artifact validation, and public
  documentation claims without a GPU or model download.
- The GPU kernel suite compares packed Triton scoring/value mixing with the
  packed PyTorch reference across supported shapes. It is a kernel-parity
  result, not an end-to-end quality result.
- The opt-in Qwen2.5 smoke test uses a 1K-token prefill plus 64 decode tokens,
  checks direct/reference greedy-token agreement, finite logits, packed-block
  consumption, and absence of historical KV concatenation. It observed maximum
  fp16-logit drift `16.935546875`; that is deliberately emitted, not hidden.

## Retired prototype claims

Older documents described a generic legacy `TurboQuantCacheV2` experiment as
if it established broad multi-model quality, compression, or throughput. Those
observations are not evidence for Forge's packed direct backend: V2 reconstructs
historical K/V state and the new profile loader rejects it. They are retained in
source history only, not in the release claim surface.

For the exact acceptance and invalidation rules, read
[docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md). For a reproduction
sequence, read [docs/REPLICATION_CHECKLIST.md](docs/REPLICATION_CHECKLIST.md).
