# TurboQuant Forge — Testing Results

**Evidence status:** the current bound artifact has a qualified 8K quality
matrix and a complete 8K systems comparison, but the top-level release report
is `failed` because 32K capacity admission fails. It is not a 32K or release
qualification.

## Evidence source

Raw release artifacts are intentionally ignored from version control because
they contain machine-specific logs and measurements. The observations below are
from:

```text
artifacts/qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0
```

The artifact is bound to
`5c9d4c34b0f0c31d015e2166ae7ad401858b202d` and was captured on the local RTX
5070 Ti Laptop GPU (12,820,480,000 reported total VRAM), WSL2, Python 3.12.3,
PyTorch 2.7.0+cu128, Transformers 4.57.3, and Triton 3.3.0.

## Latest diagnostic checkpoint

The current implementation checkpoint is not the historical release artifact
above. Clean diagnostic JSON is bound to code commit `eba1c5b` and is described
in [LOCAL_LONG_CONTEXT_RESULTS.md](docs/LOCAL_LONG_CONTEXT_RESULTS.md),
[LOCAL_FUSED_PAGE_RESULTS.md](docs/LOCAL_FUSED_PAGE_RESULTS.md), and
[LOCAL_VALUE_BIT_ABLATION.md](docs/LOCAL_VALUE_BIT_ABLATION.md). The Qwen2.5
4K/8K warmup has 8/8 complete rows but all six packed rows fail strict
full-logit gates; the fused synthetic schedule is faster than serial but is
not model-level evidence; and the v8 value ablation is larger/slower and also
fails the strict gates. These artifacts do not change the historical release
status or establish a 32K/end-to-end win.

## Current release outcome

`repro_report.json` records the following state:

| Field | Value |
| --- | --- |
| Raw report status | `failed` |
| Structural validation | Valid under the artifact's bound contract |
| Quality child | `qualified` |
| Systems child | `completed` with direct Triton dispatch |
| 32K decision | `8k_capacity_admission_failed`; skipped before a child launch |

### Exact 8K quality gate

The quality child exercises all 12 required cases: exact 4,096 and 8,192 token
inputs, depths 0.1/0.5/0.9, and seeds 0/1. Baseline and direct are correct in
12/12 cases, and their output token IDs match exactly in all 12. The paired
bootstrap delta is 0.0 percentage points with an upper 95% bound of 0.0
percentage points (`seed=1729`, `iterations=10000`).

This qualifies the exact 4K/8K NIAH matrix in this named artifact. It does not
generalize to 32K, broad long-context benchmarks, other models, or a paper-level
claim.

### Complete 8K systems comparison, not a win claim

The direct path physically stores 233,447,424 bytes: 113,770,496 bytes of K
payload, 85,327,872 of V payload, 15,998,976 of metadata, and 18,350,080 of
bounded fp16/bf16 tail. Its end-to-end measurements compare unfavorably with
baseline:

| Measurement | Triton-direct | Baseline |
| --- | ---: | ---: |
| Decode rate | 1.29 tok/s | 33.99 tok/s |
| Decode p50 | 727.09 ms | 27.86 ms |
| Peak allocated | 9,278,377,472 bytes | 9,169,062,400 bytes |
| Peak reserved | 11,876,171,776 bytes | 11,293,163,520 bytes |

Accordingly, the artifact establishes neither an end-to-end speedup nor a VRAM
win. Physical cache bytes and allocator peaks remain different measurements.

### 32K capacity stop

The admission calculation is based on the valid 8K systems child and projects
12,809,961,472 bytes at 32K against an allowed 10,405,439,078-byte budget. It
records `8k_capacity_admission_failed`; `attempt_32k` is false. No 32K quality
or systems child process was launched.

## Correctness checks separate from release evidence

- CPU tests cover bit packing, codec/store accounting, capability/profile
  validation, blockwise reference attention, artifact validation, and public
  documentation claims without a GPU or model download.
- The GPU kernel suite compares packed Triton scoring/value mixing with the
  packed PyTorch reference across supported shapes. It is a kernel-parity
  result, not an end-to-end quality result.
- The opt-in Qwen2.5 smoke test uses a 1K-token prefill plus 64 decode tokens,
  checks direct/reference greedy-token agreement, finite logits, packed-block
  consumption, and absence of historical KV concatenation. It observed maximum
  fp16-logit drift `16.935546875`; that value is diagnostic only.

## Retired invalid v4 artifact

`qualified-v4/20260718T053419Z-unknown` is retained for historical debugging,
not as release evidence. It is invalid because its requested 4K/8K rows used
only 3,656/3,659 and 7,751/7,754 effective tokens and its provenance binding
was incomplete. Older generic `TurboQuantCacheV2` prototype statements are also
not evidence for Forge's packed direct backend.

For the exact acceptance and invalidation rules, read
[docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md). For a reproduction
sequence, read [docs/REPLICATION_CHECKLIST.md](docs/REPLICATION_CHECKLIST.md).
For a cluster continuation with raw-artifact transfer rules, read
[docs/GILBRETH_HANDOFF.md](docs/GILBRETH_HANDOFF.md).
