# Forge Evidence Results

## Canonical release-ladder result: valid 8K evidence, failed release attempt

**Raw artifact ID:** `qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0`

**Raw local path:** `artifacts/qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0`

**Bound commit:** `5c9d4c34b0f0c31d015e2166ae7ad401858b202d`

**Top-level raw report status:** `failed`

**Structural validation:** valid under the artifact's bound contract

Raw artifacts are intentionally ignored by Git. Reproduce fresh evidence with:

```bash
python scripts/reproduce_release.py --mode qualified --output-dir artifacts/repro_qualified
```

The command exits nonzero whenever a declared quality, systems, capacity, or
provenance gate does not pass, while retaining its report and child logs. A
failed release attempt is not a release qualification.

## Latest local engineering checkpoint (diagnostic only)

The release artifact above is the canonical historical release-ladder result.
The latest implementation checkpoint is documented separately so its numbers
are not mistaken for a new qualification. Clean raw artifacts are bound to
`eba1c5b0239971ab2b78e1acceadda0885b6cb28` (the publication-only documentation
commit is `cd502e2`):

- [Local long-context results](LOCAL_LONG_CONTEXT_RESULTS.md): Qwen2.5 4K/8K
  warmup; all six packed rows fail strict full-logit gates, despite exact
  top-1/greedy choices in the small matrix.
- [Local fused-page results](LOCAL_FUSED_PAGE_RESULTS.md): synthetic fused
  reduction/tail is 25.2–79.5% faster than serial over 3–63 pages; this is not
  model-level evidence.
- [Local value-bit ablation](LOCAL_VALUE_BIT_ABLATION.md): the 8-bit-value
  candidate is roughly 12% larger and 49–59.5% slower than v6 and also fails
  the strict full-logit gates.

These diagnostics do not establish a 32K result or an end-to-end speed/VRAM
win. Their raw JSON is ignored by Git; preserve it externally and rerun the
validator when source, environment, model, or configuration changes.

## Qualified 8K quality matrix

The quality child is `qualified`. It contains all 12 required paired cases:

- exact 4,096- and 8,192-token inputs;
- depths 0.1, 0.5, and 0.9;
- seeds 0 and 1 at each depth and context;
- baseline and Triton-direct correctness of 12/12 each; and
- exact baseline/direct output-token-ID agreement in all 12 cases.

The paired bootstrap reports a 0.0 percentage-point delta and 0.0
percentage-point upper 95% bound (`seed=1729`, `iterations=10000`). This is
evidence for the named 4K/8K NIAH matrix only; it is not a 32K result, a broad
benchmark result, or a paper-equivalence claim.

## Complete 8K systems comparison: no end-to-end win

The 8K systems child is complete and records direct Triton dispatch. The direct
physical store occupies 233,447,424 bytes:

| Physical component | Bytes |
| --- | ---: |
| K payload | 113,770,496 |
| V payload | 85,327,872 |
| Metadata | 15,998,976 |
| fp16/bf16 tail | 18,350,080 |
| **Total direct store** | **233,447,424** |

The same artifact's end-to-end measurements do not show a speed or VRAM
improvement:

| Measurement | Triton-direct | Baseline |
| --- | ---: | ---: |
| Decode rate | 1.29 tok/s | 33.99 tok/s |
| Decode p50 | 727.09 ms | 27.86 ms |
| Peak reserved | 11,876,171,776 bytes | 11,293,163,520 bytes |
| Peak allocated | 9,278,377,472 bytes | 9,169,062,400 bytes |

Physical KV storage is measured separately from allocator peaks. In this
configuration, the compressed direct path is slower and uses more peak reserved
and allocated memory than baseline, so it does not establish an end-to-end
performance or VRAM advantage.

## 32K capacity outcome

The persisted 32K admission preflight is structurally valid and confirms the
8K systems child, but its capacity decision is false. It projects
12,809,961,472 bytes against an allowed 10,405,439,078-byte budget. The report
records `8k_capacity_admission_failed`, sets `attempt_32k: false`, and skips the
32K leg. No 32K quality or systems child was launched.

## Retired invalid v4 evidence

`qualified-v4/20260718T053419Z-unknown` remains available as a historical,
retired invalid artifact. Its requested 4K/8K rows tokenized to only
3,656/3,659 and 7,751/7,754 tokens, and its provenance binding was incomplete.
It must not be used for quality, memory, or throughput claims.

## Correctness evidence that remains separate

The real-model Qwen2.5 smoke test covers a 1K prefill plus 64-token decode and
checks direct/reference token agreement, finite logits, packed-block
consumption, and absence of historical concatenation. GPU parity suites compare
direct packed kernels with the packed PyTorch reference. These implementation
checks complement, but do not replace, the named quality and systems artifacts.

## What a future release result requires

A future result needs its own bound artifact with valid matrix/provenance,
direct dispatch, physical-byte consistency, capacity admission, and all
declared gates. See [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) and
[REPLICATION_CHECKLIST.md](REPLICATION_CHECKLIST.md).
