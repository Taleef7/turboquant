# Benchmark Protocol

TurboQuant Forge benchmarks are validator-first. A completed command is not a
result claim until its generated JSON passes structural validation and its
quality/capacity gates pass.

## Artifact contract

`scripts/reproduce_release.py` creates timestamped directories under the
requested `artifacts/` output root. A release run retains:

- `quality_matrix` JSON and CSV rows;
- `systems_benchmark` JSON and Markdown summaries;
- `qualified_run` report and persisted capacity/32K decisions;
- per-child stdout/stderr logs and invoked command metadata;
- pinned Hub model ID/revision identity, manifest/profile/codebook/source/runner
  hashes, and environment metadata with a resolved Git commit.

The artifact directory is ignored by Git. A result document must name its raw
artifact ID and reproduction command; it must not require the raw directory to
be committed.

## Quality matrix

The runner builds the expected `(context, depth, seed)` matrix before executing
anything. `context_target` means requested model-input tokens and
`context_actual` means the post-tokenization length actually supplied to the
model. Every row must be present, completed, tokenization-preserving, and
within 99%–100% of its target. It must also bind the same pinned Hub revision,
profile, codebook, and source hashes. A missing or underfilled row, OOM,
exception, unresolved Git commit, altered binding, or tokenization loss makes
the quality artifact invalid.

For every eligible paired row the runner records baseline and profile outcome.
The baseline must reach at least 95% accuracy. It reports baseline-minus-profile
accuracy and a deterministic paired bootstrap upper 95% confidence bound using
seed 1729 and 10,000 resamples. A profile is retrieval-safe only when that upper
bound is at most two percentage points.

## Systems matrix

Each systems run interleaves baseline, packed-reference, and Triton-direct
trials after warmups and explicit CUDA synchronization. The artifact records
actual packed-store bytes by category, peak allocated/reserved memory, prefill
latency, TTFT, per-token decode distribution, p50/p95, decode rate, direct
dispatch counts, and an OOM-probe sequence.

The direct benchmark must not use the legacy V2 cache and a qualified direct
trial must not silently fall back to the PyTorch reference. A direct trial must
report positive `triton_direct` dispatches, zero packed-reference dispatches,
and positive packed-block consumption whenever packed payload exists. The
reported store bytes must exactly equal its physical category breakdown.
System timing is not an end-to-end win unless a valid artifact reports the
comparator and conditions.

## Release ladder

`--mode quick` is a labeled 8K smoke. `--mode qualified` performs:

1. A 1K systems preflight and conservative capacity estimate.
2. A complete 8K quality matrix and 8K systems matrix when preflight admits it.
3. A calibrated 8K-to-32K capacity admission.
4. The 32K child legs only when preflight, 8K retrieval, 8K systems validity,
   and calibrated capacity all pass.

The capacity calculation reserves a 25% cache-growth contingency and a 10%
free-VRAM safety margin. A skipped 32K leg is a persisted decision, not a pass.

## States and interpretation

| State | Meaning | Can support a claim? |
| --- | --- | --- |
| `qualified` | Every required structural, quality, systems, and capacity gate passed. | Yes, within the exact artifact scope. |
| `failed` | Artifact is structurally valid but a semantic gate, such as retrieval safety, failed. | No success claim; use as negative evidence. |
| `invalid` | Matrix/binding/environment/run completeness is broken, or an error/OOM/tokenization loss occurred. | No. |
| `inconclusive` | The run preserves a reason but lacks the evidence required for a conclusion. | No. |

Run an artifact validator directly when inspecting an individual JSON file:

```bash
python scripts/validate_artifact.py path/to/artifact.json
```

The validator checks structure and bindings. It does not reinterpret a
structurally valid `failed` artifact as qualified; read the release report and
quality gate as well.
