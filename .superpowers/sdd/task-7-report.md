# Task 7 report: self-validating quality and systems evidence

## Delivered

- Added typed quality, systems, and release-artifact contracts in
  `scripts/benchmark_schema.py`, plus `scripts/validate_artifact.py` for
  independent command-line validation.
- NIAH now builds its full `(context, depth, seed)` matrix ahead of execution,
  persists completed/error/OOM/tokenization-loss rows, requires a complete
  validated matrix, baseline accuracy of at least 95%, and a deterministic
  paired-bootstrap upper 95% bound of at most 2pp before calling a run
  retrieval-safe.
- Added an interleaved direct systems runner using baseline,
  packed-reference, and Triton-direct paths. It records CUDA-event timings,
  actual packed-store physical bytes, allocator peaks, decode distributions,
  throughput, OOM state, and dispatch counts without using `TurboQuantCacheV2`.
- Replaced the release entry point with child-process execution through
  `sys.executable`, stdout/stderr capture, artifact-first validation, and
  explicit `qualified`, `invalid`, `failed`, and `inconclusive` states.
- Bound the profile's direct evidence path to its exact 8-bit-key and 6-bit-
  value codebook files; artifact bindings include model, profile, codebook,
  and runner-source digests.

## TDD and CPU verification

The new schema tests were written first and initially failed collection because
`scripts.benchmark_schema` did not exist. The implementation was then added.

```text
/home/taleef/.venvs/turboquant-forge/bin/python -m pytest \
  tests/test_benchmark_schema.py tests/test_artifact_validation.py \
  scripts/test_reproduce_release.py scripts/test_repro_utils.py -q
13 passed

python -m py_compile scripts/benchmark_schema.py scripts/test_long_context.py \
  scripts/run_direct_benchmark.py scripts/reproduce_release.py \
  scripts/validate_artifact.py
exit 0

git diff --check
exit 0
```

## GPU evidence (RTX 5070 Ti)

The real 1K prompt/64-token systems smoke completed and independently
validated. Its ignored raw artifact is `artifacts/task7-1k-systems.json` and
records successful baseline, packed-reference, and Triton-direct trials,
including direct dispatches and actual physical byte measurements.

The requested quick release command was then run:

```text
python scripts/reproduce_release.py --mode quick --output-dir artifacts/quick
```

It exited non-zero with an intentionally preserved `failed` report at
`artifacts/quick/20260718T041723Z-unknown/repro_report.json`. Both quality and
systems artifacts passed structural validation, but quality did not pass the
retrieval-safe gate: baseline retrieved 12/12 needles while the direct profile
retrieved 0/12; paired delta and deterministic bootstrap upper 95% were both
100pp. The 8K systems artifact completed without OOM. Because the 8K quality
gate failed, a 32K qualified attempt was correctly not made.

No public-claim document was edited.

## Review-fix verification (2026-07-18)

- Qualified mode now performs a real 1K systems preflight, records
  `qualified_32k_decision.json`, and may launch 32K only if that preflight and
  the validated 8K retrieval gate both pass. A failed prerequisite is persisted
  as `skipped`, not only described in prose.
- Artifact binding now incorporates the quality/system runners, Qwen direct
  adapter, packed cache/store, and packed Triton kernel. Systems validation
  checks that aggregate plus the standalone runner digest.
- Quality artifacts now declare top-level metadata and a typed status. The
  corrected 8K run is structurally valid but `failed`, rather than being
  ambiguously labeled valid. Systems runs record warmup outcomes and create
  timestamped self-contained child directories.

Fresh verification:

```text
python -m pytest tests -m "not gpu" -q
185 passed, 19 deselected
```

Corrected quick evidence:

```text
artifacts/quick-v2/20260718T043949Z-unknown/repro_report.json
release status: failed
quality status: failed (complete structurally valid matrix; retrieval gate failed)
systems status: completed (structurally valid)
```

## Acceptance-fix verification (2026-07-18)

- NIAH expected rows are now materialized before execution; each coordinate
  retains `pending`, `completed`, `error`, or `oom` state in the final matrix.
- Qualified execution was run at
  `artifacts/qualified-v3/20260718T045802Z-unknown`. Its real 1K/2K direct
  systems preflight passed structural validation and recorded a two-point OOM
  probe range. The conservative 32K projection was 8,762,646,528 bytes versus
  10,405,439,078 safety-adjusted free bytes. The exact formula and the source
  artifact binding are preserved in `qualified_capacity_preflight.json`.
- The 8K quality gate failed, so the run persisted
  `qualified_32k_decision.json` with `status: skipped` and
  `reason: 8k_retrieval_gate_failed`; no 32K child was launched.
- Systems artifacts now contain actual two-point direct probes and runner-local
  diagnostic output in their timestamped artifact directories.

The final quick replacement is
`artifacts/quick-v4/20260718T051339Z-unknown/repro_report.json`: it is
`failed` on the complete quality gate while systems completed and its direct
probe boundary records `[4096, 8192]` with
`not_reached_within_configured_probe_range`.

## Final admission calibration (2026-07-18)

`artifacts/qualified-v4/20260718T053419Z-unknown` reran the qualified ladder.
Its `qualified_32k_admission.json` records a calibrated 8K-systems estimate:
the systems artifact was structurally clean, but its conservative 32K projected
reservation was 12,809,961,472 bytes, exceeding the 10,405,439,078-byte
safety-adjusted free-VRAM budget. Quality independently failed its 8K
retrieval gate, so `qualified_32k_decision.json` persists the retrieval-gate
skip and no 32K child ran. Admission requires both gates.
