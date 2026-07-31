# Reproducibility scripts

These are the active entry points for the local Forge evidence workflow. Run
them from the repository root with the pinned runtime in
`requirements/runtime.lock`. The benchmark and release runners write artifacts;
the validator reads an existing artifact before treating a measurement as
evidence.

## `run_direct_benchmark.py`

Runs an interleaved baseline, packed-reference, and Triton-direct systems
benchmark for the pinned Qwen2.5 profile. It records every trial, dispatch
counter, physical-cache breakdown, and OOM probe in the output JSON.

```bash
python scripts/run_direct_benchmark.py \
  --output artifacts/direct-systems.json \
  --prompt-tokens 8192 --decode-tokens 64 --warmups 1 --trials 2
```

The runner requires CUDA and does not itself establish a quality or Pareto
claim.

## `reproduce_release.py`

Coordinates validator-first Qwen evidence legs. `quick` is a smoke workflow;
`qualified` applies the configured admission gates. A nonzero exit status is a
reported outcome, not a result to suppress.

```bash
python scripts/reproduce_release.py --mode quick --output-dir artifacts/repro
```

## `validate_artifact.py`

Validates a JSON artifact against the checked-in schema and, by default, the
current source binding. It returns a nonzero status for malformed, incomplete,
or mismatched evidence.

```bash
python scripts/validate_artifact.py artifacts/direct-systems.json
```

For the current measured evidence and its scope, see
[`docs/RESULTS.md`](../docs/RESULTS.md). The local-foundation design describes
the next evidence requirements in
[`docs/superpowers/specs/2026-07-31-turboquant-forge-v1-design.md`](../docs/superpowers/specs/2026-07-31-turboquant-forge-v1-design.md).
