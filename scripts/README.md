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

## `replay_qwen25_fixed_tokens.py`

Runs the predeclared, exact-token Qwen2.5 replay matrix at 1,024 and 2,048
prompt tokens through `baseline`, `packed_reference`, `serial_direct`, and
`parallel_direct`.  Every lane receives the same prompt and two explicit
decode token IDs.  Rows record logits, KL divergence, top-1/greedy agreement,
dispatch/fallback counters, physical/logical/cache/workspace bytes, runtime
allocator peaks, timing, and source identity.  The artifact is always
`diagnostic_only` with an empty claims list; a missing local fixture is written
as eight explicit `unavailable` rows rather than silently changing the test.

The default is local-only and does not download model weights.  The pinned WSL
fixture can be selected explicitly with `--model-path`; the runner requires
the exact snapshot directory, validates the Qwen2.5-7B architecture, and
checks the predeclared SHA-256 file-manifest digest before loading weights:

```bash
python scripts/replay_qwen25_fixed_tokens.py \
  --model-path /home/taleef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --output artifacts/qwen25-fixed-replay.json
python scripts/validate_qwen_fixed_replay.py \
  artifacts/qwen25-fixed-replay.json
```

Use `--allow-download` only when network access is intentional.  This replay
does not publish an end-to-end speed or VRAM claim; it is the fixed-token
qualification input for a later systems gate.

## `benchmark_qwen25_fixed_warmup.py`

Runs the same fixed token stream through the four Qwen2.5 lanes with one
unmeasured warmup decode followed by three measured decode tokens across three
fresh repeats.  The artifact reports per-token samples, p50/p95, first-run vs
steady-state timings, dispatch counters, cache logical/physical/unique bytes,
runtime allocator peaks, and source identity.  It is diagnostic-only and uses
the same pinned model fixture and strict quality gate as the replay:

```bash
python scripts/benchmark_qwen25_fixed_warmup.py \
  --model-path /home/taleef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --output artifacts/qwen25-fixed-warmup.json
python scripts/validate_qwen_fixed_warmup.py \
  artifacts/qwen25-fixed-warmup.json
```

For a local long-context diagnostic, use the opt-in v2 matrix at 4,096 and
8,192 prompt tokens. It keeps the same pinned fixture, v6 profile, fixed
stream seed, four backends, repeats, and quality gates as v1; only the context
matrix and config name differ. Pass the config explicitly to both commands so
the validator checks the matching matrix:

```bash
python scripts/benchmark_qwen25_fixed_warmup.py \
  --config configs/benchmarks/qwen25-fixed-warmup-local-v2.json \
  --model-path /home/taleef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --output artifacts/qwen25-fixed-warmup-local-v2.json
python scripts/validate_qwen_fixed_warmup.py \
  --config configs/benchmarks/qwen25-fixed-warmup-local-v2.json \
  artifacts/qwen25-fixed-warmup-local-v2.json
```

The v2 run is diagnostic-only and is not a replacement for the canonical v1
artifact or its 1,024/2,048-token qualification matrix.

## `benchmark_qwen25_value_ablation.py`

Runs an opt-in value-bit ablation on the same fixed 4,096/8,192-token stream:
the pinned production profile (`value_bits=6`) and the separate profile with
8-bit values. It measures one baseline lane plus serial direct decode for each
profile; the parallel fused lane is deliberately excluded so this experiment
isolates codec precision rather than page-attention behavior. The artifact
records quality, exact cache byte fields, timing, profile paths and both file
and profile hashes, plus model/config/runner provenance. It is always
diagnostic-only with an empty claims list and does not modify the v1/v2
warmup artifacts or production profile.

```bash
python scripts/benchmark_qwen25_value_ablation.py \
  --config configs/benchmarks/qwen25-value-ablation-local-v1.json \
  --model-path /home/taleef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --output artifacts/qwen25-value-ablation-local-v1.json
```

The validator is built into the runner's final check and rejects changed
profile files, model identity, runner/config hashes, fallback, parallel
dispatch, incomplete rows, or inconsistent quality/byte/timing fields.

## `benchmark_parallel_page_attention.py`

Runs the serial packed-page diagnostic used to freeze the page-attention
baseline. It uses deterministic synthetic Qwen2.5-shaped pages at 3, 7, 15,
31, and 63 pages, with one cold/JIT launch, two warmups, and ten synchronized
CUDA-event trials per row. The output is schema-v2 and explicitly labelled
`diagnostic_only`; it records page-launch counters, allocator/workspace bytes,
and a packed-reference parity check. `serial` is the default reference,
`page-state` measures independent page states, and `parallel` measures the
complete path (identity query cast/copy, page states, chronological reduction,
and fixed-stride tail merge). The identity cast/copy is included in each timed
parallel launch; the model-specific non-identity rotation kernel belongs to
the later integration stage. The parallel row reports reusable
workspace bytes separately from packed arena capacity and runtime allocator
bytes. These measurements are engineering diagnostics, not an end-to-end
model or speed claim.
The synthetic harness binds its identity to the Qwen2.5 architecture manifest;
it does not load or hash model weight files.

```bash
python scripts/benchmark_parallel_page_attention.py \
  --implementations serial \
  --page-counts 3 7 15 31 63 \
  --output artifacts/parallel-page-baseline/manifest.json
python scripts/validate_artifact.py \
  artifacts/parallel-page-baseline/manifest.json

python scripts/benchmark_parallel_page_attention.py \
  --implementations serial parallel \
  --page-counts 3 7 15 31 63 \
  --output artifacts/parallel-page-complete/manifest.json
```

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
