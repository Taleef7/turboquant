# Scripts Index

This directory contains only active scripts used by the current reproducibility workflow.

## Canonical entry points

- `reproduce_release.py`
  - One-command reproducibility runner.
  - Modes:
    - `quick`: smoke-level validation
    - `full`: Qwen release-check path (currently through 16K)

- `benchmark_throughput.py`
  - Baseline vs TurboQuant throughput/memory benchmark.
  - Supports `--output-json` for machine-readable artifacts.

- `test_long_context.py`
  - Paired NIAH harness (`baseline` vs `TurboQuant` on identical prompts).

## Demonstration scripts

- `run_baseline.py`
- `run_turboquant_v2.py`

## Validation / unit tests

- `test_math.py`
- `test_kernels.py`
- `test_long_context_harness.py`
- `test_cache_config.py`
- `test_qjl_config.py`
- `test_repro_utils.py`
- `test_reproduce_release.py`

## Multi-model benchmark

- `test_multimodel.py`

Current claims are Qwen-focused; multi-model retrieval closure is deferred.
