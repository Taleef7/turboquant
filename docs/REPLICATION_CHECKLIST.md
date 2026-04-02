# TurboQuant Replication Checklist (Qwen Scope)

Use this checklist to verify the current claims in this repository on your own machine.

## Scope Locked for This Checklist

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Retrieval closure target in this phase: **through 16K**
- Gate: baseline-vs-TurboQuant paired NIAH delta `<= 2.0pp`

## Prerequisites

- Python 3.12 virtual environment active
- Dependencies installed from `requirements.txt`
- CUDA-capable NVIDIA GPU recommended

## Step 1: Environment sanity

Run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect:
- PyTorch imports cleanly
- CUDA availability is `True` (for full runs)

## Step 2: Fast harness/config tests

Run:

```bash
pytest scripts/test_long_context_harness.py scripts/test_cache_config.py scripts/test_qjl_config.py -q
```

Expect:
- All tests pass

## Step 3: Qwen paired NIAH gate (16K)

Run:

```bash
python scripts/test_long_context.py \
  --test niah \
  --mode paired \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-context 16384 \
  --key-bits 8 \
  --value-bits 6 \
  --buffer-size 12288 \
  --trials 6 \
  --output-prefix .tmp/niah_qwen_k8_v6_b12288_ctx16k_t6
```

Expect:
- Paired summary printed at end
- `Delta (baseline-tq)` at or below `2.0` percentage points

Artifacts:
- `.tmp/niah_qwen_k8_v6_b12288_ctx16k_t6.json`
- `.tmp/niah_qwen_k8_v6_b12288_ctx16k_t6.csv`

## Step 4: Throughput baseline vs TurboQuant

Run:

```bash
python scripts/benchmark_throughput.py
```

Expect:
- Baseline and TurboQuant tokens/sec printed for prompt lengths
- Memory/compression summary printed

## Step 5: One-command reproducibility report

Run:

```bash
python scripts/reproduce_release.py --mode full --output-dir results/repro_qwen
```

Expect:
- `results/repro_qwen/repro_report.json`
- `results/repro_qwen/repro_report.md`

## Pass/Fail Criteria for This Scope

- Paired NIAH gate through 16K meets `<= 2.0pp` delta
- Repro report generated successfully
- Harness/config tests pass

## Deferred in This Scope

- 32K/64K/128K retrieval closure
- Multi-model retrieval closure at high trial counts
- Full LongBench/RULER parity runners
