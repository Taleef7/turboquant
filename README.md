# TurboQuant: KV Cache Compression for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-76B900.svg)](https://pytorch.org/get-started/locally/)
[![Scope](https://img.shields.io/badge/Validated-Qwen%2032K%20NIAH-success.svg)](TESTING_RESULTS.md)
[![Repro](https://img.shields.io/badge/Repro-Checklist-blue.svg)](docs/REPLICATION_CHECKLIST.md)

A from-scratch PyTorch implementation of **TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) for KV-cache compression on consumer GPUs.

> Reproducible, Qwen-focused TurboQuant replication with paired baseline-vs-TurboQuant NIAH validation through 16K in the current release-check path.

## Why This Matters

Long-context inference is often limited by KV-cache memory and memory bandwidth, not raw compute.
If you can shrink KV-cache without breaking quality, you can run longer contexts cheaper and faster.

This project is a practical replication effort focused on that tradeoff.

```text
Without KV compression:
  longer context -> much larger KV cache -> more VRAM pressure / memory traffic

With TurboQuant-style KV compression:
  longer context -> compressed KV cache -> lower memory footprint -> better practical scaling
```

## What This Project Does

- Compresses KV cache online during generation using a TurboQuant-style rotation + Lloyd-Max pipeline.
- Preserves completion quality with compact settings (`6-bit` keys/values) for generation-heavy workloads.
- Provides a retrieval-safe profile for long-context NIAH-style tasks.
- Includes reproducible long-context paired evaluation tooling (`baseline` vs `TurboQuant` on identical prompts).

## TurboQuant In One Diagram

```text
Input token stream
      |
      v
Model computes K,V per layer/head
      |
      +--> Baseline cache: store fp16 K,V directly
      |
      +--> TurboQuant cache:
             K: norm -> unit normalize -> random rotate -> Lloyd-Max quantize -> indices (+ optional QJL fields)
             V: per-group min/max quantize -> indices + scales
             keep recent window uncompressed (buffer)
      |
      v
Attention step dequantizes on demand for logits/value mix
```

## Current Validated Scope

- Compression: **5.2x** (target 4.5x)
- Throughput: **61-84%** of baseline depending on model/prompt length
- Completion quality: **matches baseline** in tested suites
- Retrieval quality (Qwen2.5-7B): **NIAH gate met through 32K** with `retrieval-safe-v3`
  - Gate: baseline-vs-TurboQuant delta `<= 2.0pp`
  - Observed at 32K paired matrix (t6): **1.39pp**

Important scope note:
- Full high-trial retrieval closure is currently **Qwen-focused**.
- Mistral/Gemma retrieval checks are currently smoke-level and tracked as follow-up.

## Claim Policy (Read This First)

What this repo currently claims:
- Reproducible paired baseline-vs-TurboQuant evaluation for Qwen.
- Practical retrieval closure gate for the current release-check scope (through 16K).
- Reproducible throughput/compression comparisons with provided scripts.

What this repo does not claim yet:
- Full benchmark parity with all paper/blog experiments (LongBench/RULER/vector-search).
- Generalized low-bit near-lossless parity across all models and contexts.

See `docs/PAPER_CLAIMS_STATUS.md` and `docs/PAPER_COMPARISON.md`.

## Bits: This Repo vs Original Research

| Topic | Original paper/blog | This repo (current validated scope) |
|---|---|---|
| Key path concept | `TurboQuant_prod` uses MSE + 1-bit QJL residual correction | MSE-first path with optional QJL scaffolding |
| Reported effective bit settings | Commonly highlighted around 2.5/3.5-bit variants in paper benchmarks | Completion default: key=6, value=6 |
| Retrieval-safe operating point | paper reports near-lossless at lower effective bits in its setup | Qwen paired NIAH through 32K: key=8, value=6, buffer=16384 |
| Practical note | Theory + specialized benchmark setup | Engineering-focused, reproducible commands/artifacts in this repo |

For an explicit mapping of paper experiments to repo coverage, see `docs/PAPER_COMPARISON.md`.

## Requirements (Step-by-Step)

## 1) Hardware / runtime

- NVIDIA GPU strongly recommended for full benchmarks.
- For RTX 50-series Blackwell (`sm_120`), use PyTorch nightly `cu128` builds.
- VRAM:
  - ~12-16 GB: practical for current Qwen scope (through 32K paired tests may still be slow/heavy)
  - higher VRAM helps for larger context and multi-model sweeps

## 2) System dependencies

- Python 3.12
- `pip`
- CUDA-compatible NVIDIA driver

## 3) Python packages

This repo uses `requirements.txt`:
- `torch`
- `transformers`
- `triton`
- `bitsandbytes`
- `datasets`
- `einops`
- `accelerate`

## 4) Hugging Face access (if needed)

- Qwen and Mistral are generally accessible directly.
- Llama-3.1 requires accepted model terms + auth token.
- If you plan to run gated models:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

## Quickstart

## Quick Verify (10-20 min)

```bash
source venv312/bin/activate
python scripts/reproduce_release.py --mode quick --output-dir results/repro_quick
```

Outputs:
- `results/repro_quick/repro_report.json`
- `results/repro_quick/repro_report.md`

## Full Verify (Qwen release-check path)

```bash
source venv312/bin/activate
python scripts/reproduce_release.py --mode full --output-dir results/repro_full
```

`full` mode is currently pinned to Qwen paired NIAH through 16K with higher trials.

### 1) Environment

```bash
git clone https://github.com/Taleef7/turboquant.git
cd turboquant
python -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt
```

If you are on RTX 50-series/Blackwell, install PyTorch nightly `cu128` first (then install requirements):

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

### 2) Unit Tests (fast sanity)

```bash
pytest scripts/test_math.py scripts/test_kernels.py -v
pytest scripts/test_long_context_harness.py -q
pytest scripts/test_cache_config.py scripts/test_qjl_config.py -q
```

### 3) Baseline and TurboQuant demos

```bash
python scripts/run_baseline.py
python scripts/run_turboquant_v2.py
```

## Reproducing Key Claims

### A) Qwen paired NIAH retrieval gate (through 32K)

```bash
python scripts/test_long_context.py \
  --test niah \
  --mode paired \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-context 32768 \
  --key-bits 8 \
  --value-bits 6 \
  --buffer-size 16384 \
  --trials 6 \
  --output-prefix .tmp/niah_qwen_k8_v6_b16384_ctx32k_t6
```

This command directly compares baseline and TurboQuant on identical prompts and writes reproducible artifacts.

Expected outcome:
- `Delta (baseline-tq)` at or under `2.0pp` for this matrix
- Most recent observed run: `1.39pp`

For current release-check scope, use the one-command `full` path above (16K) and the checklist in `docs/REPLICATION_CHECKLIST.md`.

### B) Multi-model completion benchmark

```bash
python scripts/test_multimodel.py --model qwen2.5-7b
python scripts/test_multimodel.py --model mistral-7b
python scripts/test_multimodel.py --model llama3.1-8b
python scripts/test_multimodel.py --model gemma2-9b
```

## C) Throughput baseline-vs-TurboQuant (Qwen)

```bash
python scripts/benchmark_throughput.py
```

## Retrieval-Safe Configuration

`configs/retrieval_profile.json` currently defines:
- `key_bits=8`
- `value_bits=6`
- `buffer_size=16384`

Tradeoff:
- Larger `buffer_size` improves retrieval robustness but increases uncompressed cache memory.

## What Is Compared Today (for users)

- Baseline vs TurboQuant completion behavior
- Baseline vs TurboQuant paired NIAH retrieval (`scripts/test_long_context.py --mode paired`)
- Throughput baseline vs TurboQuant (`scripts/benchmark_throughput.py`)

Not fully implemented yet for one-command parity with paper figures:
- LongBench(-E) runner
- RULER runner
- full multi-method table generation (PQ/KIVI/etc.)

Use `docs/PAPER_COMPARISON.md` to see exactly what is covered vs deferred.

## Project Layout

```text
turboquant/
├── core/
│   ├── turboquant_cache_v2.py      # Main DynamicCache integration
│   └── turboquant_simple.py        # Quantizers (MSE path + optional QJL scaffolding)
├── codebooks/                      # Precomputed Lloyd-Max codebooks
├── configs/
│   └── retrieval_profile.json      # Retrieval-safe profile
├── scripts/
│   ├── test_long_context.py        # Paired NIAH harness
│   ├── test_long_context_harness.py# Harness unit tests
│   ├── test_multimodel.py          # Completion benchmark across models
│   ├── run_baseline.py
│   └── run_turboquant_v2.py
├── ISSUES.md                       # Local issue/status tracker
├── TESTING_RESULTS.md              # Detailed measured results
└── UPDATED_PLAN.md                 # Current execution/status plan
```

## Known Limits / Deferred Work

- Retrieval closure beyond 32K (64K/128K) is optional/deferred for now.
- Full multi-model retrieval closure with higher trial counts is pending.
- QJL path is scaffolded but not yet shown to improve paired NIAH in current integration.

## References

- Paper: [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
- Current implementation notes: `TESTING_RESULTS.md`, `UPDATED_PLAN.md`, `ISSUES.md`
- Paper comparison mapping: `docs/PAPER_COMPARISON.md`
- Claims status matrix: `docs/PAPER_CLAIMS_STATUS.md`
- Replication checklist: `docs/REPLICATION_CHECKLIST.md`
- Contributor guide: `CONTRIBUTING.md`
- Script index: `scripts/README.md`

## Release Summary

This release stabilizes a reproducible, user-runnable TurboQuant workflow with:
- clear baseline-vs-TurboQuant paired retrieval evaluation,
- a documented retrieval-safe configuration (`retrieval-safe-v3`),
- synchronized docs/status files describing validated scope and remaining optional work.

Practical outcome for current scope:
- Qwen retrieval gate is closed through 32K with paired NIAH delta within target.

## License

MIT - see [LICENSE](LICENSE).
