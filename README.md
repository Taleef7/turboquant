# TurboQuant: KV Cache Compression for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch PyTorch implementation of **TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) for KV-cache compression on consumer GPUs.

## What This Project Does

- Compresses KV cache online during generation using a TurboQuant-style rotation + Lloyd-Max pipeline.
- Preserves completion quality with compact settings (`6-bit` keys/values) for generation-heavy workloads.
- Provides a retrieval-safe profile for long-context NIAH-style tasks.
- Includes reproducible long-context paired evaluation tooling (`baseline` vs `TurboQuant` on identical prompts).

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

## Quickstart

### 1) Environment

```bash
git clone https://github.com/Taleef7/turboquant.git
cd turboquant
python -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt
```

### 2) Unit Tests (fast sanity)

```bash
pytest scripts/test_math.py scripts/test_kernels.py -v
pytest scripts/test_long_context_harness.py -q
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

Expected outcome:
- `Delta (baseline-tq)` at or under `2.0pp` for this matrix
- Most recent observed run: `1.39pp`

### B) Multi-model completion benchmark

```bash
python scripts/test_multimodel.py --model qwen2.5-7b
python scripts/test_multimodel.py --model mistral-7b
python scripts/test_multimodel.py --model llama3.1-8b
python scripts/test_multimodel.py --model gemma2-9b
```

## Retrieval-Safe Configuration

`configs/retrieval_profile.json` currently defines:
- `key_bits=8`
- `value_bits=6`
- `buffer_size=16384`

Tradeoff:
- Larger `buffer_size` improves retrieval robustness but increases uncompressed cache memory.

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

## Release Summary

This release stabilizes a reproducible, user-runnable TurboQuant workflow with:
- clear baseline-vs-TurboQuant paired retrieval evaluation,
- a documented retrieval-safe configuration (`retrieval-safe-v3`),
- synchronized docs/status files describing validated scope and remaining optional work.

Practical outcome for current scope:
- Qwen retrieval gate is closed through 32K with paired NIAH delta within target.

## License

MIT - see [LICENSE](LICENSE).
