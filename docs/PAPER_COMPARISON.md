# TurboQuant Paper vs This Repo

This document maps the original TurboQuant evaluations to what is currently implemented in this repository, so users can quickly reproduce comparable baselines and understand remaining gaps.

Primary references:
- Paper markdown in repo: `2504.19874v1.md`
- Google Research blog: `https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/`

## At-a-Glance Mapping

| Paper experiment area | In this repo | Status |
|---|---|---|
| Needle-In-A-Haystack (NIAH) long-context retrieval | `scripts/test_long_context.py --mode paired` | **Implemented** |
| End-to-end generation on LongBench(-E) | No direct LongBench runner yet | **Not yet implemented** |
| RULER benchmark | No direct RULER runner yet | **Not yet implemented** |
| Product variant (`TurboQuant_prod`) with QJL focus | Optional QJL scaffolding exists, not yet validated as better | **Partial** |
| Memory/compression and throughput comparisons | `scripts/benchmark_throughput.py`, docs summaries | **Implemented (project-specific)** |
| Near-neighbor search experiments (PQ/RabitQ vs TurboQuant) | No ANN benchmark script yet | **Not yet implemented** |

## What Users Can Reproduce Today

## 1) Paired NIAH (Baseline vs TurboQuant)

Goal: Compare retrieval quality on identical prompts, with deterministic seeds and auditable outputs.

Run:

```bash
source venv312/bin/activate
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

Outputs:
- `.tmp/niah_qwen_k8_v6_b16384_ctx32k_t6.json`
- `.tmp/niah_qwen_k8_v6_b16384_ctx32k_t6.csv`

Main metric:
- `Delta (baseline-tq)` in percentage points (`<= 2.0pp` target for this project)

## 2) Throughput + memory-oriented comparison

Goal: Compare decode throughput and rough memory behavior for baseline vs TurboQuant configurations.

Run:

```bash
source venv312/bin/activate
python scripts/benchmark_throughput.py
```

## 3) Multi-model completion checks

Goal: Verify completion quality and throughput trends across supported model configs.

Run:

```bash
source venv312/bin/activate
python scripts/test_multimodel.py --model qwen2.5-7b
python scripts/test_multimodel.py --model mistral-7b
python scripts/test_multimodel.py --model llama3.1-8b
python scripts/test_multimodel.py --model gemma2-9b
```

## Important Differences vs Paper Setup

- This repo currently emphasizes a practical Qwen-focused retrieval closure path through 32K (paired NIAH) rather than full paper benchmark parity.
- Paper reports include LongBench(-E), additional method comparisons, and explicit `TurboQuant_prod` analysis at low bit budgets.
- Here, optional QJL path is present in code but not yet shown to outperform the MSE-first path in current NIAH measurements.

## If You Want Paper-Style Side-by-Side Comparisons

Recommended next additions:
1. LongBench(-E) runner with baseline/TurboQuant paired scoring.
2. RULER task runner with the same paired output format.
3. Standardized comparison table generator (full-cache vs TurboQuant configs) emitted from JSON artifacts.
4. Optional ANN benchmark script for the paper's near-neighbor experiment style.

Until those land, use this document as the source of truth for what is directly reproducible in this repository.
