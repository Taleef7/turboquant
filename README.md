# TurboQuant Forge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![CUDA tested](https://img.shields.io/badge/CUDA-tested%2012.8-76B900.svg)](https://pytorch.org/get-started/locally/)
[![CI](https://github.com/Taleef7/turboquant/actions/workflows/ci.yml/badge.svg)](https://github.com/Taleef7/turboquant/actions/workflows/ci.yml)

TurboQuant Forge is a clean-room, evidence-first KV-cache research backend for
Hugging Face Qwen2.5. It physically bit-packs fixed KV blocks and evaluates
them with blockwise PyTorch reference attention and direct Triton decode
kernels. Every performance, memory, and quality statement is tied to a raw,
hash-bound artifact rather than an extrapolation.

> **Research status:** active engineering prototype. The narrow Qwen2.5 path is
> implemented and tested; compressed full-logit quality and end-to-end serving
> wins are not yet qualified.

## At a glance

| Surface | Current state |
| --- | --- |
| KV storage | Physically packed fixed 256-token blocks with exact byte accounting |
| Attention | PyTorch packed reference plus direct Triton score/value kernels |
| Model adapter | Qwen2.5-7B-Instruct only; text, batch 1, one GPU, greedy decode |
| Canonical quality | Exact 4K/8K NIAH matrix qualified in the named artifact |
| 32K | Capacity admission failed before a 32K child launched |
| End-to-end win | Not established; the canonical 8K systems run is negative |
| Portability | Qwen3, Qwen3.5, and Gemma 4 are manifest-only and fail closed |

The project is designed to make a negative result useful: failed gates,
capacity stops, allocator peaks, dispatch counters, environment details, and
raw logs are retained instead of being silently converted into a success claim.

## Latest measured snapshot

These are diagnostic measurements, not new release claims. See the linked raw
artifact pages for the exact matrix, validator, and limitations.

| Experiment | Observed result |
| --- | --- |
| Qwen2.5 warmup, 4K/8K | Cache-store savings of 36.9%/44.8%; all six packed rows fail strict full-logit gates while preserving top-1/greedy choices in the small matrix |
| Fused page attention, 3–63 pages | 25.2–79.5% faster than serial synthetic scheduling, deterministic, max error `1.05e-05` |
| Value-bit ablation, v6 vs v8 | v8 is roughly 12% larger and 49–59.5% slower; both candidates fail strict full-logit gates |

These numbers are deliberately scoped to the local RTX 5070 Ti fixture. They
are not claims about Gilbreth hardware, other models, or production serving.

## Design overview

The direct path keeps prefill conventional and changes only the one-token
decode path:

1. Hugging Face Qwen2.5 produces standard K/V during prefill.
2. `PackedKVBlockStore` stores fixed blocks using physically packed low-bit
   indices plus a bounded fp16/bf16 tail.
3. A packed PyTorch reference defines correctness independently of Triton.
4. Triton kernels rotate queries, score packed pages, mix values, and merge
   page states without reconstructing the full KV history.
5. Validators bind model, profile, runner, configuration, source commit, and
   environment hashes before a result can be treated as evidence.

This separation makes it possible to optimize the GPU schedule without hiding
quality drift or confusing KV-store bytes with total allocator VRAM.

## Current status

The only implemented model adapter is Qwen2.5-7B-Instruct, using the checked-in
[direct profile](configs/profiles/qwen25-7b-direct-v1.json). It is deliberately
narrow: text-only, batch size one, standard prefill, greedy single-token decode,
one CUDA device, and full-attention layers.

The canonical release-ladder artifact is
`artifacts/qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0`, bound to
commit `5c9d4c34b0f0c31d015e2166ae7ad401858b202d`. Its report is structurally
valid but overall `failed`: the 8K quality matrix is qualified, while 32K
capacity admission fails and no 32K child is launched. The 8K systems comparison
also shows no end-to-end speed or VRAM win. See [results](docs/RESULTS.md) and
the [benchmark protocol](docs/BENCHMARK_PROTOCOL.md).

The latest clean local engineering checkpoint is separate from that release
attempt. The Qwen2.5 warmup, fused page, and value-bit artifacts are
diagnostic-only, bound to code commit `eba1c5b`, and documented in the
[local results](docs/LOCAL_LONG_CONTEXT_RESULTS.md),
[fused-page results](docs/LOCAL_FUSED_PAGE_RESULTS.md), and
[value ablation](docs/LOCAL_VALUE_BIT_ABLATION.md). They do not close the
quality, 32K, or end-to-end performance gates. The exact transfer and Slurm
workflow for continuing on Purdue Gilbreth is [the Gilbreth handoff](docs/GILBRETH_HANDOFF.md).

The first codec-focused reproduction is independently qualified as
`codec-local-v1` from clean commit
`871cbc9058d4befe4b0f5e35b687d1b5f525cb2c`: 1,408/1,408 persisted rows pass,
588/588 across-seed groups are complete, and all 108 declared Product/MSE
comparison groups are complete. It is a local, bounded reproduction with
explicit limits, not a paper-equivalence claim. See the [reproduction report](docs/REPRODUCTION.md)
and [claim ledger](docs/CLAIMS.md) for exact queries and conditions.

## What is implemented and tested

- A `PackedKVBlockStore` with fixed 256-token blocks, physically packed indices,
  bounded fp16/bf16 tail storage, and byte accounting by payload category.
- A blockwise packed PyTorch reference and Triton key-score/value-mix kernels.
  GPU parity tests compare direct packed dispatch with that reference.
- A scoped Transformers attention implementation for the pinned Qwen2.5 model:
  standard causal attention on prefill, direct packed attention only on decode.
- A validator-first quality/systems harness. It retains failed or invalid runs
  with command logs and hashes; it never turns a missing row or failed gate into
  a passing claim.
- Fail-closed capability and profile validation before cache mutation.

The pinned Qwen model smoke test exercised a 1K-token prefill plus 64-token
decode on the local GPU. It checks finite logits, packed-block consumption,
direct/reference greedy-token agreement, and guards against historical KV
concatenation. Its observed maximum fp16-logit drift was `16.935546875`; this is
diagnostic evidence only and is not retrieval-safe evidence.

## What is not claimed

- The shipped Qwen2.5 profile has a qualified **8K** NIAH matrix, but is not
  32K- or release-qualified: capacity admission failed before a 32K child could
  run.
- The named 8K systems artifact reports no end-to-end speedup or VRAM win:
  Triton-direct is slower and reserves more memory than baseline in that run.
- Qwen3, Qwen3.5, and Gemma 4 are manifests and fail-closed capability paths,
  not completed adapters. Their promotion requires their own adapter tests and
  evidence run.
- `core/turboquant_cache_v2.py` is a legacy fidelity prototype. It reconstructs
  historical KV state and is not the direct packed backend or evidence source.
- This milestone has no server, vLLM/SGLang integration, batching, beam search,
  sampling, speculative/MTP decode, distributed execution, offload, or
  multimodal cache path.

## Model portability boundary

| Family | Current state | Promotion requirement |
| --- | --- | --- |
| Qwen2.5-7B text | Direct adapter | Keep its pinned manifest/profile and pass the direct GPU/evidence suites. |
| Qwen3 dense text (`Qwen/Qwen3-8B`) | Manifest-only | Implement an adapter, prove config/layout parity, then run its own quality and systems artifacts. |
| Qwen3.5 hybrid/multimodal (`Qwen/Qwen3.5-9B`) | Manifest-only | Compress only full-attention KV layers; preserve linear-attention recurrent state and multimodal state natively. |
| Gemma 4 E4B multimodal (`google/gemma-4-E4B`) | Manifest-only | Build a dedicated adapter, validate multimodal/cache boundaries, and pass capacity preflight plus evidence. |

Read [portability](docs/PORTABILITY.md) for the exact capability rules and
non-KV exclusions.

## Install and verify

The supported GPU path is native WSL with a CUDA-capable NVIDIA GPU. CPU tests
do not download a model.

```bash
python3.12 -m venv ~/.venvs/turboquant-forge
source ~/.venvs/turboquant-forge/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/runtime.lock
python -m pip install pytest ruff
python -m pytest tests -m "not gpu" -q
```

For the kernel suite on a CUDA/Triton environment:

```bash
python -m pytest tests/test_packed_attention_kernels.py -m gpu -q
python -m pytest tests/test_packed_attention_benchmark.py -m gpu -q
```

The real-model smoke test is opt-in because it downloads pinned weights:

```bash
TURBOQUANT_RUN_MODEL_TESTS=1 python -m pytest tests/test_qwen_adapter_gpu.py -m gpu -q
```

## Reproduce evidence

Both commands retain raw JSON, CSV, Markdown, stdout, and stderr under the
ignored output directory. A nonzero exit means the profile was not qualified;
inspect the retained report rather than treating the command as a success.

```bash
python scripts/reproduce_release.py --mode quick --output-dir artifacts/repro_quick
python scripts/reproduce_release.py --mode qualified --output-dir artifacts/repro_qualified
```

The second command currently exits nonzero unless every context, provenance, and
quality gate passes. Use [the protocol](docs/BENCHMARK_PROTOCOL.md) to interpret
`qualified`, `failed`, `invalid`, and `inconclusive` states.

## Continue on Purdue Gilbreth

The repository is ready for a controlled cluster handoff, not an assumption
that local CUDA wheels or VRAM will transfer unchanged. Start with
[GILBRETH_HANDOFF.md](docs/GILBRETH_HANDOFF.md): it records the exact branch and
artifact state, the Slurm account/partition/QOS checks, environment capture,
durable raw-artifact transfer, and the ordered 16K/32K experiment matrix. A
cluster agent should begin with a tiny preflight and 4K/8K parity before
attempting longer contexts.

## Documentation map

- [Results](docs/RESULTS.md): canonical release-ladder 8K result, 32K capacity
  stop, systems comparison, and the latest diagnostic checkpoint.
- [Local long-context results](docs/LOCAL_LONG_CONTEXT_RESULTS.md): clean
  Qwen2.5 4K/8K warmup artifact with quality, storage, timing, and limits.
- [Local fused page results](docs/LOCAL_FUSED_PAGE_RESULTS.md): clean synthetic
  page-count measurements, fused launch contract, parity, and allocation data.
- [Local value-bit ablation](docs/LOCAL_VALUE_BIT_ABLATION.md): clean v6/v8
  quality, storage, timing, and provenance comparison.
- [Codec reproduction](docs/REPRODUCTION.md): canonical local matrix, GloVe
  retrieval results, runtime conditions, and limitations.
- [Claim ledger](docs/CLAIMS.md): safe statements with exact artifact/query
  bindings and the rules for resolving them.
- [Benchmark protocol](docs/BENCHMARK_PROTOCOL.md): artifact schema, matrix,
  statistical and invalid-run rules.
- [Portability](docs/PORTABILITY.md): model capability contracts and adapter
  promotion path.
- [Paper claims status](docs/PAPER_CLAIMS_STATUS.md): paper-to-repository
  coverage without equivalence claims.
- [Paper comparison](docs/PAPER_COMPARISON.md): clean-room scope and gaps.
- [Replication checklist](docs/REPLICATION_CHECKLIST.md): repeatable local
  verification steps.
- [Gilbreth handoff](docs/GILBRETH_HANDOFF.md): current branch/artifact state,
  Slurm setup, environment checks, transfer instructions, and the next
  experiment matrix for a cluster agent.
- [Issue tracker](ISSUES.md): resolved implementation work and open promotion
  tasks.

## License

MIT — see [LICENSE](LICENSE). The implementation is clean-room; it does not
copy GPL-licensed TurboQuant implementations.
