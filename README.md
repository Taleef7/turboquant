# TurboQuant Forge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-76B900.svg)](https://pytorch.org/get-started/locally/)

TurboQuant Forge is a clean-room, evidence-first KV-cache research backend for
Hugging Face Qwen2.5. It physically bit-packs fixed KV blocks and evaluates
them with blockwise PyTorch reference attention and direct Triton decode
kernels. The project treats every performance, memory, and quality statement as
an artifact-backed claim rather than an extrapolation.

## Current status

The only implemented model adapter is Qwen2.5-7B-Instruct, using the checked-in
[direct profile](configs/profiles/qwen25-7b-direct-v1.json). It is deliberately
narrow: text-only, batch size one, standard prefill, greedy single-token decode,
one CUDA device, and full-attention layers.

The latest retained evidence attempt is local artifact
`artifacts/qualified-v4/20260718T053419Z-unknown`. Its generated report recorded
`failed`, but the strengthened validator classifies it as **invalid**: its
requested 4K/8K rows used materially shorter effective inputs and its Git/binding
provenance is incomplete. Its original 32K leg was skipped. That invalidation is
the current truth, not a hidden caveat. See
[results](docs/RESULTS.md) and the [benchmark protocol](docs/BENCHMARK_PROTOCOL.md).

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

- The shipped Qwen2.5 profile is **not retrieval-qualified**. The retained NIAH
  artifact is invalid, so it must not be used to support a quality,
  memory-saving, or throughput claim.
- There is no verified end-to-end speedup or memory win. Physical bytes, peak
  allocator memory, and latency are reported only inside named raw artifacts.
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
| Qwen3 dense text | Manifest-only | Implement an adapter, prove config/layout parity, then run its own quality and systems artifacts. |
| Qwen3.5 hybrid | Manifest-only | Compress only full-attention KV layers; preserve linear-attention recurrent state natively. |
| Gemma 4 E4B | Manifest-only | Build a dedicated adapter, validate multimodal/cache boundaries, and pass capacity preflight plus evidence. |

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

## Documentation map

- [Results](docs/RESULTS.md): current artifact-backed observations, including
  the invalidated retained run.
- [Benchmark protocol](docs/BENCHMARK_PROTOCOL.md): artifact schema, matrix,
  statistical and invalid-run rules.
- [Portability](docs/PORTABILITY.md): model capability contracts and adapter
  promotion path.
- [Paper claims status](docs/PAPER_CLAIMS_STATUS.md): paper-to-repository
  coverage without equivalence claims.
- [Paper comparison](docs/PAPER_COMPARISON.md): clean-room scope and gaps.
- [Replication checklist](docs/REPLICATION_CHECKLIST.md): repeatable local
  verification steps.
- [Issue tracker](ISSUES.md): resolved implementation work and open promotion
  tasks.

## License

MIT — see [LICENSE](LICENSE). The implementation is clean-room; it does not
copy GPL-licensed TurboQuant implementations.
