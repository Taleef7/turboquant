# TurboQuant Forge: Verified Compressed Attention on a Consumer GPU

## Purpose

Evolve this repository from a functional KV-cache compression prototype into a
showcase-quality, evidence-first LLM systems project. The first release proves
that a physically packed KV cache can be consumed by a direct GPU attention path
on the local RTX 5070 Ti (12 GB, SM120) without reconstructing the entire
historical cache as fp16 tensors.

The project is not a general inference server or a claim of universal model
support. It is a reproducible compressed-attention backend, with a clear path to
additional model families.

## Scope and non-goals

### First milestone

- Qwen2.5 text generation through Hugging Face Transformers.
- Batch size one and greedy decode only.
- Full-attention KV layers, block-based append storage, and fixed-size packed
  blocks.
- A PyTorch reference codec plus direct Triton key-score and value-mix kernels.
- Deterministic quality tests and auditable systems benchmarks on the local GPU.

### Explicit non-goals

- A vLLM, SGLang, or HTTP-serving replacement.
- Distributed inference, offloading, continuous batching, or prefix sharing.
- Beam search, speculative decoding, MTP, and arbitrary cache APIs.
- Blanket low-bit, speed, memory, or model-support claims without artifacts.

## Architecture

```text
Model adapter
    -> ModelCapabilities
    -> PackedKVBlockStore
    -> Packed reference codec / reference attention
    -> Triton packed attention backend
    -> Reproduction and evidence artifacts
```

### ModelCapabilities

`ModelCapabilities` is derived from a pinned model configuration and is the only
source for architecture-dependent facts. It records the model revision, layer
types, `num_attention_heads`, `num_key_value_heads`, head dimension, RoPE
configuration, dtype, supported cache behavior, and generation-mode limits.

The adapter supports a layer only when its capability is explicitly recognized.
Unknown, sliding-window, linear-attention, multimodal, beam, batched, or
speculative paths must fail with a clear `UnsupportedCapabilityError` until an
adapter for that case is implemented and tested.

### PackedKVBlockStore

The store owns fixed-size chronological blocks. Each block contains separate
packed key and value payloads, their metadata, sequence range, and an exact
physical-byte counter. It must append a block or a bounded tail only; it never
concatenates the full historical representation.

The counter includes all allocated payload, scales, norms, alignment padding,
and fp16-window bytes. It must agree with independently measured tensor storage.
Nominal bit rates are never reported as physical compression ratios.

### Reference codec

The reference codec is deliberately simple and authoritative:

- keys: normalized rotation, Lloyd-Max indices, norm metadata, and optional
  QJL residual payload;
- values: group quantization with packed indices and explicit group metadata;
- all packers and unpackers are pure PyTorch and CPU-testable;
- QJL follows the product construction: `(b - 1)` MSE bits plus a physically
  packed one-bit residual sign payload.

The first direct-attention profile may ship without QJL if QJL does not pass its
quality and byte-accounting gates. QJL is a separately qualified profile, not a
marketing requirement.

### Direct packed attention

The direct backend consumes blocks rather than a dequantized full cache.

1. It rotates the current query once for a key block.
2. It reads packed key indices and computes scores in Triton.
3. It performs stable online softmax across all blocks.
4. It reads packed value indices and group metadata to accumulate the weighted
   value output in Triton.

The adapter must not materialize the full history of K or V as fp16 in order to
call attention. Short per-block working tiles are allowed.

## Model ladder

The first model is Qwen2.5 text. Further adapters are promoted only after the
previous adapter has complete artifacts.

| Order | Family | Adapter rule | Local role |
| --- | --- | --- | --- |
| 1 | Qwen2.5-7B text | Full-attention baseline | Primary implementation and demo |
| 2 | Qwen3 dense text | Config-driven GQA/head-dimension support | First portability proof |
| 3 | Qwen3.5-9B | Compress `full_attention`; pass linear-attention state through | Hybrid-cache proof, subject to VRAM preflight |
| 4 | Gemma 4 E4B | Dedicated config adapter | Cross-vendor proof |
| 5 | Gemma 4 12B Q4 | Same adapter with strict capacity preflight | Stretch memory-pressure evaluation |

Large Qwen3.5 and Gemma 4 models that cannot fit fairly on one 12 GB GPU are
not local acceptance targets. CPU offload or multi-GPU results are not mixed
with the consumer-GPU benchmark claims.

## Evidence contract

### Test levels

| Level | Required evidence |
| --- | --- |
| Unit-verified | Exact packing round trips, byte accounting, invalid-input and capability rejection tests |
| Kernel-verified | Packed Triton scores and values match the unpacked quantized PyTorch reference for all supported shapes |
| Model-verified | Qwen2.5 greedy decode uses the adapter; logits and outputs are compared with fp16 and quantized-reference paths |
| Benchmark-qualified | Complete quality matrix and systems artifact with pinned environment metadata |

### Numerical gates

- Index and sign packing/unpacking is bit exact.
- Reference quantization and packed-reference decoding agree exactly where the
  representation is discrete.
- Triton score and output kernels use explicit `torch.testing.assert_close`
  tolerances appropriate to the tested dtype; the exact tolerances are emitted
  with the test matrix instead of hidden in benchmark scripts.
- Every supported tail length, KV-head count, head dimension, and block size is
  tested. Unsupported shapes are rejected.

### Quality gates

Benchmark matrices enumerate their expected context, depth, and seed rows in
the artifact before execution. An OOM, exception, tokenization loss, missing
row, or incomplete metadata invalidates the run.

A profile may be named `retrieval-safe` only when all of these are true:

- the paired NIAH baseline accuracy is at least 95%;
- every expected eligible row completed;
- the upper 95% bootstrap confidence bound for baseline-minus-profile accuracy
  is at most 2.0 percentage points;
- raw prompts, answers, and row-level outcomes are present in the artifact.

RULER and LongBench-E are reported as constrained local subsets. They extend
evidence; they do not erase a failed NIAH gate.

### Systems gates

Artifacts report actual cache bytes, allocator peak allocated and reserved
memory, prefill latency, TTFT, decode latency distribution, decode throughput,
and OOM boundary. Timing uses warmups, CUDA events or explicit synchronization,
interleaved baseline/profile trials, and a reported p50/p95 rather than a
single run.

The project can claim a storage benefit only from physical-byte accounting. It
can claim a performance benefit only when the qualified benchmark reports a
favorable result with its configuration and environment. A tie or regression is
retained and explained rather than omitted.

## Reproducibility and safety

- Pin model and tokenizer revisions, package versions, codebook/profile hashes,
  GPU/driver/runtime versions, and deterministic settings in every artifact.
- Dependencies use a documented runtime/dev split and a lock or constraints
  file. Codebooks are immutable checked-in artifacts or explicit generated
  outputs with checksums; they never silently mutate the source tree.
- Benchmark subprocesses use the active Python interpreter, fail on non-zero
  commands, and preserve stdout/stderr in artifacts.
- CPU tests run without a GPU or model download. GPU tests are separately
  labeled and skip only when the required hardware/runtime is absent.

## Success criteria

The first milestone is complete only when the repository contains a tested,
physically packed block codec; a direct compressed-attention Qwen2.5 adapter;
complete local-GPU evidence artifacts; corrected documentation and claims; and
an explicit model-capability path for the ladder above. Results must demonstrate
what wins, ties, and loses on the measured hardware without extrapolation.
