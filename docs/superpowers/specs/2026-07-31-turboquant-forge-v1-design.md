# TurboQuant Forge v1: Faithful Codec and Consumer-GPU Systems Evidence

## Purpose

TurboQuant Forge v1 is a public MIT-licensed, clean-room reproduction and
systems extension of TurboQuant. Its primary audience is GPU/LLM systems hiring
teams. It must demonstrate both:

1. a paper-faithful mathematical reference for the TurboQuant MSE and product
   codecs; and
2. an independently engineered compressed-domain runtime that can establish a
   local end-to-end Pareto win on an RTX 5070 Ti Laptop GPU with 12 GB VRAM.

The implementation must never turn a storage-only reduction, incomplete matrix,
capacity projection, or microbenchmark into an end-to-end claim.

## Source and provenance boundary

The canonical first reproduction target is arXiv
`2504.19874v1`, published 2025-04-28:
`https://arxiv.org/abs/2504.19874v1`. The repository records the source URL,
retrieval date, and SHA-256 of the reference PDF or source used by each
reproduction artifact. The final OpenReview artifact
`https://openreview.net/forum?id=tO3ASKZlok` is tracked as a separate revision;
differences are documented rather than silently mixed.

The implementation is derived from paper formulas and independently written
tests. Code, constants, kernel structure, comments, and fixtures from
`0xsero/turboquant` are not copied because that repository is GPL-3.0 and is
not an official author implementation. External implementations may be cited as
comparators only. Paper-faithful components and Forge extensions use distinct
profile names and result columns.

## Two-track architecture

### Paper-faithful reference

The reference track is CPU-testable and intentionally simple:

- deterministic Haar-random orthogonal matrices generated from seeded Gaussian
  matrices by QR, with a positive-diagonal sign convention;
- the dimension-specific spherical-coordinate density
  `f_d(x) ∝ (1 - x²)^((d - 3) / 2)` on `[-1, 1]`;
- symmetric Lloyd-Max codebooks generated independently for every
  `(dimension, bits)` pair;
- `TurboQuantMSE`, which stores the input norm and quantized rotated
  coordinates;
- `TurboQuantProduct`, which uses `(b - 1)` MSE bits plus a one-bit QJL
  residual sign sketch and residual norm; and
- exact physical-byte accounting for indices, norms, residual signs, padding,
  and codebook/matrix policy.

The product reconstruction follows Algorithm 2 of the paper:

```text
x_mse = dequantize_mse(indices, norm)
r = x - x_mse
signs = sign(S @ (r / ||r||))
x_qjl = sqrt(pi / 2) / d * ||r|| * S.T @ signs
x_hat = x_mse + x_qjl
```

The reference track validates orthogonality, deterministic encoding, spherical
coordinate moments, monotone Lloyd-Max distortion, MSE distortion scaling, and
near-zero signed inner-product bias over declared transform seeds.

### Compressed-domain systems runtime

The systems track retains the current Qwen2.5 packed-store/direct-attention
boundary but evolves it independently of the paper oracle:

- standard model prefill with only final-position logits retained;
- physically packed historical KV with a bounded raw tail;
- no full-history K/V reconstruction;
- prepared device-resident constants;
- compressed-domain score and value accumulation in Triton; and
- explicit generic/reference fallbacks that cannot appear inside a qualified
  direct trial.

The next kernel architecture is a contiguous, storage-owned page arena and
parallel page-state attention:

1. one page kernel grid computes independent stable softmax states
   `(max, denominator, numerator)` for every `(page, query-head)`;
2. one deterministic reduction kernel merges page states per query head;
3. the bounded raw tail is merged separately; and
4. workspace is reused and reported, never hidden in cache-byte claims.

This avoids both the current launch-per-page cost and the register-pressure
regression observed when two complete pages were inlined into one program.
Arena construction must not duplicate an already packed history.

Structured Hadamard rotations, adaptive bit budgets, sink/outlier protection,
and serving integration are later experiments. None is added until the
paper-faithful oracle and first page-state kernel are qualified.

## Models and execution classes

### v1 model gates

- **Qwen2.5-7B-Instruct:** primary local systems model, pinned to the existing
  Hub revision. The fair local baseline is identical NF4 model weights with an
  uncompressed FP16 KV cache.
- **Llama-3.1-8B-Instruct:** paper-reproduction model. It receives its own
  manifest, adapter, profile, tests, and evidence artifacts.

Qwen3, Qwen3.5, and Gemma remain later adapter-promotion milestones. Manifests
do not count as support. Qwen3.5 additionally requires native recurrent-state
preservation for non-full-attention layers; Gemma requires explicit multimodal
cache boundaries.

### Hardware classes

- `local_rtx5070ti_12gb`: headline consumer-GPU systems evidence. Mandatory
  contexts are 1K, 2K, 4K, 8K, and 16K; 32K is a separately declared capacity
  probe.
- `gilbreth_a100_80gb`: later paper-scale 32K–104K reproduction and Llama
  qualification.
- `gilbreth_a100_40gb`, `gilbreth_a30`, and `gilbreth_a10`: portability
  diagnostics only unless each receives a complete qualified matrix.

Results from different hardware classes are never pooled into one latency,
throughput, or memory claim.

## Evidence and benchmark contract

### Artifact schema v2

Every run records:

- immutable source, model, tokenizer, profile, codebook, paper-reference, and
  runner identities;
- clean/dirty source state and exact command;
- execution class, GPU/driver/runtime/packages, allocator settings, starting
  free VRAM, temperature, power, and competing-process snapshot;
- a predeclared matrix with exact row IDs, contexts, task/sample IDs, seeds,
  warmups, trials, and output length;
- row-level status, actual token counts, raw output/token IDs or losslessly
  referenced sidecars, metrics, dispatch/fallback counters, and failures; and
- an explicit qualification decision with all failed gates.

Large raw artifacts may live outside Git, but the repository must retain their
manifest, checksums, compact row summaries, reproduction command, and durable
location policy. Missing external payloads make the artifact unavailable, not
qualified.

### Benchmark tiers

1. **Codec/math:** synthetic sphere, Gaussian, heavy-tail, and adversarial
   vectors; GloVe retrieval; MSE, relative MSE, inner-product RMSE/bias,
   top-k agreement, `1@k`, physical bytes, and encode/decode latency.
2. **Fixed-token model replay:** 1K and 2K, identical teacher-forced tokens
   across HF baseline, packed reference, paper reference, and Triton direct;
   report logit error, KL divergence, top-1 agreement, greedy agreement, and
   direct dispatch.
3. **Paired NIAH:** smoke at 1K/2K/4K/8K; qualified local matrix at
   4K/8K/16K with depths `0.05/0.25/0.5/0.75/0.95` and ten seeds.
4. **RULER constrained local subset:** one pinned task from retrieval,
   variable tracing, aggregation, and QA, with 20 samples per task at
   4K/8K/16K.
5. **LongBench-E constrained subset:** pinned samples from `qasper_e`,
   `2wikimqa_e`, `passage_retrieval_en_e`, `repobench-p_e`, and `trec_e`,
   using official prompts and metrics.
6. **Systems:** the same prompt/decode cells at 1K/2K/4K/8K/16K; one cold/JIT
   run, two warmups, ten interleaved measured trials, and no outlier deletion.

The paper-scale Gilbreth lane later adds the paper NIAH length sweep,
LongBench-E reproduction, and DBpedia/GloVe ANN comparisons.

### Gates and pass counts

Each row is `passed`, `failed`, `invalid`, `oom`, `unsupported`, or
`unavailable`. Aggregate reports show counts for every state and never report
only successful rows.

- Mathematical reference: orthogonality and deterministic fixtures pass;
  product estimator signed bias satisfies a predeclared deterministic
  `abs(mean) <= 4 * SEM + numerical_slack` smoke gate across independent
  projection seeds, centered on the physically stored float32 residual norm;
  ordinary 95% confidence intervals are reported rather than misused as a
  deterministic pass gate; distortion summaries include confidence intervals
  rather than a single transform.
- Retrieval-safe: baseline accuracy is at least 95%; every expected row
  completes; per-context baseline-minus-profile accuracy is at most 2
  percentage points; aggregate upper 95% paired-bootstrap bound is at most 2
  percentage points; packed-reference/direct mismatch count is zero.
- Physical storage: actual packed KV is at least 1.8× smaller than the
  uncompressed FP16 KV representation at the claimed 8K/16K context.
- First local Pareto milestone: at one of 8K or 16K, the lower 95% confidence
  bound for direct/baseline decode throughput exceeds `1.0`, the upper
  confidence bound for direct/baseline decode latency is below `1.0`, peak
  allocated VRAM is no higher than baseline, physical storage passes, and all
  quality gates pass.
- Strong v1 systems claim: the same Pareto gate passes at both 8K and 16K.

Peak allocated/reserved VRAM, physical KV storage, prefill latency, compression
latency, host/device TTFT, per-token decode p50/p95/p99, throughput, cold/JIT
time, and OOM boundaries are reported separately.

## Public documentation

The public repository contains:

- a claim ledger mapping every number to an artifact ID and row/query;
- a paper reproduction report separating matched, partial, contradicted, and
  unavailable results;
- context-by-context quality and systems tables with pass/fail/invalid counts;
- a Pareto report across effective physical bytes, quality, latency, and peak
  VRAM;
- a provenance/license ledger for papers, datasets, model weights, and any
  borrowed dependencies; and
- exact local and Gilbreth reproduction instructions.

Stale planning and script documentation is either updated or archived and is
included in claim-consistency tests.

## Explicit non-goals for v1

- No claim of official Google/author endorsement or implementation equivalence.
- No copying from GPL TurboQuant implementations.
- No vLLM/SGLang server, continuous batching, beam search, sampling,
  speculative/MTP decode, distributed execution, or offload in the local
  headline.
- No Qwen3/Qwen3.5/Gemma support claim without their own adapters and artifacts.
- No 32K/64K/104K success statement based on a projection or another model.
