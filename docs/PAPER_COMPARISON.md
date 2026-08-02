# TurboQuant Paper Comparison

The original TurboQuant research describes a broader algorithm and evaluation
surface than this repository. Forge intentionally starts with a narrow,
auditable direct-decode path instead of copying a reference implementation or
claiming paper-level results.

| Dimension | Original research scope | Forge implementation scope |
| --- | --- | --- |
| Code provenance | Research implementation and paper | New clean-room Python/Triton implementation under MIT; no GPL code is copied. |
| KV storage | Quantized cache representations | Physically bit-packed fixed blocks plus bounded fp16/bf16 tail; exact tensor bytes are measured. |
| Attention execution | Research-specific evaluation | Blockwise PyTorch reference and direct Triton score/value mixing for the pinned Qwen2.5 path. |
| Quality evaluation | Broad research benchmarks | Paired NIAH runner with explicit invalid/failed states; the named bound artifact qualifies an exact 4K/8K matrix, while its 32K leg is capacity-rejected. |
| Model support | Paper-dependent architectures | Qwen2.5 direct adapter only; other listed families have no adapter. |
| Serving/runtime | Research and production-adjacent contexts | No server, batching, beam search, speculative decoding, paging/offload, or distributed path. |

## Current bounded evidence and retired invalid evidence

`qualified-semantic-final/20260718T074435Z-5c9d4c34b0f0`, bound to commit
`5c9d4c34b0f0c31d015e2166ae7ad401858b202d`, qualifies its exact 4K/8K NIAH
matrix (12/12 baseline/direct correctness and exact token-ID agreement). It
also records a complete 8K systems comparison that is slower and uses higher
allocator peaks than baseline. The top-level report remains `failed`: its
32K capacity preflight projects 12,809,961,472 bytes against an allowed
10,405,439,078-byte budget, so no 32K child was launched.

`qualified-v4/20260718T053419Z-unknown` remains retired invalid history. Its
effective contexts were materially shorter than requested and its binding was
incomplete; it must not be combined with the newer artifact to create a claim.

The latest local engineering checkpoint is separate diagnostic evidence, bound
to `eba1c5b`: Qwen2.5 4K/8K packed rows fail strict full-logit gates, the fused
page schedule is synthetic-only, and the v8 value ablation is negative. See
[RESULTS.md](RESULTS.md) and [GILBRETH_HANDOFF.md](GILBRETH_HANDOFF.md) for
the raw-artifact workflow and the next cluster matrix. None of these results
establishes paper equivalence, 32K qualification, or an end-to-end win.

## Deliberate differences from the legacy prototype

`TurboQuantCacheV2` is retained only to preserve historical experiments. It
dequantizes historical K/V state and reconstructs it for attention, so it is
not the Forge direct backend. The new profile loader accepts only
`packed_kv_block_store`; an attempt to select V2 fails before the new cache can
be mutated.

## Next evidence needed for a closer comparison

1. Make 32K capacity-admissible, then produce and validate its own quality and
   systems children before extending the 8K finding.
2. Benchmark LongBench and RULER with the same model/profile/code bindings.
3. Add a separately tested QJL residual implementation before comparing
   low-bit operating points.
4. Promote each additional architecture only after its adapter, capability,
   quality, and systems evidence are all present.

Read [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) before comparing a Forge
artifact with results from another project.
