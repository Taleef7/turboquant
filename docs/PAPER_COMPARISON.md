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
| Quality evaluation | Broad research benchmarks | Paired NIAH runner with explicit invalid/failed states; retained direct evidence is invalid, so no retrieval result is claimed. |
| Model support | Paper-dependent architectures | Qwen2.5 direct adapter only; other listed families have no adapter. |
| Serving/runtime | Research and production-adjacent contexts | No server, batching, beam search, speculative decoding, paging/offload, or distributed path. |

## Why an invalid quality artifact remains valuable

The artifact `qualified-v4/20260718T053419Z-unknown` retains attempted NIAH
rows, systems logs, and a mechanical 32K admission decision. The corrected
validator rejects it because effective contexts were materially shorter than
requested and the commit/binding provenance is incomplete. That makes the next
engineering step clear: rerun the fixed protocol with full context fidelity and
resolved provenance before evaluating profile quality.

## Deliberate differences from the legacy prototype

`TurboQuantCacheV2` is retained only to preserve historical experiments. It
dequantizes historical K/V state and reconstructs it for attention, so it is
not the Forge direct backend. The new profile loader accepts only
`packed_kv_block_store`; an attempt to select V2 fails before the new cache can
be mutated.

## Next evidence needed for a closer comparison

1. Produce a Qwen2.5 profile whose complete NIAH matrix satisfies the declared
   retrieval gate and whose systems artifact remains structurally valid.
2. Benchmark LongBench and RULER with the same model/profile/code bindings.
3. Add a separately tested QJL residual implementation before comparing
   low-bit operating points.
4. Promote each additional architecture only after its adapter, capability,
   quality, and systems evidence are all present.

Read [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) before comparing a Forge
artifact with results from another project.
