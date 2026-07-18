# Portability Surface

Portability in Forge is an explicit adapter-promotion process, not a model-name
switch. `ModelCapabilities` derives layer types, attention heads, KV heads, head
dimension, RoPE settings, text/multimodal flags, and cache behavior from the
loaded configuration and a pinned manifest. `require_direct_decode_support()`
then rejects unsupported requests before a cache store can be mutated.

## Current direct contract

Only the pinned Qwen2.5-7B-Instruct manifest and `qwen25-7b-direct-v1` profile
can select direct packed decode. The request must be text-only, cache-enabled,
full attention, batch one, greedy, non-beam, non-sampling, non-speculative, and
non-MTP. A manifest/config/revision mismatch also fails closed.

The direct adapter uses standard attention for prompt prefill and packed direct
attention only for one-token decode. It is not a general DynamicCache or serving
replacement.

## Planned families

| Family | What the manifest currently proves | What it does not prove | Adapter promotion gate |
| --- | --- | --- | --- |
| Qwen3 dense | Dense full-attention configuration can be parsed and checked. | That Transformers layout, cache updates, or kernels work with Forge. | Dedicated adapter, pinned layout tests, direct/reference GPU tests, and its own quality/systems artifacts. |
| Qwen3.5 hybrid | Full-attention layers are distinguishable from linear-attention layers. | That recurrent linear state can be compressed or substituted. | Preserve every linear-attention state natively; compress only full-attention KV blocks; then run hybrid adapter and evidence tests. |
| Gemma 4 E4B | The entry is manifest-only, multimodal, and requires capacity preflight. | A text/vision interface, cache layout, or viable GPU footprint. | Dedicated adapter, explicit processor/vision boundary tests, cache-capacity preflight, and qualified evidence. |

For Qwen3.5, `compressible_layer_indices` contains only `full_attention` layers;
`native_state_layer_indices` contains the linear layers. Current direct support
rejects linear-attention layers because native-state handling has not been
implemented. This is intentional: no hybrid state is silently treated as KV
cache.

## KV-memory accounting boundary

KV-cache claims include only the packed KV payload, its metadata/alignment, and
the bounded fp16/bf16 KV window reported by the store. They exclude model
weights, activations, CUDA workspace, tokenizer state, processor state, image
features, and vision or multimodal caches. Systems artifacts report allocator
peaks separately so a KV-size number is never confused with total VRAM use.

## Promotion checklist

1. Add a complete pinned manifest and bind it to its profile.
2. Implement a small adapter with cache-phase guards and no cache mutation
   before capability validation.
3. Add CPU contract tests and GPU direct/reference parity tests for the exact
   architecture, including RoPE and cache layout.
4. Run the model-level smoke test, then a complete quality and systems evidence
   ladder.
5. Update public docs only after a named raw artifact supports the new scope.

Until all five are complete, a manifest is discovery metadata—not model support.
