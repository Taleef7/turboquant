#!/usr/bin/env python3
"""
Capture actual KV tensors from a real model forward pass and test
roundtrip compression quality on them.

This will reveal if real attention KV distributions have characteristics
that break our compression algorithm.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TURBOQUANT_FORCE_PYTHON"] = "1"

from core.turboquant_cache import TurboQuantCache


def capture_kv_tensors():
    """Capture real KV tensors from model prefill."""
    print("Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"Prompt: '{prompt}'")
    print(f"Input tokens: {inputs.input_ids.shape[1]}")

    # Run prefill with standard cache to capture KV
    cache = DynamicCache()

    with torch.no_grad():
        outputs = model(
            **inputs, past_key_values=cache, use_cache=True, return_dict=True
        )
        cache = outputs.past_key_values

        # Get predicted token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        predicted = tokenizer.decode(next_token)
        print(f"Baseline prediction: '{predicted}'")

    # Extract KV from first layer
    layer_idx = 0
    # DynamicCache stores layers as DynamicLayer objects
    # Access via cache.layers[layer_idx].key_states and .value_states
    cache_layer = cache.layers[layer_idx]
    k_real = cache_layer.key_states  # Keys: (batch, heads, seq, head_dim)
    v_real = cache_layer.value_states  # Values: (batch, heads, seq, head_dim)

    print(f"\nLayer {layer_idx} KV stats:")
    print(f"  K: shape={k_real.shape}, dtype={k_real.dtype}")
    print(
        f"  K: mean={k_real.mean():.6f}, std={k_real.std():.6f}, norm={k_real.norm():.4f}"
    )
    print(f"  K: min={k_real.min():.6f}, max={k_real.max():.6f}")
    print(f"  V: shape={v_real.shape}, dtype={v_real.dtype}")
    print(
        f"  V: mean={v_real.mean():.6f}, std={v_real.std():.6f}, norm={v_real.norm():.4f}"
    )
    print(f"  V: min={v_real.min():.6f}, max={v_real.max():.6f}")

    return k_real, v_real, model, tokenizer, inputs


def test_real_kv_roundtrip(k_real, v_real):
    """Test compression roundtrip on real KV tensors."""
    print("\n" + "=" * 60)
    print("TESTING COMPRESSION ROUNDTRIP ON REAL KV DATA")
    print("=" * 60)

    # Create cache
    cache = TurboQuantCache(
        head_dim=128,
        n_qjl=128,
        n_outliers=32,
        device=torch.device("cuda"),
        dtype=torch.float16,
        use_qjl=False,
        bit_width=3,
    )

    # Test K roundtrip
    print("\n--- Keys (K) Roundtrip ---")
    k_compressed = cache._compress(k_real)
    k_roundtrip = cache._decompress(k_compressed)

    k_mse = ((k_real.float() - k_roundtrip.float()) ** 2).mean()
    k_cosine = torch.nn.functional.cosine_similarity(
        k_real.flatten(), k_roundtrip.flatten(), dim=0
    )
    k_norm_diff = (k_real.norm() - k_roundtrip.norm()).abs()

    print(f"  Original norm: {k_real.norm():.4f}")
    print(f"  Roundtrip norm: {k_roundtrip.norm():.4f}")
    print(f"  Norm difference: {k_norm_diff:.4f}")
    print(f"  MSE: {k_mse:.6f}")
    print(f"  Cosine similarity: {k_cosine:.6f}")

    # Test V roundtrip
    print("\n--- Values (V) Roundtrip ---")
    v_compressed = cache._compress(v_real)
    v_roundtrip = cache._decompress(v_compressed)

    v_mse = ((v_real.float() - v_roundtrip.float()) ** 2).mean()
    v_cosine = torch.nn.functional.cosine_similarity(
        v_real.flatten(), v_roundtrip.flatten(), dim=0
    )
    v_norm_diff = (v_real.norm() - v_roundtrip.norm()).abs()

    print(f"  Original norm: {v_real.norm():.4f}")
    print(f"  Roundtrip norm: {v_roundtrip.norm():.4f}")
    print(f"  Norm difference: {v_norm_diff:.4f}")
    print(f"  MSE: {v_mse:.6f}")
    print(f"  Cosine similarity: {v_cosine:.6f}")

    return k_roundtrip, v_roundtrip


def test_attention_impact(k_real, v_real, k_roundtrip, v_roundtrip, model):
    """Test how compression affects attention computation."""
    print("\n" + "=" * 60)
    print("TESTING ATTENTION IMPACT")
    print("=" * 60)

    # Get query from layer 0's self_attn
    layer_0 = model.model.layers[0]

    # Create a dummy query (simulate next token query)
    batch, heads, seq, head_dim = k_real.shape
    q = torch.randn(batch, heads, 1, head_dim, dtype=torch.float16, device="cuda")

    # Attention with original KV
    attn_original = torch.matmul(q, k_real.transpose(-2, -1)) / (head_dim**0.5)
    probs_original = torch.nn.functional.softmax(attn_original, dim=-1)
    out_original = torch.matmul(probs_original, v_real)

    # Attention with roundtrip KV
    attn_roundtrip = torch.matmul(q, k_roundtrip.transpose(-2, -1)) / (head_dim**0.5)
    probs_roundtrip = torch.nn.functional.softmax(attn_roundtrip, dim=-1)
    out_roundtrip = torch.matmul(probs_roundtrip, v_roundtrip)

    # Compare
    logits_diff = (attn_original - attn_roundtrip).abs()
    probs_diff = (probs_original - probs_roundtrip).abs()
    out_diff = (out_original - out_roundtrip).abs()

    print(f"\nAttention differences:")
    print(f"  Logits: mean={logits_diff.mean():.6f}, max={logits_diff.max():.6f}")
    print(f"  Probs:  mean={probs_diff.mean():.6f}, max={probs_diff.max():.6f}")
    print(f"  Output: mean={out_diff.mean():.6f}, max={out_diff.max():.6f}")

    # Check argmax stability
    argmax_original = probs_original.argmax(dim=-1)
    argmax_roundtrip = probs_roundtrip.argmax(dim=-1)
    argmax_match = (argmax_original == argmax_roundtrip).float().mean()

    print(f"\nArgmax stability:")
    print(f"  Match rate: {argmax_match:.2%}")

    if argmax_match < 1.0:
        print(f"  WARNING: Some heads attending to different positions!")
        mismatches = (argmax_original != argmax_roundtrip).nonzero(as_tuple=True)
        print(f"  Mismatched heads: {len(mismatches[0])}/{heads}")
        for i in range(min(4, len(mismatches[0]))):
            head_idx = mismatches[1][i].item()
            orig_pos = argmax_original[0, head_idx, 0].item()
            new_pos = argmax_roundtrip[0, head_idx, 0].item()
            print(f"    Head {head_idx}: {orig_pos} -> {new_pos}")

    # Show top-5 attention positions for first few heads
    print(f"\nTop-3 attention positions (first 2 heads):")
    for head_idx in range(min(2, heads)):
        orig_topk = torch.topk(probs_original[0, head_idx, 0], k=3)
        round_topk = torch.topk(probs_roundtrip[0, head_idx, 0], k=3)

        print(f"\n  Head {head_idx}:")
        print(
            f"    Original:  pos={orig_topk.indices.tolist()}, prob={orig_topk.values.tolist()}"
        )
        print(
            f"    Roundtrip: pos={round_topk.indices.tolist()}, prob={round_topk.values.tolist()}"
        )


def main():
    print("=" * 60)
    print("REAL KV TENSOR CAPTURE AND ROUNDTRIP TEST")
    print("=" * 60)

    # Capture real KV
    k_real, v_real, model, tokenizer, inputs = capture_kv_tensors()

    # Test roundtrip
    k_roundtrip, v_roundtrip = test_real_kv_roundtrip(k_real, v_real)

    # Test attention impact
    test_attention_impact(k_real, v_real, k_roundtrip, v_roundtrip, model)

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("If cosine similarity is >0.98 and argmax match is >95%,")
    print("then compression quality is good. The bug must be elsewhere.")
    print("If quality is poor, we need to investigate why real KV data")
    print("is harder to compress than random data.")


if __name__ == "__main__":
    main()
