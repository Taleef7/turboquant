"""
Validate TurboQuantMSE on real Qwen key states.

Success Criteria:
- Normalized MSE (per-element): < 0.02
- Cosine similarity: > 0.99

Usage:
  python scripts/validate_single_vector.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from core.turboquant_simple import TurboQuantMSE, TurboQuantValueMSE


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HEAD_DIM = 128
BITS = 4


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between corresponding vectors."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])

    a_norm = torch.nn.functional.normalize(a_flat, dim=-1)
    b_norm = torch.nn.functional.normalize(b_flat, dim=-1)

    cos_sim = (a_norm * b_norm).sum(dim=-1)
    return cos_sim.mean().item()


def normalized_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute normalized MSE (per-element MSE / variance)."""
    mse = ((original - reconstructed) ** 2).mean().item()
    var = original.var().item()
    return mse / var if var > 0 else mse


def extract_key_states(model, tokenizer, prompt: str) -> dict:
    """Run forward pass and extract key states from all layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    key_states = {}

    # Forward pass with output_hidden_states to get more info
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)

    # Extract from past_key_values
    pkv = outputs.past_key_values
    print(f"Cache type: {type(pkv).__name__}")

    # DynamicCache has .layers attribute (list of layer caches)
    if hasattr(pkv, "layers") and pkv.layers:
        print(f"  Found {len(pkv.layers)} layers")
        for i, layer_cache in enumerate(pkv.layers):
            # Each layer cache should have key_states and value_states
            if hasattr(layer_cache, "key_states"):
                key_states[i] = layer_cache.key_states.detach().clone()
            elif isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                key_states[i] = layer_cache[0].detach().clone()
            elif hasattr(layer_cache, "keys"):
                key_states[i] = layer_cache.keys.detach().clone()
            else:
                # Debug: print first layer structure
                if i == 0:
                    print(f"  Layer cache type: {type(layer_cache).__name__}")
                    if hasattr(layer_cache, "__dict__"):
                        print(f"  Layer attrs: {list(layer_cache.__dict__.keys())}")

    # Alternative: try to_legacy_cache()
    if not key_states and hasattr(pkv, "to_legacy_cache"):
        print("  Trying to_legacy_cache()...")
        try:
            legacy = pkv.to_legacy_cache()
            print(f"  Legacy format: {type(legacy).__name__}, len={len(legacy)}")
            for i, layer_cache in enumerate(legacy):
                if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                    key_states[i] = layer_cache[0].detach().clone()
        except Exception as e:
            print(f"  to_legacy_cache failed: {e}")

    return key_states


def validate_key_compression():
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    print(f"Model loaded. Extracting key states...")

    # Use a reasonable prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * 50
    key_states = extract_key_states(model, tokenizer, prompt)

    print(f"Extracted keys from {len(key_states)} layers")

    # Initialize quantizers
    key_quantizer = TurboQuantMSE(head_dim=HEAD_DIM, bits=BITS, device="cuda")
    value_quantizer = TurboQuantValueMSE(head_dim=HEAD_DIM, bits=BITS, device="cuda")

    print("\n" + "=" * 60)
    print("KEY COMPRESSION VALIDATION (TurboQuantMSE)")
    print("=" * 60)

    # Test on first, middle, and last layers
    test_layers = [0, len(key_states) // 2, len(key_states) - 1]

    all_cos_sim = []
    all_norm_mse = []

    for layer_idx in test_layers:
        keys = key_states[layer_idx]  # (batch, heads, seq, head_dim)

        # Analyze input statistics
        flat_keys = keys.reshape(-1, HEAD_DIM)
        norms = torch.norm(flat_keys, dim=-1)

        print(f"\nLayer {layer_idx}:")
        print(f"  Input shape: {keys.shape}")
        print(
            f"  Key norms - min: {norms.min():.2f}, max: {norms.max():.2f}, mean: {norms.mean():.2f}"
        )

        # Compress and decompress
        compressed = key_quantizer.compress(keys)
        reconstructed = key_quantizer.decompress(compressed)

        # Compute metrics
        cos_sim = cosine_similarity(keys.float(), reconstructed.float())
        norm_mse = normalized_mse(keys.float(), reconstructed.float())

        all_cos_sim.append(cos_sim)
        all_norm_mse.append(norm_mse)

        print(f"  Cosine similarity: {cos_sim:.6f} (target: > 0.99)")
        print(f"  Normalized MSE:    {norm_mse:.6f} (target: < 0.02)")

        # Check pass/fail
        cos_pass = "PASS" if cos_sim > 0.99 else "FAIL"
        mse_pass = "PASS" if norm_mse < 0.02 else "FAIL"
        print(f"  Result: cosine={cos_pass}, mse={mse_pass}")

    # Overall metrics
    mean_cos_sim = sum(all_cos_sim) / len(all_cos_sim)
    mean_norm_mse = sum(all_norm_mse) / len(all_norm_mse)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Mean cosine similarity: {mean_cos_sim:.6f}")
    print(f"Mean normalized MSE:    {mean_norm_mse:.6f}")

    overall_pass = mean_cos_sim > 0.99 and mean_norm_mse < 0.02
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # Also test value compression briefly
    print("\n" + "=" * 60)
    print("VALUE COMPRESSION VALIDATION (TurboQuantValueMSE)")
    print("=" * 60)

    # Use layer 0 keys as proxy for values (similar distribution)
    test_values = key_states[0]
    compressed_v = value_quantizer.compress(test_values)
    reconstructed_v = value_quantizer.decompress(compressed_v)

    cos_sim_v = cosine_similarity(test_values.float(), reconstructed_v.float())
    norm_mse_v = normalized_mse(test_values.float(), reconstructed_v.float())

    print(f"Value cosine similarity: {cos_sim_v:.6f}")
    print(f"Value normalized MSE:    {norm_mse_v:.6f}")

    return overall_pass


if __name__ == "__main__":
    success = validate_key_compression()
    sys.exit(0 if success else 1)
