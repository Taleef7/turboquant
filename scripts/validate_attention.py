"""
Validate that quantized keys produce similar attention scores.

Tests that Q @ K^T with compressed keys matches Q @ K^T with original keys.
Success Criteria:
- Attention score cosine similarity: > 0.99

Usage:
  python scripts/validate_attention.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from core.turboquant_simple import TurboQuantMSE


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HEAD_DIM = 128
BITS = 4


def cosine_similarity_2d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between corresponding attention matrices."""
    # Flatten all but last dim, then compute cosine
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    # Normalize
    a_norm = F.normalize(a_flat.unsqueeze(0), dim=-1)
    b_norm = F.normalize(b_flat.unsqueeze(0), dim=-1)

    return (a_norm * b_norm).sum().item()


def extract_qkv_states(model, tokenizer, prompt: str) -> dict:
    """Extract query and key states from attention layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Just run forward pass and extract from cache
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)

    # Get keys from cache
    pkv = outputs.past_key_values
    key_states = {}

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

    return key_states, inputs


def compute_attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute scaled dot-product attention scores: Q @ K^T / sqrt(d)."""
    # query: (batch, heads, seq_q, head_dim)
    # key: (batch, heads, seq_k, head_dim)
    d = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d**0.5)
    return scores


def validate_attention():
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

    print("Model loaded. Extracting key states...")

    # Use a reasonable prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * 30
    key_states, inputs = extract_qkv_states(model, tokenizer, prompt)

    print(f"Extracted keys from {len(key_states)} layers")

    # Initialize quantizer
    key_quantizer = TurboQuantMSE(head_dim=HEAD_DIM, bits=BITS, device="cuda")

    print("\n" + "=" * 60)
    print("ATTENTION SCORE VALIDATION")
    print("=" * 60)

    # Test layers
    test_layers = [0, len(key_states) // 2, len(key_states) - 1]

    all_cos_sim = []

    for layer_idx in test_layers:
        keys = key_states[layer_idx].float()  # (batch, heads, seq, head_dim)

        # Use keys as both Q and K for simplicity (self-attention pattern)
        # In practice, Q and K come from different projections, but the pattern
        # of how compression affects scores is what we're validating
        queries = keys.clone()

        # Compute original attention scores
        orig_scores = compute_attention_scores(queries, keys)

        # Compress and decompress keys
        compressed = key_quantizer.compress(keys)
        keys_hat = key_quantizer.decompress(compressed)

        # Compute attention scores with compressed keys
        compressed_scores = compute_attention_scores(queries, keys_hat)

        # Compute cosine similarity between attention score matrices
        cos_sim = cosine_similarity_2d(orig_scores, compressed_scores)
        all_cos_sim.append(cos_sim)

        # Also compute max absolute difference
        max_diff = (orig_scores - compressed_scores).abs().max().item()
        mean_diff = (orig_scores - compressed_scores).abs().mean().item()

        print(f"\nLayer {layer_idx}:")
        print(f"  Attention scores shape: {orig_scores.shape}")
        print(f"  Cosine similarity: {cos_sim:.6f} (target: > 0.99)")
        print(f"  Max abs diff:      {max_diff:.6f}")
        print(f"  Mean abs diff:     {mean_diff:.6f}")

        # Check pass/fail
        status = "PASS" if cos_sim > 0.99 else "FAIL"
        print(f"  Result: {status}")

    # Overall metrics
    mean_cos_sim = sum(all_cos_sim) / len(all_cos_sim)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Mean attention cosine similarity: {mean_cos_sim:.6f}")

    overall_pass = mean_cos_sim > 0.99
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # Additional test: verify softmax distributions are similar
    print("\n" + "=" * 60)
    print("SOFTMAX DISTRIBUTION VALIDATION")
    print("=" * 60)

    # Use layer 14 for detailed softmax analysis (middle layer, more moderate scales)
    keys = key_states[14].float()
    queries = keys.clone()

    orig_scores = compute_attention_scores(queries, keys)
    compressed = key_quantizer.compress(keys)
    keys_hat = key_quantizer.decompress(compressed)
    compressed_scores = compute_attention_scores(queries, keys_hat)

    # Apply causal mask (lower triangular) before softmax
    seq_len = orig_scores.shape[-1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=keys.device), diagonal=1
    ).bool()
    orig_scores_masked = orig_scores.masked_fill(causal_mask, float("-inf"))
    compressed_scores_masked = compressed_scores.masked_fill(causal_mask, float("-inf"))

    # Apply softmax
    orig_probs = F.softmax(orig_scores_masked, dim=-1)
    compressed_probs = F.softmax(compressed_scores_masked, dim=-1)

    # For non-inf positions, compute KL divergence
    # Since we have causal mask, only look at valid positions
    valid_mask = ~causal_mask.unsqueeze(0).unsqueeze(0)

    # Flatten and filter valid
    orig_flat = orig_probs[valid_mask.expand_as(orig_probs)]
    comp_flat = compressed_probs[valid_mask.expand_as(compressed_probs)]

    # Compute metrics on valid attention probabilities
    eps = 1e-10
    kl_div = (
        orig_flat * (torch.log(orig_flat + eps) - torch.log(comp_flat + eps))
    ).sum() / orig_flat.numel()

    # Total variation distance (mean absolute difference)
    tv_dist = 0.5 * (orig_probs - compressed_probs).abs().sum(dim=-1).mean()

    # Max probability error
    max_prob_error = (orig_probs - compressed_probs).abs().max()

    # Cosine similarity of attention probability distributions
    attn_cos_sim = cosine_similarity_2d(orig_probs, compressed_probs)

    print(f"Attention probability cosine sim: {attn_cos_sim:.6f}")
    print(f"KL divergence (per element):      {kl_div:.6f}")
    print(f"Total variation distance:         {tv_dist:.6f}")
    print(f"Max probability error:            {max_prob_error:.6f}")

    print(f"KL divergence: {kl_div:.6f}")
    print(f"Total variation distance: {tv_dist:.6f}")

    return overall_pass


if __name__ == "__main__":
    success = validate_attention()
    sys.exit(0 if success else 1)
