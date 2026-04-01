#!/usr/bin/env python3
"""
Debug script to compare baseline vs TurboQuant KV cache states during generation.
Focus: Find why 3-bit produces garbled output despite correct static compression.

Updated for new TurboQuantCache API with:
- Per-layer rotation matrices
- Uncompressed buffer support
- Proper Lloyd-Max codebooks
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force Python fallback for correct 3-bit handling
os.environ["TURBOQUANT_FORCE_PYTHON"] = "1"

from core.turboquant_cache import TurboQuantCache


def test_baseline():
    """Run baseline with standard DynamicCache."""
    print("\n=== BASELINE (DynamicCache) ===")

    from transformers import BitsAndBytesConfig

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

    cache = DynamicCache()

    with torch.no_grad():
        # Prefill
        outputs = model(**inputs, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        print(f"Prefill token: {tokenizer.decode(next_token[0])}")

        # Decode 10 tokens
        tokens = [next_token]
        for i in range(10):
            outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token)

            decoded = tokenizer.decode(next_token[0])

            # Debug: Check cache stats
            seq_len = cache.get_seq_length()

            print(f"Token {i + 1}: '{decoded}' | Seq len: {seq_len}")

        full_text = tokenizer.decode(torch.cat(tokens, dim=-1)[0])
        print(f"\nGenerated: {full_text}")

    return cache


def test_turboquant(buffer_size: int = 128):
    """Run TurboQuant with 3-bit compression.

    Args:
        buffer_size: Number of recent tokens to keep uncompressed.
                    Set to 0 for fully compressed cache.
    """
    print(f"\n=== TURBOQUANT (3-bit, buffer_size={buffer_size}) ===")

    from transformers import BitsAndBytesConfig

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

    cache = TurboQuantCache(
        head_dim=128,
        n_qjl=128,
        n_outliers=32,
        device=torch.device("cuda"),
        dtype=torch.float16,
        use_qjl=False,
        bit_width=3,
        buffer_size=buffer_size,  # NEW: uncompressed buffer for recent tokens
    )

    with torch.no_grad():
        # Prefill
        outputs = model(**inputs, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        print(f"Prefill token: {tokenizer.decode(next_token[0])}")

        # Decode 10 tokens
        tokens = [next_token]
        for i in range(10):
            outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token)

            decoded = tokenizer.decode(next_token[0])

            # Debug: Check cache stats
            seq_len = cache.get_seq_length()

            # Check buffer and compressed sizes
            layer_idx = 0
            buffer_len = 0
            compressed_len = 0
            if cache._key_buffers[layer_idx] is not None:
                buffer_len = cache._key_buffers[layer_idx].shape[-2]
            if cache._compressed_keys[layer_idx] is not None:
                compressed_len = cache._compressed_keys[layer_idx][
                    7
                ]  # seq is at index 7

            print(
                f"Token {i + 1}: '{decoded}' | "
                f"Seq len: {seq_len}, "
                f"Buffer: {buffer_len}, "
                f"Compressed: {compressed_len}"
            )

        full_text = tokenizer.decode(torch.cat(tokens, dim=-1)[0])
        print(f"\nGenerated: {full_text}")

    return cache


def compare_caches():
    """Compare baseline vs TurboQuant cache states."""
    print("\n" + "=" * 60)
    print("COMPARING BASELINE VS TURBOQUANT")
    print("=" * 60)

    baseline_cache = test_baseline()

    # Test with buffer (should have best quality due to recent tokens being uncompressed)
    print("\n" + "-" * 60)
    print("Testing TurboQuant WITH buffer (buffer_size=128)")
    print("Recent tokens kept in full precision → best quality")
    print("-" * 60)
    turboquant_cache_with_buffer = test_turboquant(buffer_size=128)

    # Test without buffer for comparison
    print("\n" + "-" * 60)
    print("Testing TurboQuant WITHOUT buffer (buffer_size=0)")
    print("All tokens compressed → tests pure TurboQuant quality")
    print("-" * 60)
    turboquant_cache_no_buffer = test_turboquant(buffer_size=0)

    print("\n=== COMPARISON SUMMARY ===")
    print("- Baseline should produce: ' Paris is the capital of France.'")
    print("- TurboQuant WITH buffer should match baseline closely")
    print("- TurboQuant WITHOUT buffer tests pure compression quality")
    print("\nIf both TurboQuant variants produce garbled output, the bug is in")
    print("the compression algorithm itself (rotation matrices, centroids, etc.)")
    print("\nIf only WITHOUT buffer produces garbled output, the algorithm is")
    print("correct but compression quality needs improvement.")


if __name__ == "__main__":
    compare_caches()
