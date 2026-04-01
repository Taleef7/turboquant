#!/usr/bin/env python3
"""Debug script to trace TurboQuantCacheV2 during generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project to path
import sys

sys.path.insert(0, "/home/taleef/projects/turboquant")

from core.turboquant_cache_v2 import TurboQuantCacheV2


def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Short prompt for debugging
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"Prompt: {prompt}")
    print(f"Input tokens: {inputs.input_ids.shape}")

    # Test 1: Baseline generation (no cache manipulation)
    print("\n=== BASELINE ===")
    with torch.no_grad():
        baseline_out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
        )
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    print(f"Baseline: {baseline_text}")

    # Test 2: TurboQuant cache (buffer_size=0 to force compression immediately)
    print("\n=== TURBOQUANT V2 (buffer_size=0) ===")
    cache = TurboQuantCacheV2(head_dim=128, bits=4, buffer_size=0)
    with torch.no_grad():
        tq_out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )
    tq_text = tokenizer.decode(tq_out[0], skip_special_tokens=True)
    print(f"TurboQuant (buffer=0): {tq_text}")

    # Test 3: Large buffer (should behave like baseline for short sequences)
    print("\n=== TURBOQUANT V2 (buffer_size=1024 - no compression) ===")
    cache_big = TurboQuantCacheV2(head_dim=128, bits=4, buffer_size=1024)
    with torch.no_grad():
        tq_big_out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            past_key_values=cache_big,
            use_cache=True,
        )
    tq_big_text = tokenizer.decode(tq_big_out[0], skip_special_tokens=True)
    print(f"TurboQuant (buffer=1024): {tq_big_text}")

    # Check if buffer-only mode matches baseline
    print("\n=== ANALYSIS ===")
    if baseline_text == tq_big_text:
        print("✓ Buffer-only mode matches baseline (cache integration OK)")
    else:
        print("✗ Buffer-only mode differs from baseline (cache integration broken)")

    if baseline_text == tq_text:
        print("✓ Compression mode matches baseline")
    else:
        print("✗ Compression mode differs from baseline")
        print("  This indicates compression/decompression is introducing errors")

    # Debug: Check layer 0 compression fidelity
    print("\n=== LAYER 0 COMPRESSION DEBUG ===")
    cache_debug = TurboQuantCacheV2(head_dim=128, bits=4, buffer_size=0)

    # Get keys from first forward pass
    with torch.no_grad():
        outputs = model(
            **inputs,
            past_key_values=cache_debug,
            use_cache=True,
            output_attentions=False,
        )

    # Check compressed storage
    if cache_debug._compressed_keys[0] is not None:
        compressed_k = cache_debug._compressed_keys[0]
        print(f"Compressed key shape: {compressed_k['shape']}")
        print(f"Compressed key indices shape: {compressed_k['indices'].shape}")
        print(f"Compressed key norms shape: {compressed_k['norms'].shape}")

        # Decompress and check
        decompressed_k = cache_debug._decompress_keys(compressed_k, 0)
        print(f"Decompressed key shape: {decompressed_k.shape}")
        print(
            f"Decompressed key range: [{decompressed_k.min():.4f}, {decompressed_k.max():.4f}]"
        )


if __name__ == "__main__":
    main()
