#!/usr/bin/env python3
"""Test with longer prompt to reproduce quality issues."""

import sys

sys.path.insert(0, "/home/taleef/projects/turboquant")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.turboquant_cache_v2 import TurboQuantCacheV2


def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print(f"Loading {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # Create a 500-token prompt (this is what was failing)
    filler = "The quick brown fox jumps over the lazy dog. " * 500
    tokens = tokenizer.encode(filler)[:500]
    prompt = tokenizer.decode(tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

    # Baseline
    print("\n=== BASELINE (200 tokens) ===")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    baseline = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"Generated: {baseline[:200]}...")

    # 4-bit keys with buffer
    print("\n=== 4-BIT KEYS (buffer=128, 200 tokens) ===")
    cache = TurboQuantCacheV2(head_dim=128, bits=4, buffer_size=128)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=False, past_key_values=cache
        )
    compressed = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"Generated: {compressed[:200]}...")
    print(
        f"\nBuffer shape: {cache._key_buffers[0].shape if cache._key_buffers[0] is not None else None}"
    )
    print(
        f"Compressed shape: {cache._compressed_keys[0]['shape'] if cache._compressed_keys[0] is not None else None}"
    )

    # Check word overlap
    baseline_words = set(baseline.split()[:20])
    compressed_words = set(compressed.split()[:20])
    overlap = (
        len(baseline_words & compressed_words) / len(baseline_words)
        if baseline_words
        else 0
    )
    print(f"\nWord overlap (first 20): {overlap * 100:.1f}%")

    # Check repetition
    words = compressed.split()[:30]
    repetitions = sum(
        1 for i in range(len(words) - 2) if len(set(words[i : i + 3])) == 1
    )
    print(f"Repetition score (3-word repeats in first 30): {repetitions}")


if __name__ == "__main__":
    main()
