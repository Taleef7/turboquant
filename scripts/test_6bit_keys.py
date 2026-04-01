#!/usr/bin/env python3
"""Test 6-bit keys with 500 token prompt."""

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

    # 500-token prompt
    filler = "The quick brown fox jumps over the lazy dog. " * 500
    tokens = tokenizer.encode(filler)[:500]
    prompt = tokenizer.decode(tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

    # Baseline
    print("\n=== BASELINE ===")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    baseline = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"Baseline: {baseline[:150]}...")

    for bits in [4, 6, 8]:
        torch.cuda.empty_cache()
        print(f"\n=== {bits}-BIT KEYS (buffer=128) ===")
        cache = TurboQuantCacheV2(head_dim=128, bits=bits, buffer_size=128)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=100, do_sample=False, past_key_values=cache
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(f"Generated: {generated[:150]}...")

        # Check if matches baseline
        matches = baseline[:100] == generated[:100]
        print(f"Matches baseline (first 100 chars): {matches}")


if __name__ == "__main__":
    main()
