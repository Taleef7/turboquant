#!/usr/bin/env python3
"""Test 6-bit key compression for generation quality."""

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

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Baseline
    print("\n=== BASELINE ===")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # 4-bit keys (current - broken)
    print("\n=== 4-BIT KEYS (buffer=128) ===")
    cache_4bit = TurboQuantCacheV2(head_dim=128, bits=4, buffer_size=128)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, past_key_values=cache_4bit
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # 6-bit keys
    print("\n=== 6-BIT KEYS (buffer=128) ===")
    cache_6bit = TurboQuantCacheV2(head_dim=128, bits=6, buffer_size=128)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, past_key_values=cache_6bit
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # 8-bit keys
    print("\n=== 8-BIT KEYS (buffer=128) ===")
    cache_8bit = TurboQuantCacheV2(head_dim=128, bits=8, buffer_size=128)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, past_key_values=cache_8bit
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # Large buffer (no compression for short prompts)
    print("\n=== NO COMPRESSION (buffer=1024) ===")
    cache_nobuf = TurboQuantCacheV2(head_dim=128, bits=4, buffer_size=1024)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, past_key_values=cache_nobuf
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
