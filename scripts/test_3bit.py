"""
Test 3-bit TurboQuant generation quality.

Quick test to verify that 3-bit quantization produces coherent output.
Uses Python fallback to avoid Triton kernel bug.
"""

import os

os.environ["TURBOQUANT_FORCE_PYTHON"] = "1"  # Required for 3-bit

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.turboquant_cache import TurboQuantCache

# Model setup
model_id = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)

# Test prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"\nPrompt: {prompt}")
print(f"Generating with 3-bit TurboQuant...")

# Generate with 3-bit cache
cache_3bit = TurboQuantCache(
    head_dim=128,
    bit_width=3,  # 3-bit base + 4-bit outliers
    n_outliers=32,
    device=torch.device(device),
    dtype=torch.bfloat16,
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        past_key_values=cache_3bit,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated text:")
print(generated_text)
print(f"\nCache size: {cache_3bit.get_seq_length(0)} tokens")
