"""
TurboQuant V2 Generation Test.

Tests the new TurboQuantCacheV2 implementation with Qwen2.5-7B-Instruct.

Usage:
  python scripts/run_turboquant_v2.py
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.turboquant_cache_v2 import TurboQuantCacheV2

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TARGET_NEW_TOKENS = 200
PROMPT_TOKENS = 500  # Smaller prompt for faster testing
HEAD_DIM = 128


def get_dummy_prompt(n_tokens: int, tokenizer) -> str:
    filler = "The quick brown fox jumps over the lazy dog. " * 500
    tokens = tokenizer.encode(filler)[:n_tokens]
    return tokenizer.decode(tokens)


def profile_memory():
    class MemProfiler:
        def __enter__(self):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            self.mem_before = torch.cuda.memory_allocated() / 1024**2
            return self

        def __exit__(self, *_):
            self.peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            self.current_mb = torch.cuda.memory_allocated() / 1024**2

    return MemProfiler()


def run_baseline(model, tokenizer, inputs, n_tokens):
    """Run baseline generation without compression."""
    print("\n=== BASELINE (No Compression) ===")

    input_len = inputs["input_ids"].shape[1]

    with profile_memory() as prof:
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        t_total = time.perf_counter() - t0

    n_generated = outputs.sequences.shape[1] - input_len
    tps = n_generated / t_total

    print(f"Tokens generated:     {n_generated}")
    print(f"Total time:           {t_total:.2f}s")
    print(f"Throughput (TPS):     {tps:.2f} tok/s")
    print(f"Peak VRAM:            {prof.peak_mb:.1f} MB")

    # Get generated text
    generated_text = tokenizer.decode(
        outputs.sequences[0, input_len:], skip_special_tokens=True
    )
    print(f"\nGenerated text:\n{generated_text[:500]}...")

    return tps, generated_text


def run_turboquant_v2(model, tokenizer, inputs, n_tokens, buffer_size=128):
    """Run generation with TurboQuantCacheV2."""
    print(f"\n=== TURBOQUANT V2 (buffer_size={buffer_size}) ===")

    input_len = inputs["input_ids"].shape[1]

    # Create TurboQuant cache
    tq_cache = TurboQuantCacheV2(
        head_dim=HEAD_DIM,
        bits=4,
        device=torch.device("cuda"),
        dtype=torch.float16,
        buffer_size=buffer_size,
    )

    with profile_memory() as prof:
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                past_key_values=tq_cache,
                return_dict_in_generate=True,
            )
        t_total = time.perf_counter() - t0

    n_generated = outputs.sequences.shape[1] - input_len
    tps = n_generated / t_total
    compressed_kb = tq_cache.compressed_size_bytes() / 1024

    print(f"Tokens generated:     {n_generated}")
    print(f"Total time:           {t_total:.2f}s")
    print(f"Throughput (TPS):     {tps:.2f} tok/s")
    print(f"Peak VRAM:            {prof.peak_mb:.1f} MB")
    print(f"Compressed KV size:   {compressed_kb:.1f} KB")

    # Get generated text
    generated_text = tokenizer.decode(
        outputs.sequences[0, input_len:], skip_special_tokens=True
    )
    print(f"\nGenerated text:\n{generated_text[:500]}...")

    return tps, generated_text


def main():
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

    torch.cuda.empty_cache()
    mem_after_load = torch.cuda.memory_allocated() / 1024**2
    print(f"Model loaded. VRAM used: {mem_after_load:.1f} MB")

    prompt = get_dummy_prompt(PROMPT_TOKENS, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    print(f"Prompt tokens: {input_len}")

    # Run baseline
    baseline_tps, baseline_text = run_baseline(
        model, tokenizer, inputs.copy(), TARGET_NEW_TOKENS
    )

    # Clear cache
    torch.cuda.empty_cache()

    # Run TurboQuant V2
    tq_tps, tq_text = run_turboquant_v2(
        model, tokenizer, inputs.copy(), TARGET_NEW_TOKENS
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline TPS:         {baseline_tps:.2f}")
    print(f"TurboQuant V2 TPS:    {tq_tps:.2f}")
    print(f"TPS Ratio:            {tq_tps / baseline_tps * 100:.1f}%")

    # Check text quality
    baseline_words = baseline_text.split()[:20]
    tq_words = tq_text.split()[:20]

    # Simple coherence check: are words repeating excessively?
    def has_excessive_repetition(words, threshold=3):
        if len(words) < threshold:
            return False
        for i in range(len(words) - threshold + 1):
            if len(set(words[i : i + threshold])) == 1:
                return True
        return False

    baseline_coherent = not has_excessive_repetition(baseline_words)
    tq_coherent = not has_excessive_repetition(tq_words)

    print(f"\nBaseline coherent:    {'YES' if baseline_coherent else 'NO'}")
    print(f"TurboQuant coherent:  {'YES' if tq_coherent else 'NO'}")

    if tq_coherent:
        print("\n✓ TurboQuant V2 generates coherent text!")
    else:
        print("\n✗ TurboQuant V2 has quality issues - needs debugging")

    return tq_coherent


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
