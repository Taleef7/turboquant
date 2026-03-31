"""
TurboQuant KV Cache Benchmark — Compressed KV with TurboQuantCache.

Loads Qwen2.5-7B-Instruct in 4-bit, injects TurboQuantCache, generates
4,000 tokens, and reports the same metrics as run_baseline.py for comparison.

Usage:
  python scripts/run_turboquant.py

Hardware requirement: NVIDIA GPU with ≥8GB VRAM
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.turboquant_cache import TurboQuantCache

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TARGET_NEW_TOKENS = 200
PROMPT_TOKENS = 3800
HEAD_DIM = 128       # Qwen2.5-7B head dimension
N_QJL = 128          # QJL projections


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


def run_turboquant():
    print(f"Loading {MODEL_ID} in 4-bit with TurboQuant KV cache...")
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

    # Initialize TurboQuantCache
    tq_cache = TurboQuantCache(
        head_dim=HEAD_DIM,
        n_qjl=N_QJL,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    # Warm-up
    tq_cache_warmup = TurboQuantCache(
        head_dim=HEAD_DIM, n_qjl=N_QJL, device=torch.device("cuda"), dtype=torch.float16
    )
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                           past_key_values=tq_cache_warmup)

    torch.cuda.empty_cache()

    # Timed run
    with profile_memory() as prof:
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=TARGET_NEW_TOKENS,
                do_sample=False,
                past_key_values=tq_cache,
                return_dict_in_generate=True,
            )
        t_total = time.perf_counter() - t0

    n_generated = outputs.sequences.shape[1] - input_len
    tps = n_generated / t_total
    ttft_ms = (t_total / n_generated) * 1000
    compressed_kb = tq_cache.compressed_size_bytes() / 1024

    print("\n=== TURBOQUANT RESULTS ===")
    print(f"Peak VRAM:            {prof.peak_mb:.1f} MB")
    print(f"VRAM delta (KV):      {(prof.peak_mb - mem_after_load):.1f} MB")
    print(f"Compressed KV size:   {compressed_kb:.1f} KB")
    print(f"Tokens generated:     {n_generated}")
    print(f"Total time:           {t_total:.2f}s")
    print(f"Approx TTFT:          {ttft_ms:.1f} ms/token")
    print(f"Throughput (TPS):     {tps:.2f} tok/s")
    print("==========================")

    return {"peak_vram_mb": prof.peak_mb, "tps": tps, "compressed_kb": compressed_kb}


if __name__ == "__main__":
    run_turboquant()
