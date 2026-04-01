#!/usr/bin/env python3
"""
Multi-model benchmark: Test TurboQuant on various LLMs.
Tests quality and throughput across different model architectures.
"""

import torch
import time
import sys
import argparse
from typing import Dict, Tuple, Optional

sys.path.insert(0, "/home/taleef/projects/turboquant")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.turboquant_cache_v2 import TurboQuantCacheV2


# Model configurations
MODELS = {
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "head_dim": 128,
        "num_layers": 28,
        "num_heads": 28,
    },
    "llama3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "head_dim": 128,
        "num_layers": 32,
        "num_heads": 32,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "head_dim": 128,
        "num_layers": 32,
        "num_heads": 32,
    },
    "gemma2-9b": {
        "name": "google/gemma-2-9b-it",
        "head_dim": 256,  # Gemma uses larger head dim
        "num_layers": 42,
        "num_heads": 16,
    },
}


def load_model(model_key: str):
    """Load a model with 4-bit quantization."""
    config = MODELS[model_key]
    print(f"Loading {config['name']}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer, config


def test_quality(
    model, tokenizer, head_dim: int, prompt_tokens: int = 500, gen_tokens: int = 100
) -> Dict[str, str]:
    """Test generation quality with baseline vs TurboQuant."""
    device = torch.device("cuda")

    # Create prompt
    base_text = "The quick brown fox jumps over the lazy dog. " * 200
    tokens = tokenizer.encode(base_text)[:prompt_tokens]
    prompt = tokenizer.decode(tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    results = {}

    # Baseline
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=gen_tokens, do_sample=False)
    results["baseline"] = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # TurboQuant 6-bit
    torch.cuda.empty_cache()
    cache = TurboQuantCacheV2(head_dim=head_dim, bits=6, buffer_size=128)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=gen_tokens, do_sample=False, past_key_values=cache
        )
    results["turboquant_6bit"] = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return results


def benchmark_throughput(
    model, tokenizer, head_dim: int, prompt_tokens: int, gen_tokens: int = 200
) -> Dict[str, float]:
    """Benchmark throughput for baseline vs TurboQuant."""
    device = torch.device("cuda")

    # Create prompt
    base_text = "The quick brown fox jumps over the lazy dog. " * 200
    tokens = tokenizer.encode(base_text)[:prompt_tokens]
    input_ids = torch.tensor([tokens], device=device)

    results = {}

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    # Baseline
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=gen_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = out.shape[1] - input_ids.shape[1]
    results["baseline"] = generated / elapsed

    # TurboQuant 6-bit (warmup)
    for _ in range(2):
        cache = TurboQuantCacheV2(head_dim=head_dim, bits=6, buffer_size=128)
        with torch.no_grad():
            _ = model.generate(
                input_ids, max_new_tokens=10, do_sample=False, past_key_values=cache
            )

    # TurboQuant 6-bit (benchmark)
    torch.cuda.empty_cache()
    cache = TurboQuantCacheV2(head_dim=head_dim, bits=6, buffer_size=128)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=gen_tokens, do_sample=False, past_key_values=cache
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = out.shape[1] - input_ids.shape[1]
    results["turboquant_6bit"] = generated / elapsed

    return results


def run_model_tests(model_key: str):
    """Run full test suite on a model."""
    try:
        model, tokenizer, config = load_model(model_key)
    except Exception as e:
        print(f"Failed to load {model_key}: {e}")
        return None

    head_dim = config["head_dim"]

    print(f"\n{'=' * 70}")
    print(f"Testing: {config['name']}")
    print(
        f"Head dim: {head_dim}, Layers: {config['num_layers']}, Heads: {config['num_heads']}"
    )
    print(f"{'=' * 70}")

    # Quality test
    print("\n--- Quality Test (500 token prompt) ---")
    quality = test_quality(model, tokenizer, head_dim)

    baseline_preview = quality["baseline"][:100]
    tq_preview = quality["turboquant_6bit"][:100]

    print(f"Baseline:    {baseline_preview}...")
    print(f"TurboQ-6b:   {tq_preview}...")

    quality_match = quality["baseline"][:100] == quality["turboquant_6bit"][:100]
    print(f"Quality Match (first 100 chars): {'✅ YES' if quality_match else '❌ NO'}")

    # Throughput tests
    print("\n--- Throughput Test ---")
    results = {}

    for prompt_len in [100, 500, 1000]:
        try:
            throughput = benchmark_throughput(model, tokenizer, head_dim, prompt_len)
            pct = throughput["turboquant_6bit"] / throughput["baseline"] * 100
            results[prompt_len] = {
                "baseline": throughput["baseline"],
                "turboquant": throughput["turboquant_6bit"],
                "percent": pct,
            }
            print(
                f"Prompt {prompt_len:4d} tok: Baseline {throughput['baseline']:.1f} tok/s, "
                f"TurboQ {throughput['turboquant_6bit']:.1f} tok/s ({pct:.1f}%)"
            )
        except Exception as e:
            print(f"Prompt {prompt_len:4d} tok: ERROR - {e}")
            results[prompt_len] = None

    return {
        "model": config["name"],
        "head_dim": head_dim,
        "quality_match": quality_match,
        "throughput": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-model TurboQuant benchmark")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) + ["all"],
        default="qwen2.5-7b",
        help="Model to test (or 'all' for all models)",
    )
    args = parser.parse_args()

    if args.model == "all":
        models_to_test = list(MODELS.keys())
    else:
        models_to_test = [args.model]

    all_results = {}

    for model_key in models_to_test:
        result = run_model_tests(model_key)
        if result:
            all_results[model_key] = result

        # Clear memory between models
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<20} {'Quality':<10} {'500tok %':<12} {'1000tok %':<12}")
    print("-" * 54)

    for model_key, result in all_results.items():
        quality = "✅" if result["quality_match"] else "❌"

        pct_500 = "N/A"
        pct_1000 = "N/A"

        if result["throughput"].get(500):
            pct_500 = f"{result['throughput'][500]['percent']:.1f}%"
        if result["throughput"].get(1000):
            pct_1000 = f"{result['throughput'][1000]['percent']:.1f}%"

        print(f"{model_key:<20} {quality:<10} {pct_500:<12} {pct_1000:<12}")


if __name__ == "__main__":
    main()
