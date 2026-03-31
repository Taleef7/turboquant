"""
Needle-in-a-Haystack (NIAH) Evaluation for TurboQuant.

Tests whether TurboQuant's KV cache compression preserves long-context
retrieval accuracy. Inserts a known "needle" fact into an 8,000-token
"haystack" at depths 10%, 50%, and 90%, then checks if the model can
retrieve it.

The needle: "The secret password is 'TURBO_5070_TI'"

Usage:
  python scripts/eval_niah.py

Hardware requirement: NVIDIA GPU with ≥8GB VRAM
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from core.turboquant_cache import TurboQuantCache

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NEEDLE = "The secret password is 'TURBO_5070_TI'"
HAYSTACK_TOKENS = 8000
NEEDLE_DEPTHS = [0.10, 0.50, 0.90]
HEAD_DIM = 128
N_QJL = 128

HAYSTACK_TEXT = """
In the beginning, many scientists believed that the universe was static and eternal.
However, observations by Edwin Hubble in the 1920s revealed that galaxies are moving
away from each other, suggesting the universe is expanding. This discovery led to the
development of the Big Bang theory, which proposes that the universe began as an
extremely hot and dense point approximately 13.8 billion years ago.

The study of quantum mechanics transformed physics in the 20th century. Developed by
physicists like Niels Bohr, Werner Heisenberg, and Erwin Schrödinger, quantum mechanics
describes the behavior of particles at atomic and subatomic scales. The theory introduces
concepts such as wave-particle duality, the uncertainty principle, and quantum entanglement.

Machine learning and artificial intelligence have transformed numerous fields in recent
decades. Neural networks, inspired by the structure of the human brain, can learn complex
patterns from data. Deep learning architectures with many layers have achieved remarkable
results in image recognition, natural language processing, and game playing.

The human genome contains approximately 3 billion base pairs of DNA, encoding around
20,000-25,000 genes. Advances in genomics and biotechnology have enabled scientists to
sequence entire genomes rapidly and at low cost. This has opened new avenues for
understanding genetic diseases and developing personalized medicine.
""" * 100  # Repeat to get enough tokens


def build_haystack_with_needle(tokenizer, depth: float) -> str:
    """Insert needle at given fractional depth into the haystack."""
    haystack_tokens = tokenizer.encode(HAYSTACK_TEXT)[:HAYSTACK_TOKENS]
    needle_tokens = tokenizer.encode(f"\n{NEEDLE}\n")

    insert_pos = int(len(haystack_tokens) * depth)
    combined = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
    combined = combined[:HAYSTACK_TOKENS + len(needle_tokens)]
    return tokenizer.decode(combined)


def build_prompt(haystack: str) -> str:
    return (
        f"{haystack}\n\n"
        "Based on the text above, what is the secret password? "
        "Answer with just the password value, nothing else."
    )


def check_answer(response: str) -> bool:
    """Check if the needle value appears in the response."""
    return "TURBO_5070_TI" in response


def run_niah(use_turboquant: bool = True):
    print(f"\nLoading {MODEL_ID} ({'TurboQuant' if use_turboquant else 'Baseline'} cache)...")
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

    results = {}
    for depth in NEEDLE_DEPTHS:
        print(f"\n--- Testing needle at depth {depth*100:.0f}% ---")
        haystack = build_haystack_with_needle(tokenizer, depth)
        prompt = build_prompt(haystack)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        n_tokens = inputs["input_ids"].shape[1]
        print(f"  Context length: {n_tokens} tokens")

        kwargs = dict(
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        if use_turboquant:
            kwargs["past_key_values"] = TurboQuantCache(
                head_dim=HEAD_DIM, n_qjl=N_QJL,
                device=torch.device("cuda"), dtype=torch.float16
            )

        with torch.no_grad():
            outputs = model.generate(**inputs, **kwargs)

        response_ids = outputs[0][n_tokens:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        found = check_answer(response)
        results[depth] = found
        status = "✓ FOUND" if found else "✗ MISSED"
        print(f"  Response: {response[:100]!r}")
        print(f"  Result: {status}")

    mode = "TurboQuant" if use_turboquant else "Baseline"
    print(f"\n=== NIAH RESULTS ({mode}) ===")
    all_pass = True
    for depth, found in results.items():
        status = "PASS" if found else "FAIL"
        if not found:
            all_pass = False
        print(f"  Depth {depth*100:.0f}%: {status}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("================================")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Run with baseline DynamicCache instead of TurboQuantCache")
    parser.add_argument("--both", action="store_true",
                        help="Run both baseline and TurboQuant for comparison")
    args = parser.parse_args()

    if args.both:
        baseline_results = run_niah(use_turboquant=False)
        tq_results = run_niah(use_turboquant=True)
        print("\n=== COMPARISON ===")
        for depth in NEEDLE_DEPTHS:
            b = "PASS" if baseline_results[depth] else "FAIL"
            t = "PASS" if tq_results[depth] else "FAIL"
            print(f"  Depth {depth*100:.0f}%: Baseline={b}, TurboQuant={t}")
    elif args.baseline:
        run_niah(use_turboquant=False)
    else:
        run_niah(use_turboquant=True)
