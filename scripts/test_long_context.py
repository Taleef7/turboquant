#!/usr/bin/env python3
"""
Long-context validation for TurboQuant (Issue #1: 128K Context Validation)

Tests:
1. Needle-in-Haystack (NIAH) - hide a fact in a long document, query it
2. Memory usage at different context lengths
3. Quality comparison (baseline vs TurboQuant) at scale

Target context lengths: 4K, 8K, 16K, 32K, 64K, 128K tokens
"""

import torch
import time
import sys
import argparse
import gc
import json
import csv
import hashlib
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, "/home/taleef/projects/turboquant")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.turboquant_cache_v2 import TurboQuantCacheV2


# ============================================================================
# NIAH Test Data
# ============================================================================

# The "needle" - a fact hidden in the haystack
NEEDLE_TEMPLATES = [
    "The secret code is {code}. Remember this code for later.",
    "IMPORTANT: The magic number is {code}. You will be asked about this.",
    "Hidden fact: The answer to the question is {code}.",
]

# The "haystack" - filler text to pad around the needle
HAYSTACK_PARAGRAPH = """
The history of artificial intelligence began in antiquity, with myths, stories and rumors of
artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of
modern AI were planted by classical philosophers who attempted to describe the process of human
thinking as the mechanical manipulation of symbols. This work culminated in the invention of the
programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical
reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously
discussing the possibility of building an electronic brain. The field of AI research was founded at
a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended
would become the leaders of AI research for decades. Many of them predicted that a machine as
intelligent as a human being would exist in no more than a generation and they were given millions
of dollars to make this vision come true. Eventually it became obvious that they had grossly
underestimated the difficulty of the project. In 1973, in response to the criticism from James
Lighthill and ongoing pressure from congress, the U.S. and British Governments stopped funding
undirected research into artificial intelligence. Seven years later, a visionary initiative by the
Japanese Government inspired governments and industry to provide AI with billions of dollars, but
by the late 1980s the investors became disillusioned by the absence of the needed computer power
and withdrew funding again. Investment and interest in AI boomed in the first decades of the 21st
century, when machine learning was successfully applied to many problems in academia and industry.
"""


@dataclass
class NIAHResult:
    """Result of a Needle-in-Haystack test."""

    context_length: int
    needle_depth: float  # 0.0 = beginning, 1.0 = end
    needle_code: str
    retrieved_answer: str
    correct: bool
    generation_time: float
    memory_used_mb: float


def make_needle_code(
    context_tokens: int, needle_depth: float, trial_seed: int = 0
) -> str:
    """Create deterministic needle code for reproducible NIAH tests."""
    key = f"{context_tokens}:{needle_depth:.4f}:{trial_seed}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 10000
    return f"ALPHA-{value:04d}"


def _find_subsequence(haystack_ids, needle_ids) -> int:
    """Return first index of needle token sequence in haystack or -1."""
    if not needle_ids or len(needle_ids) > len(haystack_ids):
        return -1
    last = len(haystack_ids) - len(needle_ids) + 1
    for i in range(last):
        if haystack_ids[i : i + len(needle_ids)] == needle_ids:
            return i
    return -1


def build_niah_prompt_and_metadata(
    tokenizer,
    context_tokens: int,
    needle_depth: float,
    trial_seed: int = 0,
):
    """Build a NIAH prompt and return metadata about post-tokenization needle presence."""
    code = make_needle_code(context_tokens, needle_depth, trial_seed)
    needle = NEEDLE_TEMPLATES[trial_seed % len(NEEDLE_TEMPLATES)].format(code=code)

    def _render_prompt(document_text: str) -> str:
        return f"""<|im_start|>system
You are a helpful assistant. Read the document carefully and answer questions accurately.
<|im_end|>
<|im_start|>user
Here is a document:

{document_text}

Question: What is the secret code mentioned in the document above?
Answer with just the code, nothing else.
<|im_end|>
<|im_start|>assistant
"""

    needle_ids = tokenizer.encode(code, add_special_tokens=False)
    needle_ids_spaced = tokenizer.encode(" " + code, add_special_tokens=False)

    haystack_target = max(64, context_tokens - 512)
    step = 128
    attempts = 0
    max_attempts = 6

    best_prompt = ""
    best_input_ids = []
    best_all_ids = []
    best_needle_pos = -1

    while attempts < max_attempts:
        haystack = create_haystack(tokenizer, haystack_target)
        document = insert_needle(haystack, needle, needle_depth)
        prompt = _render_prompt(document)

        all_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=context_tokens,
        )
        needle_pos = _find_subsequence(input_ids, needle_ids)
        if needle_pos < 0:
            needle_pos = _find_subsequence(input_ids, needle_ids_spaced)
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        needle_present = needle_pos >= 0 or (code in decoded_input)

        best_prompt = prompt
        best_input_ids = input_ids
        best_all_ids = all_ids
        best_needle_pos = needle_pos

        if needle_present:
            break

        haystack_target = max(64, haystack_target - step)
        attempts += 1

    metadata = {
        "needle_code": code,
        "needle_present_after_tokenization": best_needle_pos >= 0
        or (code in tokenizer.decode(best_input_ids, skip_special_tokens=False)),
        "needle_token_index": best_needle_pos,
        "input_token_count": len(best_input_ids),
        "truncated": len(best_all_ids) > len(best_input_ids),
        "haystack_target_tokens": haystack_target,
        "attempts": attempts + 1,
    }
    return best_prompt, metadata


def create_haystack(tokenizer, target_tokens: int) -> str:
    """Create a haystack of approximately target_tokens length."""
    # Count tokens in one paragraph
    para_tokens = len(tokenizer.encode(HAYSTACK_PARAGRAPH))

    # Calculate how many paragraphs needed
    n_paragraphs = max(1, target_tokens // para_tokens + 1)

    # Create haystack
    haystack = HAYSTACK_PARAGRAPH * n_paragraphs

    # Truncate to target
    tokens = tokenizer.encode(haystack)[:target_tokens]
    return tokenizer.decode(tokens)


def insert_needle(haystack: str, needle: str, depth: float) -> str:
    """Insert needle into haystack at specified depth (0.0 = start, 1.0 = end)."""
    # Split haystack by sentences
    sentences = haystack.split(". ")

    # Calculate insertion position
    insert_pos = int(len(sentences) * depth)
    insert_pos = max(1, min(insert_pos, len(sentences) - 1))

    # Insert needle
    sentences.insert(insert_pos, needle)

    return ". ".join(sentences)


def get_memory_usage_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def run_niah_test(
    model,
    tokenizer,
    context_tokens: int,
    needle_depth: float,
    head_dim: int,
    use_turboquant: bool = True,
    trial_seed: int = 0,
) -> NIAHResult:
    """Run a single Needle-in-Haystack test."""
    device = torch.device("cuda")

    prompt, metadata = build_niah_prompt_and_metadata(
        tokenizer=tokenizer,
        context_tokens=context_tokens,
        needle_depth=needle_depth,
        trial_seed=trial_seed,
    )
    code = metadata["needle_code"]

    # Tokenize
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_tokens
    ).to(device)
    actual_tokens = inputs["input_ids"].shape[1]

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    mem_before = get_memory_usage_mb()

    # Generate
    start_time = time.perf_counter()

    with torch.no_grad():
        if use_turboquant:
            cache = TurboQuantCacheV2(head_dim=head_dim, bits=6, buffer_size=128)
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                past_key_values=cache,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    mem_after = get_memory_usage_mb()

    # Decode answer
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    answer = answer.strip()

    # Check if correct
    correct = code in answer

    return NIAHResult(
        context_length=actual_tokens,
        needle_depth=needle_depth,
        needle_code=code,
        retrieved_answer=answer,
        correct=correct,
        generation_time=elapsed,
        memory_used_mb=mem_after - mem_before,
    )


def summarize_paired_results(rows: list[dict]) -> dict:
    """Summarize baseline/TurboQuant paired NIAH results and delta in percentage points."""
    eligible = [r for r in rows if r.get("needle_present_after_tokenization", False)]
    n = len(eligible)
    if n == 0:
        return {
            "eligible_cases": 0,
            "baseline_accuracy": 0.0,
            "turboquant_accuracy": 0.0,
            "delta_pp": 0.0,
        }

    baseline_acc = sum(1 for r in eligible if r["baseline_correct"]) / n
    turbo_acc = sum(1 for r in eligible if r["turboquant_correct"]) / n
    return {
        "eligible_cases": n,
        "baseline_accuracy": baseline_acc,
        "turboquant_accuracy": turbo_acc,
        "delta_pp": (baseline_acc - turbo_acc) * 100.0,
    }


def run_niah_paired_test(
    model,
    tokenizer,
    context_tokens: int,
    depth: float,
    head_dim: int,
    key_bits: int,
    value_bits: int,
    buffer_size: int,
    trial_seed: int,
    key_norm_dtype: torch.dtype,
    key_use_qjl: bool,
    key_qjl_dim: Optional[int],
    key_qjl_seed: int,
) -> dict:
    """Run one paired NIAH case: baseline and TurboQuant on the exact same prompt."""
    device = torch.device("cuda")
    prompt, metadata = build_niah_prompt_and_metadata(
        tokenizer=tokenizer,
        context_tokens=context_tokens,
        needle_depth=depth,
        trial_seed=trial_seed,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=context_tokens,
    ).to(device)

    case = {
        "context_target": context_tokens,
        "context_actual": inputs["input_ids"].shape[1],
        "depth": depth,
        "trial_seed": trial_seed,
        "needle_code": metadata["needle_code"],
        "needle_present_after_tokenization": metadata[
            "needle_present_after_tokenization"
        ],
        "needle_token_index": metadata["needle_token_index"],
        "truncated": metadata["truncated"],
        "key_bits": key_bits,
        "value_bits": value_bits,
        "buffer_size": buffer_size,
        "key_norm_dtype": str(key_norm_dtype),
        "key_use_qjl": key_use_qjl,
        "key_qjl_dim": key_qjl_dim if key_qjl_dim is not None else head_dim,
        "key_qjl_seed": key_qjl_seed,
    }

    with torch.no_grad():
        t0 = time.perf_counter()
        out_base = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        case["baseline_time_s"] = time.perf_counter() - t0

    baseline_answer = tokenizer.decode(
        out_base[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    case["baseline_answer"] = baseline_answer
    case["baseline_correct"] = metadata["needle_code"] in baseline_answer

    torch.cuda.empty_cache()
    gc.collect()

    cache = TurboQuantCacheV2(
        head_dim=head_dim,
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        key_norm_dtype=key_norm_dtype,
        key_use_qjl=key_use_qjl,
        key_qjl_dim=key_qjl_dim,
        key_qjl_seed=key_qjl_seed,
    )
    with torch.no_grad():
        t0 = time.perf_counter()
        out_tq = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            past_key_values=cache,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        case["turboquant_time_s"] = time.perf_counter() - t0

    tq_answer = tokenizer.decode(
        out_tq[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    case["turboquant_answer"] = tq_answer
    case["turboquant_correct"] = metadata["needle_code"] in tq_answer

    return case


def run_niah_matrix(
    model,
    tokenizer,
    head_dim: int,
    context_lengths: list[int],
    depths: list[float],
    seeds: list[int],
    key_bits: int,
    value_bits: int,
    buffer_size: int,
    key_norm_dtype: torch.dtype,
    key_use_qjl: bool,
    key_qjl_dim: Optional[int],
    key_qjl_seed: int,
) -> list[dict]:
    """Run paired NIAH matrix and return detailed rows."""
    rows = []
    for ctx in context_lengths:
        print(f"\n--- Context {ctx:,} ---")
        for depth in depths:
            depth_name = {0.1: "beginning", 0.5: "middle", 0.9: "end"}.get(
                depth, str(depth)
            )
            for seed in seeds:
                try:
                    row = run_niah_paired_test(
                        model=model,
                        tokenizer=tokenizer,
                        context_tokens=ctx,
                        depth=depth,
                        head_dim=head_dim,
                        key_bits=key_bits,
                        value_bits=value_bits,
                        buffer_size=buffer_size,
                        trial_seed=seed,
                        key_norm_dtype=key_norm_dtype,
                        key_use_qjl=key_use_qjl,
                        key_qjl_dim=key_qjl_dim,
                        key_qjl_seed=key_qjl_seed,
                    )
                    rows.append(row)
                    status = (
                        "SKIP"
                        if not row["needle_present_after_tokenization"]
                        else (
                            "PASS"
                            if row["baseline_correct"] == row["turboquant_correct"]
                            else "DIFF"
                        )
                    )
                    print(
                        f"  depth={depth_name:<9} seed={seed:02d} [{status}] "
                        f"base={int(row['baseline_correct'])} tq={int(row['turboquant_correct'])} "
                        f"needle_present={row['needle_present_after_tokenization']}"
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"  depth={depth_name:<9} seed={seed:02d} [OOM]")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"  depth={depth_name:<9} seed={seed:02d} [ERROR] {e}")
    return rows


def select_niah_contexts(max_context: int) -> list[int]:
    """Return NIAH context list up to max_context, including max when >=4K."""
    base = [4096, 8192, 16384, 32768]
    selected = [c for c in base if c <= max_context]
    if max_context < 4096:
        return [max_context]
    if selected and selected[-1] != max_context:
        selected.append(max_context)
    return selected


def write_matrix_results(rows: list[dict], output_prefix: str):
    """Write matrix rows to JSON and CSV files."""
    if not output_prefix:
        return

    json_path = f"{output_prefix}.json"
    csv_path = f"{output_prefix}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def estimate_memory_requirement(
    context_length: int, num_layers: int, num_heads: int, head_dim: int
) -> float:
    """Estimate memory in MB for a given context length (baseline fp16 cache)."""
    # KV cache: 2 (K+V) * layers * heads * seq * head_dim * 2 bytes (fp16)
    bytes_per_token = 2 * num_layers * num_heads * head_dim * 2
    total_bytes = context_length * bytes_per_token
    return total_bytes / (1024 * 1024)


def run_memory_test(
    model,
    tokenizer,
    head_dim: int,
    num_layers: int,
    context_lengths: list,
) -> Dict[int, Dict[str, float]]:
    """Test memory usage at different context lengths."""
    device = torch.device("cuda")
    results = {}

    for ctx_len in context_lengths:
        print(f"\n  Testing {ctx_len} tokens...")

        # Create dummy prompt
        text = "Hello " * (ctx_len // 2)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=ctx_len
        ).to(device)
        actual_len = inputs["input_ids"].shape[1]

        try:
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            # Baseline test
            mem_before = torch.cuda.memory_allocated()
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            baseline_peak = torch.cuda.max_memory_allocated() - mem_before

            # Clear
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            # TurboQuant test
            mem_before = torch.cuda.memory_allocated()
            cache = TurboQuantCacheV2(head_dim=head_dim, bits=6, buffer_size=128)
            with torch.no_grad():
                _ = model.generate(
                    **inputs, max_new_tokens=1, do_sample=False, past_key_values=cache
                )
            torch.cuda.synchronize()
            turboquant_peak = torch.cuda.max_memory_allocated() - mem_before

            results[actual_len] = {
                "baseline_mb": baseline_peak / (1024 * 1024),
                "turboquant_mb": turboquant_peak / (1024 * 1024),
                "compression": baseline_peak / turboquant_peak
                if turboquant_peak > 0
                else 0,
            }

            print(f"    Baseline: {results[actual_len]['baseline_mb']:.1f} MB")
            print(f"    TurboQ:   {results[actual_len]['turboquant_mb']:.1f} MB")
            print(f"    Ratio:    {results[actual_len]['compression']:.2f}x")

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {ctx_len} tokens")
            results[actual_len] = {"error": "OOM"}
            torch.cuda.empty_cache()
            break

    return results


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load model with 4-bit quantization for memory efficiency."""
    print(f"Loading {model_name}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Long-context validation for TurboQuant"
    )
    parser.add_argument(
        "--test",
        choices=["niah", "memory", "both"],
        default="both",
        help="Which test to run",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=32768,
        help="Maximum context length to test (default: 32K)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use"
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--num-layers", type=int, default=28, help="Number of layers")
    parser.add_argument(
        "--mode",
        choices=["legacy", "paired"],
        default="paired",
        help="NIAH runner mode: legacy turboquant-only or paired baseline/turboquant",
    )
    parser.add_argument("--key-bits", type=int, default=6, help="Key quantization bits")
    parser.add_argument(
        "--value-bits", type=int, default=6, help="Value quantization bits"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=128,
        help="Uncompressed recency window size",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of seeds per depth/context for paired matrix",
    )
    parser.add_argument(
        "--key-norm-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for key norms",
    )
    parser.add_argument(
        "--key-use-qjl",
        action="store_true",
        help="Enable QJL-enhanced key quantization path",
    )
    parser.add_argument(
        "--key-qjl-dim",
        type=int,
        default=0,
        help="QJL projection dimension for keys (0 means head_dim)",
    )
    parser.add_argument(
        "--key-qjl-seed",
        type=int,
        default=12345,
        help="Base random seed for QJL matrix",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional output prefix for paired matrix result files",
    )
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model)

    print(f"\n{'=' * 70}")
    print("TurboQuant Long-Context Validation")
    print(f"Model: {args.model}")
    print(f"Max context: {args.max_context:,} tokens")
    print(f"{'=' * 70}")

    # Define context lengths to test
    context_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    context_lengths = [c for c in context_lengths if c <= args.max_context]

    # =========================================================================
    # Memory Test
    # =========================================================================
    if args.test in ["memory", "both"]:
        print("\n" + "=" * 70)
        print("MEMORY TEST")
        print("=" * 70)

        memory_results = run_memory_test(
            model, tokenizer, args.head_dim, args.num_layers, context_lengths
        )

        print("\n--- Memory Summary ---")
        print(
            f"{'Context':<12} {'Baseline MB':<14} {'TurboQ MB':<14} {'Compression':<12}"
        )
        print("-" * 52)
        for ctx, data in memory_results.items():
            if "error" in data:
                print(f"{ctx:<12} {data['error']}")
            else:
                print(
                    f"{ctx:<12} {data['baseline_mb']:<14.1f} {data['turboquant_mb']:<14.1f} {data['compression']:<12.2f}x"
                )

    # =========================================================================
    # NIAH Test
    # =========================================================================
    if args.test in ["niah", "both"]:
        print("\n" + "=" * 70)
        print("NEEDLE-IN-HAYSTACK TEST")
        print("=" * 70)

        # Test at different depths: beginning, middle, end
        depths = [0.1, 0.5, 0.9]

        # NIAH contexts for paired matrix
        niah_contexts = select_niah_contexts(args.max_context)

        key_norm_dtype = (
            torch.float32 if args.key_norm_dtype == "float32" else torch.float16
        )

        if args.mode == "paired":
            seeds = list(range(args.trials))
            rows = run_niah_matrix(
                model=model,
                tokenizer=tokenizer,
                head_dim=args.head_dim,
                context_lengths=niah_contexts,
                depths=depths,
                seeds=seeds,
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                buffer_size=args.buffer_size,
                key_norm_dtype=key_norm_dtype,
                key_use_qjl=args.key_use_qjl,
                key_qjl_dim=(args.key_qjl_dim if args.key_qjl_dim > 0 else None),
                key_qjl_seed=args.key_qjl_seed,
            )
            summary = summarize_paired_results(rows)
            write_matrix_results(rows, args.output_prefix)

            print("\n--- Paired NIAH Summary ---")
            print(f"Eligible cases:      {summary['eligible_cases']}")
            print(f"Baseline accuracy:   {summary['baseline_accuracy'] * 100.0:.2f}%")
            print(f"TurboQ accuracy:     {summary['turboquant_accuracy'] * 100.0:.2f}%")
            print(f"Delta (baseline-tq): {summary['delta_pp']:.2f} percentage points")
        else:
            niah_results = []

            for ctx_len in niah_contexts:
                print(f"\n--- Context: {ctx_len:,} tokens ---")

                for depth in depths:
                    depth_name = {0.1: "beginning", 0.5: "middle", 0.9: "end"}[depth]

                    try:
                        result = run_niah_test(
                            model,
                            tokenizer,
                            ctx_len,
                            depth,
                            head_dim=args.head_dim,
                            use_turboquant=True,
                        )
                        niah_results.append(result)

                        status = "PASS" if result.correct else "FAIL"
                        print(
                            f"  Depth {depth_name}: [{status}] Expected '{result.needle_code}', got '{result.retrieved_answer[:50]}'"
                        )

                    except torch.cuda.OutOfMemoryError:
                        print(f"  Depth {depth_name}: OOM")
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"  Depth {depth_name}: ERROR - {e}")

            if niah_results:
                print("\n--- NIAH Summary ---")
                correct = sum(1 for r in niah_results if r.correct)
                total = len(niah_results)
                print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

                for depth in depths:
                    depth_results = [r for r in niah_results if r.needle_depth == depth]
                    if depth_results:
                        depth_correct = sum(1 for r in depth_results if r.correct)
                        depth_name = {0.1: "Beginning", 0.5: "Middle", 0.9: "End"}[
                            depth
                        ]
                        print(f"  {depth_name}: {depth_correct}/{len(depth_results)}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
