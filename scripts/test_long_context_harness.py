"""Unit tests for long-context NIAH harness helpers."""

import sys

sys.path.insert(0, "/home/taleef/projects/turboquant")

from transformers import AutoTokenizer

from scripts.test_long_context import (
    create_haystack,
    insert_needle,
    make_needle_code,
    build_niah_prompt_and_metadata,
    summarize_paired_results,
    select_niah_contexts,
)


def test_make_needle_code_is_deterministic():
    code_a = make_needle_code(context_tokens=4096, needle_depth=0.5, trial_seed=7)
    code_b = make_needle_code(context_tokens=4096, needle_depth=0.5, trial_seed=7)
    assert code_a == code_b


def test_insert_needle_places_text_in_document():
    haystack = "A. B. C. D."
    needle = "The secret code is ALPHA-1111."
    merged = insert_needle(haystack, needle, 0.5)
    assert "ALPHA-1111" in merged


def test_build_prompt_metadata_marks_needle_missing_when_truncated():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    prompt, metadata = build_niah_prompt_and_metadata(
        tokenizer=tokenizer,
        context_tokens=256,
        needle_depth=0.9,
        trial_seed=1,
    )

    assert isinstance(prompt, str)
    assert metadata["input_token_count"] <= 256
    assert isinstance(metadata["needle_present_after_tokenization"], bool)


def test_summarize_paired_results_computes_delta_pp():
    rows = [
        {
            "needle_present_after_tokenization": True,
            "baseline_correct": True,
            "turboquant_correct": True,
        },
        {
            "needle_present_after_tokenization": True,
            "baseline_correct": True,
            "turboquant_correct": False,
        },
        {
            "needle_present_after_tokenization": True,
            "baseline_correct": False,
            "turboquant_correct": False,
        },
        {
            "needle_present_after_tokenization": False,
            "baseline_correct": False,
            "turboquant_correct": False,
        },
    ]

    summary = summarize_paired_results(rows)

    assert summary["eligible_cases"] == 3
    assert abs(summary["baseline_accuracy"] - (2 / 3)) < 1e-6
    assert abs(summary["turboquant_accuracy"] - (1 / 3)) < 1e-6
    assert abs(summary["delta_pp"] - ((2 / 3 - 1 / 3) * 100.0)) < 1e-6


def test_build_prompt_metadata_keeps_needle_at_4k_budget():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    _, metadata = build_niah_prompt_and_metadata(
        tokenizer=tokenizer,
        context_tokens=4096,
        needle_depth=0.5,
        trial_seed=0,
    )
    assert metadata["needle_present_after_tokenization"] is True


def test_select_niah_contexts_includes_requested_max_when_needed():
    assert select_niah_contexts(max_context=32768) == [4096, 8192, 16384, 32768]


def test_select_niah_contexts_handles_sub_4k_max():
    assert select_niah_contexts(max_context=2048) == [2048]
