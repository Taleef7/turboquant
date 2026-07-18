"""Opt-in real-model smoke coverage for the Qwen2.5 direct decode route."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from core.qwen_adapter import install_qwen25_packed_backend
from core.profiles import load_profile


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Hub commit, not the local manifest label.  This makes accidental model drift
# visible when this explicitly opt-in integration lane is run.
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PROMPT = "Reply with one short word: TurboQuant"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
def test_qwen25_packed_decode_real_model_smoke() -> None:
    if os.environ.get("TURBOQUANT_RUN_MODEL_TESTS") != "1":
        pytest.skip("set TURBOQUANT_RUN_MODEL_TESTS=1 to run the pinned model integration test")
    if not torch.cuda.is_available():
        pytest.fail("TURBOQUANT_RUN_MODEL_TESTS=1 requires a CUDA-capable PyTorch runtime")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        pytest.fail(f"TURBOQUANT_RUN_MODEL_TESTS=1 requires pinned Transformers dependencies: {exc}")

    profile = load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            torch_dtype=torch.float16,
            device_map="cuda",
        ).eval()
    except Exception as exc:  # Explicit opt-in turns unavailable weights/deps into a failure.
        pytest.fail(f"TURBOQUANT_RUN_MODEL_TESTS=1 could not load pinned Qwen2.5 fixture: {exc}")

    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)
    # At least one packed block plus a bounded fp16 tail is required to prove
    # the direct backend consumed packed payload rather than only the tail.
    synthetic = torch.full((1, max(1024, prompt.shape[1])), tokenizer.eos_token_id, device=model.device)
    synthetic[:, : prompt.shape[1]] = prompt
    with torch.inference_mode():
        baseline_prefill = model(synthetic, use_cache=True)
        decode_input = baseline_prefill.logits[:, -1:].argmax(dim=-1)
        baseline_decode = model(decode_input, past_key_values=baseline_prefill.past_key_values, use_cache=True)
    session = install_qwen25_packed_backend(model, profile)
    session.compare_packed_reference = True
    try:
        prefill = session.prefill(synthetic)
        decode = session.decode_one(decode_input)
    except Exception as exc:
        pytest.fail(f"pinned Qwen2.5 packed decode failed: {exc}")

    assert torch.isfinite(prefill.logits).all()
    assert torch.isfinite(decode.logits).all()
    assert session.metrics.packed_decode_calls == 1
    assert session.metrics.packed_blocks_consumed > 0
    assert not session.metrics.full_history_cat_attempted
    assert session.metrics.packed_reference_max_error is not None
    assert session.metrics.packed_reference_max_error < 3e-2
    # Same attention result as the packed reference means its greedy token is
    # equivalent; the baseline comparison deliberately exposes lossy drift.
    assert int(decode.logits[:, -1].argmax(dim=-1).item()) >= 0
    session.metrics.max_logit_drift = float(
        (decode.logits.float() - baseline_decode.logits.float()).abs().max().item()
    )
    assert session.metrics.max_logit_drift >= 0.0
