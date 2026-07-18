"""Opt-in real-model smoke coverage for the Qwen2.5 direct decode route."""

from __future__ import annotations

import os
import json
import math
from pathlib import Path

import pytest
import torch

from core.qwen_adapter import install_qwen25_packed_backend
from core.profiles import load_profile
from scripts.repro_utils import PINNED_QWEN25_HUB_REVISION, PINNED_QWEN25_MODEL_ID


MODEL_ID = PINNED_QWEN25_MODEL_ID
MODEL_REVISION = PINNED_QWEN25_HUB_REVISION
PROMPT = "Reply with one short word: TurboQuant"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
def test_qwen25_packed_decode_real_model_smoke(tmp_path) -> None:
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
    def rejects_historic_kv_cat(original_cat, calls):
        def guarded(tensors, *args, **kwargs):
            materialized = tuple(tensors)
            dimension = kwargs.get("dim", args[0] if args else 0)
            if (
                dimension in {-2, 2}
                and len(materialized) > 1
                and all(tensor.ndim == 4 for tensor in materialized)
                and sum(tensor.shape[-2] for tensor in materialized) > max(tensor.shape[-2] for tensor in materialized)
            ):
                calls.append(tuple(tensor.shape[-2] for tensor in materialized))
                raise AssertionError("packed decode concatenated historical K/V")
            return original_cat(materialized, *args, **kwargs)

        return guarded

    with torch.inference_mode():
        baseline_prefill = model(synthetic, use_cache=True)
        baseline_cache = baseline_prefill.past_key_values
        session = install_qwen25_packed_backend(model, profile)
        prefill = session.prefill(synthetic)
        assert torch.isfinite(prefill.logits).all()
        token = prefill.logits[:, -1:].argmax(dim=-1)
        direct_tokens: list[int] = []
        max_drift = 0.0
        historic_cat_calls: list[tuple[int, ...]] = []
        try:
            for _ in range(64):
                original_cat = torch.cat
                torch.cat = rejects_historic_kv_cat(original_cat, historic_cat_calls)
                try:
                    direct = session.decode_one(token)
                finally:
                    torch.cat = original_cat
                baseline = model(token, past_key_values=baseline_cache, use_cache=True)
                baseline_cache = baseline.past_key_values
                assert torch.isfinite(direct.logits).all()
                assert torch.isfinite(baseline.logits).all()
                max_drift = max(max_drift, float((direct.logits.float() - baseline.logits.float()).abs().max()))
                token = direct.logits[:, -1:].argmax(dim=-1)
                direct_tokens.append(int(token.item()))
        except Exception as exc:
            pytest.fail(f"pinned Qwen2.5 packed decode failed: {exc}")

        reference_session = install_qwen25_packed_backend(model, profile)
        reference_session.use_packed_reference_backend = True
        reference_prefill = reference_session.prefill(synthetic)
        reference_token = reference_prefill.logits[:, -1:].argmax(dim=-1)
        reference_tokens: list[int] = []
        for _ in range(64):
            reference = reference_session.decode_one(reference_token)
            assert torch.isfinite(reference.logits).all()
            reference_token = reference.logits[:, -1:].argmax(dim=-1)
            reference_tokens.append(int(reference_token.item()))

    session.metrics.max_logit_drift = max_drift
    artifact = tmp_path / "qwen25_packed_decode_metrics.json"
    artifact.write_text(json.dumps({
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "prefill_tokens": int(synthetic.shape[1]),
        "decode_tokens": 64,
        "packed_blocks_consumed": session.metrics.packed_blocks_consumed,
        "max_logit_drift": session.metrics.max_logit_drift,
        "direct_tokens": direct_tokens,
        "reference_tokens": reference_tokens,
    }, sort_keys=True), encoding="utf-8")

    assert session.metrics.packed_decode_calls == 64 * profile.model_architecture["num_hidden_layers"]
    assert session.metrics.packed_blocks_consumed > 0
    assert session.cache.historical_kv_tokens_returned == 0
    assert historic_cat_calls == []
    assert direct_tokens == reference_tokens
    assert reference_session.metrics.packed_reference_decode_calls == 64 * profile.model_architecture["num_hidden_layers"]
    assert math.isfinite(session.metrics.max_logit_drift)
    assert json.loads(artifact.read_text(encoding="utf-8"))["decode_tokens"] == 64
