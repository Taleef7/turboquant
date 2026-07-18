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
PREFILL_LOGIT_RTOL = 5e-4
PREFILL_LOGIT_ATOL = 5e-4
GENERATION_PARITY_TOKENS = 8


@pytest.fixture(scope="module")
def pinned_qwen25_model():
    if os.environ.get("TURBOQUANT_RUN_MODEL_TESTS") != "1":
        pytest.skip("set TURBOQUANT_RUN_MODEL_TESTS=1 to run the pinned model integration test")
    if not torch.cuda.is_available():
        pytest.fail("TURBOQUANT_RUN_MODEL_TESTS=1 requires a CUDA-capable PyTorch runtime")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        pytest.fail(f"TURBOQUANT_RUN_MODEL_TESTS=1 requires pinned Transformers dependencies: {exc}")

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

    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_qwen25_packed_prefill_matches_baseline_before_packed_decode(
    pinned_qwen25_model,
) -> None:
    """The custom AttentionInterface must return Qwen's [B, Q, H, D] layout."""
    model, tokenizer = pinned_qwen25_model
    profile = load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")
    prompt = tokenizer(
        "Explain the difference between a key and a value in one short sentence.",
        return_tensors="pt",
    ).input_ids.to(model.device)
    assert prompt.shape[1] > 1

    with torch.inference_mode():
        baseline = model(prompt, use_cache=True)
        session = install_qwen25_packed_backend(model, profile)
        packed_prefill = session.prefill(prompt)

    max_logit_drift = float(
        (packed_prefill.logits.float() - baseline.logits.float()).abs().max().item()
    )
    assert session.metrics.packed_decode_calls == 0
    assert session.metrics.standard_prefill_calls == profile.model_architecture["num_hidden_layers"]
    assert torch.equal(
        packed_prefill.logits[:, -1].argmax(dim=-1), baseline.logits[:, -1].argmax(dim=-1)
    )
    assert max_logit_drift <= PREFILL_LOGIT_ATOL
    torch.testing.assert_close(
        packed_prefill.logits,
        baseline.logits,
        rtol=PREFILL_LOGIT_RTOL,
        atol=PREFILL_LOGIT_ATOL,
        msg=f"Qwen prefill max logit drift: {max_logit_drift}",
    )


@pytest.mark.gpu
def test_qwen25_short_greedy_generation_matches_baseline_before_packed_blocks(
    pinned_qwen25_model,
) -> None:
    """Control t1..tN accounting while the direct cache is only an fp16 tail."""
    model, tokenizer = pinned_qwen25_model
    profile = load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")
    prompt = tokenizer(
        "State one concise fact about the word cache.", return_tensors="pt"
    ).input_ids.to(model.device)
    assert 1 < prompt.shape[1] <= 383
    eos_token_id = model.generation_config.eos_token_id
    eos_token_ids = (
        {int(eos_token_id)}
        if isinstance(eos_token_id, int)
        else {int(token_id) for token_id in eos_token_id or []}
    )

    with torch.inference_mode():
        baseline = model.generate(
            prompt,
            max_new_tokens=GENERATION_PARITY_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        baseline_ids = [int(token_id) for token_id in baseline[0, prompt.shape[1] :].tolist()]

        session = install_qwen25_packed_backend(model, profile)
        direct = session.prefill(prompt)
        token_ids = direct.logits[:, -1:].argmax(dim=-1)
        direct_ids: list[int] = []
        for generated_count in range(GENERATION_PARITY_TOKENS):
            token_id = int(token_ids.item())
            direct_ids.append(token_id)
            if token_id in eos_token_ids or generated_count + 1 == GENERATION_PARITY_TOKENS:
                break
            direct = session.decode_one(token_ids)
            token_ids = direct.logits[:, -1:].argmax(dim=-1)

    assert direct_ids == baseline_ids
    assert session.cache.sequence_length() <= 383


@pytest.mark.gpu
def test_qwen25_packed_decode_real_model_smoke(tmp_path, pinned_qwen25_model) -> None:
    profile = load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")
    model, tokenizer = pinned_qwen25_model

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
