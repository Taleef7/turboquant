"""CPU-only contract tests for the narrow Qwen2.5 packed decode adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.capabilities import DecodeRequest, derive_model_capabilities, load_model_manifest
from core.errors import UnsupportedCapabilityError
from core.profiles import load_profile
from core.qwen_adapter import install_qwen25_packed_backend, qwen25_packed_attention


ROOT = Path(__file__).resolve().parents[1]


class FakeAttention:
    def __init__(self, layer_idx: int, config: SimpleNamespace) -> None:
        self.layer_idx = layer_idx
        self.config = config
        self.scaling = 1 / config.head_dim**0.5


class FakeQwen:
    """Local Qwen-shaped module that drives the public attention interface."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="qwen2",
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            head_dim=128,
            layer_types=["full_attention"] * 28,
            rope_theta=1_000_000.0,
            is_text_model=True,
            is_multimodal=False,
            use_cache=True,
            _attn_implementation="sdpa",
        )
        self.model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=FakeAttention(index, self.config)) for index in range(28)]
        )
        self.device = torch.device("cpu")
        self.dtype = torch.float16

    def __call__(self, input_ids: torch.Tensor, *, past_key_values, use_cache: bool, **kwargs):
        assert use_cache is True
        module = self.model.layers[0].self_attn
        tokens = input_ids.shape[1]
        query = torch.ones((1, 28, tokens, 128), dtype=torch.float16)
        key = torch.ones((1, 4, tokens, 128), dtype=torch.float16)
        value = torch.ones_like(key)
        key, value = past_key_values.update(key, value, 0, {"cache_position": torch.arange(tokens)})
        output, _ = qwen25_packed_attention(
            module, query, key, value, None, scaling=module.scaling
        )
        assert output.shape == query.shape
        return SimpleNamespace(logits=torch.zeros((1, tokens, 11), dtype=torch.float16))


@pytest.fixture
def fake_qwen() -> FakeQwen:
    return FakeQwen()


@pytest.fixture
def profile():
    return load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")


@pytest.fixture
def capabilities(fake_qwen: FakeQwen, profile):
    return derive_model_capabilities(
        fake_qwen.config, profile.revision, load_model_manifest(profile.model_manifest_path)
    )


def test_decode_dispatches_packed_backend_after_standard_prefill(
    fake_qwen: FakeQwen, profile, capabilities
) -> None:
    session = install_qwen25_packed_backend(fake_qwen, profile, capabilities)
    session.compare_packed_reference = True
    session.prefill(torch.tensor([[1, 2, 3]]))
    # Fill enough post-RoPE cache state to make one immutable packed block.
    with session._active("prefill"):
        session.cache.update(
            torch.ones((1, 4, 381, 128), dtype=torch.float16),
            torch.ones((1, 4, 381, 128), dtype=torch.float16),
            0,
            {"cache_position": torch.arange(381)},
        )
    session.decode_one()

    assert session.metrics.standard_prefill_calls == 1
    assert session.metrics.packed_decode_calls == 1
    assert session.metrics.packed_blocks_consumed >= 1
    assert session.cache.historical_kv_tokens_returned == 0
    assert session.metrics.packed_reference_max_error is not None
    assert session.metrics.packed_reference_max_error < 1e-3


def test_layer_stores_have_distinct_rotations(fake_qwen: FakeQwen, profile, capabilities) -> None:
    session = install_qwen25_packed_backend(fake_qwen, profile, capabilities)
    assert not torch.equal(
        session.cache.store_for(0).key_codec._rotation,
        session.cache.store_for(1).key_codec._rotation,
    )


def test_install_derives_the_profile_pinned_capabilities(fake_qwen: FakeQwen, profile) -> None:
    session = install_qwen25_packed_backend(fake_qwen, profile, profile.revision)
    assert session.capabilities.revision == profile.revision


@pytest.mark.parametrize(
    "decode_request",
    [
        DecodeRequest(batch_size=2),
        DecodeRequest(num_beams=2),
        DecodeRequest(greedy=False),
    ],
)
def test_unsupported_generation_request_fails_before_cache_mutation(
    fake_qwen: FakeQwen, profile, capabilities, decode_request: DecodeRequest
) -> None:
    session = install_qwen25_packed_backend(fake_qwen, profile, capabilities)
    with pytest.raises(UnsupportedCapabilityError):
        session.prefill(torch.tensor([[1, 2, 3]]), request=decode_request)
    assert session.cache.sequence_length() == 0


def test_cache_rejects_dtype_change_before_append(fake_qwen: FakeQwen, profile, capabilities) -> None:
    session = install_qwen25_packed_backend(fake_qwen, profile, capabilities)
    with session._active("prefill"):
        session.cache.update(
            torch.ones((1, 4, 2, 128), dtype=torch.float16),
            torch.ones((1, 4, 2, 128), dtype=torch.float16),
            0,
            {"cache_position": torch.arange(2)},
        )
        with pytest.raises(UnsupportedCapabilityError, match="dtype"):
            session.cache.update(
                torch.ones((1, 4, 1, 128), dtype=torch.bfloat16),
                torch.ones((1, 4, 1, 128), dtype=torch.bfloat16),
                0,
                {"cache_position": torch.tensor([2])},
            )
    assert session.cache.store_for(0).sequence_length() == 2


def test_cache_rejects_device_change_before_append(fake_qwen: FakeQwen, profile, capabilities) -> None:
    session = install_qwen25_packed_backend(fake_qwen, profile, capabilities)
    with session._active("prefill"):
        session.cache.update(
            torch.ones((1, 4, 2, 128), dtype=torch.float16),
            torch.ones((1, 4, 2, 128), dtype=torch.float16),
            0,
        )
        with pytest.raises(UnsupportedCapabilityError, match="device"):
            session.cache.update(
                torch.ones((1, 4, 1, 128), dtype=torch.float16, device="meta"),
                torch.ones((1, 4, 1, 128), dtype=torch.float16, device="meta"),
                0,
            )
    assert session.cache.store_for(0).sequence_length() == 2


def test_cache_off_fails_before_prefill_mutates_store(fake_qwen: FakeQwen, profile, capabilities) -> None:
    session = install_qwen25_packed_backend(fake_qwen, profile, capabilities)
    with pytest.raises(UnsupportedCapabilityError, match="enabled KV cache"):
        session.prefill(torch.tensor([[1, 2, 3]]), use_cache=False)
    assert session.cache.sequence_length() == 0


def test_unsupported_qwen_layout_fails_closed(fake_qwen: FakeQwen, profile) -> None:
    fake_qwen.config.layer_types[3] = "sliding_attention"
    capabilities = derive_model_capabilities(fake_qwen.config, profile.revision)
    with pytest.raises(UnsupportedCapabilityError):
        install_qwen25_packed_backend(fake_qwen, profile, capabilities)


def _real_qwen2_attention_session(profile):
    """Build one real pinned-layout attention layer without model weights/downloads."""
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=3584,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=1_000_000.0,
        use_cache=True,
    )
    config.layer_types = ["full_attention"] * 28
    config.head_dim = 128
    config.is_text_model = True
    config.is_multimodal = False
    config._attn_implementation = "sdpa"
    attention = Qwen2Attention(config, layer_idx=0).to(dtype=torch.float16)
    placeholders = [SimpleNamespace(self_attn=FakeAttention(index, config)) for index in range(1, 28)]
    model = SimpleNamespace(
        config=config,
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention), *placeholders]),
    )
    rotary = Qwen2RotaryEmbedding(config)
    session = install_qwen25_packed_backend(model, profile, profile.revision)

    def invoke(tokens: int, cache) -> torch.Tensor:
        hidden = torch.zeros((1, tokens, config.hidden_size), dtype=torch.float16)
        positions = torch.arange(tokens).unsqueeze(0)
        cos, sin = rotary(hidden, positions)
        output, _ = attention(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=cache,
            cache_position=torch.arange(tokens),
        )
        return output

    return session, invoke


def test_real_qwen2_attention_rejects_idle_decode_before_cache_mutation(profile) -> None:
    session, invoke = _real_qwen2_attention_session(profile)
    with session:
        with pytest.raises(UnsupportedCapabilityError, match="active prefill or greedy decode"):
            invoke(1, session.cache)
    assert session.cache.sequence_length() == 0


def test_real_qwen2_attention_rejects_raw_cache_use_without_prefill_phase(profile) -> None:
    """Equivalent to an external Qwen call that supplies the cache with cache disabled."""
    session, invoke = _real_qwen2_attention_session(profile)
    with session:
        with pytest.raises(UnsupportedCapabilityError, match="active prefill or greedy decode"):
            invoke(3, session.cache)
    assert session.cache.sequence_length() == 0
    assert session.cache.rejected_updates == 1


def test_real_qwen2_attention_uses_registered_backend_for_prefill(profile) -> None:
    session, invoke = _real_qwen2_attention_session(profile)
    with session._active("prefill"):
        output = invoke(3, session.cache)
    assert output.shape == (1, 3, 3584)
    assert session.cache.sequence_length() == 3
    assert session.metrics.standard_prefill_calls == 1


def test_real_qwen2_attention_dispatches_packed_decode_without_historic_cat(profile, monkeypatch) -> None:
    session, invoke = _real_qwen2_attention_session(profile)
    with session._active("prefill"):
        invoke(3, session.cache)
        session.cache.update(
            torch.ones((1, 4, 381, 128), dtype=torch.float16),
            torch.ones((1, 4, 381, 128), dtype=torch.float16),
            0,
        )

    historic_cat_calls: list[tuple[int, ...]] = []
    original_cat = torch.cat

    def reject_historic_kv_cat(tensors, *args, **kwargs):
        materialized = tuple(tensors)
        dimension = kwargs.get("dim", args[0] if args else 0)
        if (
            dimension in {-2, 2}
            and
            len(materialized) > 1
            and all(isinstance(tensor, torch.Tensor) and tensor.ndim == 4 for tensor in materialized)
            and sum(tensor.shape[-2] for tensor in materialized) > max(tensor.shape[-2] for tensor in materialized)
        ):
            historic_cat_calls.append(tuple(tensor.shape[-2] for tensor in materialized))
            raise AssertionError("packed decode must not concatenate historical K/V tensors")
        return original_cat(materialized, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", reject_historic_kv_cat)
    with session._active("decode"):
        output = invoke(1, session.cache)

    assert output.shape == (1, 1, 3584)
    assert session.metrics.packed_decode_calls == 1
    assert session.metrics.packed_blocks_consumed >= 1
    assert session.cache.historical_kv_tokens_returned == 0
    assert historic_cat_calls == []
