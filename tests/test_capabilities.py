"""Fail-closed model capability rules without model downloads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.capabilities import (
    DecodeRequest,
    ModelRevision,
    derive_model_capabilities,
    require_direct_decode_support,
)
from core.errors import UnsupportedCapabilityError


def qwen25_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_type": "qwen2",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rope_theta": 1_000_000.0,
        "use_cache": True,
        "layer_types": ("full_attention", "full_attention"),
        "is_multimodal": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def qwen35_hybrid_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen3_5",
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=1_000_000.0,
        use_cache=True,
        layer_types=(
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ),
    )


class NeverMutatedStore:
    def __init__(self) -> None:
        self.append_calls = 0

    def append(self, *args: object, **kwargs: object) -> None:
        self.append_calls += 1
        raise AssertionError("capability rejection must happen before append")


def direct_revision() -> ModelRevision:
    return ModelRevision(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        revision="test-revision",
        manifest_sha256="a" * 64,
    )


def test_qwen25_records_architecture_facts_from_config() -> None:
    caps = derive_model_capabilities(qwen25_config(), direct_revision())

    assert caps.model_type == "qwen2"
    assert caps.layer_count == 2
    assert caps.full_attention_layer_indices == (0, 1)
    assert caps.compressible_layer_indices == (0, 1)
    assert caps.num_key_value_groups == 4
    assert caps.rope_settings == {"rope_theta": 1_000_000.0}
    assert caps.adapter_status == "direct-adapter"


def test_qwen35_exposes_only_full_attention_as_compressible() -> None:
    caps = derive_model_capabilities(qwen35_hybrid_config())

    assert caps.compressible_layer_indices == (3, 7)
    assert caps.native_state_layer_indices == (0, 1, 2, 4, 5, 6)
    assert caps.adapter_status == "manifest-only"


def test_gemma4_is_manifest_only() -> None:
    caps = derive_model_capabilities(
        SimpleNamespace(
            model_type="gemma4",
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=("full_attention", "full_attention"),
            is_multimodal=True,
        )
    )

    assert caps.adapter_status == "manifest-only"
    assert caps.is_multimodal is True


@pytest.mark.parametrize(
    ("config", "decode_request", "message"),
    [
        (qwen25_config(), DecodeRequest(batch_size=2), "batch_size"),
        (qwen25_config(), DecodeRequest(do_sample=True), "greedy"),
        (qwen25_config(), DecodeRequest(greedy=False), "greedy"),
        (qwen25_config(), DecodeRequest(num_beams=2), "greedy"),
        (
            qwen25_config(layer_types=("sliding_attention", "full_attention")),
            DecodeRequest(),
            "sliding",
        ),
        (
            qwen25_config(layer_types=("linear_attention", "linear_attention")),
            DecodeRequest(),
            "linear",
        ),
        (qwen25_config(), DecodeRequest(speculative=True), "speculative"),
        (qwen25_config(), DecodeRequest(mtp=True), "MTP"),
        (qwen25_config(is_multimodal=True), DecodeRequest(), "multimodal"),
        (
            qwen25_config(),
            DecodeRequest(includes_multimodal_cache=True),
            "multimodal",
        ),
        (qwen25_config(), DecodeRequest(multimodal_cache=True), "multimodal"),
        (
            SimpleNamespace(model_type="qwen3", num_hidden_layers=2),
            DecodeRequest(),
            "manifest-only",
        ),
    ],
)
def test_unsupported_direct_decode_routes_fail_closed(
    config: SimpleNamespace, decode_request: DecodeRequest, message: str
) -> None:
    caps = derive_model_capabilities(config, direct_revision())

    with pytest.raises(UnsupportedCapabilityError, match=message):
        require_direct_decode_support(caps, decode_request)


def test_unknown_model_rejects_before_cache_mutation() -> None:
    store = NeverMutatedStore()
    capabilities = derive_model_capabilities(SimpleNamespace(model_type="unknown"))

    with pytest.raises(UnsupportedCapabilityError, match="model_type"):
        require_direct_decode_support(
            capabilities, DecodeRequest(batch_size=1, store=store)
        )

    assert store.append_calls == 0


def test_direct_route_returns_capabilities_without_mutating_store() -> None:
    store = NeverMutatedStore()
    capabilities = derive_model_capabilities(qwen25_config(), direct_revision())

    assert (
        require_direct_decode_support(
            capabilities, DecodeRequest(batch_size=1, store=store)
        )
        is capabilities
    )
    assert store.append_calls == 0
