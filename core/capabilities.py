"""Fail-closed architecture facts and direct-decode eligibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from core.errors import UnsupportedCapabilityError


@dataclass(frozen=True)
class ModelRevision:
    """Pinned identity of a model manifest used to make an adapter decision."""

    model_id: str
    revision: str
    manifest_sha256: str


@dataclass(frozen=True)
class ModelCapabilities:
    """Architecture facts derived from a model configuration, never a benchmark name."""

    model_type: str | None
    revision: ModelRevision | None
    adapter_status: str
    layer_count: int | None
    layer_types: tuple[str, ...]
    full_attention_layer_indices: tuple[int, ...]
    sliding_window_layer_indices: tuple[int, ...]
    linear_attention_layer_indices: tuple[int, ...]
    compressible_layer_indices: tuple[int, ...]
    native_state_layer_indices: tuple[int, ...]
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    num_key_value_groups: int | None
    rope_settings: Mapping[str, Any]
    is_multimodal: bool
    cache_behavior: str


@dataclass(frozen=True)
class DecodeRequest:
    """The narrow generation contract for the first direct adapter milestone."""

    batch_size: int = 1
    greedy: bool = True
    do_sample: bool = False
    num_beams: int = 1
    speculative: bool = False
    mtp: bool = False
    includes_multimodal_cache: bool = False
    multimodal_cache: bool = False
    store: object | None = None


def _field(config: object, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _layer_types(config: object, model_type: str | None, count: int | None) -> tuple[str, ...]:
    configured = _field(config, "layer_types")
    if configured is not None:
        return tuple(str(layer) for layer in configured)
    if count is None:
        return ()
    if _field(config, "use_sliding_window", False):
        return tuple("sliding_attention" for _ in range(count))
    if model_type in {"qwen2", "qwen3", "gemma4"}:
        return tuple("full_attention" for _ in range(count))
    return ()


def _adapter_status(model_type: str | None, revision: ModelRevision | None) -> str:
    if model_type == "qwen2" and revision is not None:
        if "qwen2.5" in revision.model_id.lower():
            return "direct-adapter"
    if model_type in {"qwen3", "qwen3_5", "gemma4"}:
        return "manifest-only"
    return "not-supported"


def derive_model_capabilities(
    config: object, revision: ModelRevision | None = None
) -> ModelCapabilities:
    """Derive all adapter-relevant facts from the supplied pinned configuration."""
    raw_model_type = _field(config, "model_type")
    model_type = str(raw_model_type) if raw_model_type is not None else None
    raw_count = _field(config, "num_hidden_layers")
    layer_count = int(raw_count) if raw_count is not None else None
    layer_types = _layer_types(config, model_type, layer_count)

    full = tuple(i for i, layer in enumerate(layer_types) if layer == "full_attention")
    sliding = tuple(
        i
        for i, layer in enumerate(layer_types)
        if layer in {"sliding_attention", "sliding_window"}
    )
    linear = tuple(i for i, layer in enumerate(layer_types) if layer == "linear_attention")
    native = tuple(i for i, layer in enumerate(layer_types) if i not in full)

    raw_q_heads = _field(config, "num_attention_heads")
    q_heads = int(raw_q_heads) if raw_q_heads is not None else None
    raw_kv_heads = _field(config, "num_key_value_heads")
    kv_heads = int(raw_kv_heads) if raw_kv_heads is not None else q_heads
    raw_head_dim = _field(config, "head_dim")
    head_dim = int(raw_head_dim) if raw_head_dim is not None else None
    if head_dim is None:
        hidden_size = _field(config, "hidden_size")
        if hidden_size is not None and q_heads:
            head_dim = int(hidden_size) // q_heads
    groups = q_heads // kv_heads if q_heads and kv_heads and q_heads % kv_heads == 0 else None

    rope_settings = {
        name: value
        for name in ("rope_theta", "rope_scaling", "rope_parameters")
        if (value := _field(config, name)) is not None
    }
    cache_behavior = "enabled" if _field(config, "use_cache", False) else "disabled"
    if cache_impl := _field(config, "cache_implementation"):
        cache_behavior = str(cache_impl)

    return ModelCapabilities(
        model_type=model_type,
        revision=revision,
        adapter_status=_adapter_status(model_type, revision),
        layer_count=layer_count,
        layer_types=layer_types,
        full_attention_layer_indices=full,
        sliding_window_layer_indices=sliding,
        linear_attention_layer_indices=linear,
        compressible_layer_indices=full,
        native_state_layer_indices=native,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        num_key_value_groups=groups,
        rope_settings=MappingProxyType(rope_settings),
        is_multimodal=bool(_field(config, "is_multimodal", False)),
        cache_behavior=cache_behavior,
    )


def require_direct_decode_support(
    capabilities: ModelCapabilities, request: DecodeRequest
) -> ModelCapabilities:
    """Return eligible capabilities or fail before any cache/store operation starts."""
    if capabilities.model_type not in {"qwen2", "qwen3", "qwen3_5", "gemma4"}:
        raise UnsupportedCapabilityError(
            f"Unsupported model_type {capabilities.model_type!r}; no adapter is registered."
        )
    if capabilities.adapter_status != "direct-adapter":
        raise UnsupportedCapabilityError(
            f"{capabilities.model_type} is {capabilities.adapter_status}; "
            "direct decode requires a tested direct-adapter."
        )
    if (
        capabilities.is_multimodal
        or request.includes_multimodal_cache
        or request.multimodal_cache
    ):
        raise UnsupportedCapabilityError(
            "multimodal processor or vision cache is unsupported by direct KV decode."
        )
    if request.batch_size != 1:
        raise UnsupportedCapabilityError("direct decode requires batch_size=1.")
    if not request.greedy or request.do_sample or request.num_beams != 1:
        raise UnsupportedCapabilityError("direct decode supports greedy generation only.")
    if request.speculative:
        raise UnsupportedCapabilityError("speculative decoding is not supported.")
    if request.mtp:
        raise UnsupportedCapabilityError("MTP decoding is not supported.")
    if capabilities.sliding_window_layer_indices:
        raise UnsupportedCapabilityError("sliding-window attention layers are not supported.")
    if capabilities.linear_attention_layer_indices:
        raise UnsupportedCapabilityError("linear-attention layers require native state support.")
    if not capabilities.layer_types or len(capabilities.full_attention_layer_indices) != len(
        capabilities.layer_types
    ):
        raise UnsupportedCapabilityError("direct decode requires full-attention KV layers.")
    if capabilities.cache_behavior == "disabled":
        raise UnsupportedCapabilityError("direct decode requires an enabled KV cache.")
    return capabilities
