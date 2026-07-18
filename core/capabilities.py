"""Fail-closed architecture facts, pinned manifests, and decode eligibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from core.errors import UnsupportedCapabilityError


_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "revision",
        "model_type",
        "adapter_status",
        "architecture_complete",
        "is_text_model",
        "is_multimodal",
        "use_cache",
    }
)
_MANIFEST_ARCHITECTURE_FIELDS = frozenset(
    {
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "layer_types",
    }
)
_DIRECT_QWEN25_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class ModelRevision:
    """Pinned identity of the exact manifest used for an adapter decision."""

    model_id: str
    revision: str
    manifest_sha256: str


@dataclass(frozen=True)
class ModelManifest:
    """Validated architecture source; manifest-only entries may be incomplete."""

    path: Path
    revision: ModelRevision
    model_type: str
    adapter_status: str
    architecture_complete: bool
    layer_count: int | None
    layer_types: tuple[str, ...]
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    rope_settings: Mapping[str, Any]
    is_text_model: bool
    is_multimodal: bool
    cache_behavior: str

    def architecture_binding(self) -> dict[str, Any]:
        """Return the complete, JSON-serializable architecture identity."""
        return {
            "model_type": self.model_type,
            "adapter_status": self.adapter_status,
            "architecture_complete": self.architecture_complete,
            "num_hidden_layers": self.layer_count,
            "layer_types": list(self.layer_types),
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rope_settings": dict(self.rope_settings),
            "is_text_model": self.is_text_model,
            "is_multimodal": self.is_multimodal,
            "cache_behavior": self.cache_behavior,
        }


@dataclass(frozen=True)
class ModelCapabilities:
    """Architecture facts derived from configuration and optionally a manifest."""

    model_type: str | None
    revision: ModelRevision | None
    manifest: ModelManifest | None
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
    is_text_model: bool | None
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


def _field(config: object, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _required_fields(payload: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    missing = sorted(fields.difference(payload))
    if missing:
        raise UnsupportedCapabilityError(
            f"{label} is missing required field(s): {', '.join(missing)}."
        )


def _manifest_layer_types(payload: Mapping[str, Any]) -> tuple[str, ...]:
    layer_count = payload["num_hidden_layers"]
    layer_types = payload["layer_types"]
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise UnsupportedCapabilityError("model manifest num_hidden_layers must be positive.")
    if not isinstance(layer_types, list) or len(layer_types) != layer_count:
        raise UnsupportedCapabilityError(
            "model manifest layer_types must contain exactly num_hidden_layers entries."
        )
    valid = {"full_attention", "sliding_attention", "linear_attention"}
    if any(layer not in valid for layer in layer_types):
        raise UnsupportedCapabilityError("model manifest contains an unknown layer type.")
    return tuple(layer_types)


def load_model_manifest(path: str | Path) -> ModelManifest:
    """Load a model manifest without loading Transformers or model weights."""
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedCapabilityError(
            f"Cannot read model manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise UnsupportedCapabilityError("model manifest must be a JSON object.")
    _required_fields(payload, _MANIFEST_BASE_FIELDS, "model manifest")
    if payload["schema_version"] != _MANIFEST_SCHEMA_VERSION:
        raise UnsupportedCapabilityError("model manifest has an unsupported schema_version.")

    complete = payload["architecture_complete"]
    if not isinstance(complete, bool):
        raise UnsupportedCapabilityError("model manifest architecture_complete must be boolean.")
    if complete:
        _required_fields(payload, _MANIFEST_ARCHITECTURE_FIELDS, "model manifest")
        layer_types = _manifest_layer_types(payload)
        for field in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            if not isinstance(payload[field], int) or payload[field] <= 0:
                raise UnsupportedCapabilityError(
                    f"model manifest {field} must be a positive integer."
                )
        layer_count = payload["num_hidden_layers"]
        q_heads = payload["num_attention_heads"]
        kv_heads = payload["num_key_value_heads"]
        head_dim = payload["head_dim"]
    else:
        layer_count = q_heads = kv_heads = head_dim = None
        layer_types = ()

    rope_settings = {
        name: payload[name]
        for name in ("rope_theta", "rope_scaling", "rope_parameters")
        if name in payload
    }
    cache_behavior = "enabled" if payload["use_cache"] else "disabled"
    if payload.get("cache_implementation"):
        cache_behavior = str(payload["cache_implementation"])
    return ModelManifest(
        path=manifest_path,
        revision=ModelRevision(
            model_id=payload["model_id"],
            revision=payload["revision"],
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
        ),
        model_type=payload["model_type"],
        adapter_status=payload["adapter_status"],
        architecture_complete=complete,
        layer_count=layer_count,
        layer_types=layer_types,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        rope_settings=MappingProxyType(rope_settings),
        is_text_model=payload["is_text_model"],
        is_multimodal=payload["is_multimodal"],
        cache_behavior=cache_behavior,
    )


def _layer_types(config: object, count: int | None) -> tuple[str, ...]:
    configured = _field(config, "layer_types")
    if configured is None:
        return ()
    if not isinstance(configured, (list, tuple)):
        return ()
    return tuple(str(layer) for layer in configured)


def _revision_matches_manifest(
    revision: ModelRevision | None, manifest: ModelManifest | None
) -> bool:
    return revision is not None and manifest is not None and revision == manifest.revision


def _adapter_status(
    model_type: str | None, revision: ModelRevision | None, manifest: ModelManifest | None
) -> str:
    if _revision_matches_manifest(revision, manifest) and manifest.model_type == model_type:
        return manifest.adapter_status
    if model_type == "qwen2" and revision is not None:
        return "direct-adapter"
    if model_type in {"qwen3", "qwen3_5", "gemma4"}:
        return "manifest-only"
    return "not-supported"


def derive_model_capabilities(
    config: object,
    revision: ModelRevision | None = None,
    manifest: ModelManifest | None = None,
) -> ModelCapabilities:
    """Derive facts from the actual config; a manifest governs adapter eligibility."""
    raw_model_type = _field(config, "model_type")
    model_type = str(raw_model_type) if raw_model_type is not None else None
    raw_count = _field(config, "num_hidden_layers")
    layer_count = int(raw_count) if isinstance(raw_count, int) else None
    layer_types = _layer_types(config, layer_count)

    full = tuple(i for i, layer in enumerate(layer_types) if layer == "full_attention")
    sliding = tuple(i for i, layer in enumerate(layer_types) if layer == "sliding_attention")
    linear = tuple(i for i, layer in enumerate(layer_types) if layer == "linear_attention")
    native = tuple(i for i, layer in enumerate(layer_types) if i not in full)
    raw_q_heads = _field(config, "num_attention_heads")
    q_heads = int(raw_q_heads) if isinstance(raw_q_heads, int) else None
    raw_kv_heads = _field(config, "num_key_value_heads")
    kv_heads = int(raw_kv_heads) if isinstance(raw_kv_heads, int) else None
    raw_head_dim = _field(config, "head_dim")
    head_dim = int(raw_head_dim) if isinstance(raw_head_dim, int) else None
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
        manifest=manifest,
        adapter_status=_adapter_status(model_type, revision, manifest),
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
        is_text_model=_field(config, "is_text_model"),
        is_multimodal=bool(_field(config, "is_multimodal", False)),
        cache_behavior=cache_behavior,
    )


def _matches_pinned_manifest(capabilities: ModelCapabilities) -> bool:
    manifest = capabilities.manifest
    return bool(
        manifest
        and manifest.architecture_complete
        and capabilities.revision == manifest.revision
        and capabilities.model_type == manifest.model_type
        and capabilities.layer_count == manifest.layer_count
        and capabilities.layer_types == manifest.layer_types
        and capabilities.num_attention_heads == manifest.num_attention_heads
        and capabilities.num_key_value_heads == manifest.num_key_value_heads
        and capabilities.head_dim == manifest.head_dim
        and dict(capabilities.rope_settings) == dict(manifest.rope_settings)
        and capabilities.is_text_model == manifest.is_text_model
        and capabilities.is_multimodal == manifest.is_multimodal
        and capabilities.cache_behavior == manifest.cache_behavior
    )


def require_direct_decode_support(
    capabilities: ModelCapabilities,
    request: DecodeRequest,
    *,
    pending_store: object | None = None,
) -> ModelCapabilities:
    """Preflight a request without touching an optionally supplied pending store."""
    # The adapter has not taken ownership of this store yet. Keeping the value
    # at this boundary makes the no-mutation ordering explicit without adding a
    # cache implementation or a serving API to the capability layer.
    _ = pending_store
    if capabilities.model_type not in {"qwen2", "qwen3", "qwen3_5", "gemma4"}:
        raise UnsupportedCapabilityError(
            f"Unsupported model_type {capabilities.model_type!r}; no adapter is registered."
        )
    if capabilities.adapter_status != "direct-adapter":
        raise UnsupportedCapabilityError(
            f"{capabilities.model_type} is {capabilities.adapter_status}; "
            "direct decode requires a tested direct-adapter."
        )
    if capabilities.manifest is None or capabilities.revision is None:
        raise UnsupportedCapabilityError("direct decode requires a pinned manifest.")
    if capabilities.manifest.revision.model_id != _DIRECT_QWEN25_MODEL_ID:
        raise UnsupportedCapabilityError(
            "direct decode is pinned only for the Qwen2.5-7B-Instruct manifest."
        )
    if capabilities.is_multimodal or not capabilities.is_text_model:
        raise UnsupportedCapabilityError(
            "multimodal processor or vision cache is unsupported by direct KV decode."
        )
    if capabilities.is_multimodal or request.includes_multimodal_cache or request.multimodal_cache:
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
    if not _matches_pinned_manifest(capabilities):
        raise UnsupportedCapabilityError(
            "direct decode configuration does not match the pinned manifest layout."
        )
    return capabilities
