"""Documented-attention-interface adapter for Qwen2.5 batch-one greedy decode."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Any, Iterator

import torch

from core.capabilities import (
    DecodeRequest,
    ModelCapabilities,
    ModelRevision,
    derive_model_capabilities,
    load_model_manifest,
    require_direct_decode_support,
    require_qwen25_packed_layout,
)
from core.errors import UnsupportedCapabilityError
from core.packed_cache import PackedDynamicCache
from core.profiles import CompressionProfile


_BACKEND_NAME = "turboquant_qwen25_packed_decode"
_UNSUPPORTED_LAYOUT = (
    "TurboQuant Qwen2.5 packed decode requires Transformers >=4.57.3 "
    "AttentionInterface and supported Qwen2 attention layout."
)
_ACTIVE_SESSION: ContextVar["PackedAttentionSession | None"] = ContextVar(
    "turboquant_qwen25_packed_session", default=None
)


@dataclass
class PackedAttentionMetrics:
    standard_prefill_calls: int = 0
    packed_decode_calls: int = 0
    packed_blocks_consumed: int = 0
    packed_reference_decode_calls: int = 0
    parallel_decode_calls: int = 0
    parallel_page_launches: int = 0
    parallel_reduce_tail_launches: int = 0
    parallel_reduction_launches: int = 0
    parallel_tail_launches: int = 0
    max_logit_drift: float | None = None
    packed_reference_max_error: float | None = None


@dataclass
class _PreparedLayerDecode:
    """Reusable device constants and static block views for one attention layer."""

    device: torch.device
    block_count: int
    scale: float
    rotation: torch.Tensor
    key_codebook: torch.Tensor
    gqa_mapping: torch.Tensor
    prepared: Any
    debug: Any


def _attention_layers(model: object) -> list[object]:
    try:
        layers = model.model.layers
        attention = [layer.self_attn for layer in layers]
    except (AttributeError, TypeError) as exc:
        raise UnsupportedCapabilityError(_UNSUPPORTED_LAYOUT) from exc
    if not attention or any(not hasattr(module, "layer_idx") for module in attention):
        raise UnsupportedCapabilityError(_UNSUPPORTED_LAYOUT)
    return attention


def _standard_causal_attention(
    module: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor:
    """Delegate prefill exactly to the pinned Transformers SDPA interface.

    Qwen2 expects its attention function to return ``[batch, query, heads,
    head_dim]`` before its own reshape.  The Transformers helper also owns the
    version-specific GQA repeat, mask slicing, explicit scaling, dropout, and
    causal decision, so duplicating those details here risks divergence.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    output, _ = sdpa_attention_forward(
        module, query, key, value, attention_mask, **kwargs
    )
    return output


def qwen25_packed_attention(
    module: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Transformers ``AttentionInterface`` callable registered by this module."""
    session = _ACTIVE_SESSION.get()
    if session is None or not session.owns(module):
        raise UnsupportedCapabilityError(
            "Qwen2.5 packed attention requires an active installed session."
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or query.shape[0] != 1:
        raise UnsupportedCapabilityError("direct decode requires batch_size=1.")
    query_length = query.shape[-2]
    if query_length > 1:
        session.metrics.standard_prefill_calls += 1
        return (
            _standard_causal_attention(
                module, query, key, value, attention_mask, **kwargs
            ),
            None,
        )
    if query_length != 1 or session.phase != "decode":
        raise UnsupportedCapabilityError(
            "direct packed attention requires a standard prompt prefill first."
        )
    output = session.packed_decode(module, query, scale=kwargs.get("scaling"))
    return output, None


class PackedAttentionSession:
    """Context-scoped direct attention route; no process-global mutable cache."""

    def __init__(
        self,
        model: object,
        profile: CompressionProfile,
        capabilities: ModelCapabilities,
        *,
        maximum_sequence_length: int | None = None,
        use_parallel_page_attention: bool = False,
    ) -> None:
        self.model = model
        self.profile = profile
        self.capabilities = capabilities
        self.layers = _attention_layers(model)
        if len(self.layers) != capabilities.layer_count:
            raise UnsupportedCapabilityError(_UNSUPPORTED_LAYOUT)
        self._layer_ids = {
            id(module): index for index, module in enumerate(self.layers)
        }
        if type(use_parallel_page_attention) is not bool:  # noqa: E721
            raise TypeError("use_parallel_page_attention must be a bool.")
        if maximum_sequence_length is not None:
            if type(maximum_sequence_length) is not int:  # noqa: E721
                raise TypeError("maximum_sequence_length must be a built-in int.")
            if maximum_sequence_length < 1:
                raise ValueError("maximum_sequence_length must be positive.")
            if not use_parallel_page_attention:
                raise UnsupportedCapabilityError(
                    "maximum_sequence_length reservation requires the opt-in parallel page path."
                )
        if use_parallel_page_attention and maximum_sequence_length is None:
            raise UnsupportedCapabilityError(
                "parallel page attention requires an explicit maximum_sequence_length."
            )
        self.maximum_sequence_length = maximum_sequence_length
        self.use_parallel_page_attention = use_parallel_page_attention
        self.cache = PackedDynamicCache(
            profile,
            layer_count=len(self.layers),
            maximum_sequence_length=maximum_sequence_length
            if use_parallel_page_attention
            else None,
        )
        self.cache.set_update_guard(self._require_permitted_cache_update)
        self.metrics = PackedAttentionMetrics()
        self.phase = "idle"
        self._token: object | None = None
        self._previous_implementation: object | None = None
        self._last_logits: torch.Tensor | None = None
        self._prepared_layers: dict[int, _PreparedLayerDecode] = {}
        self._parallel_workspace: Any | None = None
        self._parallel_workspace_device: torch.device | None = None
        self._parallel_outputs: dict[int, torch.Tensor] = {}
        self._parallel_prepared_codecs: dict[int, Any] = {}
        self.compare_packed_reference = False
        self.use_packed_reference_backend = False

    def owns(self, module: object) -> bool:
        return id(module) in self._layer_ids

    def _require_permitted_cache_update(
        self, key_states: torch.Tensor, layer_idx: int
    ) -> None:
        """Reject unsupported attention calls before a K/V append can occur."""
        if _ACTIVE_SESSION.get() is not self:
            raise UnsupportedCapabilityError(
                "Qwen2.5 packed cache requires its active session."
            )
        tokens = key_states.shape[-2]
        if self.phase == "prefill" and tokens > 1:
            return
        if self.phase == "decode" and tokens == 1 and self.cache.sequence_length() > 0:
            return
        raise UnsupportedCapabilityError(
            "Qwen2.5 packed cache updates require active prefill or greedy decode."
        )

    def __enter__(self) -> "PackedAttentionSession":
        active = _ACTIVE_SESSION.get()
        if active is not None and active is not self:
            raise UnsupportedCapabilityError(
                "nested Qwen2.5 packed attention sessions are unsupported."
            )
        if self._token is None:
            self._previous_implementation = getattr(
                self.model.config, "_attn_implementation", None
            )
            self.model.config._attn_implementation = _BACKEND_NAME
            self._token = _ACTIVE_SESSION.set(self)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._token is not None:
            self.model.config._attn_implementation = self._previous_implementation
            _ACTIVE_SESSION.reset(self._token)
            self._token = None

    @contextmanager
    def _active(self, phase: str) -> Iterator[None]:
        previous = self.phase
        self.phase = phase
        with self:
            yield
        self.phase = previous

    def _validate_request(
        self, request: DecodeRequest, input_ids: torch.Tensor
    ) -> None:
        require_direct_decode_support(self.capabilities, request)
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise UnsupportedCapabilityError("direct decode requires batch_size=1.")
        if (
            input_ids.device.type != "cpu"
            and self.cache._device is not None
            and input_ids.device != self.cache._device
        ):
            raise UnsupportedCapabilityError(
                "Qwen2.5 packed decode does not support cache device changes."
            )

    def prefill(
        self,
        prompt_ids: torch.Tensor,
        *,
        request: DecodeRequest = DecodeRequest(),
        use_cache: bool = True,
    ) -> Any:
        self._validate_request(request, prompt_ids)
        if use_cache is not True:
            raise UnsupportedCapabilityError(
                "direct decode requires an enabled KV cache."
            )
        if prompt_ids.shape[1] <= 1:
            raise UnsupportedCapabilityError(
                "direct packed attention requires a standard prompt prefill first."
            )
        with self._active("prefill"):
            result = self.model(
                prompt_ids,
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=1,
            )
        if self.use_parallel_page_attention:
            self._prepare_parallel_decode_state()
        self._last_logits = result.logits
        return result

    def decode_one(
        self,
        token_ids: torch.Tensor | None = None,
        *,
        request: DecodeRequest = DecodeRequest(),
        use_cache: bool = True,
    ) -> Any:
        if token_ids is None:
            if self._last_logits is None:
                raise UnsupportedCapabilityError(
                    "direct packed attention requires a standard prompt prefill first."
                )
            token_ids = self._last_logits[:, -1:].argmax(dim=-1)
        self._validate_request(request, token_ids)
        if use_cache is not True:
            raise UnsupportedCapabilityError(
                "direct decode requires an enabled KV cache."
            )
        if token_ids.shape[1] != 1 or self.cache.sequence_length() < 1:
            raise UnsupportedCapabilityError(
                "direct packed attention requires a standard prompt prefill first."
            )
        with self._active("decode"):
            result = self.model(token_ids, past_key_values=self.cache, use_cache=True)
        self._last_logits = result.logits
        return result

    def _prepared_layer_decode(
        self,
        layer_idx: int,
        store: object,
        query: torch.Tensor,
        *,
        scale: float,
    ) -> _PreparedLayerDecode:
        """Prepare static CUDA inputs only when the layer gains a packed block."""
        from kernels.packed_attention import (
            PackedAttentionBlock,
            PackedAttentionHotPathDebugState,
            prepare_packed_attention_decode,
        )

        num_kv_heads = self.capabilities.num_key_value_heads or 0
        gqa_groups = self.capabilities.num_key_value_groups
        if num_kv_heads < 1 or gqa_groups is None or gqa_groups < 1:
            raise UnsupportedCapabilityError(_UNSUPPORTED_LAYOUT)
        block_count = store.block_count
        cached = self._prepared_layers.get(layer_idx)
        if cached is not None and (
            cached.device == query.device
            and cached.block_count == block_count
            and cached.scale == scale
        ):
            return cached
        if cached is not None and cached.device == query.device:
            rotation = cached.rotation
            key_codebook = cached.key_codebook
            gqa_mapping = cached.gqa_mapping
            debug = cached.debug
        else:
            rotation = store.key_codec._rotation.to(query.device, dtype=torch.float32)
            key_codebook = store.key_codec._centroids.to(
                query.device, dtype=torch.float32
            )
            gqa_mapping = (
                torch.arange(query.shape[1], device=query.device, dtype=torch.int32)
                // gqa_groups
            )
            debug = PackedAttentionHotPathDebugState()
        blocks = tuple(
            PackedAttentionBlock(
                packed_keys=block.keys.indices.reshape(
                    num_kv_heads, block.logical_tokens, -1
                ),
                key_norms=block.keys.norms.reshape(num_kv_heads, block.logical_tokens),
                key_codebook=key_codebook,
                key_bits=store.key_codec.layout.bits,
                packed_values=block.values.indices.reshape(
                    num_kv_heads, block.logical_tokens, -1
                ),
                value_mins=block.values.mins.reshape(
                    num_kv_heads, block.logical_tokens, -1
                ),
                value_scales=block.values.scales.reshape(
                    num_kv_heads, block.logical_tokens, -1
                ),
                value_bits=store.value_codec.layout.bits,
                value_group_size=store.value_codec.layout.group_size,
            )
            for block in store.iter_blocks()
        )
        prepared = prepare_packed_attention_decode(
            blocks,
            rotation,
            gqa_mapping,
            num_kv_heads=num_kv_heads,
            scale=scale,
            debug=debug,
        )
        cached = _PreparedLayerDecode(
            query.device,
            block_count,
            scale,
            rotation,
            key_codebook,
            gqa_mapping,
            prepared,
            debug,
        )
        self._prepared_layers[layer_idx] = cached
        return cached

    def _prepare_parallel_decode_state(self) -> None:
        """Prewarm all CUDA constants and reusable outputs after prefill."""
        if not self.use_parallel_page_attention:
            return
        device = self.cache._device
        if device is None:
            raise UnsupportedCapabilityError(
                "parallel page attention requires a populated cache after prefill."
            )
        if device.type != "cuda":
            # Keep the CPU path fail-closed at decode rather than silently
            # switching to the legacy serial backend.
            return
        if self.cache._dtype not in (torch.float16, torch.bfloat16):
            raise UnsupportedCapabilityError(
                "parallel page attention requires a stable fp16 or bf16 cache dtype."
            )
        for layer_idx, store in enumerate(self.cache._stores):
            if store is not None:
                self._parallel_prepared_codecs[
                    layer_idx
                ] = store.key_codec.prepare_device(device)
        self._parallel_workspace_for(device)

    def _parallel_workspace_for(self, device: torch.device) -> Any:
        from core.parallel_page_workspace import reserve_parallel_page_workspace

        if not self.use_parallel_page_attention or self.maximum_sequence_length is None:
            raise UnsupportedCapabilityError(
                "parallel page attention requires an explicit capacity reservation."
            )
        if device.type != "cuda":
            raise UnsupportedCapabilityError(
                "parallel page attention requires CUDA; the serial path remains available."
            )
        if self._parallel_workspace is not None:
            if self._parallel_workspace_device != device:
                raise UnsupportedCapabilityError(
                    "parallel page workspace device changes are unsupported."
                )
            return self._parallel_workspace
        tail_capacity = self.profile.block_size + self.profile.fp16_window - 1
        page_capacity = max(
            1,
            math.ceil(
                max(0, self.maximum_sequence_length - tail_capacity)
                / self.profile.block_size
            ),
        )
        self._parallel_workspace = reserve_parallel_page_workspace(
            page_capacity=page_capacity,
            device=device,
        )
        self._parallel_workspace_device = device
        # Allocate one model-facing output slot per layer during first-use
        # setup.  Decode launches then only overwrite session-owned buffers;
        # no per-token output allocation is allowed on the fast path.
        output_dtype = self.cache._dtype or torch.float16
        self._parallel_outputs = {
            layer_idx: torch.empty((1, 1, 28, 128), dtype=output_dtype, device=device)
            for layer_idx in range(len(self.layers))
        }
        return self._parallel_workspace

    def _parallel_decode(
        self,
        layer_idx: int,
        store: object,
        query: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        from kernels.qwen25_parallel_page_attention import (
            launch_qwen25_page_reduce_tail_merge,
            launch_qwen25_page_states,
            launch_qwen25_query_rotation,
        )

        if query.shape != (1, 28, 1, 128) or not query.is_contiguous():
            raise UnsupportedCapabilityError(
                "parallel page attention requires contiguous Qwen2.5 query shape [1, 28, 1, 128]."
            )
        arena = store.arena  # type: ignore[attr-defined]
        if arena is None:
            raise UnsupportedCapabilityError(
                "parallel page attention requires an arena-backed cache."
            )
        layout = arena.layout
        if (
            layout.page_tokens != 256
            or layout.num_kv_heads != 4
            or layout.head_dim != 128
            or layout.key_packed_width != 128
            or layout.value_packed_width != 96
            or layout.value_groups != 4
            or store.key_codec.layout.qjl_enabled  # type: ignore[attr-defined]
        ):
            raise UnsupportedCapabilityError(
                "parallel page attention requires the canonical Qwen2.5 packed layout."
            )
        workspace = self._parallel_workspace_for(query.device)
        prepared_codec = self._parallel_prepared_codecs.get(layer_idx)
        if prepared_codec is None:
            raise UnsupportedCapabilityError(
                "parallel decode state must be prewarmed after prompt prefill."
            )
        launch_qwen25_query_rotation(query, prepared_codec.rotation, workspace)
        launch_qwen25_page_states(
            query,
            arena,
            prepared_codec,
            workspace,
            used_pages=arena.used_pages,
            scale=scale,
        )
        output = self._parallel_outputs.get(layer_idx)
        if (
            output is None
            or output.device != query.device
            or output.dtype != query.dtype
        ):
            raise UnsupportedCapabilityError(
                "parallel page output buffers were not reserved for this cache device/dtype."
            )
        launch_qwen25_page_reduce_tail_merge(
            query,
            arena.tail_keys,
            arena.tail_values,
            workspace,
            used_pages=arena.used_pages,
            tail_length=arena.tail_length,
            tail_head_stride=arena.tail_keys.stride(1),
            tail_token_stride=arena.tail_keys.stride(2),
            scale=scale,
            output=output,
            tail_query=query,
        )
        self.metrics.parallel_decode_calls += 1
        self.metrics.parallel_page_launches += 1
        self.metrics.parallel_reduce_tail_launches += 1
        return output

    def packed_decode(
        self, module: object, query: torch.Tensor, *, scale: float | None
    ) -> torch.Tensor:
        from kernels.packed_attention import (
            PackedAttentionBlock,
            PackedAttentionDebugState,
            packed_attention_decode,
        )

        layer_idx = self._layer_ids.get(id(module))
        if layer_idx is None:
            raise UnsupportedCapabilityError(_UNSUPPORTED_LAYOUT)
        store = self.cache.store_for(layer_idx)
        if store.sequence_length() < 1:
            raise UnsupportedCapabilityError(
                "direct packed attention requires a standard prompt prefill first."
            )
        if self.use_packed_reference_backend:
            from core.reference_attention import (
                ReferenceAttentionLayout,
                packed_reference_decode,
            )

            reference = packed_reference_decode(
                query,
                store,
                ReferenceAttentionLayout(
                    head_dim=query.shape[-1],
                    num_attention_heads=query.shape[1],
                    num_key_value_heads=self.capabilities.num_key_value_heads or 0,
                ),
                scale=float(scale) if scale is not None else None,
            )
            self.metrics.packed_reference_decode_calls += 1
            return reference.transpose(1, 2).contiguous()
        attention_scale = (
            float(scale) if scale is not None else 1 / math.sqrt(query.shape[-1])
        )
        if self.use_parallel_page_attention:
            output = self._parallel_decode(
                layer_idx, store, query, scale=attention_scale
            )
            packed_blocks_consumed = store.block_count
            parallel_output = True
        else:
            tail = next(store.iter_fp16_window(), None)
            window = None if tail is None else (tail[0].squeeze(0), tail[1].squeeze(0))
            parallel_output = False
            if query.is_cuda:
                prepared = self._prepared_layer_decode(
                    layer_idx, store, query, scale=attention_scale
                )
                output = prepared.prepared.launch(query.squeeze(0), fp16_window=window)
                packed_blocks_consumed = prepared.block_count
            else:
                blocks = (
                    PackedAttentionBlock(
                        packed_keys=block.keys.indices.reshape(
                            query.shape[1] // self.capabilities.num_key_value_groups,
                            block.logical_tokens,
                            -1,
                        ),
                        key_norms=block.keys.norms.reshape(
                            query.shape[1] // self.capabilities.num_key_value_groups,
                            block.logical_tokens,
                        ),
                        key_codebook=store.key_codec._centroids.to(query.device),
                        key_bits=store.key_codec.layout.bits,
                        packed_values=block.values.indices.reshape(
                            query.shape[1] // self.capabilities.num_key_value_groups,
                            block.logical_tokens,
                            -1,
                        ),
                        value_mins=block.values.mins.reshape(
                            query.shape[1] // self.capabilities.num_key_value_groups,
                            block.logical_tokens,
                            -1,
                        ),
                        value_scales=block.values.scales.reshape(
                            query.shape[1] // self.capabilities.num_key_value_groups,
                            block.logical_tokens,
                            -1,
                        ),
                        value_bits=store.value_codec.layout.bits,
                        value_group_size=store.value_codec.layout.group_size,
                    )
                    for block in store.iter_blocks()
                )
                debug = PackedAttentionDebugState()
                output = packed_attention_decode(
                    query.squeeze(0),
                    blocks,
                    store.key_codec._rotation.to(query.device),
                    torch.arange(query.shape[1], device=query.device, dtype=torch.int32)
                    // self.capabilities.num_key_value_groups,
                    scale=attention_scale,
                    fp16_window=window,
                    debug=debug,
                    allow_torch_fallback=True,
                )
                packed_blocks_consumed = debug.blocks_consumed
        model_output = (
            output
            if parallel_output
            else output.unsqueeze(0).transpose(1, 2).contiguous()
        )
        self.metrics.packed_decode_calls += 1
        self.metrics.packed_blocks_consumed += packed_blocks_consumed
        if self.compare_packed_reference:
            from core.reference_attention import (
                ReferenceAttentionLayout,
                packed_reference_decode,
            )

            reference = packed_reference_decode(
                query,
                store,
                ReferenceAttentionLayout(
                    head_dim=query.shape[-1],
                    num_attention_heads=query.shape[1],
                    num_key_value_heads=self.capabilities.num_key_value_heads or 0,
                ),
                scale=float(scale) if scale is not None else None,
            )
            reference_model = reference.transpose(1, 2).contiguous()
            self.metrics.packed_reference_max_error = float(
                (model_output.float() - reference_model.float()).abs().max().item()
            )
        return model_output


def install_qwen25_packed_backend(
    model: object,
    profile: CompressionProfile,
    revision: ModelCapabilities | object | None = None,
    *,
    maximum_sequence_length: int | None = None,
    use_parallel_page_attention: bool = False,
) -> PackedAttentionSession:
    """Install a session object for the sole supported Qwen2.5 direct profile."""
    try:
        import transformers
        from transformers import AttentionInterface
    except ImportError as exc:  # pragma: no cover - dependency-specific boundary.
        raise UnsupportedCapabilityError(_UNSUPPORTED_LAYOUT) from exc
    if tuple(int(part) for part in transformers.__version__.split(".")[:3]) < (
        4,
        57,
        3,
    ):
        raise UnsupportedCapabilityError(_UNSUPPORTED_LAYOUT)
    if not hasattr(AttentionInterface, "register"):
        raise UnsupportedCapabilityError(_UNSUPPORTED_LAYOUT)
    if not isinstance(profile, CompressionProfile):
        raise TypeError("profile must be a CompressionProfile.")
    if revision is not None and not isinstance(
        revision, (ModelCapabilities, ModelRevision)
    ):
        raise TypeError("revision must be a ModelRevision, ModelCapabilities, or None.")
    if isinstance(revision, ModelRevision) and revision != profile.revision:
        raise UnsupportedCapabilityError(
            "Qwen2.5 packed decode revision does not match the profile."
        )
    capabilities = revision if isinstance(revision, ModelCapabilities) else None
    if capabilities is None:
        capabilities = derive_model_capabilities(
            model.config,
            profile.revision,
            load_model_manifest(profile.model_manifest_path),
        )
    require_direct_decode_support(capabilities, DecodeRequest())
    require_qwen25_packed_layout(capabilities)
    AttentionInterface.register(_BACKEND_NAME, qwen25_packed_attention)
    return PackedAttentionSession(
        model,
        profile,
        capabilities,
        maximum_sequence_length=maximum_sequence_length,
        use_parallel_page_attention=use_parallel_page_attention,
    )
