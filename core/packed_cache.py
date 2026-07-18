"""Narrow packed KV cache for the Qwen2.5 direct-decode adapter.

This is intentionally not a ``DynamicCache`` replacement.  It implements only
the cache calls made by the verified Qwen2.5 attention layout and retains no
full-history fp16 tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from core.errors import UnsupportedCapabilityError
from core.packed_store import PackedKVBlockStore, PackedStoreRuntimeLayout, create_packed_kv_block_store

if TYPE_CHECKING:
    from core.profiles import CompressionProfile


class PackedDynamicCache:
    """One packed block store per verified full-attention Qwen2.5 layer."""

    def __init__(self, profile: "CompressionProfile", *, layer_count: int) -> None:
        self.profile = profile
        self._layer_count = layer_count
        self._stores: list[PackedKVBlockStore | None] = [None] * layer_count
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def _validate_update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> None:
        if type(layer_idx) is not int or not 0 <= layer_idx < self._layer_count:
            raise UnsupportedCapabilityError("unsupported Qwen2.5 attention layer index.")
        expected = (
            1,
            self.profile.model_architecture["num_key_value_heads"],
            self.profile.model_architecture["head_dim"],
        )
        if not isinstance(key_states, torch.Tensor) or not isinstance(value_states, torch.Tensor):
            raise UnsupportedCapabilityError("Qwen2.5 packed decode requires tensor KV states.")
        if (
            key_states.ndim != 4
            or value_states.shape != key_states.shape
            or key_states.shape[0] != expected[0]
            or key_states.shape[1] != expected[1]
            or key_states.shape[-1] != expected[2]
            or key_states.shape[-2] < 1
        ):
            raise UnsupportedCapabilityError("unsupported Qwen2.5 KV tensor layout.")
        if key_states.dtype not in (torch.float16, torch.bfloat16) or value_states.dtype != key_states.dtype:
            raise UnsupportedCapabilityError("Qwen2.5 packed decode requires stable fp16 or bf16 KV dtype.")
        if key_states.device != value_states.device:
            raise UnsupportedCapabilityError("Qwen2.5 packed decode requires key/value on one device.")
        if self._device is not None and key_states.device != self._device:
            raise UnsupportedCapabilityError("Qwen2.5 packed decode does not support cache device changes.")
        if self._dtype is not None and key_states.dtype != self._dtype:
            raise UnsupportedCapabilityError("Qwen2.5 packed decode does not support cache dtype changes.")

    def _new_store(self, key_states: torch.Tensor, layer_idx: int) -> PackedKVBlockStore:
        runtime = PackedStoreRuntimeLayout(
            head_dim=key_states.shape[-1],
            num_kv_heads=key_states.shape[1],
            norm_dtype=key_states.dtype,
        )
        store = create_packed_kv_block_store(self.profile, runtime)
        # A layer-specific rotation is required: matching rotations make layer
        # compression accidentally share an implementation detail.
        store.key_codec._rotation = store.key_codec._rotation.roll(layer_idx, dims=0)
        return store

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode post-RoPE K/V and return only the current local segment.

        The caller's custom attention backend consumes the packed stores; it
        must never treat this return value as a concatenated historical cache.
        """
        _ = cache_kwargs
        self._validate_update(key_states, value_states, layer_idx)
        store = self._stores[layer_idx]
        if store is None:
            store = self._new_store(key_states, layer_idx)
        # All validation happens before this append so failed mode/device/dtype
        # transitions cannot alter packed cache state.
        store.append(key_states, value_states)
        self._stores[layer_idx] = store
        self._device = key_states.device
        self._dtype = key_states.dtype
        return key_states, value_states

    def store_for(self, layer_idx: int) -> PackedKVBlockStore:
        if type(layer_idx) is not int or not 0 <= layer_idx < self._layer_count:
            raise UnsupportedCapabilityError("unsupported Qwen2.5 attention layer index.")
        store = self._stores[layer_idx]
        if store is not None:
            return store
        dtype = self._dtype or torch.float16
        device = self._device or torch.device("cpu")
        seed = torch.empty(
            (1, self.profile.model_architecture["num_key_value_heads"], 1, self.profile.model_architecture["head_dim"]),
            dtype=dtype,
            device=device,
        )
        store = self._new_store(seed, layer_idx)
        self._stores[layer_idx] = store
        return store

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not 0 <= layer_idx < self._layer_count:
            return 0
        store = self._stores[layer_idx]
        return 0 if store is None else store.sequence_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    def sequence_length(self) -> int:
        return self.get_seq_length(0)

