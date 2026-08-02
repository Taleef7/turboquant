"""Append-only, fixed-size storage for physically packed KV cache blocks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import torch

from core.codec import (
    KeyCodec,
    KeyLayout,
    PackedKeyPayload,
    PackedValuePayload,
    ValueCodec,
    ValueLayout,
)
from core.errors import UnsupportedCapabilityError
from core.packed_page_arena import PackedPageArena
from core.packing import tensor_nbytes
from core.profiles import CacheImplementation, CompressionProfile


_BLOCK_SIZE = 256
_WINDOW_DTYPES = frozenset({torch.float16, torch.bfloat16})


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a built-in int.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True)
class PhysicalByteBreakdown:
    """Exact bytes occupied by allocated payload tensors, by storage category."""

    key_payload: int
    value_payload: int
    metadata: int
    fp16_window: int
    padding: int

    def __post_init__(self) -> None:
        for name in (
            "key_payload",
            "value_payload",
            "metadata",
            "fp16_window",
            "padding",
        ):
            _require_int(getattr(self, name), name)

    @property
    def total_nbytes(self) -> int:
        return (
            self.key_payload
            + self.value_payload
            + self.metadata
            + self.fp16_window
            + self.padding
        )

    @property
    def total(self) -> int:
        """Alias retained for simple reporting call sites."""
        return self.total_nbytes


@dataclass(frozen=True)
class PackedKVBlock:
    """One chronological, immutable-size packed key/value interval."""

    start_token: int
    end_token: int
    keys: PackedKeyPayload
    values: PackedValuePayload

    def __post_init__(self) -> None:
        _require_int(self.start_token, "start_token")
        _require_int(self.end_token, "end_token", minimum=1)
        if self.end_token - self.start_token != _BLOCK_SIZE:
            raise ValueError(
                f"PackedKVBlock must contain exactly {_BLOCK_SIZE} tokens."
            )
        if (
            self.keys.logical_tokens != _BLOCK_SIZE
            or self.values.logical_tokens != _BLOCK_SIZE
        ):
            raise ValueError(
                "key and value payloads must both contain one complete block."
            )

    @property
    def logical_tokens(self) -> int:
        return self.end_token - self.start_token

    @property
    def key_payload(self) -> PackedKeyPayload:
        return self.keys

    @property
    def value_payload(self) -> PackedValuePayload:
        return self.values


@dataclass(frozen=True)
class PackedStoreRuntimeLayout:
    """Runtime K/V layout facts that a profile cannot safely infer on its own."""

    head_dim: int
    num_kv_heads: int
    norm_dtype: torch.dtype
    qjl_enabled: bool = False
    qjl_dim: int | None = None

    def __post_init__(self) -> None:
        _require_int(self.head_dim, "head_dim", minimum=3)
        _require_int(self.num_kv_heads, "num_kv_heads", minimum=1)
        if self.norm_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("norm_dtype must be fp16, bf16, or fp32.")
        if not isinstance(self.qjl_enabled, bool):
            raise TypeError("qjl_enabled must be a bool.")
        if self.qjl_dim is not None:
            _require_int(self.qjl_dim, "qjl_dim", minimum=1)


class PackedKVBlockStore:
    """Own chronological packed blocks plus one bounded, unencoded tail.

    Only the pending tail is ever concatenated. Historical blocks are encoded once
    and are neither replaced nor concatenated during later appends.
    """

    def __init__(
        self,
        key_codec: KeyCodec,
        value_codec: ValueCodec,
        *,
        block_size: int = _BLOCK_SIZE,
        fp16_window: int = 0,
        num_kv_heads: int | None = None,
        arena: PackedPageArena | None = None,
    ) -> None:
        if not isinstance(key_codec, KeyCodec):
            raise TypeError("key_codec must be a KeyCodec.")
        if not isinstance(value_codec, ValueCodec):
            raise TypeError("value_codec must be a ValueCodec.")
        if key_codec.layout.head_dim != value_codec.layout.head_dim:
            raise ValueError("key and value codecs must have the same head_dim.")
        if _require_int(block_size, "block_size", minimum=1) != _BLOCK_SIZE:
            raise ValueError(f"block_size is fixed at {_BLOCK_SIZE}.")
        fp16_window = _require_int(fp16_window, "fp16_window")
        if fp16_window > _BLOCK_SIZE:
            raise ValueError(f"fp16_window cannot exceed {_BLOCK_SIZE}.")
        if num_kv_heads is not None:
            _require_int(num_kv_heads, "num_kv_heads", minimum=1)
        if arena is not None and not isinstance(arena, PackedPageArena):
            raise TypeError("arena must be a PackedPageArena or None.")
        self._key_codec = key_codec
        self._value_codec = value_codec
        self._fp16_window = fp16_window
        self._num_kv_heads = num_kv_heads
        self._arena = arena
        if arena is not None:
            self._validate_arena_layout(arena)
        self._blocks: list[PackedKVBlock] = []
        self._pending_keys: torch.Tensor | None = None
        self._pending_values: torch.Tensor | None = None
        self._sequence_length = 0

    def _validate_arena_layout(self, arena: PackedPageArena) -> None:
        if self._key_codec.layout.qjl_enabled:
            raise UnsupportedCapabilityError(
                "arena-backed storage does not support QJL key payloads yet."
            )
        layout = arena.layout
        if layout.page_tokens != self.block_size:
            raise ValueError("arena page_tokens must match the fixed block size.")
        if layout.head_dim != self._key_codec.layout.head_dim:
            raise ValueError("arena head_dim must match the codec layout.")
        if self._num_kv_heads is not None and layout.num_kv_heads != self._num_kv_heads:
            raise ValueError("arena num_kv_heads must match the configured head count.")
        expected_key_width = math.ceil(layout.head_dim * self._key_codec.mse_bits / 8)
        expected_value_width = math.ceil(
            layout.head_dim * self._value_codec.layout.bits / 8
        )
        expected_groups = layout.head_dim // self._value_codec.layout.group_size
        if layout.key_packed_width != expected_key_width:
            raise ValueError("arena key_packed_width does not match the key codec.")
        if layout.value_packed_width != expected_value_width:
            raise ValueError("arena value_packed_width does not match the value codec.")
        if layout.value_groups != expected_groups:
            raise ValueError("arena value_groups does not match the value codec.")
        if self._value_codec.layout.metadata_dtype != arena.kv_dtype:
            raise TypeError("arena kv_dtype must match value metadata dtype.")

    def _validate_append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        head_dim = self._key_codec.layout.head_dim
        for label, tensor in (("keys", keys), ("values", values)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{label} must be a torch.Tensor.")
            if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] != head_dim:
                raise ValueError(
                    f"{label} must have shape [1, num_kv_heads, tokens, {head_dim}], got {tuple(tensor.shape)}."
                )
            if tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
                raise ValueError(f"{label} must contain at least one head and token.")
            if tensor.dtype not in _WINDOW_DTYPES:
                raise TypeError(f"{label} must use fp16 or bf16 to preserve the tail.")
        if keys.shape != values.shape:
            raise ValueError("keys and values must have identical shapes.")
        if self._num_kv_heads is not None and keys.shape[1] != self._num_kv_heads:
            raise ValueError("keys and values must match the configured num_kv_heads.")
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError("keys and values must have the same device and dtype.")
        if self._arena is not None:
            if keys.device != self._arena.device:
                raise ValueError(
                    "arena-backed appends must stay on the reserved device."
                )
            if keys.dtype != self._arena.kv_dtype:
                raise TypeError("arena-backed appends must stay in the reserved dtype.")
        if self._pending_keys is not None and (
            self._pending_keys.shape[1] != keys.shape[1]
            or self._pending_keys.device != keys.device
            or self._pending_keys.dtype != keys.dtype
        ):
            raise ValueError(
                "all appends must retain key/value head count, device, and dtype."
            )

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append K/V tokens, encoding only complete chronological 256-token blocks."""
        self._validate_append(keys, values)
        if self._arena is not None:
            self._append_arena(keys, values)
            return
        if self._pending_keys is None:
            self._pending_keys = keys.clone()
            self._pending_values = values.clone()
        else:
            # Allocate only the single pending tail plus this incoming append.
            # Historical packed blocks are never joined or copied here.
            self._pending_keys = self._extend_pending(self._pending_keys, keys)
            self._pending_values = self._extend_pending(self._pending_values, values)
        self._sequence_length += keys.shape[-2]

        while self._pending_keys.shape[-2] >= _BLOCK_SIZE + self._fp16_window:
            block_keys = self._pending_keys[..., :_BLOCK_SIZE, :]
            block_values = self._pending_values[..., :_BLOCK_SIZE, :]
            start = self._sequence_length - self._pending_keys.shape[-2]
            self._blocks.append(
                PackedKVBlock(
                    start_token=start,
                    end_token=start + _BLOCK_SIZE,
                    keys=self._key_codec.encode(block_keys),
                    values=self._value_codec.encode(block_values),
                )
            )
            self._pending_keys = self._pending_keys[..., _BLOCK_SIZE:, :].clone()
            self._pending_values = self._pending_values[..., _BLOCK_SIZE:, :].clone()

    def _append_arena(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append through reserved storage while keeping the legacy path intact."""
        assert self._arena is not None
        old_tail_keys, old_tail_values = self._arena.tail_views()
        old_tail_tokens = self._arena.tail_length
        incoming_tokens = keys.shape[-2]
        total_pending = old_tail_tokens + incoming_tokens
        tail_capacity = self._arena.layout.tail_capacity
        pages_to_flush = max(
            0,
            math.ceil((total_pending - tail_capacity) / self.block_size),
        )
        if self._arena.used_pages + pages_to_flush > self._arena.page_capacity:
            raise OverflowError(
                "packed page arena capacity is exhausted before append."
            )
        if pages_to_flush and total_pending < pages_to_flush * self.block_size:
            raise OverflowError(
                "arena tail capacity cannot retain a partial page before the next flush."
            )
        if pages_to_flush == 0:
            self._arena.append_tail(keys, values)
            self._sequence_length = self._arena.sequence_length
            return

        combined_keys = torch.empty(
            (1, keys.shape[1], total_pending, keys.shape[-1]),
            dtype=keys.dtype,
            device=keys.device,
        )
        combined_values = torch.empty_like(combined_keys)
        if old_tail_tokens:
            combined_keys[..., :old_tail_tokens, :].copy_(old_tail_keys)
            combined_values[..., :old_tail_tokens, :].copy_(old_tail_values)
        combined_keys[..., old_tail_tokens:, :].copy_(keys)
        combined_values[..., old_tail_tokens:, :].copy_(values)

        # Prepare the immutable key constants once for this device.  The old
        # payload API retains an unpack closure for reference attention; the
        # arena path only needs raw arrays and should not build those closures
        # for every historical page.
        prepared = self._key_codec.prepare_device(keys.device)
        encoded: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        for page_id in range(pages_to_flush):
            start = page_id * self.block_size
            stop = start + self.block_size
            (
                key_indices,
                key_norms,
                key_signs,
                key_gamma,
            ) = self._key_codec._encode_arrays(
                combined_keys[..., start:stop, :], prepared=prepared
            )
            if key_signs is not None or key_gamma is not None:
                raise UnsupportedCapabilityError(
                    "arena-backed storage does not support QJL key payloads yet."
                )
            value_indices, value_mins, value_scales = self._value_codec._encode_arrays(
                combined_values[..., start:stop, :]
            )
            encoded.append(
                (key_indices, key_norms, value_indices, value_mins, value_scales)
            )
        self._arena.clear_tail()
        for key_indices, key_norms, value_indices, value_mins, value_scales in encoded:
            self._arena.append_encoded_page(
                key_indices=key_indices,
                key_norms=key_norms,
                value_indices=value_indices,
                value_mins=value_mins,
                value_scales=value_scales,
            )
        remainder_start = pages_to_flush * self.block_size
        remainder_keys = combined_keys[..., remainder_start:, :]
        remainder_values = combined_values[..., remainder_start:, :]
        if remainder_keys.shape[-2]:
            self._arena.append_tail(remainder_keys, remainder_values)
        self._sequence_length = self._arena.sequence_length

    @staticmethod
    def _extend_pending(existing: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
        result = torch.empty(
            (
                *existing.shape[:-2],
                existing.shape[-2] + incoming.shape[-2],
                existing.shape[-1],
            ),
            dtype=existing.dtype,
            device=existing.device,
        )
        result[..., : existing.shape[-2], :].copy_(existing)
        result[..., existing.shape[-2] :, :].copy_(incoming)
        return result

    def iter_blocks(self) -> Iterator[PackedKVBlock]:
        """Iterate chronological compressed blocks without reconstructing history."""
        if self._arena is not None:
            raise UnsupportedCapabilityError(
                "arena-backed storage exposes page_views instead of payload closures."
            )
        yield from self._blocks

    @property
    def block_count(self) -> int:
        """Number of immutable packed blocks, without materializing an iterator."""
        return self._arena.used_pages if self._arena is not None else len(self._blocks)

    @property
    def arena(self) -> PackedPageArena | None:
        """Return the opt-in fixed-capacity arena, if this store owns one."""
        return self._arena

    def iter_arena_pages(self) -> Iterator[dict[str, torch.Tensor]]:
        if self._arena is None:
            raise UnsupportedCapabilityError("store has no arena-backed pages.")
        yield from self._arena.iter_page_views()

    @property
    def key_codec(self) -> KeyCodec:
        """Codec metadata required by the CPU-only reference attention path."""
        return self._key_codec

    @property
    def value_codec(self) -> ValueCodec:
        """Value codec metadata required by the CPU-only reference path."""
        return self._value_codec

    @property
    def block_size(self) -> int:
        return _BLOCK_SIZE

    @property
    def num_kv_heads(self) -> int | None:
        return self._num_kv_heads

    def iter_fp16_window(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield the single unencoded tail (alignment tail plus recency window)."""
        if self._arena is not None:
            if self._arena.tail_length:
                yield self._arena.tail_views()
            return
        if self._pending_keys is not None:
            assert self._pending_values is not None
            yield self._pending_keys, self._pending_values

    def sequence_length(self) -> int:
        return (
            self._arena.sequence_length
            if self._arena is not None
            else self._sequence_length
        )

    def physical_nbytes(self) -> PhysicalByteBreakdown:
        if self._arena is not None:
            pages = self._arena.used_pages
            page = self._arena.page_views(0) if pages else None
            key_payload = 0
            value_payload = 0
            metadata = 0
            if page is not None:
                key_payload = pages * (
                    page["key_indices"].numel() * page["key_indices"].element_size()
                )
                value_payload = pages * (
                    page["value_indices"].numel() * page["value_indices"].element_size()
                )
                metadata = pages * sum(
                    page[name].numel() * page[name].element_size()
                    for name in ("key_norms", "value_mins", "value_scales")
                )
            tail = self._arena.tail_views()
            window_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in tail
            )
            return PhysicalByteBreakdown(
                key_payload=key_payload,
                value_payload=value_payload,
                metadata=metadata,
                fp16_window=window_bytes,
                padding=0,
            )
        key_payload = 0
        value_payload = 0
        metadata = 0
        for block in self._blocks:
            key_payload += tensor_nbytes(block.keys.indices)
            value_payload += tensor_nbytes(block.values.indices)
            metadata += tensor_nbytes(block.keys.norms)
            metadata += tensor_nbytes(block.values.mins) + tensor_nbytes(
                block.values.scales
            )
            if block.keys.qjl_signs is not None:
                key_payload += tensor_nbytes(block.keys.qjl_signs)
            if block.keys.qjl_gamma is not None:
                metadata += tensor_nbytes(block.keys.qjl_gamma)
        window_bytes = 0
        if self._pending_keys is not None:
            assert self._pending_values is not None
            window_bytes = tensor_nbytes(self._pending_keys) + tensor_nbytes(
                self._pending_values
            )
        return PhysicalByteBreakdown(
            key_payload=key_payload,
            value_payload=value_payload,
            metadata=metadata,
            fp16_window=window_bytes,
            padding=0,
        )


def create_packed_kv_block_store(
    profile: CompressionProfile,
    runtime: PackedStoreRuntimeLayout,
    *,
    arena: PackedPageArena | None = None,
) -> PackedKVBlockStore:
    """Build the only supported packed-store backend from validated profile data."""
    if not isinstance(profile, CompressionProfile):
        raise TypeError("profile must be a CompressionProfile.")
    if not isinstance(runtime, PackedStoreRuntimeLayout):
        raise TypeError("runtime must be a PackedStoreRuntimeLayout.")
    if profile.cache_implementation is not CacheImplementation.PACKED_KV_BLOCK_STORE:
        raise UnsupportedCapabilityError(
            f"cache_implementation {profile.cache_implementation!r} is not supported."
        )
    profile_head_dim = profile.model_architecture.get("head_dim")
    profile_kv_heads = profile.model_architecture.get("num_key_value_heads")
    if runtime.head_dim != profile_head_dim:
        raise UnsupportedCapabilityError(
            "runtime head_dim does not match the profile binding."
        )
    if runtime.num_kv_heads != profile_kv_heads:
        raise UnsupportedCapabilityError(
            "runtime num_kv_heads does not match the profile binding."
        )
    return PackedKVBlockStore(
        KeyCodec(
            KeyLayout(
                head_dim=runtime.head_dim,
                bits=profile.key_bits,
                norm_dtype=runtime.norm_dtype,
                qjl_enabled=runtime.qjl_enabled,
            ),
            qjl_dim=runtime.qjl_dim,
        ),
        ValueCodec(
            ValueLayout(
                head_dim=runtime.head_dim,
                bits=profile.value_bits,
                group_size=profile.value_group_size,
                metadata_dtype=runtime.norm_dtype
                if arena is not None
                else torch.float16,
            )
        ),
        block_size=profile.block_size,
        fp16_window=profile.fp16_window,
        num_kv_heads=runtime.num_kv_heads,
        arena=arena,
    )
