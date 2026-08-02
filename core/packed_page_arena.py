"""Fixed-capacity storage for packed KV pages and a bounded raw tail.

The arena owns every historical payload allocation.  Appends only copy into
reserved slots, so page and tail base pointers remain stable for the lifetime
of a session.  Model adapters opt into this class explicitly; the legacy
``PackedKVBlockStore`` path remains unchanged by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import sys
from typing import Iterator

import torch


_SUPPORTED_KV_DTYPES = frozenset({torch.float16, torch.bfloat16})
_MAX_ALLOCATION_BYTES = sys.maxsize


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int:  # noqa: E721
        raise TypeError(f"{name} must be a built-in int.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _checked_bytes(shape: tuple[int, ...], element_size: int, label: str) -> int:
    try:
        elements = math.prod(shape)
        bytes_required = elements * element_size
    except (OverflowError, ValueError) as exc:
        raise OverflowError(f"{label} allocation size overflowed.") from exc
    if bytes_required > _MAX_ALLOCATION_BYTES:
        raise OverflowError(f"{label} allocation exceeds the supported byte limit.")
    return bytes_required


def _dtype_itemsize(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    raise TypeError(f"unsupported arena dtype: {dtype}")


@dataclass(frozen=True)
class PackedPageArenaLayout:
    """Static dimensions for one fixed-capacity packed-page arena."""

    page_capacity: int
    page_tokens: int
    num_kv_heads: int
    head_dim: int
    key_packed_width: int
    value_packed_width: int
    value_groups: int
    tail_capacity: int

    def __post_init__(self) -> None:
        _require_int(self.page_capacity, "page_capacity", minimum=1)
        _require_int(self.page_tokens, "page_tokens", minimum=1)
        _require_int(self.num_kv_heads, "num_kv_heads", minimum=1)
        _require_int(self.head_dim, "head_dim", minimum=1)
        _require_int(self.key_packed_width, "key_packed_width", minimum=1)
        _require_int(self.value_packed_width, "value_packed_width", minimum=1)
        _require_int(self.value_groups, "value_groups", minimum=1)
        _require_int(self.tail_capacity, "tail_capacity", minimum=0)


@dataclass(frozen=True)
class PackedArenaByteBreakdown:
    """Logical, reserved, and runtime bytes attributed to an arena."""

    logical_payload: int
    arena_capacity: int
    tail_logical: int
    tail_capacity: int
    shared_constants: int
    workspace: int
    padding: int
    unique_resident: int

    def __post_init__(self) -> None:
        for name in (
            "logical_payload",
            "arena_capacity",
            "tail_logical",
            "tail_capacity",
            "shared_constants",
            "workspace",
            "padding",
            "unique_resident",
        ):
            _require_int(getattr(self, name), name, minimum=0)

    @property
    def total_nbytes(self) -> int:
        return (
            self.arena_capacity
            + self.tail_capacity
            + self.shared_constants
            + self.workspace
            + self.padding
        )

    @property
    def total(self) -> int:
        return self.total_nbytes


class PackedPageArena:
    """Preallocated packed pages plus one fixed-stride raw K/V tail."""

    def __init__(
        self,
        layout: PackedPageArenaLayout,
        *,
        device: torch.device,
        kv_dtype: torch.dtype,
        key_indices: torch.Tensor,
        key_norms: torch.Tensor,
        value_indices: torch.Tensor,
        value_mins: torch.Tensor,
        value_scales: torch.Tensor,
        tail_keys: torch.Tensor,
        tail_values: torch.Tensor,
    ) -> None:
        self.layout = layout
        self.device = device
        self.kv_dtype = kv_dtype
        self.key_indices = key_indices
        self.key_norms = key_norms
        self.value_indices = value_indices
        self.value_mins = value_mins
        self.value_scales = value_scales
        self.tail_keys = tail_keys
        self.tail_values = tail_values
        self._used_pages = 0
        self._tail_length = 0
        self._pending_key_page: int | None = None

    @classmethod
    def reserve(
        cls,
        layout: PackedPageArenaLayout,
        *,
        device: torch.device,
        kv_dtype: torch.dtype,
    ) -> "PackedPageArena":
        """Reserve all page and tail tensors before any token is appended."""
        if not isinstance(layout, PackedPageArenaLayout):
            raise TypeError("layout must be a PackedPageArenaLayout.")
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device.")
        if kv_dtype not in _SUPPORTED_KV_DTYPES:
            raise TypeError("kv_dtype must be torch.float16 or torch.bfloat16.")

        page = layout.page_capacity
        heads = layout.num_kv_heads
        tokens = layout.page_tokens
        dim = layout.head_dim
        key_width = layout.key_packed_width
        value_width = layout.value_packed_width
        groups = layout.value_groups
        tail = layout.tail_capacity
        # Check every allocation before calling torch.empty so a malformed
        # capacity cannot leave a partially reserved arena behind.
        kv_itemsize = _dtype_itemsize(kv_dtype)
        _checked_bytes((page, heads, tokens, key_width), 1, "key_indices")
        _checked_bytes((page, heads, tokens), kv_itemsize, "key_norms")
        _checked_bytes((page, heads, tokens, value_width), 1, "value_indices")
        _checked_bytes((page, heads, tokens, groups), kv_itemsize, "value_mins")
        _checked_bytes((page, heads, tokens, groups), kv_itemsize, "value_scales")
        _checked_bytes((1, heads, tail, dim), kv_itemsize, "tail_keys")
        _checked_bytes((1, heads, tail, dim), kv_itemsize, "tail_values")

        return cls(
            layout,
            device=device,
            kv_dtype=kv_dtype,
            key_indices=torch.empty(
                (page, heads, tokens, key_width), dtype=torch.uint8, device=device
            ),
            key_norms=torch.empty((page, heads, tokens), dtype=kv_dtype, device=device),
            value_indices=torch.empty(
                (page, heads, tokens, value_width), dtype=torch.uint8, device=device
            ),
            value_mins=torch.empty(
                (page, heads, tokens, groups), dtype=kv_dtype, device=device
            ),
            value_scales=torch.empty(
                (page, heads, tokens, groups), dtype=kv_dtype, device=device
            ),
            tail_keys=torch.empty((1, heads, tail, dim), dtype=kv_dtype, device=device),
            tail_values=torch.empty(
                (1, heads, tail, dim), dtype=kv_dtype, device=device
            ),
        )

    @property
    def used_pages(self) -> int:
        return self._used_pages

    @property
    def tail_length(self) -> int:
        return self._tail_length

    @property
    def sequence_length(self) -> int:
        return self._used_pages * self.layout.page_tokens + self._tail_length

    @property
    def page_capacity(self) -> int:
        return self.layout.page_capacity

    def _page_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        *,
        width: int | None,
        dtype: torch.dtype,
        rank: int,
    ) -> torch.Tensor:
        heads = self.layout.num_kv_heads
        tokens = self.layout.page_tokens
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        expected = (heads, tokens) if rank == 2 else (heads, tokens, width)
        flattened = (heads * tokens,) if rank == 2 else (heads * tokens, width)
        if tensor.shape == expected:
            view = tensor
        elif tensor.shape == flattened:
            view = tensor.reshape(expected)
        else:
            raise ValueError(
                f"{name} must have shape {expected} or {flattened}, got {tuple(tensor.shape)}."
            )
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}.")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must use {dtype}.")
        return view

    def append_encoded_page(
        self,
        *,
        key_indices: torch.Tensor,
        key_norms: torch.Tensor,
        value_indices: torch.Tensor,
        value_mins: torch.Tensor,
        value_scales: torch.Tensor,
        page_id: int | None = None,
    ) -> int:
        """Copy one complete encoded page into the next reserved slot."""
        target = self._used_pages if page_id is None else page_id
        if type(target) is not int:  # noqa: E721
            raise TypeError("page_id must be a built-in int.")
        if target != self._used_pages:
            raise ValueError("pages must be appended in chronological order.")
        if self._pending_key_page is not None:
            raise RuntimeError(
                "complete the pending key page before appending another page."
            )
        if target >= self.layout.page_capacity:
            raise OverflowError("packed page arena capacity is exhausted.")
        key_indices_view = self._page_tensor(
            key_indices,
            "key_indices",
            width=self.layout.key_packed_width,
            dtype=torch.uint8,
            rank=3,
        )
        key_norms_view = self._page_tensor(
            key_norms, "key_norms", width=None, dtype=self.kv_dtype, rank=2
        )
        value_indices_view = self._page_tensor(
            value_indices,
            "value_indices",
            width=self.layout.value_packed_width,
            dtype=torch.uint8,
            rank=3,
        )
        value_mins_view = self._page_tensor(
            value_mins,
            "value_mins",
            width=self.layout.value_groups,
            dtype=self.kv_dtype,
            rank=3,
        )
        value_scales_view = self._page_tensor(
            value_scales,
            "value_scales",
            width=self.layout.value_groups,
            dtype=self.kv_dtype,
            rank=3,
        )
        # All shape/device/dtype/capacity checks happen before the first copy.
        self.key_indices[target].copy_(key_indices_view)
        self.key_norms[target].copy_(key_norms_view)
        self.value_indices[target].copy_(value_indices_view)
        self.value_mins[target].copy_(value_mins_view)
        self.value_scales[target].copy_(value_scales_view)
        self._used_pages += 1
        return target

    def append_key_page(
        self,
        *,
        key_indices: torch.Tensor,
        key_norms: torch.Tensor,
        page_id: int | None = None,
    ) -> int:
        """Write key arrays while keeping the page transactionally invisible.

        This is intentionally a small primitive for codecs that can write keys
        directly into the arena.  ``used_pages`` advances only after
        :meth:`write_value_page` validates and copies all value arrays, so a
        failed value encode cannot expose a partially initialized page.
        """
        target = self._used_pages if page_id is None else page_id
        if type(target) is not int:  # noqa: E721
            raise TypeError("page_id must be a built-in int.")
        if target != self._used_pages:
            raise ValueError("pages must be appended in chronological order.")
        if self._pending_key_page is not None:
            raise RuntimeError("a key page is already pending value completion.")
        if target >= self.layout.page_capacity:
            raise OverflowError("packed page arena capacity is exhausted.")
        key_indices_view = self._page_tensor(
            key_indices,
            "key_indices",
            width=self.layout.key_packed_width,
            dtype=torch.uint8,
            rank=3,
        )
        key_norms_view = self._page_tensor(
            key_norms, "key_norms", width=None, dtype=self.kv_dtype, rank=2
        )
        self.key_indices[target].copy_(key_indices_view)
        self.key_norms[target].copy_(key_norms_view)
        self._pending_key_page = target
        return target

    def write_value_page(
        self,
        *,
        value_indices: torch.Tensor,
        value_mins: torch.Tensor,
        value_scales: torch.Tensor,
        page_id: int,
    ) -> None:
        """Fill value arrays for an already appended key page."""
        if type(page_id) is not int:  # noqa: E721
            raise TypeError("page_id must be a built-in int.")
        if self._pending_key_page != page_id:
            raise ValueError("value arrays must complete the pending key page.")
        value_indices_view = self._page_tensor(
            value_indices,
            "value_indices",
            width=self.layout.value_packed_width,
            dtype=torch.uint8,
            rank=3,
        )
        value_mins_view = self._page_tensor(
            value_mins,
            "value_mins",
            width=self.layout.value_groups,
            dtype=self.kv_dtype,
            rank=3,
        )
        value_scales_view = self._page_tensor(
            value_scales,
            "value_scales",
            width=self.layout.value_groups,
            dtype=self.kv_dtype,
            rank=3,
        )
        self.value_indices[page_id].copy_(value_indices_view)
        self.value_mins[page_id].copy_(value_mins_view)
        self.value_scales[page_id].copy_(value_scales_view)
        self._pending_key_page = None
        self._used_pages += 1

    def append_tail(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Copy raw K/V tokens into the next fixed-stride tail slots."""
        if self._pending_key_page is not None:
            raise RuntimeError("complete the pending key page before appending a tail.")
        expected_prefix = (1, self.layout.num_kv_heads)
        expected_suffix = (self.layout.head_dim,)
        for name, tensor in (("keys", keys), ("values", values)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if (
                tensor.ndim != 4
                or tuple(tensor.shape[:2]) != expected_prefix
                or tensor.shape[-1] != expected_suffix[0]
                or tensor.shape[-2] < 1
            ):
                raise ValueError(
                    f"{name} must have shape [1, {self.layout.num_kv_heads}, tokens, {self.layout.head_dim}]."
                )
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on {self.device}.")
            if tensor.dtype != self.kv_dtype:
                raise TypeError(f"{name} must use {self.kv_dtype}.")
        if keys.shape != values.shape:
            raise ValueError("keys and values must have identical shapes.")
        tokens = keys.shape[-2]
        if self._tail_length + tokens > self.layout.tail_capacity:
            raise OverflowError("raw-tail arena capacity is exhausted.")
        start, stop = self._tail_length, self._tail_length + tokens
        self.tail_keys[..., start:stop, :].copy_(keys)
        self.tail_values[..., start:stop, :].copy_(values)
        self._tail_length = stop

    def clear_tail(self) -> None:
        if self._pending_key_page is not None:
            raise RuntimeError(
                "complete the pending key page before clearing the tail."
            )
        self._tail_length = 0

    def tail_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tail_keys[..., : self._tail_length, :], self.tail_values[
            ..., : self._tail_length, :
        ]

    def page_views(self, page_id: int) -> dict[str, torch.Tensor]:
        if type(page_id) is not int:  # noqa: E721
            raise TypeError("page_id must be a built-in int.")
        if not 0 <= page_id < self._used_pages:
            raise IndexError("page_id is outside the used page range.")
        return {
            "key_indices": self.key_indices[page_id],
            "key_norms": self.key_norms[page_id],
            "value_indices": self.value_indices[page_id],
            "value_mins": self.value_mins[page_id],
            "value_scales": self.value_scales[page_id],
        }

    def iter_page_views(self) -> Iterator[dict[str, torch.Tensor]]:
        for page_id in range(self._used_pages):
            yield self.page_views(page_id)

    def data_pointers(self) -> dict[str, int]:
        return {
            name: getattr(self, name).data_ptr()
            for name in (
                "key_indices",
                "key_norms",
                "value_indices",
                "value_mins",
                "value_scales",
                "tail_keys",
                "tail_values",
            )
        }

    def _page_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.key_indices[0],
                self.key_norms[0],
                self.value_indices[0],
                self.value_mins[0],
                self.value_scales[0],
            )
        )

    def _tail_capacity_bytes(self) -> int:
        return (
            self.tail_keys.numel() * self.tail_keys.element_size()
            + self.tail_values.numel() * self.tail_values.element_size()
        )

    def _unique_storage_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in (
            self.key_indices,
            self.key_norms,
            self.value_indices,
            self.value_mins,
            self.value_scales,
            self.tail_keys,
            self.tail_values,
        ):
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                total += storage.nbytes()
        return total

    def byte_breakdown(
        self,
        *,
        shared_constants: int = 0,
        workspace: int = 0,
        padding: int = 0,
    ) -> PackedArenaByteBreakdown:
        for name, value in (
            ("shared_constants", shared_constants),
            ("workspace", workspace),
            ("padding", padding),
        ):
            _require_int(value, name, minimum=0)
        page_bytes = self._page_bytes()
        tail_capacity_bytes = self._tail_capacity_bytes()
        tail_logical = (
            self._tail_length
            * self.layout.num_kv_heads
            * self.layout.head_dim
            * 2
            * self.kv_dtype.itemsize
        )
        unique_resident = (
            self._unique_storage_bytes() + shared_constants + workspace + padding
        )
        return PackedArenaByteBreakdown(
            logical_payload=self._used_pages * page_bytes,
            arena_capacity=self.layout.page_capacity * page_bytes,
            tail_logical=tail_logical,
            tail_capacity=tail_capacity_bytes,
            shared_constants=shared_constants,
            workspace=workspace,
            padding=padding,
            unique_resident=unique_resident,
        )

    def storage_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self._used_pages}:{self._tail_length}".encode())
        for tensor in (
            self.key_indices,
            self.key_norms,
            self.value_indices,
            self.value_mins,
            self.value_scales,
            self.tail_keys,
            self.tail_values,
        ):
            digest.update(
                tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
        return digest.hexdigest()

    def snapshot_state(self) -> tuple[int, int, str]:
        return self._used_pages, self._tail_length, self.storage_fingerprint()
