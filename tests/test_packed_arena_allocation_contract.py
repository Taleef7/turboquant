"""Allocation and copy-volume contracts for the opt-in arena backend."""

from __future__ import annotations

import torch
import pytest

from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packed_page_arena import PackedPageArena, PackedPageArenaLayout
from core.packed_store import PackedKVBlockStore


def _make_store(page_capacity: int = 40) -> tuple[PackedKVBlockStore, PackedPageArena]:
    arena = PackedPageArena.reserve(
        PackedPageArenaLayout(
            page_capacity=page_capacity,
            page_tokens=256,
            num_kv_heads=4,
            head_dim=128,
            key_packed_width=128,
            value_packed_width=96,
            value_groups=4,
            tail_capacity=383,
        ),
        device=torch.device("cpu"),
        kv_dtype=torch.float16,
    )
    return (
        PackedKVBlockStore(
            KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16)),
            ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
            num_kv_heads=4,
            arena=arena,
        ),
        arena,
    )


def test_8k_prefill_copies_only_final_arena_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large prefill must not materialize a prompt-sized combined tensor."""
    store, arena = _make_store()
    storage_pointers = {
        tensor.untyped_storage().data_ptr()
        for tensor in (
            arena.key_indices,
            arena.key_norms,
            arena.value_indices,
            arena.value_mins,
            arena.value_scales,
            arena.tail_keys,
            arena.tail_values,
        )
    }
    copies: list[int] = []
    original_copy = torch.Tensor.copy_

    def track_copy(
        self: torch.Tensor, source: torch.Tensor, *args: object, **kwargs: object
    ):
        if self.untyped_storage().data_ptr() in storage_pointers:
            copies.append(self.numel() * self.element_size())
        return original_copy(self, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", track_copy)
    tokens = 8192
    keys = torch.randn((1, 4, tokens, 128), dtype=torch.float16)
    store.append(keys, keys + 1)

    pages = 31
    tail_tokens = 256
    page_bytes = (
        4 * 256 * 128 + 4 * 256 * 2 + 4 * 256 * 96 + 4 * 256 * 4 * 2 + 4 * 256 * 4 * 2
    )
    tail_bytes = 2 * 4 * tail_tokens * 128 * 2
    assert arena.used_pages == pages
    assert arena.tail_length == tail_tokens
    assert len(copies) == pages * 5 + 2
    assert sum(copies) == pages * page_bytes + tail_bytes
