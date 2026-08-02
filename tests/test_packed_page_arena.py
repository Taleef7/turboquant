"""Fixed-capacity packed-page arena and opt-in store contracts."""

from __future__ import annotations

import math

import pytest
import torch

from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packed_page_arena import (
    PackedArenaByteBreakdown,
    PackedPageArena,
    PackedPageArenaLayout,
)
from core.packed_store import PackedKVBlockStore
from core.profiles import load_profile


def make_layout(*, pages: int = 3, tail: int = 383) -> PackedPageArenaLayout:
    return PackedPageArenaLayout(
        page_capacity=pages,
        page_tokens=256,
        num_kv_heads=4,
        head_dim=128,
        key_packed_width=128,
        value_packed_width=96,
        value_groups=4,
        tail_capacity=tail,
    )


def _payload(page_count: int = 1) -> dict[str, torch.Tensor]:
    return {
        "key_indices": torch.randint(
            0, 256, (4, page_count * 256, 128), dtype=torch.uint8
        ),
        "key_norms": torch.randn((4, page_count * 256), dtype=torch.float16),
        "value_indices": torch.randint(
            0, 64, (4, page_count * 256, 96), dtype=torch.uint8
        ),
        "value_mins": torch.randn((4, page_count * 256, 4), dtype=torch.float16),
        "value_scales": torch.rand((4, page_count * 256, 4), dtype=torch.float16),
    }


def test_reserve_has_exact_payload_and_tail_shapes_and_dtypes() -> None:
    arena = PackedPageArena.reserve(
        make_layout(), device=torch.device("cpu"), kv_dtype=torch.float16
    )

    assert arena.key_indices.shape == (3, 4, 256, 128)
    assert arena.key_norms.shape == (3, 4, 256)
    assert arena.value_indices.shape == (3, 4, 256, 96)
    assert arena.value_mins.shape == (3, 4, 256, 4)
    assert arena.value_scales.shape == (3, 4, 256, 4)
    assert arena.tail_keys.shape == (1, 4, 383, 128)
    assert arena.tail_values.shape == (1, 4, 383, 128)
    assert arena.key_indices.dtype == torch.uint8
    assert arena.value_indices.dtype == torch.uint8
    assert all(
        tensor.dtype == torch.float16
        for tensor in (
            arena.key_norms,
            arena.value_mins,
            arena.value_scales,
            arena.tail_keys,
            arena.tail_values,
        )
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("page_capacity", 0),
        ("page_capacity", -1),
        ("page_tokens", 0),
        ("num_kv_heads", 0),
        ("head_dim", 0),
        ("key_packed_width", 0),
        ("value_packed_width", 0),
        ("value_groups", 0),
        ("tail_capacity", -1),
    ),
)
def test_layout_rejects_invalid_capacity_and_dimensions(field: str, value: int) -> None:
    values = make_layout().__dict__
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        PackedPageArenaLayout(**values)


def test_reserve_rejects_non_builtin_int_device_and_dtype() -> None:
    values = make_layout().__dict__
    values["page_capacity"] = True
    with pytest.raises(TypeError):
        PackedPageArenaLayout(**values)
    with pytest.raises(TypeError):
        PackedPageArena.reserve(make_layout(), device="cpu", kv_dtype=torch.float16)
    with pytest.raises(TypeError):
        PackedPageArena.reserve(
            make_layout(), device=torch.device("cpu"), kv_dtype=torch.float32
        )


def test_reserve_rejects_shape_byte_overflow_before_allocation() -> None:
    values = make_layout(pages=2**62).__dict__
    with pytest.raises(OverflowError):
        PackedPageArena.reserve(
            PackedPageArenaLayout(**values),
            device=torch.device("cpu"),
            kv_dtype=torch.float16,
        )


def test_byte_breakdown_matches_exact_capacity_and_logical_formulas() -> None:
    arena = PackedPageArena.reserve(
        make_layout(), device=torch.device("cpu"), kv_dtype=torch.float16
    )
    arena.append_encoded_page(**_payload())
    keys = torch.randn((1, 4, 17, 128), dtype=torch.float16)
    arena.append_tail(keys, keys + 1)
    page_bytes = (
        4 * 256 * 128 + 4 * 256 * 2 + 4 * 256 * 96 + 4 * 256 * 4 * 2 + 4 * 256 * 4 * 2
    )
    full_tail_bytes = 2 * 4 * 383 * 128 * 2
    logical_tail_bytes = 2 * 4 * 17 * 128 * 2
    breakdown = arena.byte_breakdown(shared_constants=11, workspace=13, padding=7)
    assert breakdown == PackedArenaByteBreakdown(
        logical_payload=page_bytes,
        arena_capacity=3 * page_bytes,
        tail_logical=logical_tail_bytes,
        tail_capacity=full_tail_bytes,
        shared_constants=11,
        workspace=13,
        padding=7,
        unique_resident=3 * page_bytes + full_tail_bytes + 11 + 13 + 7,
    )
    assert breakdown.total_nbytes == 3 * page_bytes + full_tail_bytes + 11 + 13 + 7


def test_page_and_tail_pointers_stay_stable_and_prior_page_is_immutable() -> None:
    arena = PackedPageArena.reserve(
        make_layout(), device=torch.device("cpu"), kv_dtype=torch.float16
    )
    pointers = arena.data_pointers()
    first = _payload()
    arena.append_encoded_page(**first)
    page_names = (
        "key_indices",
        "key_norms",
        "value_indices",
        "value_mins",
        "value_scales",
    )
    first_bytes = {name: getattr(arena, name)[0].clone() for name in page_names}
    second = _payload()
    arena.append_encoded_page(**second)
    assert arena.data_pointers() == pointers
    for name, expected in first_bytes.items():
        assert torch.equal(getattr(arena, name)[0], expected)


def test_store_arena_tail_append_reuses_reserved_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arena = PackedPageArena.reserve(
        make_layout(pages=1, tail=383),
        device=torch.device("cpu"),
        kv_dtype=torch.float16,
    )
    store = PackedKVBlockStore(
        KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16)),
        ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
        num_kv_heads=4,
        arena=arena,
    )
    keys = torch.randn((1, 4, 128, 128), dtype=torch.float16)
    store.append(keys, keys + 1)
    pointers = arena.data_pointers()

    def fail_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("tail-only arena append allocated a new tensor")

    monkeypatch.setattr(torch, "empty", fail_allocation)
    monkeypatch.setattr(torch, "empty_like", fail_allocation)
    monkeypatch.setattr(torch, "cat", fail_allocation)
    store.append(keys[..., :1, :], keys[..., :1, :] + 1)

    assert arena.data_pointers() == pointers
    assert arena.tail_length == 129


def test_store_arena_reuses_one_prepared_constant_bundle_across_pages() -> None:
    codec = KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16))
    arena = PackedPageArena.reserve(
        make_layout(pages=3, tail=383),
        device=torch.device("cpu"),
        kv_dtype=torch.float16,
    )
    store = PackedKVBlockStore(
        codec,
        ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
        num_kv_heads=4,
        arena=arena,
    )
    pointers = arena.data_pointers()
    keys = torch.randn((1, 4, 640, 128), dtype=torch.float16)
    store.append(keys, keys + 1)

    prepared = codec.prepare_device(torch.device("cpu"))
    assert codec.prepare_device(torch.device("cpu")) is prepared
    assert len(codec._prepared_devices) == 1
    assert arena.used_pages == 2
    assert arena.data_pointers() == pointers


@pytest.mark.parametrize("tokens", [127, 128, 255, 256, 383, 384, 639, 640])
def test_store_arena_preserves_chronological_pages_and_tail_at_boundaries(
    tokens: int,
) -> None:
    layout = make_layout(pages=4, tail=383)
    arena = PackedPageArena.reserve(
        layout, device=torch.device("cpu"), kv_dtype=torch.float16
    )
    store = PackedKVBlockStore(
        KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16)),
        ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
        fp16_window=128,
        num_kv_heads=4,
        arena=arena,
    )
    keys = torch.arange(4 * tokens * 128, dtype=torch.float16).reshape(
        1, 4, tokens, 128
    )
    values = keys + 1
    store.append(keys, values)
    expected_pages = max(0, math.ceil((tokens - 383) / 256))
    assert arena.used_pages == expected_pages
    assert arena.tail_length == tokens - expected_pages * 256
    assert store.sequence_length() == tokens
    assert store.block_count == expected_pages
    if expected_pages:
        encoded = store.key_codec.encode(keys[..., :256, :])
        page = arena.page_views(0)
        assert torch.equal(page["key_indices"].reshape(4 * 256, -1), encoded.indices)
        assert torch.equal(page["key_norms"].reshape(4 * 256), encoded.norms)


def test_store_arena_flush_keeps_pointers_and_prior_page_bytes_stable() -> None:
    arena = PackedPageArena.reserve(
        make_layout(pages=2, tail=383),
        device=torch.device("cpu"),
        kv_dtype=torch.float16,
    )
    store = PackedKVBlockStore(
        KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16)),
        ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
        fp16_window=128,
        num_kv_heads=4,
        arena=arena,
    )
    pointers = arena.data_pointers()
    first_keys = torch.randn((1, 4, 383, 128), dtype=torch.float16)
    store.append(first_keys, first_keys + 1)
    store.append(first_keys[..., :1, :], first_keys[..., :1, :] + 1)
    first_page = {name: tensor.clone() for name, tensor in arena.page_views(0).items()}
    store.append(first_keys[..., :256, :], first_keys[..., :256, :] + 1)
    assert arena.data_pointers() == pointers
    for name, expected in first_page.items():
        assert torch.equal(arena.page_views(0)[name], expected)


def test_store_arena_page_overflow_fails_before_mutation() -> None:
    arena = PackedPageArena.reserve(
        make_layout(pages=1, tail=383),
        device=torch.device("cpu"),
        kv_dtype=torch.float16,
    )
    store = PackedKVBlockStore(
        KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16)),
        ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
        fp16_window=128,
        num_kv_heads=4,
        arena=arena,
    )
    keys = torch.randn((1, 4, 640, 128), dtype=torch.float16)
    before = arena.snapshot_state()
    with pytest.raises(OverflowError):
        store.append(keys, keys + 1)
    assert arena.snapshot_state() == before


def test_store_arena_dtype_mismatch_at_flush_fails_before_mutation() -> None:
    arena = PackedPageArena.reserve(
        make_layout(pages=2, tail=383),
        device=torch.device("cpu"),
        kv_dtype=torch.float16,
    )
    store = PackedKVBlockStore(
        KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16)),
        ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
        fp16_window=128,
        num_kv_heads=4,
        arena=arena,
    )
    fp16 = torch.randn((1, 4, 383, 128), dtype=torch.float16)
    store.append(fp16, fp16 + 1)
    before = arena.snapshot_state()
    bf16 = torch.randn((1, 4, 1, 128), dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        store.append(bf16, bf16 + 1)
    assert arena.snapshot_state() == before


def test_store_arena_device_mismatch_at_flush_fails_before_mutation() -> None:
    arena = PackedPageArena.reserve(
        make_layout(pages=2, tail=383),
        device=torch.device("cpu"),
        kv_dtype=torch.float16,
    )
    store = PackedKVBlockStore(
        KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16)),
        ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
        fp16_window=128,
        num_kv_heads=4,
        arena=arena,
    )
    fp16 = torch.randn((1, 4, 383, 128), dtype=torch.float16)
    store.append(fp16, fp16 + 1)
    before = arena.snapshot_state()
    meta = torch.empty((1, 4, 1, 128), device="meta", dtype=torch.float16)
    with pytest.raises(ValueError):
        store.append(meta, meta)
    assert arena.snapshot_state() == before


def test_exact_capacity_append_succeeds_and_one_more_token_fails_before_mutation() -> (
    None
):
    arena = PackedPageArena.reserve(
        make_layout(pages=1, tail=383),
        device=torch.device("cpu"),
        kv_dtype=torch.float16,
    )
    keys = torch.randn((1, 4, 383, 128), dtype=torch.float16)
    arena.append_tail(keys, keys + 1)
    before = arena.snapshot_state()
    with pytest.raises(OverflowError):
        arena.append_tail(keys[..., :1, :], keys[..., :1, :] + 1)
    assert arena.snapshot_state() == before


def test_cache_opt_in_reserves_arena_only_when_capacity_is_declared() -> None:
    profile = load_profile("configs/profiles/qwen25-7b-direct-v1.json")
    from core.packed_cache import PackedDynamicCache

    legacy = PackedDynamicCache(profile, layer_count=1)
    assert legacy.maximum_sequence_length is None
    arena_cache = PackedDynamicCache(
        profile, layer_count=1, maximum_sequence_length=640
    )
    assert arena_cache.maximum_sequence_length == 640
    keys = torch.randn((1, 4, 1, 128), dtype=torch.float16)
    store = arena_cache._new_store(keys, 0)
    assert store.arena is not None
    assert store.arena.layout.page_capacity >= 1
    assert store.arena.layout.tail_capacity == 383


def test_cache_declared_capacity_rejects_overflow_before_store_mutation() -> None:
    profile = load_profile("configs/profiles/qwen25-7b-direct-v1.json")
    from core.packed_cache import PackedDynamicCache

    cache = PackedDynamicCache(profile, layer_count=1, maximum_sequence_length=384)
    keys = torch.randn((1, 4, 383, 128), dtype=torch.float16)
    cache.update(keys, keys + 1, 0)
    store = cache.store_for(0)
    before = store.arena.snapshot_state()  # type: ignore[union-attr]
    with pytest.raises(OverflowError):
        cache.update(keys[..., :2, :], keys[..., :2, :] + 1, 0)
    assert store.arena.snapshot_state() == before  # type: ignore[union-attr]
