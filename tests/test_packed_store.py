"""Append-only storage invariants for physically packed KV blocks."""

from __future__ import annotations

import inspect

import pytest
import torch

from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packed_store import PackedKVBlockStore
from core.profiles import load_profile
from core.turboquant_cache_v2 import TurboQuantCacheV2


def make_store(*, fp16_window: int = 0) -> PackedKVBlockStore:
    return PackedKVBlockStore(
        KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=torch.float16)),
        ValueCodec(ValueLayout(head_dim=128, bits=6, group_size=32)),
        fp16_window=fp16_window,
    )


def test_append_flushes_fixed_blocks_without_replacing_prior_payloads() -> None:
    store = make_store()
    keys = torch.randn((1, 2, 300, 128), dtype=torch.float16)
    values = torch.randn((1, 2, 300, 128), dtype=torch.float16)

    store.append(keys[..., :100, :], values[..., :100, :])
    store.append(keys[..., 100:256, :], values[..., 100:256, :])
    first = next(store.iter_blocks())
    pointers = (first.keys.indices.data_ptr(), first.values.indices.data_ptr())
    store.append(keys[..., 256:, :], values[..., 256:, :])

    assert store.sequence_length() == 300
    assert len(tuple(store.iter_blocks())) == 1
    assert (first.keys.indices.data_ptr(), first.values.indices.data_ptr()) == pointers
    window = list(store.iter_fp16_window())
    assert len(window) == 1
    assert window[0][0].shape[-2] == 44
    assert torch.equal(window[0][0], keys[..., 256:, :])
    assert torch.equal(window[0][1], values[..., 256:, :])


def test_append_keeps_one_bounded_pending_window_and_sequence_order() -> None:
    store = make_store(fp16_window=128)
    keys = torch.randn((1, 1, 700, 128), dtype=torch.bfloat16)
    values = torch.randn((1, 1, 700, 128), dtype=torch.bfloat16)

    for start, stop in ((0, 7), (7, 299), (299, 513), (513, 700)):
        store.append(keys[..., start:stop, :], values[..., start:stop, :])

    blocks = tuple(store.iter_blocks())
    pending = list(store.iter_fp16_window())
    assert [block.logical_tokens for block in blocks] == [256, 256]
    assert len(pending) == 1
    assert pending[0][0].shape[-2] == 188
    reconstructed = [block.keys.unpack_reference() for block in blocks] + [pending[0][0]]
    assert torch.equal(torch.cat(reconstructed, dim=-2)[..., 512:, :], keys[..., 512:, :])
    assert store.sequence_length() == 700


def test_byte_breakdown_equals_all_stored_tensor_payloads() -> None:
    store = make_store()
    keys = torch.randn((1, 1, 300, 128), dtype=torch.float16)
    values = torch.randn((1, 1, 300, 128), dtype=torch.float16)
    store.append(keys, values)

    breakdown = store.physical_nbytes()
    tensors = []
    for block in store.iter_blocks():
        tensors.extend(
            [
                block.keys.indices,
                block.keys.norms,
                block.values.indices,
                block.values.mins,
                block.values.scales,
            ]
        )
    for key_window, value_window in store.iter_fp16_window():
        tensors.extend([key_window, value_window])
    actual = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    assert breakdown.total_nbytes == actual
    assert breakdown.padding == 0


def test_store_rejects_non_kv_shape_or_non_low_precision_window() -> None:
    store = make_store()
    with pytest.raises((TypeError, ValueError)):
        store.append(torch.zeros((1, 1, 128), dtype=torch.float16), torch.zeros((1, 1, 128), dtype=torch.float16))
    with pytest.raises((TypeError, ValueError)):
        store.append(torch.zeros((1, 1, 1, 128), dtype=torch.float32), torch.zeros((1, 1, 1, 128), dtype=torch.float32))


def test_legacy_v2_cache_is_not_a_direct_profile_route() -> None:
    assert TurboQuantCacheV2.LEGACY is True
    profile = load_profile("configs/profiles/qwen25-7b-direct-v1.json")
    assert profile.cache_implementation != "TurboQuantCacheV2"


def test_store_has_no_full_history_reconstruction_or_concatenation_helper() -> None:
    public_methods = {name for name in dir(PackedKVBlockStore) if not name.startswith("_")}

    assert not any("reconstruct" in name for name in public_methods)
    assert "torch.cat" not in inspect.getsource(PackedKVBlockStore)
