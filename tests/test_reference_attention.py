"""CPU equivalence and memory-bound tests for packed reference attention."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packed_store import PackedKVBlockStore
from core.reference_attention import (
    ReferenceAttentionDebugState,
    ReferenceAttentionLayout,
    packed_reference_decode,
)


def make_store(*, head_dim: int, fp16_window: int = 0) -> PackedKVBlockStore:
    return PackedKVBlockStore(
        KeyCodec(KeyLayout(head_dim=head_dim, bits=8, norm_dtype=torch.float16)),
        ValueCodec(ValueLayout(head_dim=head_dim, bits=6, group_size=32)),
        fp16_window=fp16_window,
        num_kv_heads=2,
    )


def sdpa_on_unpacked_quantized_history(
    query: torch.Tensor, store: PackedKVBlockStore, layout: ReferenceAttentionLayout
) -> torch.Tensor:
    """Oracle intentionally permitted to materialize full quantized history."""
    keys = [block.keys.unpack_reference() for block in store.iter_blocks()]
    values = [block.values.unpack_reference() for block in store.iter_blocks()]
    for key_window, value_window in store.iter_fp16_window():
        keys.append(key_window)
        values.append(value_window)
    all_keys = torch.cat(keys, dim=-2)
    all_values = torch.cat(values, dim=-2)
    repeat = layout.num_attention_heads // layout.num_key_value_heads
    return F.scaled_dot_product_attention(
        query,
        all_keys.repeat_interleave(repeat, dim=1),
        all_values.repeat_interleave(repeat, dim=1),
        dropout_p=0.0,
        is_causal=False,
        scale=1 / math.sqrt(layout.head_dim),
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tail_tokens", [1, 255, 256])
def test_blockwise_reference_matches_unpacked_quantized_attention(
    head_dim: int, dtype: torch.dtype, tail_tokens: int
) -> None:
    torch.manual_seed(10_000 + head_dim + tail_tokens)
    layout = ReferenceAttentionLayout(
        head_dim=head_dim, num_attention_heads=4, num_key_value_heads=2
    )
    store = make_store(head_dim=head_dim)
    keys = torch.randn((1, 2, 256 + tail_tokens, head_dim), dtype=dtype)
    values = torch.randn_like(keys)
    query = torch.randn((1, 4, 1, head_dim), dtype=dtype)
    store.append(keys, values)

    output = packed_reference_decode(
        query, store, layout, scale=1 / math.sqrt(head_dim)
    )
    expected = sdpa_on_unpacked_quantized_history(query, store, layout)

    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


def test_blockwise_reference_handles_mixed_packed_and_fp16_window_history() -> None:
    torch.manual_seed(42)
    layout = ReferenceAttentionLayout(
        head_dim=128, num_attention_heads=4, num_key_value_heads=2
    )
    store = make_store(head_dim=128, fp16_window=32)
    keys = torch.randn((1, 2, 545, 128), dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn((1, 4, 3, 128), dtype=torch.float16)
    store.append(keys, values)

    assert len(tuple(store.iter_blocks())) == 2
    assert next(store.iter_fp16_window())[0].shape[-2] == 33
    output = packed_reference_decode(
        query, store, layout, scale=1 / math.sqrt(128)
    )

    torch.testing.assert_close(
        output,
        sdpa_on_unpacked_quantized_history(query, store, layout),
        rtol=2e-3,
        atol=2e-3,
    )


def test_extra_old_block_changes_output_without_increasing_decoded_block_bound() -> None:
    torch.manual_seed(7)
    layout = ReferenceAttentionLayout(
        head_dim=64, num_attention_heads=4, num_key_value_heads=2
    )
    query = torch.randn((1, 4, 1, 64), dtype=torch.float16)
    current_keys = torch.randn((1, 2, 256, 64), dtype=torch.float16)
    current_values = torch.randn_like(current_keys)
    older_keys = torch.full((1, 2, 256, 64), 0.25, dtype=torch.float16)
    older_values = torch.full_like(older_keys, 4.0)

    short_store = make_store(head_dim=64)
    short_store.append(current_keys, current_values)
    short_debug = ReferenceAttentionDebugState()
    short_output = packed_reference_decode(query, short_store, layout, debug=short_debug)

    long_store = make_store(head_dim=64)
    long_store.append(older_keys, older_values)
    long_store.append(current_keys, current_values)
    long_debug = ReferenceAttentionDebugState()
    long_output = packed_reference_decode(query, long_store, layout, debug=long_debug)

    assert not torch.allclose(long_output, short_output)
    assert short_debug.max_decoded_tokens == long_debug.max_decoded_tokens == 256
    assert long_debug.max_temporary_tokens <= 256


def test_reference_attention_deterministic_lossy_drift_fixture() -> None:
    torch.manual_seed(123)
    layout = ReferenceAttentionLayout(
        head_dim=64, num_attention_heads=4, num_key_value_heads=2
    )
    store = make_store(head_dim=64)
    keys = torch.randn((1, 2, 257, 64), dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn((1, 4, 1, 64), dtype=torch.float16)
    store.append(keys, values)

    reference = packed_reference_decode(query, store, layout)
    quantized_keys = torch.cat(
        [block.keys.unpack_reference() for block in store.iter_blocks()]
        + [key_window for key_window, _ in store.iter_fp16_window()],
        dim=-2,
    )
    quantized_logits = (
        query.float()
        @ quantized_keys.float().repeat_interleave(2, dim=1).transpose(-1, -2)
    ) * (1 / math.sqrt(64))
    original_logits = (
        query.float() @ keys.float().repeat_interleave(2, dim=1).transpose(-1, -2)
    ) * (1 / math.sqrt(64))
    original = F.scaled_dot_product_attention(
        query,
        keys.repeat_interleave(2, dim=1),
        values.repeat_interleave(2, dim=1),
        dropout_p=0.0,
        is_causal=False,
        scale=1 / math.sqrt(64),
    )

    # The codec is intentionally lossy; this fixture guards drift, not fp16 identity.
    assert float((quantized_logits - original_logits).abs().max()) < 0.1
    assert float((reference - original).abs().max()) < 0.06
