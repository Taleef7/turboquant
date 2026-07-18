"""Exact direct-profile GPU parity coverage for the Qwen2.5 cache shape."""

from __future__ import annotations

import pytest
import torch

from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packed_store import PackedKVBlockStore
from core.reference_attention import ReferenceAttentionLayout, packed_reference_decode
from kernels.packed_attention import (
    PackedAttentionBlock,
    PackedAttentionDebugState,
    packed_attention_decode,
)


RTOL = 3e-3
ATOL = 3e-3


def _require_gpu_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; Qwen profile parity requires a CUDA GPU.")
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("Triton is unavailable; Qwen profile parity requires Triton.")


@pytest.mark.gpu
def test_qwen25_direct_profile_triton_matches_packed_reference_with_blocks_and_tail():
    """Profile shape: 8-bit K, 6-bit V, d128, 4 KV heads, and 7-way GQA."""
    _require_gpu_triton()
    torch.manual_seed(25_706)
    device = torch.device("cuda")
    head_dim, kv_heads, gqa_ratio = 128, 4, 7
    packed_tokens, fp16_tail = 512, 128
    dtype = torch.float16
    key_codec = KeyCodec(KeyLayout(head_dim=head_dim, bits=8, norm_dtype=dtype))
    value_codec = ValueCodec(ValueLayout(head_dim=head_dim, bits=6, group_size=32))
    store = PackedKVBlockStore(
        key_codec, value_codec, fp16_window=fp16_tail, num_kv_heads=kv_heads
    )
    keys = torch.randn(
        (1, kv_heads, packed_tokens + fp16_tail, head_dim), device=device, dtype=dtype
    )
    values = torch.randn_like(keys)
    store.append(keys, values)
    query = torch.randn((kv_heads * gqa_ratio, 1, head_dim), device=device, dtype=dtype)
    expected = packed_reference_decode(
        query.unsqueeze(0),
        store,
        ReferenceAttentionLayout(
            head_dim=head_dim,
            num_attention_heads=kv_heads * gqa_ratio,
            num_key_value_heads=kv_heads,
        ),
    ).squeeze(0)

    def to_kernel_block(block: object) -> PackedAttentionBlock:
        payload = block
        return PackedAttentionBlock(
            packed_keys=payload.keys.indices.reshape(kv_heads, 256, -1),
            key_norms=payload.keys.norms.reshape(kv_heads, 256),
            key_codebook=key_codec._centroids.to(device),
            key_bits=8,
            packed_values=payload.values.indices.reshape(kv_heads, 256, -1),
            value_mins=payload.values.mins.reshape(kv_heads, 256, -1),
            value_scales=payload.values.scales.reshape(kv_heads, 256, -1),
            value_bits=6,
            value_group_size=32,
        )

    blocks = [to_kernel_block(block) for block in store.iter_blocks()]
    assert len(blocks) == 2
    tail = tuple(tensor.squeeze(0) for tensor in next(store.iter_fp16_window()))
    assert tail[0].shape == (kv_heads, fp16_tail, head_dim)
    debug = PackedAttentionDebugState()
    actual, execution = packed_attention_decode(
        query,
        blocks,
        key_codec._rotation.to(device),
        torch.arange(kv_heads * gqa_ratio, device=device, dtype=torch.int32) // gqa_ratio,
        fp16_window=tail,
        debug=debug,
        return_execution=True,
    )

    assert execution.kernel == "triton"
    assert execution.used_torch_fallback is False
    assert debug.blocks_consumed == 2
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
