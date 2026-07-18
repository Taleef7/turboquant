"""Prepared-only fused Qwen packed-page attention contracts."""

from __future__ import annotations

import math

import pytest
import torch

from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packed_store import PackedKVBlockStore
from core.reference_attention import ReferenceAttentionLayout, packed_reference_decode
from kernels.packed_attention import (
    PackedAttentionBlock,
    PackedAttentionHotPathDebugState,
    packed_attention_decode,
    prepare_packed_attention_decode,
)


_HEADS = 28
_KV_HEADS = 4
_GQA_RATIO = 7
_HEAD_DIM = 128
_PAGE_TOKENS = 256


def _require_gpu_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; fused Qwen-page coverage requires CUDA.")
    pytest.importorskip("triton", reason="Triton is unavailable for fused Qwen-page coverage")


def _fixture(
    pages: int,
) -> tuple[
    tuple[PackedAttentionBlock, ...],
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    PackedKVBlockStore,
]:
    torch.manual_seed(27_100 + pages)
    device = torch.device("cuda")
    key_codec = KeyCodec(KeyLayout(head_dim=_HEAD_DIM, bits=8, norm_dtype=torch.float16))
    value_codec = ValueCodec(ValueLayout(head_dim=_HEAD_DIM, bits=6, group_size=32))
    store = PackedKVBlockStore(
        key_codec,
        value_codec,
        fp16_window=128,
        num_kv_heads=_KV_HEADS,
    )
    tokens = pages * _PAGE_TOKENS + 128
    keys = torch.randn((1, _KV_HEADS, tokens, _HEAD_DIM), device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    store.append(keys, values)
    blocks = tuple(
        PackedAttentionBlock(
            packed_keys=block.keys.indices.reshape(_KV_HEADS, _PAGE_TOKENS, -1),
            key_norms=block.keys.norms.reshape(_KV_HEADS, _PAGE_TOKENS),
            key_codebook=key_codec._centroids.to(device),
            key_bits=8,
            packed_values=block.values.indices.reshape(_KV_HEADS, _PAGE_TOKENS, -1),
            value_mins=block.values.mins.reshape(_KV_HEADS, _PAGE_TOKENS, -1),
            value_scales=block.values.scales.reshape(_KV_HEADS, _PAGE_TOKENS, -1),
            value_bits=6,
            value_group_size=32,
        )
        for block in store.iter_blocks()
    )
    assert len(blocks) == pages
    if blocks:
        # d128 at six bits ends in the 96th byte.  The fused decoder must not
        # load beyond it for the final code.
        assert blocks[0].packed_values.shape[-1] == 96
    tail = tuple(tensor.squeeze(0) for tensor in next(store.iter_fp16_window()))
    query = torch.randn((_HEADS, 1, _HEAD_DIM), device=device, dtype=torch.float16)
    mapping = torch.arange(_HEADS, device=device, dtype=torch.int32) // _GQA_RATIO
    return blocks, tail, query, mapping, store


@pytest.mark.gpu
@pytest.mark.parametrize("pages", [0, 1, 2, 31])
def test_prepared_qwen_fuses_each_canonical_packed_page_without_scores(
    monkeypatch: pytest.MonkeyPatch,
    pages: int,
) -> None:
    _require_gpu_triton()
    blocks, tail, query, mapping, store = _fixture(pages)
    expected_reference = packed_reference_decode(
        query.unsqueeze(0),
        store,
        ReferenceAttentionLayout(
            head_dim=_HEAD_DIM,
            num_attention_heads=_HEADS,
            num_key_value_heads=_KV_HEADS,
        ),
    ).squeeze(0)
    expected_direct = packed_attention_decode(
        query,
        blocks,
        store.key_codec._rotation.to(query.device),
        mapping,
        scale=1 / math.sqrt(_HEAD_DIM),
        fp16_window=tail,
    )
    debug = PackedAttentionHotPathDebugState()
    prepared = prepare_packed_attention_decode(
        blocks,
        store.key_codec._rotation.to(query.device),
        mapping,
        num_kv_heads=_KV_HEADS,
        debug=debug,
    )
    original_empty = torch.empty

    def reject_packed_score_materialization(*size, **kwargs):
        shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
        if tuple(shape) == (_HEADS, 1, _PAGE_TOKENS):
            raise AssertionError("prepared canonical Qwen path materialized packed scores")
        return original_empty(*size, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(torch, "empty", reject_packed_score_materialization)
        actual = prepared.launch(query, fp16_window=tail)

    assert getattr(debug, "fused_packed_page_launches", 0) == pages
    assert debug.packed_score_launches == 0
    assert debug.packed_value_launches == 0
    torch.testing.assert_close(actual, expected_direct, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(actual, expected_reference, rtol=3e-3, atol=3e-3)
