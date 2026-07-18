"""Direct-profile Triton contracts for the raw fp16 Qwen2.5 tail."""

from __future__ import annotations

import math

import pytest
import torch

from kernels.packed_attention import (
    PackedAttentionHotPathDebugState,
    PackedAttentionTailDebugState,
    prepare_packed_attention_decode,
    qwen25_fp16_tail_online_update,
)


_HEADS = 28
_KV_HEADS = 4
_GQA_RATIO = 7
_HEAD_DIM = 128
_RTOL = 3e-3
_ATOL = 3e-3


def _require_gpu_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; direct Qwen tail coverage requires CUDA.")
    pytest.importorskip("triton", reason="Triton is unavailable for direct Qwen tail coverage")


def _tail_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    prior_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    scale: float,
    mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mapping is None:
        mapping = torch.arange(_HEADS, device=query.device) // _GQA_RATIO
    mapped_keys = keys.float()[mapping]
    mapped_values = values.float()[mapping]
    scores = torch.einsum("hqd,htd->hqt", query.float(), mapped_keys) * scale
    old_max, old_denominator, old_numerator = prior_state
    block_max = scores.max(dim=-1).values
    new_max = torch.maximum(old_max.float(), block_max)
    old_scale = torch.where(
        torch.isfinite(old_max), torch.exp(old_max - new_max), torch.zeros_like(old_max)
    )
    weights = torch.exp(scores.float() - new_max.unsqueeze(-1))
    return (
        new_max,
        old_denominator.float() * old_scale + weights.sum(dim=-1),
        old_numerator.float() * old_scale.unsqueeze(-1)
        + torch.einsum("hqt,htd->hqd", weights, mapped_values),
    )


def _tail_fixture(tokens: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    torch.manual_seed(27_000 + tokens)
    device = torch.device("cuda")
    query = torch.randn((_HEADS, 1, _HEAD_DIM), device=device, dtype=torch.float16)
    keys = torch.randn((_KV_HEADS, tokens, _HEAD_DIM), device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    prior_state = (
        torch.randn((_HEADS, 1), device=device),
        torch.rand((_HEADS, 1), device=device) + 0.1,
        torch.randn((_HEADS, 1, _HEAD_DIM), device=device),
    )
    return query, keys, values, prior_state


@pytest.mark.gpu
@pytest.mark.parametrize("tokens", [1, 127, 128, 255, 256, 383])
def test_qwen25_fp16_tail_kernel_matches_online_reference_with_one_dispatch(tokens: int) -> None:
    _require_gpu_triton()
    query, keys, values, prior_state = _tail_fixture(tokens)
    expected = _tail_reference(
        query, values=values, keys=keys, prior_state=prior_state, scale=1 / math.sqrt(_HEAD_DIM)
    )
    debug = PackedAttentionTailDebugState()

    actual = qwen25_fp16_tail_online_update(
        query,
        keys,
        values,
        prior_state=prior_state,
        scale=1 / math.sqrt(_HEAD_DIM),
        debug=debug,
    )

    assert debug.dispatches == 1
    assert debug.tail_tokens == tokens
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=_RTOL, atol=_ATOL)


@pytest.mark.gpu
def test_prepared_qwen_tail_avoids_gqa_expanded_tensor_materialization(monkeypatch) -> None:
    _require_gpu_triton()
    query, keys, values, prior_state = _tail_fixture(383)
    expected = _tail_reference(
        query, values=values, keys=keys, prior_state=(
            torch.full_like(prior_state[0], -torch.inf),
            torch.zeros_like(prior_state[1]),
            torch.zeros_like(prior_state[2]),
        ), scale=1 / math.sqrt(_HEAD_DIM)
    )
    debug = PackedAttentionHotPathDebugState()
    prepared = prepare_packed_attention_decode(
        (),
        torch.eye(_HEAD_DIM, device="cuda"),
        torch.arange(_HEADS, device="cuda", dtype=torch.int32) // _GQA_RATIO,
        num_kv_heads=_KV_HEADS,
        debug=debug,
    )
    original_getitem = torch.Tensor.__getitem__

    def reject_gqa_expansion(tensor: torch.Tensor, index):
        if (
            tensor.ndim == 3
            and tensor.shape[0] == _KV_HEADS
            and tensor.shape[-1] == _HEAD_DIM
            and isinstance(index, torch.Tensor)
            and index.numel() == _HEADS
        ):
            raise AssertionError("direct Qwen tail materialized GQA-expanded K/V")
        return original_getitem(tensor, index)

    monkeypatch.setattr(torch.Tensor, "__getitem__", reject_gqa_expansion)
    actual = prepared.launch(query, fp16_window=(keys, values))

    assert debug.tail_kernel_dispatches == 1
    torch.testing.assert_close(
        actual,
        (expected[2] / expected[1].unsqueeze(-1)).to(query.dtype),
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.gpu
def test_prepared_qwen_tail_falls_back_for_permuted_valid_gqa_mapping() -> None:
    _require_gpu_triton()
    query, keys, values, prior_state = _tail_fixture(383)
    canonical_mapping = torch.arange(_HEADS, device="cuda", dtype=torch.int32) // _GQA_RATIO
    permuted_mapping = canonical_mapping.roll(_GQA_RATIO).contiguous()
    assert not torch.equal(permuted_mapping, canonical_mapping)
    expected = _tail_reference(
        query,
        keys=keys,
        values=values,
        prior_state=(
            torch.full_like(prior_state[0], -torch.inf),
            torch.zeros_like(prior_state[1]),
            torch.zeros_like(prior_state[2]),
        ),
        scale=1 / math.sqrt(_HEAD_DIM),
        mapping=permuted_mapping,
    )
    debug = PackedAttentionHotPathDebugState()
    prepared = prepare_packed_attention_decode(
        (),
        torch.eye(_HEAD_DIM, device="cuda"),
        permuted_mapping,
        num_kv_heads=_KV_HEADS,
        debug=debug,
    )

    actual = prepared.launch(query, fp16_window=(keys, values))

    assert debug.tail_kernel_dispatches == 0
    torch.testing.assert_close(
        actual,
        (expected[2] / expected[1].unsqueeze(-1)).to(query.dtype),
        rtol=_RTOL,
        atol=_ATOL,
    )
