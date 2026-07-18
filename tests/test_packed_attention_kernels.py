"""GPU parity contracts for clean-room packed-attention Triton kernels."""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import pytest
import torch

from core.packing import pack_indices
from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packed_store import PackedKVBlockStore
from core.reference_attention import ReferenceAttentionLayout, packed_reference_decode
from kernels.packed_attention import (
    PackedAttentionBlock,
    PackedAttentionDebugState,
    PackedAttentionExecution,
    packed_attention_decode,
    packed_key_scores,
    packed_key_scores_reference,
    packed_key_scores_rotated,
    packed_value_mix,
    packed_value_mix_reference,
    rotate_query,
)


_RTOL = 3e-3
_ATOL = 3e-3


def _require_gpu_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; packed Triton parity requires a CUDA GPU.")
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("Triton is unavailable; packed Triton parity requires Triton.")


@dataclass(frozen=True)
class PackedCase:
    key_bits: int
    value_bits: int
    head_dim: int
    kv_heads: int
    gqa_ratio: int
    tokens: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return (
            f"kb{self.key_bits}-vb{self.value_bits}-d{self.head_dim}-"
            f"kv{self.kv_heads}-gqa{self.gqa_ratio}-t{self.tokens}-"
            f"{str(self.dtype).removeprefix('torch.')}"
        )


# A compact covering matrix keeps GPU runtime practical while exercising every
# required dimension, including all packed widths and ragged token boundaries.
_CASES = [
    PackedCase(6, 4, 64, 1, 1, 1, torch.float16),
    PackedCase(8, 6, 128, 2, 4, 17, torch.bfloat16),
    PackedCase(6, 8, 256, 8, 8, 255, torch.float16),
    PackedCase(8, 4, 64, 8, 4, 256, torch.bfloat16),
    PackedCase(6, 6, 128, 1, 8, 17, torch.float16),
    PackedCase(8, 8, 256, 2, 1, 255, torch.bfloat16),
]


def _case_id(case: PackedCase) -> str:
    return f"{case.name}-rtol{_RTOL:g}-atol{_ATOL:g}"


def _fixture(case: PackedCase) -> dict[str, torch.Tensor | int | float]:
    torch.manual_seed(7_000 + case.head_dim + case.tokens + case.value_bits)
    device = torch.device("cuda")
    heads = case.kv_heads * case.gqa_ratio
    query_tokens = 2
    group_size = 32
    groups = case.head_dim // group_size
    codebook = torch.linspace(
        -1.0, 1.0, 1 << case.key_bits, device=device, dtype=torch.float32
    )
    query = torch.randn(
        (heads, query_tokens, case.head_dim), device=device, dtype=case.dtype
    )
    rotation = torch.linalg.qr(
        torch.randn((case.head_dim, case.head_dim), device=device, dtype=torch.float32)
    ).Q.contiguous()
    key_indices = torch.randint(
        0,
        1 << case.key_bits,
        (case.kv_heads * case.tokens, case.head_dim),
        device=device,
        dtype=torch.uint8,
    )
    packed_keys = pack_indices(key_indices, case.key_bits).reshape(
        case.kv_heads, case.tokens, -1
    ).contiguous()
    norms = (0.2 + torch.rand((case.kv_heads, case.tokens), device=device)).to(case.dtype)
    value_indices = torch.randint(
        0,
        1 << case.value_bits,
        (case.kv_heads * case.tokens, case.head_dim),
        device=device,
        dtype=torch.uint8,
    )
    packed_values = pack_indices(value_indices, case.value_bits).reshape(
        case.kv_heads, case.tokens, -1
    ).contiguous()
    mins = torch.randn((case.kv_heads, case.tokens, groups), device=device, dtype=case.dtype)
    scales = (0.01 + torch.rand((case.kv_heads, case.tokens, groups), device=device)).to(case.dtype)
    gqa_mapping = torch.arange(heads, device=device, dtype=torch.int32) // case.gqa_ratio
    return {
        "query": query,
        "rotation": rotation,
        "packed_keys": packed_keys,
        "codebook": codebook,
        "norms": norms,
        "packed_values": packed_values,
        "mins": mins,
        "scales": scales,
        "gqa_mapping": gqa_mapping,
        "group_size": group_size,
        "scale": 1 / math.sqrt(case.head_dim),
    }


@pytest.mark.gpu
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_packed_score_kernel_matches_reference(case: PackedCase) -> None:
    _require_gpu_triton()
    data = _fixture(case)
    actual, execution = packed_key_scores(
        data["query"], data["packed_keys"], data["codebook"], data["norms"],
        data["rotation"], data["gqa_mapping"], key_bits=case.key_bits,
        scale=data["scale"], return_execution=True,
    )
    expected = packed_key_scores_reference(
        data["query"], data["packed_keys"], data["codebook"], data["norms"],
        data["rotation"], data["gqa_mapping"], key_bits=case.key_bits,
        scale=data["scale"],
    )
    assert isinstance(execution, PackedAttentionExecution)
    assert execution.kernel == "triton"
    assert not execution.used_torch_fallback
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.gpu
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_packed_value_mix_kernel_matches_reference(case: PackedCase) -> None:
    _require_gpu_triton()
    data = _fixture(case)
    scores, _ = packed_key_scores(
        data["query"], data["packed_keys"], data["codebook"], data["norms"],
        data["rotation"], data["gqa_mapping"], key_bits=case.key_bits,
        scale=data["scale"], return_execution=True,
    )
    prior_max = torch.full_like(scores[..., 0], -torch.inf, dtype=torch.float32)
    prior_denominator = torch.zeros_like(prior_max)
    prior_numerator = torch.zeros(
        (*prior_max.shape, case.head_dim), device="cuda", dtype=torch.float32
    )
    actual, execution = packed_value_mix(
        scores, data["packed_values"], data["mins"], data["scales"],
        data["gqa_mapping"], value_bits=case.value_bits, group_size=data["group_size"],
        prior_state=(prior_max, prior_denominator, prior_numerator), return_execution=True,
    )
    expected = packed_value_mix_reference(
        scores, data["packed_values"], data["mins"], data["scales"],
        data["gqa_mapping"], value_bits=case.value_bits, group_size=data["group_size"],
        prior_state=(prior_max, prior_denominator, prior_numerator),
    )
    assert isinstance(execution, PackedAttentionExecution)
    assert execution.kernel == "triton"
    assert not execution.used_torch_fallback
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=_RTOL, atol=_ATOL)


def test_packed_score_requires_explicit_cpu_reference_fallback() -> None:
    query = torch.ones((1, 1, 64), dtype=torch.float16)
    packed_keys = pack_indices(torch.zeros((1, 64), dtype=torch.uint8), 6).reshape(1, 1, -1)
    codebook = torch.linspace(-1, 1, 64)
    norms = torch.ones((1, 1), dtype=torch.float16)
    rotation = torch.eye(64)
    mapping = torch.zeros((1,), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="allow_torch_fallback"):
        packed_key_scores(
            query, packed_keys, codebook, norms, rotation, mapping,
            key_bits=6, scale=1 / 8,
        )
    _, execution = packed_key_scores(
        query, packed_keys, codebook, norms, rotation, mapping,
        key_bits=6, scale=1 / 8, allow_torch_fallback=True, return_execution=True,
    )
    assert execution.kernel == "torch_reference"
    assert execution.used_torch_fallback


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [4, 6, 8])
def test_packed_decoder_final_byte_is_explicitly_guarded(bits: int) -> None:
    """The final index must not issue a `base + 1` load past row storage."""
    _require_gpu_triton()
    head_dim, tokens = 64, 1
    query = torch.ones((1, 1, head_dim), device="cuda", dtype=torch.float16)
    indices = torch.full((1, head_dim), (1 << bits) - 1, device="cuda", dtype=torch.uint8)
    packed = pack_indices(indices, bits).reshape(1, tokens, -1)
    codebook = torch.arange(1 << bits, device="cuda", dtype=torch.float32)
    norms = torch.ones((1, tokens), device="cuda", dtype=torch.float16)
    rotation = torch.eye(head_dim, device="cuda")
    mapping = torch.zeros((1,), device="cuda", dtype=torch.int32)
    actual, execution = packed_key_scores(
        query, packed, codebook, norms, rotation, mapping,
        key_bits=bits, scale=1.0, return_execution=True,
    )
    expected = packed_key_scores_reference(
        query, packed, codebook, norms, rotation, mapping, key_bits=bits, scale=1.0
    )
    assert execution.kernel == "triton" and not execution.used_torch_fallback
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)
    source = Path("kernels/packed_attention.py").read_text(encoding="utf-8")
    assert "byte_offset[None, :] + 1 < PACKED_WIDTH" in source


@pytest.mark.gpu
def test_triton_rotation_and_rotated_score_match_reference() -> None:
    _require_gpu_triton()
    case = _CASES[2]
    data = _fixture(case)
    rotated, execution = rotate_query(data["query"], data["rotation"], return_execution=True)
    expected_rotated = data["query"].float() @ data["rotation"].float().T
    assert execution.kernel == "triton" and not execution.used_torch_fallback
    torch.testing.assert_close(rotated, expected_rotated, rtol=_RTOL, atol=_ATOL)
    scores, execution = packed_key_scores_rotated(
        rotated, data["packed_keys"], data["codebook"], data["norms"],
        data["gqa_mapping"], key_bits=case.key_bits, scale=data["scale"],
        return_execution=True,
    )
    expected = packed_key_scores_reference(
        data["query"], data["packed_keys"], data["codebook"], data["norms"],
        data["rotation"], data["gqa_mapping"], key_bits=case.key_bits,
        scale=data["scale"],
    )
    assert execution.kernel == "triton" and not execution.used_torch_fallback
    torch.testing.assert_close(scores, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.gpu
def test_packed_decode_matches_task4_blocks_and_fp16_window_without_joining_history() -> None:
    _require_gpu_triton()
    torch.manual_seed(2505)
    head_dim, kv_heads, gqa_ratio, tokens, window = 64, 2, 4, 600, 32
    dtype = torch.float16
    key_codec = KeyCodec(KeyLayout(head_dim=head_dim, bits=6, norm_dtype=dtype))
    value_codec = ValueCodec(ValueLayout(head_dim=head_dim, bits=6, group_size=32))
    store = PackedKVBlockStore(
        key_codec, value_codec, fp16_window=window, num_kv_heads=kv_heads
    )
    keys = torch.randn((1, kv_heads, tokens, head_dim), device="cuda", dtype=dtype)
    values = torch.randn_like(keys)
    store.append(keys, values)
    query = torch.randn((kv_heads * gqa_ratio, 2, head_dim), device="cuda", dtype=dtype)
    layout = ReferenceAttentionLayout(
        head_dim=head_dim,
        num_attention_heads=kv_heads * gqa_ratio,
        num_key_value_heads=kv_heads,
    )
    expected = packed_reference_decode(query.unsqueeze(0), store, layout).squeeze(0)

    def to_kernel_block(block: object) -> PackedAttentionBlock:
        payload = block  # Typed locally to keep the test focused on public payload layout.
        return PackedAttentionBlock(
            packed_keys=payload.keys.indices.reshape(kv_heads, 256, -1),
            key_norms=payload.keys.norms.reshape(kv_heads, 256),
            key_codebook=key_codec._centroids.to("cuda"),
            key_bits=6,
            packed_values=payload.values.indices.reshape(kv_heads, 256, -1),
            value_mins=payload.values.mins.reshape(kv_heads, 256, -1),
            value_scales=payload.values.scales.reshape(kv_heads, 256, -1),
            value_bits=6,
            value_group_size=32,
        )

    blocks = [to_kernel_block(block) for block in store.iter_blocks()]
    assert len(blocks) == 2
    fp16_window = tuple(tensor.squeeze(0) for tensor in next(store.iter_fp16_window()))
    debug = PackedAttentionDebugState()
    actual, execution = packed_attention_decode(
        query,
        (block for block in blocks),
        key_codec._rotation.to("cuda"),
        torch.arange(kv_heads * gqa_ratio, device="cuda", dtype=torch.int32) // gqa_ratio,
        fp16_window=fp16_window,
        debug=debug,
        return_execution=True,
    )
    assert execution.kernel == "triton" and not execution.used_torch_fallback
    assert debug.blocks_consumed == 2
    assert debug.max_block_tokens == 256
    assert not debug.joined_history_attempted
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)
