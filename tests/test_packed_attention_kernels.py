"""GPU parity contracts for clean-room packed-attention Triton kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from core.packing import pack_indices
from kernels.packed_attention import (
    PackedAttentionExecution,
    packed_key_scores,
    packed_key_scores_reference,
    packed_value_mix,
    packed_value_mix_reference,
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
