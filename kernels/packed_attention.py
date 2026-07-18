"""Clean-room Triton kernels for direct attention over byte-packed KV payloads.

The public GPU entry points never silently decode with PyTorch: callers that
want a CPU/reference result must opt in through ``allow_torch_fallback`` or
call the explicit ``*_reference`` functions.  Packed indices are decoded from
their byte stream in registers; no decoded key/value block is materialized by
the Triton path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import torch

from core.packing import unpack_indices

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on CPU-only installations.
    _TRITON_AVAILABLE = False


_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})


@dataclass(frozen=True)
class PackedAttentionExecution:
    """Dispatch provenance returned by qualified packed-kernel calls."""

    kernel: Literal["triton", "torch_reference"]
    used_torch_fallback: bool


@dataclass
class PackedAttentionDebugState:
    """Bounded-history accounting for the independent decode coordinator."""

    blocks_consumed: int = 0
    max_block_tokens: int = 0
    joined_history_attempted: bool = False


@dataclass(frozen=True)
class PackedAttentionBlock:
    """One chronological, independently packed KV block for decode coordination."""

    packed_keys: torch.Tensor
    key_norms: torch.Tensor
    key_codebook: torch.Tensor
    key_bits: int
    packed_values: torch.Tensor
    value_mins: torch.Tensor
    value_scales: torch.Tensor
    value_bits: int
    value_group_size: int


PACKED_ATTENTION_COMPUTE_SANITIZER_COMMAND = (
    "compute-sanitizer --tool memcheck --error-exitcode 1 "
    "/home/taleef/.venvs/turboquant-forge/bin/python -m pytest "
    "tests/test_packed_attention_kernels.py -m gpu -q"
)


if _TRITON_AVAILABLE:

    @triton.jit
    def _rotate_query_kernel(
        query_ptr,
        rotation_ptr,
        rotated_query_ptr,
        QUERY_TOKENS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_out = tl.program_id(1)
        output_dims = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        input_dims = tl.arange(0, HEAD_DIM)
        query = tl.load(query_ptr + pid_hq * HEAD_DIM + input_dims).to(tl.float32)
        rotation = tl.load(
            rotation_ptr + output_dims[:, None] * HEAD_DIM + input_dims[None, :],
            mask=output_dims[:, None] < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.sum(rotation * query[None, :], axis=1)
        tl.store(
            rotated_query_ptr + pid_hq * HEAD_DIM + output_dims,
            rotated,
            mask=output_dims < HEAD_DIM,
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 16}, num_warps=4),
            triton.Config({"BLOCK_T": 32}, num_warps=4),
            triton.Config({"BLOCK_T": 64}, num_warps=8),
        ],
        key=["HEAD_DIM", "TOKEN_COUNT"],
    )
    @triton.jit
    def _packed_key_scores_kernel(
        query_ptr,
        packed_keys_ptr,
        codebook_ptr,
        norms_ptr,
        mapping_ptr,
        scores_ptr,
        QUERY_TOKENS: tl.constexpr,
        TOKEN_COUNT: tl.constexpr,
        PACKED_WIDTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_BITS: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_t = tl.program_id(1)
        head = pid_hq // QUERY_TOKENS
        query_token = pid_hq % QUERY_TOKENS
        token = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        dims = tl.arange(0, HEAD_DIM)
        kv_head = tl.load(mapping_ptr + head).to(tl.int64)
        query = tl.load(query_ptr + (head * QUERY_TOKENS + query_token) * HEAD_DIM + dims)
        bit_position = dims * KEY_BITS
        byte_offset = bit_position // 8
        shift = bit_position % 8
        row = (kv_head * TOKEN_COUNT + token)[:, None]
        base = row * PACKED_WIDTH + byte_offset[None, :]
        mask = token[:, None] < TOKEN_COUNT
        low = tl.load(packed_keys_ptr + base, mask=mask, other=0).to(tl.int32)
        high = tl.load(
            packed_keys_ptr + base + 1,
            mask=mask & (byte_offset[None, :] + 1 < PACKED_WIDTH),
            other=0,
        ).to(tl.int32)
        indices = ((low | (high << 8)) >> shift[None, :]) & ((1 << KEY_BITS) - 1)
        values = tl.load(codebook_ptr + indices).to(tl.float32)
        dot = tl.sum(values * query[None, :], axis=1)
        norms = tl.load(norms_ptr + kv_head * TOKEN_COUNT + token, mask=token < TOKEN_COUNT, other=0.0)
        output = (head * QUERY_TOKENS + query_token) * TOKEN_COUNT + token
        tl.store(scores_ptr + output, dot * norms * SCALE, mask=token < TOKEN_COUNT)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 256}, num_warps=4),
            triton.Config({"BLOCK_T": 256}, num_warps=8),
        ],
        key=["TOKEN_COUNT"],
    )
    @triton.jit
    def _online_state_kernel(
        scores_ptr,
        prior_max_ptr,
        prior_denominator_ptr,
        max_ptr,
        denominator_ptr,
        QUERY_TOKENS: tl.constexpr,
        TOKEN_COUNT: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        token = tl.arange(0, BLOCK_T)
        scores = tl.load(
            scores_ptr + pid * TOKEN_COUNT + token,
            mask=token < TOKEN_COUNT,
            other=-float("inf"),
        ).to(tl.float32)
        old_max = tl.load(prior_max_ptr + pid).to(tl.float32)
        old_denominator = tl.load(prior_denominator_ptr + pid).to(tl.float32)
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(old_max, block_max)
        old_scale = tl.where(old_max == -float("inf"), 0.0, tl.exp(old_max - new_max))
        weights = tl.exp(scores - new_max)
        denominator = old_denominator * old_scale + tl.sum(weights, axis=0)
        tl.store(max_ptr + pid, new_max)
        tl.store(denominator_ptr + pid, denominator)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 256}, num_warps=4),
            triton.Config({"BLOCK_T": 256}, num_warps=8),
        ],
        key=["HEAD_DIM", "TOKEN_COUNT"],
    )
    @triton.jit
    def _packed_value_numerator_kernel(
        scores_ptr,
        packed_values_ptr,
        mins_ptr,
        scales_ptr,
        mapping_ptr,
        prior_max_ptr,
        prior_numerator_ptr,
        max_ptr,
        numerator_ptr,
        QUERY_TOKENS: tl.constexpr,
        TOKEN_COUNT: tl.constexpr,
        PACKED_WIDTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_hq = tl.program_id(0)
        pid_d = tl.program_id(1)
        head = pid_hq // QUERY_TOKENS
        dims = pid_d * HEAD_DIM + tl.arange(0, HEAD_DIM)
        token = tl.arange(0, BLOCK_T)
        kv_head = tl.load(mapping_ptr + head).to(tl.int64)
        scores = tl.load(
            scores_ptr + pid_hq * TOKEN_COUNT + token,
            mask=token < TOKEN_COUNT,
            other=-float("inf"),
        ).to(tl.float32)
        new_max = tl.load(max_ptr + pid_hq).to(tl.float32)
        old_max = tl.load(prior_max_ptr + pid_hq).to(tl.float32)
        old_scale = tl.where(old_max == -float("inf"), 0.0, tl.exp(old_max - new_max))
        weights = tl.exp(scores - new_max)
        bit_position = dims * VALUE_BITS
        byte_offset = bit_position // 8
        shift = bit_position % 8
        row = (kv_head * TOKEN_COUNT + token)[:, None]
        base = row * PACKED_WIDTH + byte_offset[None, :]
        mask = token[:, None] < TOKEN_COUNT
        low = tl.load(packed_values_ptr + base, mask=mask, other=0).to(tl.int32)
        high = tl.load(
            packed_values_ptr + base + 1,
            mask=mask & (byte_offset[None, :] + 1 < PACKED_WIDTH),
            other=0,
        ).to(tl.int32)
        indices = ((low | (high << 8)) >> shift[None, :]) & ((1 << VALUE_BITS) - 1)
        group = dims // GROUP_SIZE
        metadata = row * (HEAD_DIM // GROUP_SIZE) + group[None, :]
        mins = tl.load(mins_ptr + metadata, mask=mask, other=0.0).to(tl.float32)
        scales = tl.load(scales_ptr + metadata, mask=mask, other=0.0).to(tl.float32)
        values = mins + indices.to(tl.float32) * scales
        previous = tl.load(prior_numerator_ptr + pid_hq * HEAD_DIM + dims).to(tl.float32)
        contribution = tl.sum(values * weights[:, None], axis=0)
        tl.store(numerator_ptr + pid_hq * HEAD_DIM + dims, previous * old_scale + contribution)


def _require_packed_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must be a torch.uint8 tensor.")
    if tensor.ndim != 3 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(f"{name} must have shape [kv_heads, tokens, packed_bytes].")


def _validate_bits(bits: int, name: str) -> None:
    if type(bits) is not int or bits not in {4, 6, 8}:
        raise ValueError(f"{name} must be one of 4, 6, or 8.")


def _validate_score_inputs(
    query: torch.Tensor,
    packed_keys: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    rotation: torch.Tensor,
    gqa_mapping: torch.Tensor,
    key_bits: int,
) -> tuple[int, int, int, int]:
    _validate_bits(key_bits, "key_bits")
    _require_packed_tensor(packed_keys, "packed_keys")
    if not isinstance(query, torch.Tensor) or query.ndim != 3 or query.dtype not in _FLOAT_DTYPES:
        raise ValueError("query must have shape [heads, query_tokens, head_dim] and float dtype.")
    heads, query_tokens, head_dim = query.shape
    kv_heads, tokens, packed_width = packed_keys.shape
    if packed_keys.device != query.device:
        raise ValueError("packed_keys must be on query.device.")
    if packed_width != math.ceil(head_dim * key_bits / 8):
        raise ValueError("packed_keys width does not match head_dim and key_bits.")
    if codebook.shape != (1 << key_bits,) or codebook.device != query.device:
        raise ValueError("codebook must contain exactly 2**key_bits entries on query.device.")
    if norms.shape != (kv_heads, tokens) or norms.device != query.device:
        raise ValueError("norms must have shape [kv_heads, tokens] on query.device.")
    if rotation.shape != (head_dim, head_dim) or rotation.device != query.device:
        raise ValueError("rotation must have shape [head_dim, head_dim] on query.device.")
    if gqa_mapping.shape != (heads,) or gqa_mapping.device != query.device:
        raise ValueError("gqa_mapping must have shape [heads] on query.device.")
    if int(gqa_mapping.min()) < 0 or int(gqa_mapping.max()) >= kv_heads:
        raise ValueError("gqa_mapping contains an invalid KV-head index.")
    return heads, query_tokens, head_dim, tokens


def packed_key_scores_reference(
    query: torch.Tensor,
    packed_keys: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    rotation: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    key_bits: int,
    scale: float,
) -> torch.Tensor:
    """Direct PyTorch oracle; it intentionally unpacks the complete key tile."""
    heads, _, head_dim, tokens = _validate_score_inputs(
        query, packed_keys, codebook, norms, rotation, gqa_mapping, key_bits
    )
    decoded = unpack_indices(
        packed_keys.reshape(-1, packed_keys.shape[-1]), key_bits, head_dim
    ).reshape(packed_keys.shape[0], tokens, head_dim)
    rotated_query = query.float() @ rotation.float().T
    keys = codebook[decoded.long()].float() * norms.float().unsqueeze(-1)
    mapped_keys = keys[gqa_mapping.long()]
    if mapped_keys.shape[0] != heads:
        raise AssertionError("validated GQA mapping did not preserve query heads.")
    return torch.einsum("hqd,htd->hqt", rotated_query, mapped_keys) * float(scale)


def rotate_query(
    query: torch.Tensor,
    rotation: torch.Tensor,
    *,
    return_execution: bool = False,
    allow_torch_fallback: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PackedAttentionExecution]:
    """Rotate query heads with Triton before packed-key scoring."""
    if (
        not isinstance(query, torch.Tensor)
        or query.ndim != 3
        or query.dtype not in _FLOAT_DTYPES
        or rotation.shape != (query.shape[-1], query.shape[-1])
        or rotation.device != query.device
    ):
        raise ValueError("query/rotation must be compatible [heads, tokens, head_dim] tensors.")
    if not (_TRITON_AVAILABLE and query.is_cuda):
        if not allow_torch_fallback:
            raise RuntimeError(
                "rotate_query requires CUDA and Triton; use allow_torch_fallback=True "
                "only for unit/reference use."
            )
        result = query.float() @ rotation.float().T
        execution = PackedAttentionExecution("torch_reference", True)
        return (result, execution) if return_execution else result
    heads, query_tokens, head_dim = query.shape
    output = torch.empty(
        (heads, query_tokens, head_dim), device=query.device, dtype=torch.float32
    )
    grid = (heads * query_tokens, triton.cdiv(head_dim, 32))
    _rotate_query_kernel[grid](
        query.contiguous(), rotation.float().contiguous(), output,
        QUERY_TOKENS=query_tokens, HEAD_DIM=head_dim, BLOCK_OUT=32,
    )
    execution = PackedAttentionExecution("triton", False)
    return (output, execution) if return_execution else output


def packed_key_scores_rotated(
    rotated_query: torch.Tensor,
    packed_keys: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    key_bits: int,
    scale: float,
    return_execution: bool = False,
    allow_torch_fallback: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PackedAttentionExecution]:
    """Triton score kernel over a query already rotated by :func:`rotate_query`."""
    _validate_bits(key_bits, "key_bits")
    _require_packed_tensor(packed_keys, "packed_keys")
    if not isinstance(rotated_query, torch.Tensor) or rotated_query.ndim != 3:
        raise ValueError("rotated_query must have shape [heads, query_tokens, head_dim].")
    heads, query_tokens, head_dim = rotated_query.shape
    kv_heads, tokens, packed_width = packed_keys.shape
    if (
        packed_keys.device != rotated_query.device
        or codebook.shape != (1 << key_bits,)
        or codebook.device != rotated_query.device
        or norms.shape != (kv_heads, tokens)
        or norms.device != rotated_query.device
        or gqa_mapping.shape != (heads,)
        or gqa_mapping.device != rotated_query.device
        or packed_width != math.ceil(head_dim * key_bits / 8)
    ):
        raise ValueError("rotated score inputs have incompatible shapes or devices.")
    if int(gqa_mapping.min()) < 0 or int(gqa_mapping.max()) >= kv_heads:
        raise ValueError("gqa_mapping contains an invalid KV-head index.")
    if not (_TRITON_AVAILABLE and rotated_query.is_cuda):
        if not allow_torch_fallback:
            raise RuntimeError(
                "packed_key_scores_rotated requires CUDA and Triton; use an explicit "
                "reference function or allow_torch_fallback=True for unit-only use."
            )
        decoded = unpack_indices(
            packed_keys.reshape(-1, packed_width), key_bits, head_dim
        ).reshape(kv_heads, tokens, head_dim)
        result = torch.einsum(
            "hqd,htd->hqt",
            rotated_query.float(),
            codebook[decoded.long()].float()[gqa_mapping.long()]
            * norms.float()[gqa_mapping.long()].unsqueeze(-1),
        ) * float(scale)
        execution = PackedAttentionExecution("torch_reference", True)
        return (result, execution) if return_execution else result
    output = torch.empty(
        (heads, query_tokens, tokens), device=rotated_query.device, dtype=torch.float32
    )
    grid = lambda meta: (heads * query_tokens, triton.cdiv(tokens, meta["BLOCK_T"]))
    _packed_key_scores_kernel[grid](
        rotated_query.contiguous(), packed_keys.contiguous(), codebook.float().contiguous(),
        norms.contiguous(), gqa_mapping.contiguous(), output,
        QUERY_TOKENS=query_tokens, TOKEN_COUNT=tokens, PACKED_WIDTH=packed_width,
        HEAD_DIM=head_dim, KEY_BITS=key_bits, SCALE=float(scale),
    )
    execution = PackedAttentionExecution("triton", False)
    return (output, execution) if return_execution else output


def packed_key_scores(
    query: torch.Tensor,
    packed_keys: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    rotation: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    key_bits: int,
    scale: float,
    return_execution: bool = False,
    allow_torch_fallback: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PackedAttentionExecution]:
    """Score byte-packed, rotated key indices without a decoded-key allocation."""
    _validate_score_inputs(
        query, packed_keys, codebook, norms, rotation, gqa_mapping, key_bits
    )
    if not (_TRITON_AVAILABLE and query.is_cuda):
        if not allow_torch_fallback:
            raise RuntimeError(
                "packed_key_scores requires CUDA and Triton; use the explicit reference "
                "function or allow_torch_fallback=True for unit-only fallback."
            )
        result = packed_key_scores_reference(
            query, packed_keys, codebook, norms, rotation, gqa_mapping,
            key_bits=key_bits, scale=scale,
        )
        execution = PackedAttentionExecution("torch_reference", True)
        return (result, execution) if return_execution else result
    rotated_query, rotation_execution = rotate_query(
        query, rotation, return_execution=True, allow_torch_fallback=False
    )
    result, score_execution = packed_key_scores_rotated(
        rotated_query, packed_keys, codebook, norms, gqa_mapping, key_bits=key_bits,
        scale=scale, return_execution=True, allow_torch_fallback=False,
    )
    execution = PackedAttentionExecution(
        "torch_reference" if rotation_execution.used_torch_fallback else score_execution.kernel,
        rotation_execution.used_torch_fallback or score_execution.used_torch_fallback,
    )
    return (result, execution) if return_execution else result


def _validate_value_inputs(
    scores: torch.Tensor,
    packed_values: torch.Tensor,
    mins: torch.Tensor,
    scales: torch.Tensor,
    gqa_mapping: torch.Tensor,
    value_bits: int,
    group_size: int,
    prior_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[int, int, int, int]:
    _validate_bits(value_bits, "value_bits")
    _require_packed_tensor(packed_values, "packed_values")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise ValueError("scores must have shape [heads, query_tokens, tokens].")
    heads, query_tokens, tokens = scores.shape
    kv_heads, packed_tokens, packed_width = packed_values.shape
    if tokens > 256:
        raise ValueError("packed_value_mix accepts at most one 256-token packed block.")
    if (
        packed_values.device != scores.device
        or mins.device != scores.device
        or scales.device != scores.device
        or gqa_mapping.device != scores.device
    ):
        raise ValueError("packed value inputs and gqa_mapping must share scores.device.")
    if packed_tokens != tokens:
        raise ValueError("scores and packed_values must have the same token count.")
    head_dim = (packed_width * 8) // value_bits
    if packed_width != math.ceil(head_dim * value_bits / 8) or head_dim % group_size:
        raise ValueError("packed_values width is incompatible with value_bits/group_size.")
    groups = head_dim // group_size
    if mins.shape != (kv_heads, tokens, groups) or scales.shape != mins.shape:
        raise ValueError("mins/scales must have shape [kv_heads, tokens, head_dim/group_size].")
    if gqa_mapping.shape != (heads,) or int(gqa_mapping.min()) < 0 or int(gqa_mapping.max()) >= kv_heads:
        raise ValueError("gqa_mapping must map every query head to a valid KV head.")
    old_max, old_denominator, old_numerator = prior_state
    if old_max.shape != (heads, query_tokens) or old_denominator.shape != old_max.shape:
        raise ValueError("prior max and denominator must match scores without tokens.")
    if old_numerator.shape != (heads, query_tokens, head_dim):
        raise ValueError("prior numerator must include head_dim values.")
    return heads, query_tokens, tokens, head_dim


def packed_value_mix_reference(
    scores: torch.Tensor,
    packed_values: torch.Tensor,
    mins: torch.Tensor,
    scales: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    value_bits: int,
    group_size: int,
    prior_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct PyTorch online-softmax oracle for packed grouped values."""
    _, _, tokens, head_dim = _validate_value_inputs(
        scores, packed_values, mins, scales, gqa_mapping, value_bits, group_size, prior_state
    )
    old_max, old_denominator, old_numerator = prior_state
    decoded = unpack_indices(
        packed_values.reshape(-1, packed_values.shape[-1]), value_bits, head_dim
    ).reshape(*packed_values.shape[:2], head_dim).float()
    values = mins.float().unsqueeze(-1) + decoded.reshape(
        *decoded.shape[:2], head_dim // group_size, group_size
    ) * scales.float().unsqueeze(-1)
    values = values.reshape(*decoded.shape)[gqa_mapping.long()]
    block_max = scores.float().max(dim=-1).values
    new_max = torch.maximum(old_max.float(), block_max)
    old_scale = torch.where(torch.isfinite(old_max), torch.exp(old_max - new_max), 0.0)
    weights = torch.exp(scores.float() - new_max.unsqueeze(-1))
    denominator = old_denominator.float() * old_scale + weights.sum(dim=-1)
    numerator = old_numerator.float() * old_scale.unsqueeze(-1) + torch.einsum(
        "hqt,htd->hqd", weights, values
    )
    return new_max, denominator, numerator


def packed_value_mix(
    scores: torch.Tensor,
    packed_values: torch.Tensor,
    mins: torch.Tensor,
    scales: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    value_bits: int,
    group_size: int,
    prior_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    return_execution: bool = False,
    allow_torch_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor], PackedAttentionExecution
]:
    """Update fp32 online-softmax state from packed grouped values."""
    heads, query_tokens, tokens, head_dim = _validate_value_inputs(
        scores, packed_values, mins, scales, gqa_mapping, value_bits, group_size, prior_state
    )
    if not (_TRITON_AVAILABLE and scores.is_cuda):
        if not allow_torch_fallback:
            raise RuntimeError(
                "packed_value_mix requires CUDA and Triton; use the explicit reference "
                "function or allow_torch_fallback=True for unit-only fallback."
            )
        result = packed_value_mix_reference(
            scores, packed_values, mins, scales, gqa_mapping, value_bits=value_bits,
            group_size=group_size, prior_state=prior_state,
        )
        execution = PackedAttentionExecution("torch_reference", True)
        return (result, execution) if return_execution else result
    old_max, old_denominator, old_numerator = prior_state
    new_max = torch.empty_like(old_max, dtype=torch.float32)
    new_denominator = torch.empty_like(old_denominator, dtype=torch.float32)
    new_numerator = torch.empty_like(old_numerator, dtype=torch.float32)
    _online_state_kernel[(heads * query_tokens,)](
        scores.contiguous(), old_max.contiguous(), old_denominator.contiguous(), new_max,
        new_denominator, QUERY_TOKENS=query_tokens, TOKEN_COUNT=tokens,
    )
    _packed_value_numerator_kernel[(heads * query_tokens, 1)](
        scores.contiguous(), packed_values.contiguous(), mins.contiguous(), scales.contiguous(),
        gqa_mapping.contiguous(), old_max.contiguous(), old_numerator.contiguous(), new_max,
        new_numerator, QUERY_TOKENS=query_tokens, TOKEN_COUNT=tokens,
        PACKED_WIDTH=packed_values.shape[-1], HEAD_DIM=head_dim, GROUP_SIZE=group_size,
        VALUE_BITS=value_bits,
    )
    result = (new_max, new_denominator, new_numerator)
    execution = PackedAttentionExecution("triton", False)
    return (result, execution) if return_execution else result


def packed_attention_decode(
    query: torch.Tensor,
    blocks: Iterable[PackedAttentionBlock],
    rotation: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    scale: float | None = None,
    fp16_window: tuple[torch.Tensor, torch.Tensor] | None = None,
    debug: PackedAttentionDebugState | None = None,
    return_execution: bool = False,
    allow_torch_fallback: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PackedAttentionExecution]:
    """Decode packed blocks in sequence with fp32 online state and a local tail.

    ``blocks`` is consumed one at a time; it is never joined into an all-history
    decoded tensor.  The optional fp16 window is intentionally processed as its
    own bounded local segment.  It is not a substitute for a packed-kernel
    fallback, so the returned dispatch provenance still reports whether a
    packed score/value kernel was replaced by the reference implementation.
    """
    if not isinstance(query, torch.Tensor) or query.ndim != 3:
        raise ValueError("query must have shape [heads, query_tokens, head_dim].")
    if debug is not None and not isinstance(debug, PackedAttentionDebugState):
        raise TypeError("debug must be PackedAttentionDebugState or None.")
    heads, query_tokens, head_dim = query.shape
    attention_scale = 1 / math.sqrt(head_dim) if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError("scale must be finite.")
    state = (
        torch.full((heads, query_tokens), -torch.inf, device=query.device, dtype=torch.float32),
        torch.zeros((heads, query_tokens), device=query.device, dtype=torch.float32),
        torch.zeros((heads, query_tokens, head_dim), device=query.device, dtype=torch.float32),
    )
    saw_history = False
    used_torch_fallback = False
    for block in blocks:
        if not isinstance(block, PackedAttentionBlock):
            raise TypeError("blocks must yield PackedAttentionBlock instances.")
        if block.packed_keys.shape[:2] != block.packed_values.shape[:2]:
            raise ValueError("packed key and value blocks must share KV-head and token dimensions.")
        if debug is not None:
            debug.blocks_consumed += 1
            debug.max_block_tokens = max(debug.max_block_tokens, block.packed_keys.shape[1])
        scores, score_execution = packed_key_scores(
            query, block.packed_keys, block.key_codebook, block.key_norms, rotation,
            gqa_mapping, key_bits=block.key_bits, scale=attention_scale,
            return_execution=True, allow_torch_fallback=allow_torch_fallback,
        )
        state, value_execution = packed_value_mix(
            scores, block.packed_values, block.value_mins, block.value_scales,
            gqa_mapping, value_bits=block.value_bits, group_size=block.value_group_size,
            prior_state=state, return_execution=True,
            allow_torch_fallback=allow_torch_fallback,
        )
        used_torch_fallback |= (
            score_execution.used_torch_fallback or value_execution.used_torch_fallback
        )
        saw_history = True
    if fp16_window is not None:
        window_keys, window_values = fp16_window
        if (
            window_keys.ndim != 3
            or window_values.shape != window_keys.shape
            or window_keys.shape[-1] != head_dim
            or window_keys.shape[0] < 1
        ):
            raise ValueError("fp16_window must contain matching [kv_heads, tokens, head_dim] tensors.")
        if window_keys.shape[1] < 1:
            raise ValueError("fp16_window must contain at least one token.")
        if gqa_mapping.shape != (heads,):
            raise ValueError("gqa_mapping must have one KV-head mapping per query head.")
        if int(gqa_mapping.min()) < 0 or int(gqa_mapping.max()) >= window_keys.shape[0]:
            raise ValueError("gqa_mapping contains an invalid fp16-window KV head.")
        mapped_keys = window_keys.float()[gqa_mapping.long()]
        mapped_values = window_values.float()[gqa_mapping.long()]
        window_scores = torch.einsum("hqd,htd->hqt", query.float(), mapped_keys)
        window_scores = window_scores * attention_scale
        old_max, old_denominator, old_numerator = state
        block_max = window_scores.max(dim=-1).values
        new_max = torch.maximum(old_max, block_max)
        old_scale = torch.where(
            torch.isfinite(old_max), torch.exp(old_max - new_max), torch.zeros_like(old_max)
        )
        weights = torch.exp(window_scores - new_max.unsqueeze(-1))
        state = (
            new_max,
            old_denominator * old_scale + weights.sum(dim=-1),
            old_numerator * old_scale.unsqueeze(-1)
            + torch.einsum("hqt,htd->hqd", weights, mapped_values),
        )
        saw_history = True
    if not saw_history:
        raise ValueError("cannot decode packed attention with no KV history.")
    result = (state[2] / state[1].unsqueeze(-1)).to(query.dtype)
    execution = PackedAttentionExecution(
        "torch_reference" if used_torch_fallback else "triton", used_torch_fallback
    )
    return (result, execution) if return_execution else result
