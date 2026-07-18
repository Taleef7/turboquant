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


@dataclass
class PackedAttentionBenchmarkDebug:
    """Instrumentation proving prepared benchmark launches do not validate."""

    validation_passes: int = 0
    validation_reductions: int = 0
    timed_launches: int = 0


@dataclass
class PackedAttentionHotPathDebugState:
    """Opt-in accounting for the prepared per-layer decode hot path.

    Static validation intentionally performs the two GQA bound reductions once
    during preparation.  A subsequent ``launch`` must not perform scalar
    reductions or copy static constants to the device.
    """

    static_validation_passes: int = 0
    static_validation_reductions: int = 0
    constant_residency_checks: int = 0
    rotation_launches: int = 0
    packed_score_launches: int = 0
    packed_value_launches: int = 0
    tail_kernel_dispatches: int = 0


@dataclass
class PackedAttentionTailDebugState:
    """Dispatch accounting for the specialized raw Qwen2.5 fp16 tail kernel."""

    dispatches: int = 0
    tail_tokens: int = 0


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


_QWEN25_DIRECT_HEADS = 28
_QWEN25_DIRECT_KV_HEADS = 4
_QWEN25_DIRECT_GQA_RATIO = 7
_QWEN25_DIRECT_HEAD_DIM = 128
_QWEN25_DIRECT_MAX_TAIL_TOKENS = 383


if _TRITON_AVAILABLE:

    @triton.jit
    def _qwen25_fp16_tail_online_kernel(
        query_ptr,
        keys_ptr,
        values_ptr,
        old_max_ptr,
        old_denominator_ptr,
        old_numerator_ptr,
        new_max_ptr,
        new_denominator_ptr,
        new_numerator_ptr,
        tail_tokens,
        SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """One program per Q head; stream the bounded raw fp16 tail in registers."""
        head = tl.program_id(0)
        kv_head = head // 7
        dims = tl.arange(0, 128)
        query = tl.load(
            query_ptr + head * 128 + dims
        ).to(tl.float32)
        running_max = tl.load(old_max_ptr + head).to(tl.float32)
        running_denominator = tl.load(old_denominator_ptr + head).to(tl.float32)
        running_numerator = tl.load(
            old_numerator_ptr + head * 128 + dims
        ).to(tl.float32)
        for token_start in tl.static_range(
            0, 383, BLOCK_T
        ):
            tokens = token_start + tl.arange(0, BLOCK_T)
            token_mask = tokens < tail_tokens
            offsets = (
                (kv_head * tail_tokens + tokens)[:, None]
                * 128
                + dims[None, :]
            )
            keys = tl.load(
                keys_ptr + offsets,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            products = tl.reshape(keys * query[None, :], (BLOCK_T, 8, 16))
            partial_scores = tl.sum(products, axis=2)
            scores = tl.zeros((BLOCK_T,), dtype=tl.float32)
            compensation = tl.zeros((BLOCK_T,), dtype=tl.float32)
            partial_axis = tl.arange(0, 8)
            for partial_index in tl.static_range(0, 8):
                partial = tl.sum(
                    partial_scores
                    * (partial_axis[None, :] == partial_index),
                    axis=1,
                )
                compensated = partial - compensation
                updated = scores + compensated
                compensation = (updated - scores) - compensated
                scores = updated
            scores = scores * SCALE
            scores = tl.where(token_mask, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.where(
                running_max == -float("inf"),
                0.0,
                tl.exp(running_max - new_max),
            )
            weights = tl.exp(scores - new_max)
            running_denominator = (
                running_denominator * old_scale + tl.sum(weights, axis=0)
            )
            values = tl.load(
                values_ptr + offsets,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            running_numerator = (
                running_numerator * old_scale
                + tl.sum(values * weights[:, None], axis=0)
            )
            running_max = new_max
        tl.store(new_max_ptr + head, running_max)
        tl.store(new_denominator_ptr + head, running_denominator)
        tl.store(
            new_numerator_ptr + head * 128 + dims,
            running_numerator,
        )

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


@dataclass(frozen=True)
class _PreparedRotateQuery:
    query: torch.Tensor
    rotation: torch.Tensor
    heads: int
    query_tokens: int
    head_dim: int
    debug: PackedAttentionBenchmarkDebug | None = None

    def launch(self) -> torch.Tensor:
        if self.debug is not None:
            self.debug.timed_launches += 1
        output = torch.empty(
            (self.heads, self.query_tokens, self.head_dim),
            device=self.query.device,
            dtype=torch.float32,
        )
        grid = (self.heads * self.query_tokens, triton.cdiv(self.head_dim, 32))
        _rotate_query_kernel[grid](
            self.query,
            self.rotation,
            output,
            QUERY_TOKENS=self.query_tokens,
            HEAD_DIM=self.head_dim,
            BLOCK_OUT=32,
        )
        return output


@dataclass(frozen=True)
class _PreparedPackedKeyScores:
    rotated_query: torch.Tensor
    packed_keys: torch.Tensor
    codebook: torch.Tensor
    norms: torch.Tensor
    gqa_mapping: torch.Tensor
    heads: int
    query_tokens: int
    head_dim: int
    tokens: int
    packed_width: int
    key_bits: int
    scale: float
    debug: PackedAttentionBenchmarkDebug | None = None

    def launch(self) -> torch.Tensor:
        if self.debug is not None:
            self.debug.timed_launches += 1
        output = torch.empty(
            (self.heads, self.query_tokens, self.tokens),
            device=self.rotated_query.device,
            dtype=torch.float32,
        )
        grid = lambda meta: (
            self.heads * self.query_tokens,
            triton.cdiv(self.tokens, meta["BLOCK_T"]),
        )
        _packed_key_scores_kernel[grid](
            self.rotated_query,
            self.packed_keys,
            self.codebook,
            self.norms,
            self.gqa_mapping,
            output,
            QUERY_TOKENS=self.query_tokens,
            TOKEN_COUNT=self.tokens,
            PACKED_WIDTH=self.packed_width,
            HEAD_DIM=self.head_dim,
            KEY_BITS=self.key_bits,
            SCALE=self.scale,
        )
        return output


@dataclass(frozen=True)
class _PreparedPackedValueMix:
    scores: torch.Tensor
    packed_values: torch.Tensor
    mins: torch.Tensor
    scales: torch.Tensor
    gqa_mapping: torch.Tensor
    old_max: torch.Tensor
    old_denominator: torch.Tensor
    old_numerator: torch.Tensor
    heads: int
    query_tokens: int
    tokens: int
    head_dim: int
    value_bits: int
    group_size: int
    debug: PackedAttentionBenchmarkDebug | None = None

    def launch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.debug is not None:
            self.debug.timed_launches += 1
        new_max = torch.empty_like(self.old_max, dtype=torch.float32)
        new_denominator = torch.empty_like(self.old_denominator, dtype=torch.float32)
        new_numerator = torch.empty_like(self.old_numerator, dtype=torch.float32)
        _online_state_kernel[(self.heads * self.query_tokens,)](
            self.scores,
            self.old_max,
            self.old_denominator,
            new_max,
            new_denominator,
            QUERY_TOKENS=self.query_tokens,
            TOKEN_COUNT=self.tokens,
        )
        _packed_value_numerator_kernel[(self.heads * self.query_tokens, 1)](
            self.scores,
            self.packed_values,
            self.mins,
            self.scales,
            self.gqa_mapping,
            self.old_max,
            self.old_numerator,
            new_max,
            new_numerator,
            QUERY_TOKENS=self.query_tokens,
            TOKEN_COUNT=self.tokens,
            PACKED_WIDTH=self.packed_values.shape[-1],
            HEAD_DIM=self.head_dim,
            GROUP_SIZE=self.group_size,
            VALUE_BITS=self.value_bits,
        )
        return new_max, new_denominator, new_numerator


def _benchmark_debug_validation(
    debug: PackedAttentionBenchmarkDebug | None, *, reductions: int
) -> None:
    if debug is not None:
        debug.validation_passes += 1
        debug.validation_reductions += reductions


def prepare_rotate_query(
    query: torch.Tensor,
    rotation: torch.Tensor,
    *,
    debug: PackedAttentionBenchmarkDebug | None = None,
) -> _PreparedRotateQuery:
    """Validate/prepare a rotation launch once, outside a benchmark timer."""
    if not (_TRITON_AVAILABLE and isinstance(query, torch.Tensor) and query.is_cuda):
        raise RuntimeError("prepared rotation launch requires CUDA and Triton.")
    if (
        query.ndim != 3
        or query.dtype not in _FLOAT_DTYPES
        or rotation.shape != (query.shape[-1], query.shape[-1])
        or rotation.device != query.device
    ):
        raise ValueError("query/rotation must be compatible [heads, tokens, head_dim] tensors.")
    _benchmark_debug_validation(debug, reductions=0)
    heads, query_tokens, head_dim = query.shape
    return _PreparedRotateQuery(
        query.contiguous(), rotation.float().contiguous(), heads, query_tokens, head_dim, debug
    )


def prepare_packed_key_scores_rotated(
    rotated_query: torch.Tensor,
    packed_keys: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    key_bits: int,
    scale: float,
    debug: PackedAttentionBenchmarkDebug | None = None,
) -> _PreparedPackedKeyScores:
    """Validate/prepare a score launch once, excluding reductions from timing."""
    if not (_TRITON_AVAILABLE and isinstance(rotated_query, torch.Tensor) and rotated_query.is_cuda):
        raise RuntimeError("prepared score launch requires CUDA and Triton.")
    _validate_bits(key_bits, "key_bits")
    _require_packed_tensor(packed_keys, "packed_keys")
    if rotated_query.ndim != 3:
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
    _benchmark_debug_validation(debug, reductions=2)
    return _PreparedPackedKeyScores(
        rotated_query.contiguous(),
        packed_keys.contiguous(),
        codebook.float().contiguous(),
        norms.contiguous(),
        gqa_mapping.contiguous(),
        heads,
        query_tokens,
        head_dim,
        tokens,
        packed_width,
        key_bits,
        float(scale),
        debug,
    )


def prepare_packed_value_mix(
    scores: torch.Tensor,
    packed_values: torch.Tensor,
    mins: torch.Tensor,
    scales: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    value_bits: int,
    group_size: int,
    prior_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    debug: PackedAttentionBenchmarkDebug | None = None,
) -> _PreparedPackedValueMix:
    """Validate/prepare the two-kernel online value launch outside timing."""
    if not (_TRITON_AVAILABLE and isinstance(scores, torch.Tensor) and scores.is_cuda):
        raise RuntimeError("prepared value launch requires CUDA and Triton.")
    heads, query_tokens, tokens, head_dim = _validate_value_inputs(
        scores,
        packed_values,
        mins,
        scales,
        gqa_mapping,
        value_bits,
        group_size,
        prior_state,
    )
    _benchmark_debug_validation(debug, reductions=2)
    old_max, old_denominator, old_numerator = prior_state
    return _PreparedPackedValueMix(
        scores.contiguous(),
        packed_values.contiguous(),
        mins.contiguous(),
        scales.contiguous(),
        gqa_mapping.contiguous(),
        old_max.contiguous(),
        old_denominator.contiguous(),
        old_numerator.contiguous(),
        heads,
        query_tokens,
        tokens,
        head_dim,
        value_bits,
        group_size,
        debug,
    )


def qwen25_fp16_tail_online_update(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    prior_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scale: float,
    debug: PackedAttentionTailDebugState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge one raw Qwen2.5 fp16 tail into a packed-attention online state.

    This intentionally targets the direct Qwen2.5 decode layout only: 28 query
    heads, four KV heads, GQA ratio seven, head dimension 128, and a bounded
    tail of at most 383 tokens.  The kernel maps query to KV heads internally,
    so it never materializes an expanded ``[28, tokens, 128]`` K/V tensor.
    """
    if (
        not isinstance(query, torch.Tensor)
        or query.shape != (_QWEN25_DIRECT_HEADS, 1, _QWEN25_DIRECT_HEAD_DIM)
        or query.dtype != torch.float16
    ):
        raise ValueError(
            "direct Qwen2.5 fp16 tail requires query [28, 1, 128] torch.float16."
        )
    if not (_TRITON_AVAILABLE and torch.cuda.is_available() and query.is_cuda):
        raise RuntimeError("direct Qwen2.5 fp16 tail update requires CUDA and Triton.")
    if (
        not isinstance(keys, torch.Tensor)
        or not isinstance(values, torch.Tensor)
        or keys.ndim != 3
        or keys.shape != values.shape
        or keys.shape[0] != _QWEN25_DIRECT_KV_HEADS
        or keys.shape[-1] != _QWEN25_DIRECT_HEAD_DIM
        or not 1 <= keys.shape[1] <= _QWEN25_DIRECT_MAX_TAIL_TOKENS
        or keys.dtype != torch.float16
        or values.dtype != torch.float16
        or keys.device != query.device
        or values.device != query.device
        or not keys.is_contiguous()
        or not values.is_contiguous()
    ):
        raise ValueError(
            "direct Qwen2.5 fp16 tail requires contiguous K/V [4, tokens, 128] "
            "torch.float16 tensors on the query device, with 1 <= tokens <= 383."
        )
    if not isinstance(prior_state, tuple) or len(prior_state) != 3:
        raise ValueError("prior_state must be (max, denominator, numerator).")
    old_max, old_denominator, old_numerator = prior_state
    if (
        not all(isinstance(tensor, torch.Tensor) for tensor in prior_state)
        or old_max.shape != (_QWEN25_DIRECT_HEADS, 1)
        or old_denominator.shape != (_QWEN25_DIRECT_HEADS, 1)
        or old_numerator.shape
        != (_QWEN25_DIRECT_HEADS, 1, _QWEN25_DIRECT_HEAD_DIM)
        or old_max.dtype != torch.float32
        or old_denominator.dtype != torch.float32
        or old_numerator.dtype != torch.float32
        or any(tensor.device != query.device for tensor in prior_state)
        or not all(tensor.is_contiguous() for tensor in prior_state)
    ):
        raise ValueError(
            "direct Qwen2.5 fp16 tail requires contiguous fp32 online state "
            "([28, 1], [28, 1], [28, 1, 128]) on the query device."
        )
    if not math.isfinite(scale):
        raise ValueError("scale must be finite.")

    new_max = torch.empty_like(old_max)
    new_denominator = torch.empty_like(old_denominator)
    new_numerator = torch.empty_like(old_numerator)
    _qwen25_fp16_tail_online_kernel[(_QWEN25_DIRECT_HEADS,)](
        query.contiguous(),
        keys,
        values,
        old_max,
        old_denominator,
        old_numerator,
        new_max,
        new_denominator,
        new_numerator,
        keys.shape[1],
        SCALE=float(scale),
        BLOCK_T=16,
        num_warps=4,
    )
    if debug is not None:
        debug.dispatches += 1
        debug.tail_tokens = keys.shape[1]
    return new_max, new_denominator, new_numerator


@dataclass(frozen=True)
class PreparedPackedAttentionDecode:
    """Validated static packed layout for one Qwen attention layer.

    The immutable packed blocks and their device-resident constants are
    prepared outside token generation.  ``launch`` receives only the dynamic
    query and bounded fp16 tail, and deliberately bypasses the public generic
    helpers so their validation reductions cannot synchronize every block.
    """

    blocks: tuple[PackedAttentionBlock, ...]
    rotation: torch.Tensor
    gqa_mapping: torch.Tensor
    gqa_mapping_long: torch.Tensor
    heads: int
    head_dim: int
    num_kv_heads: int
    scale: float
    static_constant_ptrs: tuple[int, ...]
    static_constant_device: torch.device
    debug: PackedAttentionHotPathDebugState | None = None

    def verify_static_constants_resident(self) -> None:
        """Explicitly verify a prepared plan's immutable device constants.

        This is a test/debug operation.  It deliberately stays out of
        :meth:`launch`, whose per-token setup must not scan every packed block.
        """
        constants = (
            self.rotation,
            self.gqa_mapping,
            self.gqa_mapping_long,
            *(block.key_codebook for block in self.blocks),
        )
        if (
            len(constants) != len(self.static_constant_ptrs)
            or any(
                tensor.device != self.static_constant_device
                or tensor.data_ptr() != pointer
                for tensor, pointer in zip(constants, self.static_constant_ptrs, strict=True)
            )
        ):
            raise RuntimeError("prepared packed-attention constants lost device residency.")
        if self.debug is not None:
            self.debug.constant_residency_checks += 1

    def launch(
        self,
        query: torch.Tensor,
        *,
        fp16_window: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run one decode token without revalidating static GPU inputs."""
        if (
            not isinstance(query, torch.Tensor)
            or query.ndim != 3
            or query.shape[0] != self.heads
            or query.shape[-1] != self.head_dim
            or query.dtype not in _FLOAT_DTYPES
            or query.device != self.rotation.device
        ):
            raise ValueError("query must match the prepared heads, head_dim, dtype, and device.")
        query_tokens = query.shape[1]
        if query_tokens < 1:
            raise ValueError("query must contain at least one token.")
        rotated_query = torch.empty(
            (self.heads, query_tokens, self.head_dim),
            device=query.device,
            dtype=torch.float32,
        )
        _rotate_query_kernel[(self.heads * query_tokens, triton.cdiv(self.head_dim, 32))](
            query.contiguous(),
            self.rotation,
            rotated_query,
            QUERY_TOKENS=query_tokens,
            HEAD_DIM=self.head_dim,
            BLOCK_OUT=32,
        )
        if self.debug is not None:
            self.debug.rotation_launches += 1
        state = (
            torch.full(
                (self.heads, query_tokens),
                -torch.inf,
                device=query.device,
                dtype=torch.float32,
            ),
            torch.zeros((self.heads, query_tokens), device=query.device, dtype=torch.float32),
            torch.zeros(
                (self.heads, query_tokens, self.head_dim),
                device=query.device,
                dtype=torch.float32,
            ),
        )
        saw_history = False
        for block in self.blocks:
            tokens = block.packed_keys.shape[1]
            scores = torch.empty(
                (self.heads, query_tokens, tokens),
                device=query.device,
                dtype=torch.float32,
            )
            grid = lambda meta: (
                self.heads * query_tokens,
                triton.cdiv(tokens, meta["BLOCK_T"]),
            )
            _packed_key_scores_kernel[grid](
                rotated_query,
                block.packed_keys,
                block.key_codebook,
                block.key_norms,
                self.gqa_mapping,
                scores,
                QUERY_TOKENS=query_tokens,
                TOKEN_COUNT=tokens,
                PACKED_WIDTH=block.packed_keys.shape[-1],
                HEAD_DIM=self.head_dim,
                KEY_BITS=block.key_bits,
                SCALE=self.scale,
            )
            old_max, old_denominator, old_numerator = state
            new_max = torch.empty_like(old_max, dtype=torch.float32)
            new_denominator = torch.empty_like(old_denominator, dtype=torch.float32)
            new_numerator = torch.empty_like(old_numerator, dtype=torch.float32)
            _online_state_kernel[(self.heads * query_tokens,)](
                scores,
                old_max,
                old_denominator,
                new_max,
                new_denominator,
                QUERY_TOKENS=query_tokens,
                TOKEN_COUNT=tokens,
            )
            _packed_value_numerator_kernel[(self.heads * query_tokens, 1)](
                scores,
                block.packed_values,
                block.value_mins,
                block.value_scales,
                self.gqa_mapping,
                old_max,
                old_numerator,
                new_max,
                new_numerator,
                QUERY_TOKENS=query_tokens,
                TOKEN_COUNT=tokens,
                PACKED_WIDTH=block.packed_values.shape[-1],
                HEAD_DIM=self.head_dim,
                GROUP_SIZE=block.value_group_size,
                VALUE_BITS=block.value_bits,
            )
            state = (new_max, new_denominator, new_numerator)
            if self.debug is not None:
                self.debug.packed_score_launches += 1
                self.debug.packed_value_launches += 1
            saw_history = True
        if fp16_window is not None:
            window_keys, window_values = fp16_window
            if (
                not isinstance(window_keys, torch.Tensor)
                or not isinstance(window_values, torch.Tensor)
                or window_keys.ndim != 3
                or window_values.shape != window_keys.shape
                or window_keys.shape[0] != self.num_kv_heads
                or window_keys.shape[1] < 1
                or window_keys.shape[-1] != self.head_dim
                or window_keys.device != query.device
                or window_values.device != query.device
            ):
                raise ValueError("fp16_window must match the prepared KV-head layout and device.")
            old_max, old_denominator, old_numerator = state
            if (
                self.heads == _QWEN25_DIRECT_HEADS
                and self.num_kv_heads == _QWEN25_DIRECT_KV_HEADS
                and self.head_dim == _QWEN25_DIRECT_HEAD_DIM
                and query_tokens == 1
                and query.dtype == torch.float16
                and window_keys.dtype == torch.float16
                and window_values.dtype == torch.float16
                and window_keys.shape[1] <= _QWEN25_DIRECT_MAX_TAIL_TOKENS
                and window_keys.is_contiguous()
                and window_values.is_contiguous()
            ):
                state = qwen25_fp16_tail_online_update(
                    query,
                    window_keys,
                    window_values,
                    prior_state=(old_max, old_denominator, old_numerator),
                    scale=self.scale,
                )
                if self.debug is not None:
                    self.debug.tail_kernel_dispatches += 1
            else:
                mapped_keys = window_keys.float()[self.gqa_mapping_long]
                mapped_values = window_values.float()[self.gqa_mapping_long]
                window_scores = torch.einsum("hqd,htd->hqt", query.float(), mapped_keys)
                window_scores = window_scores * self.scale
                block_max = window_scores.max(dim=-1).values
                new_max = torch.maximum(old_max, block_max)
                old_scale = torch.where(
                    torch.isfinite(old_max),
                    torch.exp(old_max - new_max),
                    torch.zeros_like(old_max),
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
        return (state[2] / state[1].unsqueeze(-1)).to(query.dtype)


def prepare_packed_attention_decode(
    blocks: Iterable[PackedAttentionBlock],
    rotation: torch.Tensor,
    gqa_mapping: torch.Tensor,
    *,
    num_kv_heads: int | None = None,
    scale: float | None = None,
    debug: PackedAttentionHotPathDebugState | None = None,
) -> PreparedPackedAttentionDecode:
    """Validate static CUDA state once for repeated one-layer decode launches."""
    if not (_TRITON_AVAILABLE and isinstance(rotation, torch.Tensor) and rotation.is_cuda):
        raise RuntimeError("prepared packed-attention decode requires CUDA and Triton.")
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("rotation must have shape [head_dim, head_dim].")
    if not isinstance(gqa_mapping, torch.Tensor) or gqa_mapping.ndim != 1:
        raise ValueError("gqa_mapping must have shape [heads].")
    if gqa_mapping.device != rotation.device:
        raise ValueError("rotation and gqa_mapping must share a device.")
    prepared_blocks = tuple(blocks)
    head_dim = rotation.shape[0]
    inferred_kv_heads = None
    for block in prepared_blocks:
        if not isinstance(block, PackedAttentionBlock):
            raise TypeError("blocks must yield PackedAttentionBlock instances.")
        _validate_bits(block.key_bits, "block.key_bits")
        _validate_bits(block.value_bits, "block.value_bits")
        _require_packed_tensor(block.packed_keys, "block.packed_keys")
        _require_packed_tensor(block.packed_values, "block.packed_values")
        kv_heads, tokens, key_width = block.packed_keys.shape
        if inferred_kv_heads is None:
            inferred_kv_heads = kv_heads
        if (
            kv_heads != inferred_kv_heads
            or block.packed_values.shape[:2] != (kv_heads, tokens)
            or key_width != math.ceil(head_dim * block.key_bits / 8)
            or block.packed_values.shape[-1] != math.ceil(head_dim * block.value_bits / 8)
            or block.key_norms.shape != (kv_heads, tokens)
            or block.key_codebook.shape != (1 << block.key_bits,)
            or block.value_mins.shape != (kv_heads, tokens, head_dim // block.value_group_size)
            or block.value_scales.shape != block.value_mins.shape
            or any(
                tensor.device != rotation.device
                for tensor in (
                    block.packed_keys,
                    block.key_norms,
                    block.key_codebook,
                    block.packed_values,
                    block.value_mins,
                    block.value_scales,
                )
            )
        ):
            raise ValueError("packed block layout does not match the prepared rotation/device.")
    kv_heads = inferred_kv_heads if inferred_kv_heads is not None else num_kv_heads
    if type(kv_heads) is not int or kv_heads < 1:
        raise ValueError("num_kv_heads is required when preparing an empty block sequence.")
    if num_kv_heads is not None and num_kv_heads != kv_heads:
        raise ValueError("num_kv_heads does not match the packed block layout.")
    if gqa_mapping.shape[0] < 1 or int(gqa_mapping.min()) < 0 or int(gqa_mapping.max()) >= kv_heads:
        raise ValueError("gqa_mapping contains an invalid KV-head index.")
    attention_scale = 1 / math.sqrt(head_dim) if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError("scale must be finite.")
    if debug is not None:
        if not isinstance(debug, PackedAttentionHotPathDebugState):
            raise TypeError("debug must be PackedAttentionHotPathDebugState or None.")
        debug.static_validation_passes += 1
        debug.static_validation_reductions += 2
    prepared_rotation = rotation.float().contiguous()
    prepared_gqa_mapping = gqa_mapping.contiguous()
    prepared_gqa_mapping_long = gqa_mapping.to(dtype=torch.long).contiguous()
    constants = (
        prepared_rotation,
        prepared_gqa_mapping,
        prepared_gqa_mapping_long,
        *(block.key_codebook for block in prepared_blocks),
    )
    return PreparedPackedAttentionDecode(
        prepared_blocks,
        prepared_rotation,
        prepared_gqa_mapping,
        prepared_gqa_mapping_long,
        gqa_mapping.shape[0],
        head_dim,
        kv_heads,
        attention_scale,
        tuple(tensor.data_ptr() for tensor in constants),
        rotation.device,
        debug,
    )


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
