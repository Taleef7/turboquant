"""Atomic-free packed-page attention for the canonical Qwen2.5 layout.

The page-state launch computes one independent stable softmax state per
``(page, query_head)``.  Task 5 adds a deterministic chronological reduction and
a fixed-stride raw-tail merge.  All three launches write preallocated tensors;
none allocates a tensor, uses atomics, or silently falls back to PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from core.packed_page_arena import PackedPageArena
    from core.codec import PreparedKeyCodecDevice

try:  # pragma: no cover - availability is exercised by GPU-marked tests.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU environments intentionally skip GPU tests.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


_QUERY_HEADS = 28
_KV_HEADS = 4
_HEAD_DIM = 128
_PAGE_TOKENS = 256
_KEY_WIDTH = 128
_VALUE_WIDTH = 96
_VALUE_GROUPS = 4
_KEY_BITS = 8
_VALUE_BITS = 6
_VALUE_GROUP_SIZE = 32
_GQA_RATIO = _QUERY_HEADS // _KV_HEADS
_SUPPORTED_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class ParallelPageWorkspace:
    """Preallocated outputs for one independent page-state launch."""

    rotated_query: torch.Tensor
    page_max: torch.Tensor
    page_denominator: torch.Tensor
    page_numerator: torch.Tensor
    reduced_max: torch.Tensor
    reduced_denominator: torch.Tensor
    reduced_numerator: torch.Tensor


def _require_triton_cuda() -> None:
    if not _TRITON_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError(
            "launch_qwen25_page_states requires a CUDA device and Triton; "
            "use the explicit PyTorch oracle in tests for CPU reference work."
        )


def page_state_grid(used_pages: int) -> tuple[int, int]:
    """Return the invariant Triton grid: one program per page/query head."""
    if type(used_pages) is not int or used_pages < 0:  # noqa: E721
        raise ValueError("used_pages must be a non-negative built-in int.")
    return used_pages, _QUERY_HEADS


def _validate_query(
    query: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, int]:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor.")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("query must use fp16, bf16, or fp32.")
    if query.device != device:
        raise ValueError("query must reside on the arena device.")
    if query.ndim == 2:
        if query.shape != (_QUERY_HEADS, _HEAD_DIM):
            raise ValueError("query must have shape [28, 128].")
        return query, query.shape[0]
    if query.ndim == 4 and tuple(query.shape) == (1, _QUERY_HEADS, 1, _HEAD_DIM):
        return query.reshape(_QUERY_HEADS, _HEAD_DIM), _QUERY_HEADS
    raise ValueError("query must have shape [28, 128] or [1, 28, 1, 128].")


def _validate_workspace(
    workspace: ParallelPageWorkspace,
    *,
    device: torch.device,
    used_pages: int,
) -> None:
    if not isinstance(workspace, ParallelPageWorkspace):
        raise TypeError("workspace must be a ParallelPageWorkspace.")
    if workspace.rotated_query.shape != (_QUERY_HEADS, _HEAD_DIM):
        raise ValueError("workspace.rotated_query must have shape [28, 128].")
    for name, tensor in (
        ("rotated_query", workspace.rotated_query),
        ("page_max", workspace.page_max),
        ("page_denominator", workspace.page_denominator),
        ("page_numerator", workspace.page_numerator),
        ("reduced_max", workspace.reduced_max),
        ("reduced_denominator", workspace.reduced_denominator),
        ("reduced_numerator", workspace.reduced_numerator),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.device != device:
            raise ValueError(f"workspace.{name} must be a tensor on {device}.")
    if workspace.page_max.ndim != 2 or workspace.page_max.shape[0] < used_pages:
        raise ValueError("workspace.page_max must cover used pages.")
    if workspace.page_denominator.shape != workspace.page_max.shape:
        raise ValueError("page denominator shape must match page max shape.")
    if (
        workspace.page_max.shape[1] != _QUERY_HEADS
        or workspace.page_numerator.ndim != 3
        or workspace.page_numerator.shape[0] < used_pages
        or workspace.page_numerator.shape[1] != _QUERY_HEADS
        or workspace.page_numerator.shape[2] != _HEAD_DIM
    ):
        raise ValueError("page workspace tensors must use [pages, 28, 128] layout.")
    if workspace.page_max.dtype != torch.float32:
        raise TypeError("page_max must use torch.float32.")
    if workspace.page_denominator.dtype != torch.float32:
        raise TypeError("page_denominator must use torch.float32.")
    if workspace.page_numerator.dtype != torch.float32:
        raise TypeError("page_numerator must use torch.float32.")
    if workspace.rotated_query.dtype != torch.float32:
        raise TypeError("workspace.rotated_query must use torch.float32.")
    if workspace.reduced_max.shape != (_QUERY_HEADS,):
        raise ValueError("workspace.reduced_max must have shape [28].")
    if workspace.reduced_denominator.shape != (_QUERY_HEADS,):
        raise ValueError("workspace.reduced_denominator must have shape [28].")
    if workspace.reduced_numerator.shape != (_QUERY_HEADS, _HEAD_DIM):
        raise ValueError("workspace.reduced_numerator must have shape [28, 128].")
    for name, tensor in (
        ("reduced_max", workspace.reduced_max),
        ("reduced_denominator", workspace.reduced_denominator),
        ("reduced_numerator", workspace.reduced_numerator),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"workspace.{name} must use torch.float32.")
    if not all(
        tensor.is_contiguous()
        for tensor in (
            workspace.rotated_query,
            workspace.page_max,
            workspace.page_denominator,
            workspace.page_numerator,
            workspace.reduced_max,
            workspace.reduced_denominator,
            workspace.reduced_numerator,
        )
    ):
        raise ValueError("workspace tensors must be contiguous.")


def _validate_arena(
    arena: "PackedPageArena", prepared: "PreparedKeyCodecDevice"
) -> torch.device:
    from core.packed_page_arena import PackedPageArena

    if not isinstance(arena, PackedPageArena):
        raise TypeError("arena must be a PackedPageArena.")
    layout = arena.layout
    if (
        layout.page_tokens != _PAGE_TOKENS
        or layout.num_kv_heads != _KV_HEADS
        or layout.head_dim != _HEAD_DIM
        or layout.key_packed_width != _KEY_WIDTH
        or layout.value_packed_width != _VALUE_WIDTH
        or layout.value_groups != _VALUE_GROUPS
    ):
        raise ValueError(
            "arena does not match the canonical Qwen2.5 packed page layout."
        )
    if arena.kv_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("arena kv_dtype must use fp16 or bf16.")
    from core.codec import PreparedKeyCodecDevice

    if not isinstance(prepared, PreparedKeyCodecDevice):
        raise TypeError("prepared must be a PreparedKeyCodecDevice.")
    if prepared.centroids.device != arena.device or prepared.centroids.shape != (256,):
        raise ValueError("prepared key centroids must be [256] on the arena device.")
    if prepared.qjl_projection is not None:
        raise ValueError(
            "QJL prepared constants are not supported by this page kernel."
        )
    return arena.device


if _TRITON_AVAILABLE:

    @triton.jit
    def _qwen25_query_rotation_kernel(
        query_ptr,
        rotation_ptr,
        rotated_query_ptr,
        BLOCK_OUT: tl.constexpr,
    ):
        """Rotate one contiguous [28, 128] query without an output allocation."""
        head = tl.program_id(0)
        output_block = tl.program_id(1)
        output_dims = output_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        input_dims = tl.arange(0, 128)
        query_values = tl.load(query_ptr + head * 128 + input_dims).to(tl.float32)
        rotation = tl.load(
            rotation_ptr + output_dims[:, None] * 128 + input_dims[None, :],
            mask=output_dims[:, None] < 128,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.sum(rotation * query_values[None, :], axis=1)
        tl.store(
            rotated_query_ptr + head * 128 + output_dims,
            rotated,
            mask=output_dims < 128,
        )

    @triton.jit
    def _qwen25_page_states_kernel(
        rotated_query_ptr,
        packed_keys_ptr,
        key_centroids_ptr,
        key_norms_ptr,
        packed_values_ptr,
        value_mins_ptr,
        value_scales_ptr,
        page_max_ptr,
        page_denominator_ptr,
        page_numerator_ptr,
        SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Compute one independent stable softmax state for one page/head."""
        page = tl.program_id(0)
        head = tl.program_id(1)
        kv_head = head // 7
        dims = tl.arange(0, 128)
        query = tl.load(rotated_query_ptr + head * 128 + dims).to(tl.float32)

        page_key_base = page * 4 * 256 * 128
        page_norm_base = page * 4 * 256
        page_value_base = page * 4 * 256 * 96
        page_meta_base = page * 4 * 256 * 4
        row_base = kv_head * 256

        page_maximum = -float("inf")
        for token_start in tl.static_range(0, 256, BLOCK_T):
            tokens = token_start + tl.arange(0, BLOCK_T)
            rows = row_base + tokens
            key_offsets = page_key_base + rows[:, None] * 128 + dims[None, :]
            key_indices = tl.load(packed_keys_ptr + key_offsets).to(tl.int32)
            decoded_keys = tl.load(key_centroids_ptr + key_indices).to(tl.float32)
            norms = tl.load(
                key_norms_ptr + page_norm_base + rows, mask=tokens < 256, other=0.0
            ).to(tl.float32)
            scores = tl.sum(decoded_keys * query[None, :], axis=1) * norms * SCALE
            page_maximum = tl.maximum(page_maximum, tl.max(scores, axis=0))

        denominator = tl.zeros((), dtype=tl.float32)
        denominator_compensation = tl.zeros((), dtype=tl.float32)
        numerator = tl.zeros((128,), dtype=tl.float32)
        numerator_compensation = tl.zeros((128,), dtype=tl.float32)
        value_bit_position = dims * 6
        value_byte_offset = value_bit_position // 8
        value_shift = value_bit_position % 8
        value_group = dims // 32

        for token_start in tl.static_range(0, 256, BLOCK_T):
            tokens = token_start + tl.arange(0, BLOCK_T)
            rows = row_base + tokens
            key_offsets = page_key_base + rows[:, None] * 128 + dims[None, :]
            key_indices = tl.load(packed_keys_ptr + key_offsets).to(tl.int32)
            decoded_keys = tl.load(key_centroids_ptr + key_indices).to(tl.float32)
            norms = tl.load(key_norms_ptr + page_norm_base + rows).to(tl.float32)
            scores = tl.sum(decoded_keys * query[None, :], axis=1) * norms * SCALE
            weights = tl.exp(scores - page_maximum)

            denominator_tile = tl.sum(weights, axis=0)
            denominator_adjusted = denominator_tile - denominator_compensation
            denominator_updated = denominator + denominator_adjusted
            denominator_compensation = (
                denominator_updated - denominator - denominator_adjusted
            )
            denominator = denominator_updated

            value_offsets = (
                page_value_base + rows[:, None] * 96 + value_byte_offset[None, :]
            )
            low = tl.load(packed_values_ptr + value_offsets).to(tl.int32)
            high = tl.load(
                packed_values_ptr + value_offsets + 1,
                mask=value_byte_offset[None, :] + 1 < 96,
                other=0,
            ).to(tl.int32)
            value_indices = ((low | (high << 8)) >> value_shift[None, :]) & 63
            metadata = page_meta_base + rows[:, None] * 4 + value_group[None, :]
            mins = tl.load(value_mins_ptr + metadata).to(tl.float32)
            scales = tl.load(value_scales_ptr + metadata).to(tl.float32)
            values = mins + value_indices.to(tl.float32) * scales
            numerator_tile = tl.sum(values * weights[:, None], axis=0)
            numerator_adjusted = numerator_tile - numerator_compensation
            numerator_updated = numerator + numerator_adjusted
            numerator_compensation = numerator_updated - numerator - numerator_adjusted
            numerator = numerator_updated

        output_state = page * 28 + head
        tl.store(page_max_ptr + output_state, page_maximum)
        tl.store(page_denominator_ptr + output_state, denominator)
        tl.store(
            page_numerator_ptr + output_state * 128 + dims,
            numerator,
        )

    @triton.jit
    def _qwen25_page_reduction_kernel(
        page_max_ptr,
        page_denominator_ptr,
        page_numerator_ptr,
        reduced_max_ptr,
        reduced_denominator_ptr,
        reduced_numerator_ptr,
        USED_PAGES: tl.constexpr,
    ):
        """Merge page states in ascending page order for one query head."""
        head = tl.program_id(0)
        dims = tl.arange(0, 128)
        merged_max = tl.full((), -float("inf"), dtype=tl.float32)
        merged_denominator = tl.zeros((), dtype=tl.float32)
        merged_numerator = tl.zeros((128,), dtype=tl.float32)

        for page in tl.static_range(0, USED_PAGES):
            state = page * 28 + head
            page_maximum = tl.load(page_max_ptr + state)
            page_denominator = tl.load(page_denominator_ptr + state)
            page_numerator = tl.load(page_numerator_ptr + state * 128 + dims)
            next_max = tl.maximum(merged_max, page_maximum)
            previous_scale = tl.where(
                merged_denominator > 0.0,
                tl.exp(merged_max - next_max),
                0.0,
            )
            page_scale = tl.exp(page_maximum - next_max)
            merged_denominator = (
                merged_denominator * previous_scale + page_denominator * page_scale
            )
            merged_numerator = (
                merged_numerator * previous_scale + page_numerator * page_scale
            )
            merged_max = next_max

        tl.store(reduced_max_ptr + head, merged_max)
        tl.store(reduced_denominator_ptr + head, merged_denominator)
        tl.store(reduced_numerator_ptr + head * 128 + dims, merged_numerator)

    @triton.jit
    def _qwen25_tail_merge_kernel(
        query_ptr,
        tail_keys_ptr,
        tail_values_ptr,
        reduced_max_ptr,
        reduced_denominator_ptr,
        reduced_numerator_ptr,
        output_ptr,
        tail_length,
        tail_head_stride,
        tail_token_stride,
        output_head_stride,
        output_dim_stride,
        SCALE: tl.constexpr,
        TAIL_CAPACITY: tl.constexpr,
    ):
        """Merge a fixed-stride raw tail after the reduced packed pages."""
        head = tl.program_id(0)
        kv_head = head // 7
        dims = tl.arange(0, 128)
        # The caller chooses the tail query space.  The adapter passes the raw
        # query for model-space tail keys; low-level callers default to the
        # codec-rotated workspace query.
        query = tl.load(query_ptr + head * 128 + dims).to(tl.float32)
        merged_max = tl.load(reduced_max_ptr + head)
        merged_denominator = tl.load(reduced_denominator_ptr + head)
        merged_numerator = tl.load(reduced_numerator_ptr + head * 128 + dims)

        for token in tl.range(0, TAIL_CAPACITY):
            token_id = tl.full((), token, dtype=tl.int32)
            active = token_id < tail_length
            key_offsets = kv_head * tail_head_stride + token * tail_token_stride + dims
            key = tl.load(tail_keys_ptr + key_offsets, mask=active, other=0.0)
            score = tl.sum(query * key.to(tl.float32), axis=0) * SCALE
            score = tl.where(active, score, -float("inf"))
            next_max = tl.maximum(merged_max, score)
            previous_scale = tl.where(
                merged_denominator > 0.0,
                tl.exp(merged_max - next_max),
                0.0,
            )
            token_scale = tl.where(active, tl.exp(score - next_max), 0.0)
            value_offsets = (
                kv_head * tail_head_stride + token * tail_token_stride + dims
            )
            value = tl.load(
                tail_values_ptr + value_offsets,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            merged_denominator = merged_denominator * previous_scale + token_scale
            merged_numerator = merged_numerator * previous_scale + value * token_scale
            merged_max = next_max

        normalized = tl.where(
            merged_denominator > 0.0,
            merged_numerator / merged_denominator,
            0.0,
        )
        output_offsets = head * output_head_stride + dims * output_dim_stride
        tl.store(output_ptr + output_offsets, normalized)


def launch_qwen25_page_states(
    query: torch.Tensor,
    arena: "PackedPageArena",
    prepared_codec: "PreparedKeyCodecDevice",
    workspace: ParallelPageWorkspace,
    *,
    used_pages: int,
    scale: float,
) -> None:
    """Launch exactly one Triton program per ``(page, query_head)`` pair."""
    _require_triton_cuda()
    device = _validate_arena(arena, prepared_codec)
    if type(used_pages) is not int:  # noqa: E721
        raise TypeError("used_pages must be a built-in int.")
    if not 0 <= used_pages <= arena.used_pages:
        raise ValueError("used_pages must be within the arena's committed page range.")
    if (
        not isinstance(scale, (float, int))
        or isinstance(scale, bool)
        or not math.isfinite(scale)
    ):
        raise ValueError("scale must be finite.")
    query_2d, query_heads = _validate_query(query, device)
    if query_heads != _QUERY_HEADS:
        raise ValueError("query must contain exactly 28 query heads.")
    _validate_workspace(workspace, device=device, used_pages=used_pages)
    if prepared_codec.rotation.device != device or prepared_codec.rotation.shape != (
        _HEAD_DIM,
        _HEAD_DIM,
    ):
        raise ValueError("prepared rotation must be [128, 128] on the arena device.")
    if workspace.rotated_query.data_ptr() == query_2d.data_ptr():
        # The kernel only reads rotated_query.  Refuse accidental aliasing with
        # raw query rather than silently interpreting unrotated values.
        raise ValueError("workspace.rotated_query must not alias the raw query.")
    if used_pages == 0:
        return
    _qwen25_page_states_kernel[page_state_grid(used_pages)](
        workspace.rotated_query,
        arena.key_indices,
        prepared_codec.centroids,
        arena.key_norms,
        arena.value_indices,
        arena.value_mins,
        arena.value_scales,
        workspace.page_max,
        workspace.page_denominator,
        workspace.page_numerator,
        SCALE=float(scale),
        BLOCK_T=16,
        num_warps=4,
    )


def launch_qwen25_query_rotation(
    query: torch.Tensor,
    rotation: torch.Tensor,
    workspace: ParallelPageWorkspace,
) -> None:
    """Rotate one decode query into the reusable float32 workspace."""
    _require_triton_cuda()
    if not isinstance(workspace, ParallelPageWorkspace):
        raise TypeError("workspace must be a ParallelPageWorkspace.")
    if not isinstance(rotation, torch.Tensor):
        raise TypeError("rotation must be a torch.Tensor.")
    device = workspace.rotated_query.device
    query_2d, _ = _validate_query(query, device)
    _validate_workspace(workspace, device=device, used_pages=0)
    if (
        rotation.device != device
        or rotation.shape != (_HEAD_DIM, _HEAD_DIM)
        or rotation.dtype != torch.float32
        or not rotation.is_contiguous()
    ):
        raise ValueError(
            "rotation must be contiguous float32 [128, 128] on the workspace device."
        )
    if not query_2d.is_contiguous():
        raise ValueError("query must be contiguous for allocation-free rotation.")
    _qwen25_query_rotation_kernel[
        (
            _QUERY_HEADS,
            4,
        )
    ](
        query_2d,
        rotation,
        workspace.rotated_query,
        BLOCK_OUT=32,
    )


def launch_qwen25_page_reduction(
    workspace: ParallelPageWorkspace,
    *,
    used_pages: int,
) -> None:
    """Merge page states chronologically with one program per query head."""
    _require_triton_cuda()
    if not isinstance(workspace, ParallelPageWorkspace):
        raise TypeError("workspace must be a ParallelPageWorkspace.")
    if type(used_pages) is not int:  # noqa: E721
        raise TypeError("used_pages must be a built-in int.")
    if used_pages < 0 or used_pages > workspace.page_max.shape[0]:
        raise ValueError("used_pages must be within the workspace page capacity.")
    device = workspace.rotated_query.device
    _validate_workspace(workspace, device=device, used_pages=used_pages)
    if used_pages == 0:
        workspace.reduced_max.fill_(-float("inf"))
        workspace.reduced_denominator.zero_()
        workspace.reduced_numerator.zero_()
        return
    _qwen25_page_reduction_kernel[(_QUERY_HEADS,)](
        workspace.page_max,
        workspace.page_denominator,
        workspace.page_numerator,
        workspace.reduced_max,
        workspace.reduced_denominator,
        workspace.reduced_numerator,
        USED_PAGES=used_pages,
        num_warps=4,
    )


def _validate_tail(
    tail_keys: torch.Tensor,
    tail_values: torch.Tensor,
    *,
    device: torch.device,
    tail_length: int,
    tail_head_stride: int,
    tail_token_stride: int,
) -> int:
    if not isinstance(tail_keys, torch.Tensor) or not isinstance(
        tail_values, torch.Tensor
    ):
        raise TypeError("tail_keys and tail_values must be tensors.")
    if tail_keys.device != device or tail_values.device != device:
        raise ValueError("tail tensors must reside on the workspace device.")
    if (
        tail_keys.ndim != 4
        or tail_keys.shape[0] != 1
        or tail_keys.shape[1] != _KV_HEADS
    ):
        raise ValueError("tail tensors must have shape [1, 4, capacity, 128].")
    if tail_keys.shape[-1] != _HEAD_DIM or tail_values.shape != tail_keys.shape:
        raise ValueError(
            "tail tensors must have matching [1, 4, capacity, 128] shapes."
        )
    if (
        tail_keys.dtype not in _SUPPORTED_FLOAT_DTYPES
        or tail_values.dtype != tail_keys.dtype
    ):
        raise TypeError(
            "tail tensors must use fp16, bf16, or fp32 with matching dtypes."
        )
    if tail_keys.stride(-1) != 1 or tail_values.stride(-1) != 1:
        raise ValueError("tail feature dimensions must have unit stride.")
    if type(tail_length) is not int:  # noqa: E721
        raise TypeError("tail_length must be a built-in int.")
    if not 0 <= tail_length <= tail_keys.shape[-2]:
        raise ValueError("tail_length must fit the supplied tail capacity.")
    if type(tail_head_stride) is not int or type(tail_token_stride) is not int:  # noqa: E721
        raise TypeError("tail strides must be built-in ints.")
    if tail_head_stride <= 0 or tail_token_stride <= 0:
        raise ValueError("tail strides must be positive.")
    if tail_head_stride != tail_keys.stride(
        1
    ) or tail_head_stride != tail_values.stride(1):
        raise ValueError("tail_head_stride must match both tail tensor strides.")
    if tail_token_stride != tail_keys.stride(
        2
    ) or tail_token_stride != tail_values.stride(2):
        raise ValueError("tail_token_stride must match both tail tensor strides.")
    return tail_keys.shape[-2]


def launch_qwen25_tail_merge(
    query: torch.Tensor,
    tail_keys: torch.Tensor,
    tail_values: torch.Tensor,
    workspace: ParallelPageWorkspace,
    *,
    tail_length: int,
    tail_head_stride: int,
    tail_token_stride: int,
    scale: float,
    output: torch.Tensor,
    tail_query: torch.Tensor | None = None,
) -> None:
    """Merge reduced pages and a fixed-stride tail into ``output``.

    By default ``query`` is interpreted in codec-rotated space, matching the
    low-level page-kernel contract.  Adapters that retain raw model-space tail
    keys may pass ``tail_query`` explicitly; this avoids allocating a rotated
    tail-key buffer while preserving the public kernel behavior.
    """
    _require_triton_cuda()
    if not isinstance(workspace, ParallelPageWorkspace):
        raise TypeError("workspace must be a ParallelPageWorkspace.")
    device = workspace.rotated_query.device
    query_2d, _ = _validate_query(query, device)
    tail_query_2d = workspace.rotated_query
    if tail_query is not None:
        tail_query_2d, _ = _validate_query(tail_query, device)
    _validate_workspace(workspace, device=device, used_pages=0)
    tail_capacity = _validate_tail(
        tail_keys,
        tail_values,
        device=device,
        tail_length=tail_length,
        tail_head_stride=tail_head_stride,
        tail_token_stride=tail_token_stride,
    )
    if (
        not isinstance(scale, (float, int))
        or isinstance(scale, bool)
        or not math.isfinite(scale)
    ):
        raise ValueError("scale must be finite.")
    if not isinstance(output, torch.Tensor) or output.device != device:
        raise ValueError("output must be a tensor on the workspace device.")
    if output.ndim == 2 and tuple(output.shape) == (_QUERY_HEADS, _HEAD_DIM):
        output_2d = output
    elif output.ndim == 4 and tuple(output.shape) in {
        (1, _QUERY_HEADS, 1, _HEAD_DIM),
        (1, 1, _QUERY_HEADS, _HEAD_DIM),
    }:
        output_2d = output.view(_QUERY_HEADS, _HEAD_DIM)
    else:
        raise ValueError(
            "output must have shape [28, 128], [1, 28, 1, 128], or [1, 1, 28, 128]."
        )
    if output.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError("output must use fp16, bf16, or fp32.")
    if output_2d.stride(-1) != 1:
        raise ValueError("output feature dimension must have unit stride.")
    if output_2d.stride(0) <= 0:
        raise ValueError("output head stride must be positive.")
    _qwen25_tail_merge_kernel[(_QUERY_HEADS,)](
        tail_query_2d,
        tail_keys,
        tail_values,
        workspace.reduced_max,
        workspace.reduced_denominator,
        workspace.reduced_numerator,
        output_2d,
        tail_length,
        tail_head_stride,
        tail_token_stride,
        output_2d.stride(0),
        output_2d.stride(1),
        SCALE=float(scale),
        TAIL_CAPACITY=tail_capacity,
        num_warps=4,
    )


__all__ = [
    "ParallelPageWorkspace",
    "launch_qwen25_page_reduction",
    "launch_qwen25_page_states",
    "launch_qwen25_query_rotation",
    "launch_qwen25_tail_merge",
    "page_state_grid",
]
