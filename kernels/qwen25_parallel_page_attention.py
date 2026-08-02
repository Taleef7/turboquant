"""Independent packed-page attention states for the canonical Qwen2.5 layout.

Task 4 deliberately stops at the page-state stage.  Each Triton program owns
one ``(page, query_head)`` pair and writes an independent stable softmax state
to preallocated workspace.  Chronological reduction and raw-tail merging are
separate Task 5 work; this module therefore never performs atomics or silently
falls back to PyTorch.
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


__all__ = ["ParallelPageWorkspace", "launch_qwen25_page_states", "page_state_grid"]
