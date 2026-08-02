"""Fixed-capacity ownership and byte accounting for page-attention workspace."""

from __future__ import annotations

import sys

import torch

from kernels.qwen25_parallel_page_attention import ParallelPageWorkspace


_QUERY_HEADS = 28
_HEAD_DIM = 128
_MAX_ALLOCATION_BYTES = sys.maxsize
_FLOAT32_BYTES = 4
_PAGE_BYTES = (
    2 * _QUERY_HEADS * _FLOAT32_BYTES + _QUERY_HEADS * _HEAD_DIM * _FLOAT32_BYTES
)
_FIXED_BYTES = (
    _QUERY_HEADS * _HEAD_DIM * _FLOAT32_BYTES
    + 2 * _QUERY_HEADS * _FLOAT32_BYTES
    + _QUERY_HEADS * _HEAD_DIM * _FLOAT32_BYTES
)


def _require_page_capacity(page_capacity: object) -> int:
    if type(page_capacity) is not int:  # noqa: E721
        raise TypeError("page_capacity must be a built-in int.")
    if page_capacity < 1:
        raise ValueError("page_capacity must be at least one.")
    # Every workspace tensor is float32; check the aggregate allocation before
    # calling torch.empty so malformed capacities cannot partially reserve.
    if page_capacity * _PAGE_BYTES + _FIXED_BYTES > _MAX_ALLOCATION_BYTES:
        raise OverflowError("page workspace allocation exceeds the byte limit.")
    return page_capacity


def _canonical_device(device: torch.device) -> torch.device:
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device.")
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def reserve_parallel_page_workspace(
    *,
    page_capacity: int,
    device: torch.device,
) -> ParallelPageWorkspace:
    """Reserve one reusable float32 workspace for a session's max pages."""
    page_capacity = _require_page_capacity(page_capacity)
    device = _canonical_device(device)
    return ParallelPageWorkspace(
        rotated_query=torch.empty(
            (_QUERY_HEADS, _HEAD_DIM), dtype=torch.float32, device=device
        ),
        page_max=torch.empty(
            (page_capacity, _QUERY_HEADS), dtype=torch.float32, device=device
        ),
        page_denominator=torch.empty(
            (page_capacity, _QUERY_HEADS), dtype=torch.float32, device=device
        ),
        page_numerator=torch.empty(
            (page_capacity, _QUERY_HEADS, _HEAD_DIM), dtype=torch.float32, device=device
        ),
        reduced_max=torch.empty((_QUERY_HEADS,), dtype=torch.float32, device=device),
        reduced_denominator=torch.empty(
            (_QUERY_HEADS,), dtype=torch.float32, device=device
        ),
        reduced_numerator=torch.empty(
            (_QUERY_HEADS, _HEAD_DIM), dtype=torch.float32, device=device
        ),
    )


def parallel_page_workspace_nbytes(workspace: ParallelPageWorkspace) -> int:
    """Return allocated bytes for every workspace tensor exactly once."""
    if not isinstance(workspace, ParallelPageWorkspace):
        raise TypeError("workspace must be a ParallelPageWorkspace.")
    tensors = (
        workspace.rotated_query,
        workspace.page_max,
        workspace.page_denominator,
        workspace.page_numerator,
        workspace.reduced_max,
        workspace.reduced_denominator,
        workspace.reduced_numerator,
    )
    pointers: set[int] = set()
    total = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("workspace fields must be tensors.")
        pointer = tensor.untyped_storage().data_ptr()
        if pointer not in pointers:
            pointers.add(pointer)
            total += tensor.untyped_storage().nbytes()
    return total


__all__ = ["parallel_page_workspace_nbytes", "reserve_parallel_page_workspace"]
