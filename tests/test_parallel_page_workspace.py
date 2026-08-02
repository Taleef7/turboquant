"""GPU contracts for deterministic page reduction, tail merge, and reuse."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from core.parallel_page_workspace import (
    parallel_page_workspace_nbytes,
    reserve_parallel_page_workspace,
)
from kernels.qwen25_parallel_page_attention import (
    ParallelPageWorkspace,
    launch_qwen25_page_reduction,
    launch_qwen25_page_states,
    launch_qwen25_tail_merge,
)
from tests.test_qwen_parallel_page_kernel import _make_fixture, _oracle


def _require_gpu_triton() -> None:
    pytest.importorskip("triton", reason="Triton is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


@pytest.mark.parametrize("page_capacity", [0, -1, True, 1.0])
def test_workspace_reservation_rejects_invalid_capacity(page_capacity: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        reserve_parallel_page_workspace(
            page_capacity=page_capacity,
            device=torch.device("cpu"),  # type: ignore[arg-type]
        )


def test_cpu_workspace_reservation_and_storage_deduplication() -> None:
    workspace = reserve_parallel_page_workspace(
        page_capacity=3, device=torch.device("cpu")
    )
    assert workspace.page_max.shape == (3, 28)
    assert parallel_page_workspace_nbytes(workspace) == 72_576
    shared = torch.empty((28, 128), dtype=torch.float32)
    aliased = ParallelPageWorkspace(
        rotated_query=shared,
        page_max=shared,
        page_denominator=shared,
        page_numerator=shared,
        reduced_max=shared,
        reduced_denominator=shared,
        reduced_numerator=shared,
    )
    assert parallel_page_workspace_nbytes(aliased) == shared.untyped_storage().nbytes()


@pytest.mark.gpu
@pytest.mark.parametrize("page_count", [1, 2, 3, 31, 63])
def test_page_reduction_matches_chronological_oracle(page_count: int) -> None:
    _require_gpu_triton()
    arena, _codec, prepared, query, workspace = _make_fixture(page_count)
    scale = 1 / math.sqrt(128)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=page_count, scale=scale
    )
    launch_qwen25_page_reduction(workspace, used_pages=page_count)
    torch.cuda.synchronize()
    page_max, page_denominator, page_numerator = _oracle(
        arena, prepared, query, page_count, scale=scale
    )
    merged_max = page_max[0].clone()
    merged_denominator = page_denominator[0].clone()
    merged_numerator = page_numerator[0].clone()
    for page in range(1, page_count):
        next_max = torch.maximum(merged_max, page_max[page])
        old_scale = torch.exp(merged_max - next_max)
        page_scale = torch.exp(page_max[page] - next_max)
        merged_denominator = (
            merged_denominator * old_scale + page_denominator[page] * page_scale
        )
        merged_numerator = merged_numerator * old_scale.unsqueeze(-1) + page_numerator[
            page
        ] * page_scale.unsqueeze(-1)
        merged_max = next_max
    torch.testing.assert_close(workspace.reduced_max, merged_max, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(
        workspace.reduced_denominator,
        merged_denominator,
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        workspace.reduced_numerator,
        merged_numerator,
        rtol=3e-3,
        atol=3e-3,
    )
    first = workspace.reduced_numerator.clone()
    launch_qwen25_page_reduction(workspace, used_pages=page_count)
    torch.cuda.synchronize()
    assert torch.equal(first, workspace.reduced_numerator)


@pytest.mark.gpu
def test_page_reduction_zero_pages_initializes_empty_state() -> None:
    _require_gpu_triton()
    workspace = reserve_parallel_page_workspace(
        page_capacity=3, device=torch.device("cuda")
    )
    workspace.reduced_max.fill_(123.0)
    workspace.reduced_denominator.fill_(123.0)
    workspace.reduced_numerator.fill_(123.0)
    launch_qwen25_page_reduction(workspace, used_pages=0)
    torch.cuda.synchronize()
    assert torch.isneginf(workspace.reduced_max).all()
    assert torch.equal(
        workspace.reduced_denominator, torch.zeros_like(workspace.reduced_denominator)
    )
    assert torch.equal(
        workspace.reduced_numerator, torch.zeros_like(workspace.reduced_numerator)
    )


@pytest.mark.gpu
def test_tail_merge_zero_pages_and_zero_tail_is_finite_zero() -> None:
    _require_gpu_triton()
    workspace = reserve_parallel_page_workspace(
        page_capacity=1, device=torch.device("cuda")
    )
    launch_qwen25_page_reduction(workspace, used_pages=0)
    query = torch.randn((28, 128), device="cuda", dtype=torch.float16)
    tail_keys = torch.empty((1, 4, 383, 128), device="cuda", dtype=torch.float16)
    tail_values = torch.empty_like(tail_keys)
    output = torch.full((28, 128), 7.0, device="cuda", dtype=torch.float32)
    workspace.rotated_query.copy_(query.float())
    launch_qwen25_tail_merge(
        query,
        tail_keys,
        tail_values,
        workspace,
        tail_length=0,
        tail_head_stride=tail_keys.stride(1),
        tail_token_stride=tail_keys.stride(2),
        scale=1 / math.sqrt(128),
        output=output,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.gpu
def test_reduction_order_is_observable_on_rounding_sensitive_states() -> None:
    _require_gpu_triton()
    workspace = reserve_parallel_page_workspace(
        page_capacity=3, device=torch.device("cuda")
    )
    workspace.page_max.copy_(
        torch.tensor(
            [[8.0327606] * 28, [1.7483339] * 28, [0.8897810] * 28],
            device="cuda",
            dtype=torch.float32,
        )
    )
    workspace.page_denominator.copy_(
        torch.tensor(
            [[59443.121] * 28, [88786.664] * 28, [45104.133] * 28],
            device="cuda",
            dtype=torch.float32,
        )
    )
    workspace.page_numerator.copy_(workspace.page_denominator.unsqueeze(-1))
    launch_qwen25_page_reduction(workspace, used_pages=3)
    ascending = workspace.reduced_denominator.clone()
    workspace.page_denominator.copy_(workspace.page_denominator.flip(0))
    workspace.page_numerator.copy_(workspace.page_numerator.flip(0))
    launch_qwen25_page_reduction(workspace, used_pages=3)
    torch.cuda.synchronize()
    assert not torch.equal(ascending, workspace.reduced_denominator)


@pytest.mark.gpu
def test_reduction_and_tail_kernels_are_atomic_free() -> None:
    _require_gpu_triton()
    module = __import__(
        "kernels.qwen25_parallel_page_attention",
        fromlist=["_qwen25_page_reduction_kernel", "_qwen25_tail_merge_kernel"],
    )
    source = inspect.getsource(
        module._qwen25_page_reduction_kernel.fn
    ) + inspect.getsource(module._qwen25_tail_merge_kernel.fn)
    assert "atomic" not in source.lower()


@pytest.mark.gpu
@pytest.mark.parametrize("tail_length", [1, 127, 128, 255, 256, 320, 383])
def test_tail_merge_honors_fixed_strides_and_matches_reference(
    tail_length: int,
) -> None:
    _require_gpu_triton()
    arena, _codec, prepared, query, workspace = _make_fixture(
        1, seed=1200 + tail_length
    )
    scale = 1 / math.sqrt(128)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=1, scale=scale
    )
    launch_qwen25_page_reduction(workspace, used_pages=1)
    # Add a one-token gap between the view and the backing tail to ensure the
    # kernel uses the supplied head/token strides rather than assuming dense rows.
    capacity = 383
    backing_keys = torch.randn(
        (1, 4, capacity + 5, 128), device="cuda", dtype=torch.float16
    )
    backing_values = torch.randn_like(backing_keys)
    tail_keys = backing_keys[:, :, 1 : capacity + 1, :]
    tail_values = backing_values[:, :, 1 : capacity + 1, :]
    output = torch.empty((28, 128), device="cuda", dtype=torch.float32)
    launch_qwen25_tail_merge(
        query,
        tail_keys,
        tail_values,
        workspace,
        tail_length=tail_length,
        tail_head_stride=tail_keys.stride(1),
        tail_token_stride=tail_keys.stride(2),
        scale=scale,
        output=output,
    )
    torch.cuda.synchronize()
    rotated = query.float() @ prepared.rotation.T  # type: ignore[union-attr]
    page_max, page_denominator, page_numerator = _oracle(
        arena, prepared, query, 1, scale=scale
    )
    expected = torch.empty((28, 128), device="cuda", dtype=torch.float32)
    for head in range(28):
        maximum = page_max[0, head]
        denominator = page_denominator[0, head]
        numerator = page_numerator[0, head]
        for token in range(tail_length):
            kv_head = head // 7
            score = (
                torch.dot(rotated[head], tail_keys[0, kv_head, token].float()) * scale
            )
            next_max = torch.maximum(maximum, score)
            old_scale = torch.exp(maximum - next_max)
            token_scale = torch.exp(score - next_max)
            denominator = denominator * old_scale + token_scale
            numerator = (
                numerator * old_scale
                + tail_values[0, kv_head, token].float() * token_scale
            )
            maximum = next_max
        expected[head] = numerator / denominator
    torch.testing.assert_close(output, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.gpu
def test_tail_merge_does_not_call_contiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_gpu_triton()
    arena, _codec, prepared, query, workspace = _make_fixture(1, seed=8181)
    scale = 1 / math.sqrt(128)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=1, scale=scale
    )
    launch_qwen25_page_reduction(workspace, used_pages=1)
    tail_keys = torch.randn((1, 4, 8, 128), device="cuda", dtype=torch.float16)
    tail_values = torch.randn_like(tail_keys)
    output = torch.empty((28, 128), device="cuda", dtype=torch.float32)

    def fail_contiguous(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("tail merge called contiguous()")

    monkeypatch.setattr(torch.Tensor, "contiguous", fail_contiguous)
    launch_qwen25_tail_merge(
        query,
        tail_keys,
        tail_values,
        workspace,
        tail_length=8,
        tail_head_stride=tail_keys.stride(1),
        tail_token_stride=tail_keys.stride(2),
        scale=scale,
        output=output,
    )
    torch.cuda.synchronize()


@pytest.mark.gpu
def test_tail_merge_rejects_nonpositive_supplied_stride() -> None:
    _require_gpu_triton()
    arena, _codec, prepared, query, workspace = _make_fixture(1, seed=8182)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=1, scale=1 / math.sqrt(128)
    )
    launch_qwen25_page_reduction(workspace, used_pages=1)
    tail_keys = torch.randn((1, 4, 8, 128), device="cuda", dtype=torch.float16)
    tail_values = torch.randn_like(tail_keys)
    output = torch.empty((28, 128), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="strides must be positive"):
        launch_qwen25_tail_merge(
            query,
            tail_keys,
            tail_values,
            workspace,
            tail_length=8,
            tail_head_stride=0,
            tail_token_stride=tail_keys.stride(2),
            scale=1 / math.sqrt(128),
            output=output,
        )


@pytest.mark.gpu
def test_workspace_is_fixed_capacity_reusable_and_exactly_accounted() -> None:
    _require_gpu_triton()
    workspace = reserve_parallel_page_workspace(
        page_capacity=3, device=torch.device("cuda")
    )
    pointers = tuple(tensor.data_ptr() for tensor in workspace.__dict__.values())
    assert parallel_page_workspace_nbytes(workspace) == 72_576
    for _ in range(28):
        assert (
            tuple(tensor.data_ptr() for tensor in workspace.__dict__.values())
            == pointers
        )
    assert workspace.page_max.shape == (3, 28)


@pytest.mark.gpu
def test_reduction_and_tail_merge_make_no_torch_tensor_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu_triton()
    arena, _codec, prepared, query, workspace = _make_fixture(1, seed=4040)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=1, scale=1 / math.sqrt(128)
    )
    launch_qwen25_page_reduction(workspace, used_pages=1)
    tail_keys = arena.tail_keys
    tail_values = arena.tail_values
    tail_keys[..., :8, :].normal_()
    tail_values[..., :8, :].normal_()
    output = torch.empty((28, 128), device="cuda", dtype=torch.float16)
    for name in ("empty", "empty_like", "full", "zeros"):
        monkeypatch.setattr(
            torch, name, lambda *_a, **_k: (_ for _ in ()).throw(AssertionError(name))
        )
    launch_qwen25_page_reduction(workspace, used_pages=1)
    launch_qwen25_tail_merge(
        query,
        tail_keys,
        tail_values,
        workspace,
        tail_length=8,
        tail_head_stride=tail_keys.stride(1),
        tail_token_stride=tail_keys.stride(2),
        scale=1 / math.sqrt(128),
        output=output,
    )
    torch.cuda.synchronize()
