"""GPU parity and launch contracts for independent Qwen packed-page states."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from core.codec import KeyCodec, KeyLayout
from core.packed_page_arena import PackedPageArena, PackedPageArenaLayout
from core.packing import pack_indices, unpack_indices
from kernels.qwen25_parallel_page_attention import (
    ParallelPageWorkspace,
    launch_qwen25_page_states,
    page_state_grid,
)


def _require_gpu_triton() -> None:
    pytest.importorskip(
        "triton", reason="Triton is unavailable for page-state coverage"
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def _make_fixture(
    page_count: int,
    *,
    dtype: torch.dtype = torch.float16,
    seed: int = 812,
) -> tuple[PackedPageArena, KeyCodec, object, torch.Tensor, ParallelPageWorkspace]:
    _require_gpu_triton()
    device = torch.device("cuda")
    layout = PackedPageArenaLayout(
        page_capacity=page_count,
        page_tokens=256,
        num_kv_heads=4,
        head_dim=128,
        key_packed_width=128,
        value_packed_width=96,
        value_groups=4,
        tail_capacity=383,
    )
    arena = PackedPageArena.reserve(layout, device=device, kv_dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed + page_count)
    codec = KeyCodec(KeyLayout(head_dim=128, bits=8, norm_dtype=dtype))
    prepared = codec.prepare_device(device)
    for page_id in range(page_count):
        key_indices = torch.randint(
            0, 256, (4, 256, 128), generator=generator, dtype=torch.uint8
        ).to(device)
        key_norms = (
            torch.rand((4, 256), generator=generator, dtype=torch.float32)
            .clamp_min(0.1)
            .to(dtype=dtype, device=device)
        )
        raw_values = torch.randint(
            0, 64, (4 * 256, 128), generator=generator, dtype=torch.uint8
        )
        value_indices = pack_indices(raw_values, 6).reshape(4, 256, 96).to(device)
        value_mins = torch.randn(
            (4, 256, 4), generator=generator, dtype=torch.float32
        ).to(dtype=dtype, device=device)
        value_scales = (
            torch.rand((4, 256, 4), generator=generator, dtype=torch.float32)
            .clamp_min(0.01)
            .to(dtype=dtype, device=device)
        )
        arena.append_encoded_page(
            page_id=page_id,
            key_indices=key_indices,
            key_norms=key_norms,
            value_indices=value_indices,
            value_mins=value_mins,
            value_scales=value_scales,
        )
    query = torch.randn((28, 128), generator=generator, dtype=torch.float16).to(device)
    rotated_query = query.float() @ prepared.rotation.T
    workspace = ParallelPageWorkspace(
        rotated_query=rotated_query.clone(),
        page_max=torch.full((page_count, 28), float("nan"), device=device),
        page_denominator=torch.full((page_count, 28), float("nan"), device=device),
        page_numerator=torch.full((page_count, 28, 128), float("nan"), device=device),
        reduced_max=torch.empty((28,), device=device),
        reduced_denominator=torch.empty((28,), device=device),
        reduced_numerator=torch.empty((28, 128), device=device),
    )
    return arena, codec, prepared, query, workspace


def _oracle(
    arena: PackedPageArena,
    prepared: object,
    query: torch.Tensor,
    page_count: int,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotated = query.float() @ prepared.rotation.T  # type: ignore[union-attr]
    page_max = torch.empty((page_count, 28), device=query.device, dtype=torch.float32)
    page_denominator = torch.empty_like(page_max)
    page_numerator = torch.empty((page_count, 28, 128), device=query.device)
    for page_id in range(page_count):
        page = arena.page_views(page_id)
        decoded_keys = prepared.centroids[page["key_indices"].long()]  # type: ignore[union-attr]
        keys = decoded_keys * page["key_norms"].float().unsqueeze(-1)
        values = unpack_indices(page["value_indices"].reshape(-1, 96), 6, 128).reshape(
            4, 256, 128
        )
        values = values.reshape(4, 256, 4, 32)
        values = page["value_mins"].float().unsqueeze(-1) + values.float() * page[
            "value_scales"
        ].float().unsqueeze(-1)
        values = values.reshape(4, 256, 128)
        for head in range(28):
            scores = torch.einsum("d,nd->n", rotated[head], keys[head // 7]) * scale
            maximum = scores.max()
            weights = torch.exp(scores - maximum)
            page_max[page_id, head] = maximum
            page_denominator[page_id, head] = weights.sum()
            page_numerator[page_id, head] = torch.einsum(
                "n,nd->d", weights, values[head // 7]
            )
    return page_max, page_denominator, page_numerator


@pytest.mark.gpu
@pytest.mark.parametrize("page_count", [1, 2, 3, 31, 63])
def test_page_state_matches_float32_oracle(page_count: int) -> None:
    arena, _codec, prepared, query, workspace = _make_fixture(page_count)
    scale = 1 / math.sqrt(128)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=page_count, scale=scale
    )
    torch.cuda.synchronize()
    expected = _oracle(arena, prepared, query, page_count, scale=scale)
    torch.testing.assert_close(workspace.page_max, expected[0], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        workspace.page_denominator, expected[1], rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(
        workspace.page_numerator, expected[2], rtol=3e-3, atol=3e-3
    )


@pytest.mark.gpu
def test_page_state_handles_zero_and_extreme_scores_and_final_byte() -> None:
    arena, _codec, prepared, query, workspace = _make_fixture(1, seed=1919)
    page = arena.page_views(0)
    final_byte_raw = torch.zeros(
        (4 * 256, 128), device=page["value_indices"].device, dtype=torch.uint8
    )
    final_byte_raw[:, -1] = 63
    page["value_indices"].copy_(pack_indices(final_byte_raw, 6).reshape(4, 256, 96))
    assert torch.equal(
        unpack_indices(page["value_indices"].reshape(-1, 96), 6, 128)[:, -1],
        torch.full(
            (4 * 256,), 63, device=page["value_indices"].device, dtype=torch.uint8
        ),
    )
    zero_query = torch.zeros_like(query)
    workspace.rotated_query.zero_()
    launch_qwen25_page_states(
        zero_query, arena, prepared, workspace, used_pages=1, scale=1000.0
    )
    torch.cuda.synchronize()
    assert torch.equal(workspace.page_max, torch.zeros_like(workspace.page_max))
    assert torch.allclose(
        workspace.page_denominator, torch.full_like(workspace.page_denominator, 256.0)
    )
    extreme_query = query * 100.0
    workspace.rotated_query.copy_(extreme_query.float() @ prepared.rotation.T)
    launch_qwen25_page_states(
        extreme_query, arena, prepared, workspace, used_pages=1, scale=1.0
    )
    torch.cuda.synchronize()
    assert torch.isfinite(workspace.page_denominator).all()
    assert torch.isfinite(workspace.page_numerator).all()


@pytest.mark.gpu
def test_page_state_accepts_bfloat16_arena() -> None:
    arena, _codec, prepared, query, workspace = _make_fixture(1, dtype=torch.bfloat16)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=1, scale=1 / math.sqrt(128)
    )
    torch.cuda.synchronize()
    assert torch.isfinite(workspace.page_max).all()
    assert torch.isfinite(workspace.page_denominator).all()


@pytest.mark.gpu
def test_page_state_launch_has_one_grid_and_no_hot_path_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arena, _codec, prepared, query, workspace = _make_fixture(3)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=3, scale=1 / math.sqrt(128)
    )
    torch.cuda.synchronize()

    def fail_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("page-state launch allocated a hot-path tensor")

    for name in ("empty", "empty_like", "full", "zeros"):
        monkeypatch.setattr(torch, name, fail_allocation)
    launch_qwen25_page_states(
        query, arena, prepared, workspace, used_pages=3, scale=1 / math.sqrt(128)
    )
    torch.cuda.synchronize()
    assert page_state_grid(3) == (3, 28)
    module = __import__(
        "kernels.qwen25_parallel_page_attention",
        fromlist=["_qwen25_page_states_kernel"],
    )
    source = inspect.getsource(module._qwen25_page_states_kernel.fn)
    assert "atomic" not in source.lower()
    assert (
        "_qwen25_page_states_kernel[page_state_grid(used_pages)]"
        in inspect.getsource(module.launch_qwen25_page_states)
    )


@pytest.mark.gpu
def test_page_state_requires_explicit_cuda_and_does_not_fallback() -> None:
    arena, _codec, prepared, query, workspace = _make_fixture(1)
    cpu_query = query.cpu()
    with pytest.raises(ValueError, match="query must reside on the arena device"):
        launch_qwen25_page_states(
            cpu_query,
            arena,
            prepared,
            workspace,
            used_pages=1,
            scale=1.0,
        )
