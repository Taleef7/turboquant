"""Task 6 integration contracts for the opt-in Qwen2.5 page path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.capabilities import derive_model_capabilities, load_model_manifest
from core.errors import UnsupportedCapabilityError
from core.packed_store import PackedStoreRuntimeLayout, create_packed_kv_block_store
from core.profiles import load_profile
from core.qwen_adapter import PackedAttentionSession, qwen25_packed_attention
from core.reference_attention import ReferenceAttentionLayout, packed_reference_decode


ROOT = Path(__file__).resolve().parents[1]


def _model_and_profile() -> tuple[SimpleNamespace, object, object]:
    config = SimpleNamespace(
        model_type="qwen2",
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        layer_types=["full_attention"] * 28,
        rope_theta=1_000_000.0,
        is_text_model=True,
        is_multimodal=False,
        use_cache=True,
    )
    attention = [SimpleNamespace(layer_idx=index) for index in range(28)]
    model = SimpleNamespace(
        config=config,
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=module) for module in attention]
        ),
    )
    profile = load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")
    capabilities = derive_model_capabilities(
        config, profile.revision, load_model_manifest(profile.model_manifest_path)
    )
    return model, profile, capabilities


def _parallel_session(*, maximum_sequence_length: int = 640) -> PackedAttentionSession:
    model, profile, capabilities = _model_and_profile()
    return PackedAttentionSession(
        model,
        profile,
        capabilities,
        maximum_sequence_length=maximum_sequence_length,
        use_parallel_page_attention=True,
    )


def test_parallel_path_requires_explicit_capacity_and_is_opt_in() -> None:
    model, profile, capabilities = _model_and_profile()
    with pytest.raises(UnsupportedCapabilityError, match="opt-in parallel"):
        PackedAttentionSession(
            model,
            profile,
            capabilities,
            maximum_sequence_length=640,
        )
    with pytest.raises(UnsupportedCapabilityError, match="explicit maximum"):
        PackedAttentionSession(
            model,
            profile,
            capabilities,
            use_parallel_page_attention=True,
        )
    with pytest.raises(TypeError, match="built-in int"):
        PackedAttentionSession(
            model,
            profile,
            capabilities,
            maximum_sequence_length=True,
            use_parallel_page_attention=True,
        )
    with pytest.raises(ValueError, match="positive"):
        PackedAttentionSession(
            model,
            profile,
            capabilities,
            maximum_sequence_length=0,
            use_parallel_page_attention=True,
        )


def test_parallel_capacity_rejects_overflow_before_mutating_arena() -> None:
    session = _parallel_session(maximum_sequence_length=384)
    keys = torch.randn((1, 4, 383, 128), dtype=torch.float16)
    values = torch.randn_like(keys)
    with session._active("prefill"):
        session.cache.update(keys, values, 0)
        store = session.cache.store_for(0)
        assert store.arena is not None
        arena = store.arena
        pointers = arena.data_pointers()
        with pytest.raises(OverflowError, match="capacity"):
            session.cache.update(
                torch.randn((1, 4, 2, 128), dtype=torch.float16),
                torch.randn((1, 4, 2, 128), dtype=torch.float16),
                0,
            )
    assert arena.used_pages == 0
    assert arena.tail_length == 383
    assert arena.data_pointers() == pointers
    assert session.cache.sequence_length() == 383


def test_parallel_path_fails_closed_without_cuda_instead_of_using_serial_fallback() -> (
    None
):
    session = _parallel_session(maximum_sequence_length=383)
    keys = torch.randn((1, 4, 2, 128), dtype=torch.float16)
    with session._active("prefill"):
        session.cache.update(keys, keys.clone(), 0)
    with pytest.raises(UnsupportedCapabilityError, match="requires CUDA"):
        session.packed_decode(
            session.layers[0],
            torch.randn((1, 28, 1, 128), dtype=torch.float16),
            scale=1 / 128**0.5,
        )


def _reference_store(
    session: PackedAttentionSession,
    keys: torch.Tensor,
    values: torch.Tensor,
    layer_idx: int,
):
    runtime = PackedStoreRuntimeLayout(
        head_dim=128,
        num_kv_heads=4,
        norm_dtype=keys.dtype,
    )
    store = create_packed_kv_block_store(session.profile, runtime)
    store.key_codec._rotation = store.key_codec._rotation.roll(layer_idx, dims=0)
    store.append(keys, values)
    return store


def _reference(query: torch.Tensor, store: object) -> torch.Tensor:
    return packed_reference_decode(
        query,
        store,
        ReferenceAttentionLayout(
            head_dim=128,
            num_attention_heads=28,
            num_key_value_heads=4,
        ),
        scale=1 / 128**0.5,
    )


def _require_gpu_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; Qwen2.5 page integration requires CUDA.")
    pytest.importorskip("triton", reason="Triton is unavailable")


@pytest.mark.gpu
def test_parallel_decode_dispatch_reuses_workspace_and_arena_pointers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu_triton()
    torch.manual_seed(26_006)
    session = _parallel_session()
    device = torch.device("cuda")
    keys = torch.randn((1, 4, 640, 128), device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    with session._active("prefill"):
        session.cache.update(keys, values, 0)
        session.cache.update(keys, values, 1)
    session._prepare_parallel_decode_state()
    assert set(session._parallel_prepared_codecs) == {0, 1}
    query = torch.randn((1, 28, 1, 128), device=device, dtype=torch.float16)
    expected = {
        layer_idx: _reference(query, _reference_store(session, keys, values, layer_idx))
        .transpose(1, 2)
        .contiguous()
        for layer_idx in (0, 1)
    }
    stores = {layer_idx: session.cache.store_for(layer_idx) for layer_idx in (0, 1)}
    arena_pointers = {
        layer_idx: stores[layer_idx].arena.data_pointers() for layer_idx in (0, 1)
    }

    workspace = session._parallel_workspace
    assert workspace is not None
    workspace_pointers = tuple(
        tensor.data_ptr() for tensor in workspace.__dict__.values()
    )

    def reject_hot_path_allocation(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("parallel decode allocated after first-use setup")

    from core.codec import KeyCodec

    def reject_late_codec_prepare(*args: object, **kwargs: object) -> object:
        raise AssertionError("parallel decode prepared a codec after prefill")

    with monkeypatch.context() as guarded:
        guarded.setattr(KeyCodec, "prepare_device", reject_late_codec_prepare)
        guarded.setattr(torch, "empty", reject_hot_path_allocation)
        with session._active("decode"):
            actual0, _ = qwen25_packed_attention(
                session.layers[0],
                query,
                keys[..., :1, :],
                values[..., :1, :],
                None,
                scaling=1 / 128**0.5,
            )
        actual0_repeat = session.packed_decode(
            session.layers[0], query, scale=1 / 128**0.5
        )
        actual1 = session.packed_decode(session.layers[1], query, scale=1 / 128**0.5)

    for actual, reference in (
        (actual0, expected[0]),
        (actual0_repeat, expected[0]),
        (actual1, expected[1]),
    ):
        assert actual.shape == (1, 1, 28, 128)
        torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
    assert session._parallel_workspace is workspace
    assert (
        tuple(tensor.data_ptr() for tensor in workspace.__dict__.values())
        == workspace_pointers
    )
    assert all(
        stores[idx].arena.data_pointers() == arena_pointers[idx] for idx in (0, 1)
    )
    assert session.metrics.packed_decode_calls == 3
    assert session.metrics.parallel_decode_calls == 3
    assert session.metrics.parallel_page_launches == 3
    assert session.metrics.parallel_reduce_tail_launches == 3
    assert session.metrics.parallel_reduction_launches == 0
    assert session.metrics.parallel_tail_launches == 0


@pytest.mark.gpu
def test_parallel_decode_handles_nonidentity_layer_rotation_and_tail_only_history() -> (
    None
):
    _require_gpu_triton()
    torch.manual_seed(26_007)
    session = _parallel_session(maximum_sequence_length=383)
    device = torch.device("cuda")
    keys = torch.randn((1, 4, 17, 128), device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    with session._active("prefill"):
        session.cache.update(keys, values, 1)
    session._prepare_parallel_decode_state()
    store = session.cache.store_for(1)
    assert store.arena is not None and store.arena.used_pages == 0
    rotation = store.key_codec.prepare_device(device).rotation
    assert not torch.allclose(rotation, torch.eye(128, device=device), rtol=0, atol=0)
    query = torch.randn((1, 28, 1, 128), device=device, dtype=torch.float16)
    actual = session.packed_decode(session.layers[1], query, scale=1 / 128**0.5)
    expected = (
        _reference(query, _reference_store(session, keys, values, 1))
        .transpose(1, 2)
        .contiguous()
    )
    assert actual.shape == expected.shape == (1, 1, 28, 128)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert session.metrics.parallel_decode_calls == 1


@pytest.mark.gpu
def test_short_tail_prewarm_covers_every_decoded_layer_without_late_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 17-token prompt must not lazily prepare layer constants at decode."""
    _require_gpu_triton()
    torch.manual_seed(26_008)
    session = _parallel_session(maximum_sequence_length=383)
    device = torch.device("cuda")
    keys = torch.randn((1, 4, 17, 128), device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    with session._active("prefill"):
        session.cache.update(keys, values, 0)
        session.cache.update(keys, values, 1)
    session._prepare_parallel_decode_state()
    assert set(session._parallel_prepared_codecs) == {0, 1}
    query = torch.randn((1, 28, 1, 128), device=device, dtype=torch.float16)
    expected = {
        layer_idx: _reference(query, _reference_store(session, keys, values, layer_idx))
        .transpose(1, 2)
        .contiguous()
        for layer_idx in (0, 1)
    }
    workspace = session._parallel_workspace
    assert workspace is not None
    workspace_pointers = tuple(
        tensor.data_ptr() for tensor in workspace.__dict__.values()
    )
    from core.codec import KeyCodec

    def reject_late_prepare(*args: object, **kwargs: object) -> object:
        raise AssertionError("short-tail decode prepared a codec after prefill")

    def reject_allocation(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("short-tail decode allocated after prefill setup")

    with monkeypatch.context() as guarded:
        guarded.setattr(KeyCodec, "prepare_device", reject_late_prepare)
        guarded.setattr(torch, "empty", reject_allocation)
        actual0 = session.packed_decode(session.layers[0], query, scale=1 / 128**0.5)
        actual1 = session.packed_decode(session.layers[1], query, scale=1 / 128**0.5)
    torch.testing.assert_close(actual0, expected[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual1, expected[1], rtol=1e-2, atol=1e-2)
    assert (
        tuple(tensor.data_ptr() for tensor in workspace.__dict__.values())
        == workspace_pointers
    )
