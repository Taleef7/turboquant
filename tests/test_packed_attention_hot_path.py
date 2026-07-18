"""GPU regression coverage for the prepared packed-attention decode hot path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.capabilities import derive_model_capabilities, load_model_manifest
from core.codec import KeyCodec, KeyLayout, ValueCodec, ValueLayout
from core.packed_store import PackedKVBlockStore
from core.profiles import load_profile
from core.qwen_adapter import PackedAttentionSession
from core.reference_attention import ReferenceAttentionLayout, packed_reference_decode
from kernels.packed_attention import (
    PackedAttentionBlock,
    PackedAttentionHotPathDebugState,
    PreparedPackedAttentionDecode,
    prepare_packed_attention_decode,
)


ROOT = Path(__file__).resolve().parents[1]


def _require_gpu_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; prepared hot-path coverage requires CUDA.")
    pytest.importorskip("triton", reason="Triton is unavailable for prepared hot-path coverage")


@pytest.mark.gpu
def test_prepared_decode_rotates_once_without_hot_path_syncs_or_constant_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One prepared layer reuses its static packed layout across all blocks."""
    _require_gpu_triton()
    torch.manual_seed(25_718)
    device = torch.device("cuda")
    head_dim, kv_heads, gqa_ratio = 128, 2, 4
    packed_tokens, fp16_tail = 512, 64
    dtype = torch.float16
    key_codec = KeyCodec(KeyLayout(head_dim=head_dim, bits=8, norm_dtype=dtype))
    value_codec = ValueCodec(ValueLayout(head_dim=head_dim, bits=6, group_size=32))
    store = PackedKVBlockStore(
        key_codec, value_codec, fp16_window=fp16_tail, num_kv_heads=kv_heads
    )
    keys = torch.randn(
        (1, kv_heads, packed_tokens + fp16_tail, head_dim), device=device, dtype=dtype
    )
    values = torch.randn_like(keys)
    store.append(keys, values)
    query = torch.randn((kv_heads * gqa_ratio, 1, head_dim), device=device, dtype=dtype)
    expected = packed_reference_decode(
        query.unsqueeze(0),
        store,
        ReferenceAttentionLayout(
            head_dim=head_dim,
            num_attention_heads=kv_heads * gqa_ratio,
            num_key_value_heads=kv_heads,
        ),
    ).squeeze(0)

    def to_kernel_block(block: object) -> PackedAttentionBlock:
        payload = block
        return PackedAttentionBlock(
            packed_keys=payload.keys.indices.reshape(kv_heads, 256, -1),
            key_norms=payload.keys.norms.reshape(kv_heads, 256),
            key_codebook=key_codec._centroids.to(device),
            key_bits=8,
            packed_values=payload.values.indices.reshape(kv_heads, 256, -1),
            value_mins=payload.values.mins.reshape(kv_heads, 256, -1),
            value_scales=payload.values.scales.reshape(kv_heads, 256, -1),
            value_bits=6,
            value_group_size=32,
        )

    blocks = tuple(to_kernel_block(block) for block in store.iter_blocks())
    assert len(blocks) == 2
    tail = tuple(tensor.squeeze(0) for tensor in next(store.iter_fp16_window()))
    mapping = torch.arange(kv_heads * gqa_ratio, device=device, dtype=torch.int32) // gqa_ratio
    debug = PackedAttentionHotPathDebugState()
    prepared = prepare_packed_attention_decode(
        blocks,
        key_codec._rotation.to(device),
        mapping,
        debug=debug,
    )

    static_constant_ptrs = {
        prepared.rotation.data_ptr(),
        prepared.gqa_mapping.data_ptr(),
        prepared.gqa_mapping_long.data_ptr(),
        *(block.key_codebook.data_ptr() for block in prepared.blocks),
    }
    original_to = torch.Tensor.to
    original_item = torch.Tensor.item
    original_int = torch.Tensor.__int__

    def reject_static_constant_transfer(tensor: torch.Tensor, *args, **kwargs):
        if tensor.is_cuda and tensor.data_ptr() in static_constant_ptrs:
            raise AssertionError("prepared launch copied a static device constant")
        return original_to(tensor, *args, **kwargs)

    def reject_cuda_scalar_conversion(tensor: torch.Tensor, *args, **kwargs):
        if tensor.is_cuda:
            raise AssertionError("prepared launch converted a CUDA tensor to a host scalar")
        return original_item(tensor, *args, **kwargs)

    def reject_cuda_int_conversion(tensor: torch.Tensor):
        if tensor.is_cuda:
            raise AssertionError("prepared launch converted a CUDA tensor to a Python int")
        return original_int(tensor)

    def reject_residency_verification(_: PreparedPackedAttentionDecode) -> None:
        raise AssertionError("prepared launch must not scan static constants")

    with monkeypatch.context() as guarded:
        guarded.setattr(torch.Tensor, "to", reject_static_constant_transfer)
        guarded.setattr(torch.Tensor, "item", reject_cuda_scalar_conversion)
        guarded.setattr(torch.Tensor, "__int__", reject_cuda_int_conversion)
        guarded.setattr(
            PreparedPackedAttentionDecode,
            "verify_static_constants_resident",
            reject_residency_verification,
            raising=False,
        )
        actual = prepared.launch(query, fp16_window=tail)

    assert debug.static_validation_reductions == 2
    assert debug.constant_residency_checks == 0
    prepared.verify_static_constants_resident()
    assert debug.constant_residency_checks == 1
    assert debug.rotation_launches == 1
    assert debug.packed_score_launches == len(blocks)
    assert debug.packed_value_launches == len(blocks)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.gpu
def test_qwen_layer_reuses_prepared_device_constants_between_decode_tokens() -> None:
    """The adapter must not rebuild its static layer state for every token."""
    _require_gpu_triton()
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
    session = PackedAttentionSession(model, profile, capabilities)
    device = torch.device("cuda")
    keys = torch.randn((1, 4, 640, 128), device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    with session._active("prefill"):
        session.cache.update(keys, values, 0)
    assert session.cache.store_for(0).block_count == 2
    query = torch.randn((1, 28, 1, 128), device=device, dtype=torch.float16)

    first = session.packed_decode(attention[0], query, scale=1 / 128**0.5)
    cached = session._prepared_layers[0]
    second = session.packed_decode(attention[0], query, scale=1 / 128**0.5)

    assert session._prepared_layers[0] is cached
    assert cached.rotation.is_cuda and cached.key_codebook.is_cuda
    assert cached.gqa_mapping.is_cuda
    assert cached.debug.static_validation_reductions == 2
    assert cached.debug.constant_residency_checks == 0
    cached.prepared.verify_static_constants_resident()
    assert cached.debug.constant_residency_checks == 1
    assert cached.debug.rotation_launches == 2
    torch.testing.assert_close(second, first, rtol=3e-3, atol=3e-3)


def _qwen_session_with_history(tokens: int) -> tuple[PackedAttentionSession, object, torch.Tensor]:
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
    session = PackedAttentionSession(model, profile, capabilities)
    device = torch.device("cuda")
    keys = torch.randn((1, 4, tokens, 128), device=device, dtype=torch.float16)
    with session._active("prefill"):
        session.cache.update(keys, torch.randn_like(keys), 0)
    query = torch.randn((1, 28, 1, 128), device=device, dtype=torch.float16)
    return session, attention[0], query


def _qwen_reference(session: PackedAttentionSession, query: torch.Tensor) -> torch.Tensor:
    reference = packed_reference_decode(
        query,
        session.cache.store_for(0),
        ReferenceAttentionLayout(
            head_dim=128,
            num_attention_heads=28,
            num_key_value_heads=4,
        ),
    )
    return reference.transpose(1, 2).contiguous()


@pytest.mark.gpu
def test_qwen_prepared_plan_reuses_blocks_when_only_the_tail_grows() -> None:
    _require_gpu_triton()
    session, attention, query = _qwen_session_with_history(640)
    first = session.packed_decode(attention, query, scale=1 / 128**0.5)
    first_plan = session._prepared_layers[0]
    torch.testing.assert_close(first, _qwen_reference(session, query), rtol=3e-3, atol=3e-3)

    with session._active("decode"):
        one_token = torch.randn((1, 4, 1, 128), device="cuda", dtype=torch.float16)
        session.cache.update(one_token, torch.randn_like(one_token), 0)
    assert session.cache.store_for(0).block_count == first_plan.block_count

    actual = session.packed_decode(attention, query, scale=1 / 128**0.5)
    assert session._prepared_layers[0] is first_plan
    torch.testing.assert_close(actual, _qwen_reference(session, query), rtol=3e-3, atol=3e-3)


@pytest.mark.gpu
def test_qwen_prepared_plan_refreshes_when_tail_append_flushes_a_block() -> None:
    _require_gpu_triton()
    session, attention, query = _qwen_session_with_history(639)
    first = session.packed_decode(attention, query, scale=1 / 128**0.5)
    first_plan = session._prepared_layers[0]
    assert first_plan.block_count == 1
    torch.testing.assert_close(first, _qwen_reference(session, query), rtol=3e-3, atol=3e-3)

    with session._active("decode"):
        one_token = torch.randn((1, 4, 1, 128), device="cuda", dtype=torch.float16)
        session.cache.update(one_token, torch.randn_like(one_token), 0)
    assert session.cache.store_for(0).block_count == first_plan.block_count + 1

    actual = session.packed_decode(attention, query, scale=1 / 128**0.5)
    assert session._prepared_layers[0] is not first_plan
    torch.testing.assert_close(actual, _qwen_reference(session, query), rtol=3e-3, atol=3e-3)
