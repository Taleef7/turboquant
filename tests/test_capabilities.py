"""Fail-closed model capability rules without model downloads."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
from dataclasses import replace

import pytest
import core

from core.capabilities import (
    DecodeRequest,
    ModelRevision,
    derive_model_capabilities,
    load_model_manifest,
    require_direct_decode_support,
)
from core.errors import UnsupportedCapabilityError


def qwen25_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_type": "qwen2",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rope_theta": 1_000_000.0,
        "use_cache": True,
        "layer_types": ("full_attention", "full_attention"),
        "is_multimodal": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def qwen35_hybrid_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen3_5",
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=1_000_000.0,
        use_cache=True,
        layer_types=(
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ),
    )


class NeverMutatedStore:
    def __init__(self) -> None:
        self.append_calls = 0

    def append(self, *args: object, **kwargs: object) -> None:
        self.append_calls += 1
        raise AssertionError("capability rejection must happen before append")


def direct_revision() -> ModelRevision:
    return ModelRevision(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        revision="test-revision",
        manifest_sha256="a" * 64,
    )


ROOT = Path(__file__).resolve().parents[1]
QWEN25_MANIFEST_PATH = ROOT / "configs" / "models" / "qwen25-7b.json"


def manifest_config(path: Path) -> SimpleNamespace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SimpleNamespace(**payload)


def direct_config(**overrides: object) -> SimpleNamespace:
    payload = json.loads(QWEN25_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_qwen25_records_architecture_facts_from_config() -> None:
    caps = derive_model_capabilities(qwen25_config(), direct_revision())

    assert caps.model_type == "qwen2"
    assert caps.layer_count == 2
    assert caps.full_attention_layer_indices == (0, 1)
    assert caps.compressible_layer_indices == (0, 1)
    assert caps.num_key_value_groups == 4
    assert caps.rope_settings == {"rope_theta": 1_000_000.0}
    assert caps.adapter_status == "direct-adapter"


def test_capability_rope_metadata_is_recursively_immutable() -> None:
    caps = derive_model_capabilities(
        qwen25_config(rope_scaling={"nested": {"factor": 8.0}}), direct_revision()
    )

    with pytest.raises(TypeError):
        caps.rope_settings["rope_scaling"]["nested"]["factor"] = 16.0


def test_manifest_metadata_is_recursively_immutable(tmp_path: Path) -> None:
    payload = json.loads(QWEN25_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["rope_scaling"] = {"nested": {"factor": 8.0}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_model_manifest(path)

    with pytest.raises(TypeError):
        manifest.rope_settings["rope_scaling"]["nested"]["factor"] = 16.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", 7),
        ("revision", False),
        ("model_type", ["qwen2"]),
        ("adapter_status", 1),
        ("architecture_complete", 1),
        ("is_text_model", "true"),
        ("is_multimodal", 0),
        ("use_cache", "true"),
        ("num_hidden_layers", True),
        ("num_attention_heads", "28"),
        ("num_key_value_heads", 4.0),
        ("head_dim", False),
        ("rope_theta", "1000000"),
        ("cache_implementation", 7),
    ],
)
def test_manifest_rejects_non_primitive_architecture_truth(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(QWEN25_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnsupportedCapabilityError, match=field):
        load_model_manifest(path)


def test_manifest_loader_is_available_from_the_core_public_api() -> None:
    assert core.load_model_manifest is load_model_manifest


def test_qwen35_exposes_only_full_attention_as_compressible() -> None:
    caps = derive_model_capabilities(qwen35_hybrid_config())

    assert caps.compressible_layer_indices == (3, 7)
    assert caps.native_state_layer_indices == (0, 1, 2, 4, 5, 6)
    assert caps.adapter_status == "manifest-only"


def test_gemma4_is_manifest_only() -> None:
    caps = derive_model_capabilities(
        SimpleNamespace(
            model_type="gemma4",
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=("full_attention", "full_attention"),
            is_multimodal=True,
        )
    )

    assert caps.adapter_status == "manifest-only"
    assert caps.is_multimodal is True


@pytest.mark.parametrize(
    ("config", "decode_request", "message"),
    [
        (direct_config(), DecodeRequest(batch_size=2), "batch_size"),
        (direct_config(), DecodeRequest(do_sample=True), "greedy"),
        (direct_config(), DecodeRequest(greedy=False), "greedy"),
        (direct_config(), DecodeRequest(num_beams=2), "greedy"),
        (
            direct_config(layer_types=["sliding_attention"] * 28),
            DecodeRequest(),
            "sliding",
        ),
        (
            direct_config(layer_types=["linear_attention"] * 28),
            DecodeRequest(),
            "linear",
        ),
        (direct_config(), DecodeRequest(speculative=True), "speculative"),
        (direct_config(), DecodeRequest(mtp=True), "MTP"),
        (direct_config(is_multimodal=True), DecodeRequest(), "multimodal"),
        (
            direct_config(),
            DecodeRequest(includes_multimodal_cache=True),
            "multimodal",
        ),
        (direct_config(), DecodeRequest(multimodal_cache=True), "multimodal"),
        (
            SimpleNamespace(model_type="qwen3", num_hidden_layers=2),
            DecodeRequest(),
            "manifest-only",
        ),
    ],
)
def test_unsupported_direct_decode_routes_fail_closed(
    config: SimpleNamespace, decode_request: DecodeRequest, message: str
) -> None:
    manifest = load_model_manifest(QWEN25_MANIFEST_PATH) if config.model_type == "qwen2" else None
    revision = manifest.revision if manifest else direct_revision()
    caps = derive_model_capabilities(config, revision, manifest)

    with pytest.raises(UnsupportedCapabilityError, match=message):
        require_direct_decode_support(caps, decode_request)


def test_unknown_model_rejects_before_cache_mutation() -> None:
    store = NeverMutatedStore()
    capabilities = derive_model_capabilities(SimpleNamespace(model_type="unknown"))

    with pytest.raises(UnsupportedCapabilityError, match="model_type"):
        require_direct_decode_support(
            capabilities, DecodeRequest(batch_size=1), pending_store=store
        )

    assert store.append_calls == 0


def test_direct_route_returns_capabilities_without_mutating_store() -> None:
    store = NeverMutatedStore()
    manifest = load_model_manifest(QWEN25_MANIFEST_PATH)
    capabilities = derive_model_capabilities(
        manifest_config(QWEN25_MANIFEST_PATH), manifest.revision, manifest
    )

    assert (
        require_direct_decode_support(
            capabilities, DecodeRequest(batch_size=1), pending_store=store
        )
        is capabilities
    )
    assert store.append_calls == 0


def test_direct_route_requires_pinned_manifest_and_exact_config_layout() -> None:
    manifest = load_model_manifest(QWEN25_MANIFEST_PATH)
    malformed_config = qwen25_config(
        layer_types=("full_attention",), is_text_model=True
    )
    malformed = derive_model_capabilities(
        malformed_config, manifest.revision, manifest
    )
    without_manifest = derive_model_capabilities(
        manifest_config(QWEN25_MANIFEST_PATH), manifest.revision
    )

    with pytest.raises(UnsupportedCapabilityError, match="manifest layout"):
        require_direct_decode_support(malformed, DecodeRequest())
    with pytest.raises(UnsupportedCapabilityError, match="pinned manifest"):
        require_direct_decode_support(without_manifest, DecodeRequest())


def test_direct_route_rejects_a_non_qwen25_manifest_identity() -> None:
    manifest = load_model_manifest(QWEN25_MANIFEST_PATH)
    wrong_identity = replace(
        manifest,
        revision=ModelRevision(
            model_id="example/not-qwen25",
            revision=manifest.revision.revision,
            manifest_sha256=manifest.revision.manifest_sha256,
        ),
    )
    caps = derive_model_capabilities(
        manifest_config(QWEN25_MANIFEST_PATH), wrong_identity.revision, wrong_identity
    )

    with pytest.raises(UnsupportedCapabilityError, match="Qwen2.5"):
        require_direct_decode_support(caps, DecodeRequest())


@pytest.mark.parametrize(
    ("filename", "expected_status", "compressible", "native"),
    [
        ("qwen25-7b.json", "direct-adapter", (0, 1, 2), ()),
        ("qwen3.json", "manifest-only", (0, 1, 2), ()),
        ("qwen35-9b.json", "manifest-only", (3, 7, 11), (0, 1, 2, 4, 5, 6)),
        ("gemma4-e4b.json", "manifest-only", (), ()),
    ],
)
def test_every_shipped_manifest_loads_with_coherent_capabilities(
    filename: str,
    expected_status: str,
    compressible: tuple[int, ...],
    native: tuple[int, ...],
) -> None:
    path = ROOT / "configs" / "models" / filename
    manifest = load_model_manifest(path)
    caps = derive_model_capabilities(manifest_config(path), manifest.revision, manifest)

    assert caps.adapter_status == expected_status
    assert caps.compressible_layer_indices[: len(compressible)] == compressible
    assert caps.native_state_layer_indices[: len(native)] == native
