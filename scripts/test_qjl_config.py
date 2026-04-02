"""Unit tests for optional QJL-enhanced key quantization path."""

import sys

import torch

sys.path.insert(0, "/home/taleef/projects/turboquant")

from core.turboquant_simple import TurboQuantMSE
from core.turboquant_cache_v2 import TurboQuantCacheV2


def test_turboquant_mse_with_qjl_emits_qjl_fields():
    quant = TurboQuantMSE(
        head_dim=32,
        bits=6,
        seed=2,
        device="cpu",
        use_qjl=True,
        qjl_dim=32,
    )
    x = torch.randn(4, 32)
    compressed = quant.compress(x)

    assert "qjl_bits" in compressed
    assert "qjl_gamma" in compressed
    assert compressed["qjl_bits"].shape == (4, 32)
    assert compressed["qjl_gamma"].shape == (4,)


def test_turboquant_mse_qjl_changes_reconstruction():
    quant_no_qjl = TurboQuantMSE(head_dim=32, bits=6, seed=2, device="cpu")
    quant_qjl = TurboQuantMSE(
        head_dim=32,
        bits=6,
        seed=2,
        device="cpu",
        use_qjl=True,
        qjl_dim=32,
        qjl_seed=11,
    )

    x = torch.randn(4, 32)
    compressed_no_qjl = quant_no_qjl.compress(x)
    compressed_qjl = quant_qjl.compress(x)

    recon_no_qjl = quant_no_qjl.decompress(compressed_no_qjl)
    recon_qjl = quant_qjl.decompress(compressed_qjl)

    assert not torch.allclose(recon_no_qjl, recon_qjl)


def test_cache_propagates_qjl_configuration_to_key_quantizer():
    cache = TurboQuantCacheV2(
        head_dim=32,
        key_bits=6,
        value_bits=6,
        key_use_qjl=True,
        key_qjl_dim=32,
        key_qjl_seed=17,
        device=torch.device("cpu"),
    )

    key_quant = cache._get_key_quantizer(0)
    assert key_quant.use_qjl is True
    assert key_quant.qjl_dim == 32


def test_cache_appends_qjl_fields_when_flushing_buffer():
    cache = TurboQuantCacheV2(
        head_dim=32,
        key_bits=6,
        value_bits=6,
        key_use_qjl=True,
        key_qjl_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
        buffer_size=0,
    )

    key_states_1 = torch.randn(1, 1, 3, 32)
    value_states_1 = torch.randn(1, 1, 3, 32)
    cache.update(key_states_1, value_states_1, layer_idx=0)

    key_states_2 = torch.randn(1, 1, 2, 32)
    value_states_2 = torch.randn(1, 1, 2, 32)
    cache.update(key_states_2, value_states_2, layer_idx=0)

    compressed = cache._compressed_keys[0]
    assert compressed is not None
    assert "qjl_bits" in compressed
    assert "qjl_gamma" in compressed
    assert compressed["shape"][2] == 5
