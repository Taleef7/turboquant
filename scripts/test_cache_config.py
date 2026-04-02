"""Unit tests for TurboQuant cache configuration behavior."""

import sys

import torch

sys.path.insert(0, "/home/taleef/projects/turboquant")

from core.turboquant_cache_v2 import TurboQuantCacheV2
from core.turboquant_simple import TurboQuantMSE


def test_turboquant_mse_supports_configurable_norm_dtype():
    q = TurboQuantMSE(
        head_dim=8,
        bits=6,
        seed=1,
        device="cpu",
        norm_dtype=torch.float32,
    )
    x = torch.randn(2, 8)
    compressed = q.compress(x)
    assert compressed["norms"].dtype == torch.float32


def test_cache_supports_separate_key_and_value_bits():
    cache = TurboQuantCacheV2(
        head_dim=32,
        bits=6,
        key_bits=8,
        value_bits=4,
        device=torch.device("cpu"),
    )

    key_quant = cache._get_key_quantizer(layer_idx=0)
    value_quant = cache._get_value_quantizer(layer_idx=0)

    assert key_quant.bits == 8
    assert value_quant.bits == 4


def test_cache_propagates_key_norm_dtype_to_key_quantizer():
    cache = TurboQuantCacheV2(
        head_dim=32,
        key_bits=8,
        value_bits=6,
        key_norm_dtype=torch.float32,
        device=torch.device("cpu"),
    )

    key_quant = cache._get_key_quantizer(layer_idx=0)
    assert key_quant.norm_dtype == torch.float32
