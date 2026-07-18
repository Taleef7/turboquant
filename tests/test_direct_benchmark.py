"""Fair-prefill contracts for the systems benchmark runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from core.profiles import load_profile
from core.qwen_adapter import qwen25_packed_attention
from scripts import run_direct_benchmark


ROOT = Path(__file__).resolve().parents[1]


class _FakeAttention:
    def __init__(self, layer_idx: int, config: SimpleNamespace) -> None:
        self.layer_idx = layer_idx
        self.config = config
        self.scaling = 1 / config.head_dim**0.5
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.is_causal = True


class _BenchmarkFakeQwen:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
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
            _attn_implementation="sdpa",
        )
        self.model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=_FakeAttention(index, self.config)) for index in range(28)]
        )
        self.device = torch.device("cpu")
        self.dtype = torch.float16
        self.forward_kwargs: list[dict[str, object]] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values=None,
        use_cache: bool,
        **kwargs,
    ):
        assert use_cache is True
        self.forward_kwargs.append(dict(kwargs))
        tokens = input_ids.shape[1]
        logits_tokens = 1 if kwargs.get("logits_to_keep") == 1 else tokens
        if past_key_values is None or not hasattr(past_key_values, "update"):
            return SimpleNamespace(
                logits=torch.zeros((1, logits_tokens, 11), dtype=torch.float16),
                past_key_values=SimpleNamespace(),
            )
        module = self.model.layers[0].self_attn
        query = torch.ones((1, 28, tokens, 128), dtype=torch.float16)
        key = torch.ones((1, 4, tokens, 128), dtype=torch.float16)
        value = torch.ones_like(key)
        key, value = past_key_values.update(key, value, 0, {"cache_position": torch.arange(tokens)})
        output, _ = qwen25_packed_attention(
            module, query, key, value, None, scaling=module.scaling
        )
        assert output.shape == (1, tokens, 28, 128)
        return SimpleNamespace(
            logits=torch.zeros((1, logits_tokens, 11), dtype=torch.float16),
            past_key_values=past_key_values,
        )


class _FakeCudaEvent:
    def record(self) -> None:
        return None

    def elapsed_time(self, _: "_FakeCudaEvent") -> float:
        return 1.0


def _patch_cuda_measurement(monkeypatch) -> None:
    monkeypatch.setattr(run_direct_benchmark.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(run_direct_benchmark.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(run_direct_benchmark.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(run_direct_benchmark.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(run_direct_benchmark.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(run_direct_benchmark.torch.cuda, "Event", lambda enable_timing: _FakeCudaEvent())


def test_systems_runner_uses_one_logit_prefill_for_baseline_and_direct(monkeypatch) -> None:
    _patch_cuda_measurement(monkeypatch)
    profile = load_profile(ROOT / "configs" / "profiles" / "qwen25-7b-direct-v1.json")
    model = _BenchmarkFakeQwen()
    prompt = torch.tensor([[1, 2, 3]])

    baseline = run_direct_benchmark._run_once(model, profile, prompt, "baseline", 1)
    baseline_prefill_kwargs = model.forward_kwargs[0]
    model.forward_kwargs.clear()
    direct = run_direct_benchmark._run_once(model, profile, prompt, "triton_direct", 1)
    direct_prefill_kwargs = model.forward_kwargs[0]

    assert baseline["status"] == direct["status"] == "completed"
    assert baseline_prefill_kwargs["logits_to_keep"] == 1
    assert direct_prefill_kwargs["logits_to_keep"] == 1
