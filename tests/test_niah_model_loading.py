"""The quality runner must bind model loading to the exact Hub revision."""

from __future__ import annotations

from scripts import test_long_context as niah


EXPECTED_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_HUB_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def test_quality_runner_loads_the_pinned_qwen_hub_revision(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append((model_id, kwargs.get("revision")))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append((model_id, kwargs.get("revision")))
            return cls()

        def eval(self):
            return self

    monkeypatch.setattr(niah, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(niah, "AutoModelForCausalLM", FakeModel)
    monkeypatch.setattr(niah, "BitsAndBytesConfig", lambda **kwargs: kwargs)

    model, tokenizer = niah.load_model()

    assert niah.MODEL_ID == EXPECTED_MODEL_ID
    assert niah.MODEL_REVISION == EXPECTED_HUB_REVISION
    assert calls == [
        (EXPECTED_MODEL_ID, EXPECTED_HUB_REVISION),
        (EXPECTED_MODEL_ID, EXPECTED_HUB_REVISION),
    ]
    assert model is not None
    assert tokenizer.pad_token == tokenizer.eos_token
