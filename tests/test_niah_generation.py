"""Token-level greedy-generation contracts for paired NIAH evidence."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.test_long_context import collect_greedy_session_output


class _FakeSession:
    def __init__(self, next_token_logits: list[torch.Tensor]) -> None:
        self._next_token_logits = iter(next_token_logits)
        self.received_token_ids: list[int] = []

    def decode_one(self, token_ids: torch.Tensor) -> SimpleNamespace:
        self.received_token_ids.append(int(token_ids.item()))
        return SimpleNamespace(logits=next(self._next_token_logits))


def _logits_for(token_id: int) -> torch.Tensor:
    logits = torch.full((1, 1, 16), -100.0)
    logits[:, :, token_id] = 1.0
    return logits


def test_session_generation_records_t1_and_stops_after_eos() -> None:
    session = _FakeSession([_logits_for(9), _logits_for(12)])

    output_ids, termination_reason = collect_greedy_session_output(
        session,
        _logits_for(7),
        max_new_tokens=30,
        eos_token_ids={9},
    )

    assert output_ids == [7, 9]
    assert termination_reason == "eos"
    assert session.received_token_ids == [7]
