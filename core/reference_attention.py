"""Blockwise CPU reference attention over :mod:`core.packed_store`.

This module deliberately has no model or GPU-kernel integration.  It is the
numerical oracle for packed KV blocks: each block is decoded independently and
combined using an fp32 online softmax, so compressed history is never joined.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from core.packed_store import PackedKVBlockStore


_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})


@dataclass(frozen=True)
class ReferenceAttentionLayout:
    """Attention dimensions needed to map query heads onto KV heads."""

    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int

    def __post_init__(self) -> None:
        for name in ("head_dim", "num_attention_heads", "num_key_value_heads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive built-in int.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads for GQA."
            )


@dataclass
class ReferenceAttentionDebugState:
    """Optional accounting for the largest per-segment reference temporary."""

    max_decoded_tokens: int = 0
    max_temporary_tokens: int = 0
    max_temporary_elements: int = 0
    max_rotated_key_tokens: int = 0
    max_fp32_value_tokens: int = 0

    def observe(
        self, tokens: int, *, decoded: bool, block_size: int, elements: int
    ) -> None:
        if type(tokens) is not int or tokens < 1:
            raise ValueError("temporary token count must be a positive built-in int.")
        self.max_temporary_tokens = max(self.max_temporary_tokens, tokens)
        self.max_temporary_elements = max(self.max_temporary_elements, elements)
        if tokens > block_size:
            raise AssertionError("reference attention segment exceeded one packed block.")
        if decoded:
            self.max_decoded_tokens = max(self.max_decoded_tokens, tokens)

    def observe_prepared_segment(
        self,
        rotated_keys: torch.Tensor,
        fp32_values: torch.Tensor,
        *,
        block_size: int,
    ) -> None:
        """Record the local rotation/cast temporaries before attention scoring."""
        key_tokens = rotated_keys.shape[-2]
        value_tokens = fp32_values.shape[-2]
        if key_tokens != value_tokens:
            raise AssertionError("prepared key and value segments must have equal length.")
        self.observe(
            key_tokens,
            decoded=False,
            block_size=block_size,
            elements=max(rotated_keys.numel(), fp32_values.numel()),
        )
        self.max_rotated_key_tokens = max(self.max_rotated_key_tokens, key_tokens)
        self.max_fp32_value_tokens = max(self.max_fp32_value_tokens, value_tokens)


def _validate_request(
    query: torch.Tensor,
    store: PackedKVBlockStore,
    layout: ReferenceAttentionLayout,
    scale: float | None,
) -> float:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor.")
    if not isinstance(store, PackedKVBlockStore):
        raise TypeError("store must be a PackedKVBlockStore.")
    if not isinstance(layout, ReferenceAttentionLayout):
        raise TypeError("layout must be a ReferenceAttentionLayout.")
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError("query must have shape [1, num_attention_heads, tokens, head_dim].")
    if query.shape[1] != layout.num_attention_heads or query.shape[-1] != layout.head_dim:
        raise ValueError("query does not match the reference attention layout.")
    if query.shape[-2] < 1 or query.dtype not in _FLOAT_DTYPES:
        raise ValueError("query must contain tokens and use fp16, bf16, or fp32.")
    if store.key_codec.layout.head_dim != layout.head_dim:
        raise ValueError("store head_dim does not match the reference attention layout.")
    if store.num_kv_heads is not None and store.num_kv_heads != layout.num_key_value_heads:
        raise ValueError("store KV-head count does not match the reference attention layout.")
    attention_scale = 1 / math.sqrt(layout.head_dim) if scale is None else scale
    if not isinstance(attention_scale, (float, int)) or not math.isfinite(attention_scale):
        raise ValueError("scale must be a finite number.")
    return float(attention_scale)


def packed_reference_decode(
    query: torch.Tensor,
    store: PackedKVBlockStore,
    layout: ReferenceAttentionLayout,
    *,
    scale: float | None = None,
    debug: ReferenceAttentionDebugState | None = None,
) -> torch.Tensor:
    """Attend to packed blocks and the bounded fp16 tail without joining history.

    Scores and online-softmax state remain fp32.  The result is cast back to
    ``query.dtype`` only at the model-facing boundary.
    """
    attention_scale = _validate_request(query, store, layout, scale)
    if debug is not None and not isinstance(debug, ReferenceAttentionDebugState):
        raise TypeError("debug must be a ReferenceAttentionDebugState or None.")

    group_size = layout.num_attention_heads // layout.num_key_value_heads
    rotated_query = store.key_codec.rotate_query_reference(query)
    state_shape = rotated_query.shape[:-1]
    running_max = torch.full(state_shape, -torch.inf, device=query.device)
    running_sum = torch.zeros(state_shape, device=query.device)
    weighted_value_sum = torch.zeros(
        (*state_shape, layout.head_dim), device=query.device
    )
    saw_history = False

    def consume_segment(
        rotated_keys: torch.Tensor,
        values: torch.Tensor,
        key_norms: torch.Tensor | None,
        *,
        decoded: bool,
    ) -> None:
        nonlocal running_max, running_sum, weighted_value_sum, saw_history
        tokens = rotated_keys.shape[-2]
        # An exact block flush leaves the store with an empty alignment tail.
        # It is storage state, not an attention segment.
        if tokens == 0:
            return
        if debug is not None and decoded:
            debug.max_decoded_tokens = max(debug.max_decoded_tokens, tokens)
        query_by_kv_head = rotated_query.reshape(
            1,
            layout.num_key_value_heads,
            group_size,
            query.shape[-2],
            layout.head_dim,
        )
        scores = torch.einsum(
            "bhgqd,bhkd->bhgqk", query_by_kv_head, rotated_keys.float()
        ).reshape(1, layout.num_attention_heads, query.shape[-2], tokens)
        if key_norms is not None:
            norms = key_norms.reshape(layout.num_key_value_heads, tokens).float()
            scores = scores * norms.repeat_interleave(group_size, dim=0).view(
                1, layout.num_attention_heads, 1, tokens
            )
        scores = scores * attention_scale
        block_max = scores.max(dim=-1).values
        new_max = torch.maximum(running_max, block_max)
        old_scale = torch.where(
            torch.isfinite(running_max),
            torch.exp(running_max - new_max),
            torch.zeros_like(new_max),
        )
        block_weights = torch.exp(scores - new_max.unsqueeze(-1))
        value_contribution = torch.einsum(
            "bhgqt,bhtd->bhgqd",
            block_weights.reshape(
                1,
                layout.num_key_value_heads,
                group_size,
                query.shape[-2],
                tokens,
            ),
            values.float(),
        ).reshape(*state_shape, layout.head_dim)
        if debug is not None:
            debug.max_temporary_elements = max(
                debug.max_temporary_elements,
                scores.numel(),
                block_weights.numel(),
                value_contribution.numel(),
            )
        running_sum = running_sum * old_scale + block_weights.sum(dim=-1)
        weighted_value_sum = (
            weighted_value_sum * old_scale.unsqueeze(-1)
            + value_contribution
        )
        running_max = new_max
        saw_history = True

    def prepare_and_consume_in_block_chunks(
        raw_keys: torch.Tensor,
        raw_values: torch.Tensor,
        *,
        decoded: bool,
    ) -> None:
        total_tokens = raw_keys.shape[-2]
        for start in range(0, total_tokens, store.block_size):
            stop = min(start + store.block_size, total_tokens)
            rotated_keys = store.key_codec.rotate_reference_vectors(
                raw_keys[..., start:stop, :]
            )
            fp32_values = raw_values[..., start:stop, :].float()
            if debug is not None:
                debug.observe_prepared_segment(
                    rotated_keys, fp32_values, block_size=store.block_size
                )
            consume_segment(
                rotated_keys,
                fp32_values,
                None,
                decoded=decoded,
            )

    for block in store.iter_blocks():
        decoded_keys = block.keys.unpack_reference_fp32()
        prepare_and_consume_in_block_chunks(
            decoded_keys,
            block.values.unpack_reference(),
            decoded=True,
        )
    for key_window, value_window in store.iter_fp16_window():
        prepare_and_consume_in_block_chunks(key_window, value_window, decoded=False)

    if not saw_history:
        raise ValueError("cannot decode attention from an empty packed store.")
    return (weighted_value_sum / running_sum.unsqueeze(-1)).to(query.dtype)
