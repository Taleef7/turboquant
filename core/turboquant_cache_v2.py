"""
TurboQuantCache V2 - Simplified implementation using TurboQuantMSE.

Key differences from v1:
- No QJL (MSE-only quantization proven better for generation)
- Uses TurboQuantMSE from core/turboquant_simple.py
- No outlier channels (uniform bit-width)
- Much simpler codebase

Usage:
    cache = TurboQuantCacheV2(head_dim=128)
    outputs = model.generate(..., past_key_values=cache)
"""

from __future__ import annotations
import torch
from typing import Optional, Tuple, List, Dict
from transformers import DynamicCache

from core.turboquant_simple import TurboQuantMSE, TurboQuantValueMSE


class TurboQuantCacheV2(DynamicCache):
    """
    KV Cache that compresses keys and values using TurboQuantMSE.

    This is a simplified implementation that:
    - Uses unit norm normalization before rotation
    - Uses Lloyd-Max quantization (no QJL)
    - Stores norms separately for rescaling

    Args:
        head_dim: Dimension of each attention head (128 for Qwen2.5-7B).
        bits: Quantization bit width (default 4).
        device: Device for tensors.
        dtype: Float dtype for decompressed output. Default float16.
        buffer_size: Number of recent tokens to keep uncompressed.
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 6,  # 6-bit keys required for quality with long contexts
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        buffer_size: int = 128,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.buffer_size = buffer_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        # Per-layer quantizers (lazy init for layer-specific rotation matrices)
        self._key_quantizers: Dict[int, TurboQuantMSE] = {}
        self._value_quantizers: Dict[int, TurboQuantValueMSE] = {}

        # Compressed storage: list per layer
        # Each entry is a dict from TurboQuantMSE.compress()
        self._compressed_keys: List[Optional[dict]] = []
        self._compressed_values: List[Optional[dict]] = []

        # Uncompressed buffer for recent tokens (per layer)
        self._key_buffers: List[Optional[torch.Tensor]] = []
        self._value_buffers: List[Optional[torch.Tensor]] = []

    def _get_key_quantizer(self, layer_idx: int) -> TurboQuantMSE:
        """Get or create key quantizer for a specific layer."""
        if layer_idx not in self._key_quantizers:
            # Use layer-specific seed for rotation matrix diversity
            seed = 42 + layer_idx * 7
            self._key_quantizers[layer_idx] = TurboQuantMSE(
                head_dim=self.head_dim,
                bits=self.bits,
                seed=seed,
                device=str(self.device),
            )
        return self._key_quantizers[layer_idx]

    def _get_value_quantizer(self, layer_idx: int) -> TurboQuantValueMSE:
        """Get or create value quantizer for a specific layer."""
        if layer_idx not in self._value_quantizers:
            self._value_quantizers[layer_idx] = TurboQuantValueMSE(
                head_dim=self.head_dim,
                bits=self.bits,
                device=str(self.device),
            )
        return self._value_quantizers[layer_idx]

    def _compress_keys(self, keys: torch.Tensor, layer_idx: int) -> dict:
        """Compress keys using TurboQuantMSE."""
        quantizer = self._get_key_quantizer(layer_idx)
        return quantizer.compress(keys)

    def _decompress_keys(self, compressed: dict, layer_idx: int) -> torch.Tensor:
        """Decompress keys."""
        quantizer = self._get_key_quantizer(layer_idx)
        return quantizer.decompress(compressed).to(self.dtype)

    def _compress_values(self, values: torch.Tensor, layer_idx: int) -> dict:
        """Compress values using TurboQuantValueMSE."""
        quantizer = self._get_value_quantizer(layer_idx)
        return quantizer.compress(values)

    def _decompress_values(self, compressed: dict, layer_idx: int) -> torch.Tensor:
        """Decompress values."""
        quantizer = self._get_value_quantizer(layer_idx)
        return quantizer.decompress(compressed).to(self.dtype)

    def _append_compressed(
        self, existing: dict, new_tokens: torch.Tensor, layer_idx: int, kind: str
    ) -> dict:
        """Append new tokens to existing compressed storage."""
        if kind == "k":
            quantizer = self._get_key_quantizer(layer_idx)
        else:
            quantizer = self._get_value_quantizer(layer_idx)

        # Compress new tokens
        new_compressed = quantizer.compress(new_tokens)

        # Merge: concatenate indices along sequence dimension
        orig_shape = existing["shape"]
        new_shape = new_compressed["shape"]

        # Shapes are (batch, heads, seq, head_dim)
        batch, heads, seq_old, d = orig_shape
        _, _, seq_new, _ = new_shape

        # Reshape indices for concatenation
        old_indices = existing["indices"].reshape(batch * heads, seq_old, -1)
        new_indices = new_compressed["indices"].reshape(batch * heads, seq_new, -1)
        merged_indices = torch.cat([old_indices, new_indices], dim=1)
        merged_indices = merged_indices.reshape(-1, d)

        result = {
            "indices": merged_indices.to(torch.uint8),
            "shape": (batch, heads, seq_old + seq_new, d),
        }

        if kind == "k":
            # Keys have norms (from TurboQuantMSE)
            old_norms = existing["norms"].reshape(batch * heads, seq_old)
            new_norms = new_compressed["norms"].reshape(batch * heads, seq_new)
            merged_norms = torch.cat([old_norms, new_norms], dim=1).reshape(-1)
            result["norms"] = merged_norms.to(torch.float16)
        else:
            # Values have mins/scales (from TurboQuantValueMSE)
            n_groups = existing["mins"].shape[-1]
            old_mins = existing["mins"].reshape(batch * heads, seq_old, n_groups)
            new_mins = new_compressed["mins"].reshape(batch * heads, seq_new, n_groups)
            result["mins"] = torch.cat([old_mins, new_mins], dim=1).reshape(
                -1, n_groups
            )

            old_scales = existing["scales"].reshape(batch * heads, seq_old, n_groups)
            new_scales = new_compressed["scales"].reshape(
                batch * heads, seq_new, n_groups
            )
            result["scales"] = torch.cat([old_scales, new_scales], dim=1).reshape(
                -1, n_groups
            )

        return result

    def _flush_buffer_to_compressed(self, layer_idx: int, n_flush: int):
        """Move oldest n_flush tokens from buffer to compressed storage."""
        key_buffer = self._key_buffers[layer_idx]
        value_buffer = self._value_buffers[layer_idx]

        if key_buffer is None or key_buffer.shape[-2] <= n_flush:
            return

        # Split buffer
        keys_to_compress = key_buffer[..., :n_flush, :]
        values_to_compress = value_buffer[..., :n_flush, :]

        # Update buffers to keep only recent tokens
        self._key_buffers[layer_idx] = key_buffer[..., n_flush:, :]
        self._value_buffers[layer_idx] = value_buffer[..., n_flush:, :]

        # Compress the flushed tokens
        if self._compressed_keys[layer_idx] is None:
            self._compressed_keys[layer_idx] = self._compress_keys(
                keys_to_compress, layer_idx
            )
            self._compressed_values[layer_idx] = self._compress_values(
                values_to_compress, layer_idx
            )
        else:
            self._compressed_keys[layer_idx] = self._append_compressed(
                self._compressed_keys[layer_idx], keys_to_compress, layer_idx, "k"
            )
            self._compressed_values[layer_idx] = self._append_compressed(
                self._compressed_values[layer_idx], values_to_compress, layer_idx, "v"
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by each attention layer on every forward pass.

        Args:
            key_states:   (batch, heads, seq_new, head_dim)
            value_states: (batch, heads, seq_new, head_dim)
            layer_idx:    Which transformer layer is calling.

        Returns:
            (keys_all, values_all) each (batch, heads, seq_total, head_dim)
        """
        # Ensure storage lists are long enough
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)
            self._compressed_values.append(None)
            self._key_buffers.append(None)
            self._value_buffers.append(None)

        # Convert to target dtype
        key_states = key_states.to(self.dtype)
        value_states = value_states.to(self.dtype)

        if self.buffer_size > 0:
            # Buffer-enabled path
            if self._key_buffers[layer_idx] is None:
                self._key_buffers[layer_idx] = key_states
                self._value_buffers[layer_idx] = value_states
            else:
                self._key_buffers[layer_idx] = torch.cat(
                    [self._key_buffers[layer_idx], key_states], dim=-2
                )
                self._value_buffers[layer_idx] = torch.cat(
                    [self._value_buffers[layer_idx], value_states], dim=-2
                )

            # Flush if buffer exceeds size
            current_buffer_size = self._key_buffers[layer_idx].shape[-2]
            if current_buffer_size > self.buffer_size:
                n_flush = current_buffer_size - self.buffer_size
                self._flush_buffer_to_compressed(layer_idx, n_flush)

            # Build output: compressed (decompressed) + buffer
            output_parts_k = []
            output_parts_v = []

            if self._compressed_keys[layer_idx] is not None:
                out_k_compressed = self._decompress_keys(
                    self._compressed_keys[layer_idx], layer_idx
                )
                out_v_compressed = self._decompress_values(
                    self._compressed_values[layer_idx], layer_idx
                )
                output_parts_k.append(out_k_compressed)
                output_parts_v.append(out_v_compressed)

            output_parts_k.append(self._key_buffers[layer_idx])
            output_parts_v.append(self._value_buffers[layer_idx])

            out_k = (
                torch.cat(output_parts_k, dim=-2)
                if len(output_parts_k) > 1
                else output_parts_k[0]
            )
            out_v = (
                torch.cat(output_parts_v, dim=-2)
                if len(output_parts_v) > 1
                else output_parts_v[0]
            )
        else:
            # No-buffer path: compress everything
            if self._compressed_keys[layer_idx] is None:
                self._compressed_keys[layer_idx] = self._compress_keys(
                    key_states, layer_idx
                )
                self._compressed_values[layer_idx] = self._compress_values(
                    value_states, layer_idx
                )
            else:
                self._compressed_keys[layer_idx] = self._append_compressed(
                    self._compressed_keys[layer_idx], key_states, layer_idx, "k"
                )
                self._compressed_values[layer_idx] = self._append_compressed(
                    self._compressed_values[layer_idx], value_states, layer_idx, "v"
                )

            out_k = self._decompress_keys(self._compressed_keys[layer_idx], layer_idx)
            out_v = self._decompress_values(
                self._compressed_values[layer_idx], layer_idx
            )

        return out_k, out_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for the given layer."""
        total = 0

        # Compressed tokens
        if (
            layer_idx < len(self._compressed_keys)
            and self._compressed_keys[layer_idx] is not None
        ):
            total += self._compressed_keys[layer_idx]["shape"][2]

        # Buffer tokens
        if (
            layer_idx < len(self._key_buffers)
            and self._key_buffers[layer_idx] is not None
        ):
            total += self._key_buffers[layer_idx].shape[-2]

        return total

    def get_max_length(self) -> Optional[int]:
        return None

    def compressed_size_bytes(self) -> int:
        """Estimate compressed storage size in bytes."""
        total = 0

        for ck, cv in zip(self._compressed_keys, self._compressed_values):
            if ck is not None:
                total += ck["indices"].nelement()  # uint8
                total += ck["norms"].nelement() * 2  # float16
            if cv is not None:
                total += cv["indices"].nelement()  # uint8
                # Values use mins/scales instead of norms
                if "mins" in cv:
                    total += cv["mins"].nelement() * 2  # float16
                    total += cv["scales"].nelement() * 2  # float16

        # Buffer storage
        for kb, vb in zip(self._key_buffers, self._value_buffers):
            if kb is not None:
                total += kb.nelement() * 2  # float16
            if vb is not None:
                total += vb.nelement() * 2  # float16

        return total
