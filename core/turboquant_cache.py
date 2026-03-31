"""
TurboQuant HuggingFace Cache Integration.

Subclasses transformers.DynamicCache to intercept key/value states
and compress them using TurboQuant before storage. Decompression
happens just-in-time when the cache is read for attention.

Compression storage per token vector (head_dim=128, k=128):
  idx_all:  128 × 1 byte  = 128 bytes  (int8 centroid indices)
  qjl_bits: 128 × 1 byte  = 128 bytes  (int8 ±1 signs)
  gamma:         4 bytes               (float32 residual norm)
  is_outlier: stored once per (batch*heads) slice (bool mask)
vs FP16:    128 × 2 bytes = 256 bytes

NOTE: With int8 storage (1 byte/element), actual ratio ≈ 0.97× vs FP16.
True ~4.4× compression requires bit-packing (2-3 bits/index, 1 bit/sign).
This implementation uses int8 for correctness; bit-packing is future work.

Usage:
    cache = TurboQuantCache(head_dim=128)
    outputs = model.generate(..., past_key_values=cache)
"""

from __future__ import annotations
import torch
from typing import Optional, Tuple, List
from transformers import DynamicCache

from utils.math_utils import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    get_centroids_2bit,
    get_centroids_3bit,
)
from kernels.compress_kv import compress_kv_python, build_outlier_mask
from kernels.decompress_kv import decompress_kv_python

try:
    from kernels.compress_kv import triton_compress_kv
    from kernels.decompress_kv import triton_decompress_kv
    _USE_TRITON = torch.cuda.is_available()
except ImportError:
    _USE_TRITON = False


class TurboQuantCache(DynamicCache):
    """
    KV Cache that compresses keys and values using TurboQuant on insert
    and decompresses just-in-time on read.

    Args:
        head_dim: Dimension of each attention head (128 for Qwen2.5-7B).
        n_qjl: Number of QJL projections k. Default 128.
        n_outliers: Number of 3-bit outlier channels. Default 32.
        device: Device for random matrices.
        dtype: Float dtype for decompressed output. Default float16.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_qjl: int = 128,
        n_outliers: int = 32,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        seed: int = 42,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_qjl = n_qjl
        self.n_outliers = n_outliers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        torch.manual_seed(seed)
        self._Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=self.device)
        self._S  = generate_qjl_matrix(head_dim, n_qjl, dtype=torch.float32, device=self.device)
        self._c2 = get_centroids_2bit(head_dim, device=self.device)
        self._c3 = get_centroids_3bit(head_dim, device=self.device)

        # Compressed storage: list per layer.
        # Entry format: (all_idx, all_qjl, all_gamma, all_masks, batch, heads, seq, d)
        self._compressed_keys:   List[Optional[tuple]] = []
        self._compressed_values: List[Optional[tuple]] = []

    # ------------------------------------------------------------------
    # Internal compress / decompress helpers
    # ------------------------------------------------------------------

    def _compress(self, x: torch.Tensor) -> tuple:
        """
        Compress (batch, heads, seq, head_dim) → compressed tuple.

        Flattens batch*heads, compresses each (seq, head_dim) slice with
        TurboQuant. Stores idx_all, qjl_bits, gamma, and the is_outlier mask
        per slice.

        Returns:
            (all_idx, all_qjl, all_gamma, all_masks, batch, heads, seq, d)
        """
        batch, heads, seq, d = x.shape
        x_flat = x.reshape(batch * heads, seq, d).to(torch.float32)

        all_idx, all_qjl, all_gamma, all_masks = [], [], [], []
        for i in range(batch * heads):
            xv   = x_flat[i]                              # (seq, d)
            mask = build_outlier_mask(xv, self.n_outliers)

            if _USE_TRITON:
                # Triton path: inputs must be on correct device
                idx_all, qjl, gamma = triton_compress_kv(
                    xv.to(self.device).to(torch.float16),
                    self._Pi, self._S, self._c2, self._c3,
                    mask.to(self.device),
                )
            else:
                idx_all, qjl, gamma = compress_kv_python(
                    xv.to(self.device), self._Pi, self._S,
                    self._c2, self._c3, mask.to(self.device),
                )

            all_idx.append(idx_all)
            all_qjl.append(qjl)
            all_gamma.append(gamma)
            all_masks.append(mask)

        return (all_idx, all_qjl, all_gamma, all_masks, batch, heads, seq, d)

    def _decompress(self, compressed: tuple) -> torch.Tensor:
        """
        Decompress stored tuple → (batch, heads, seq, head_dim) in self.dtype.
        """
        all_idx, all_qjl, all_gamma, all_masks, batch, heads, seq, d = compressed

        slices = []
        for i in range(batch * heads):
            if _USE_TRITON:
                xv = triton_decompress_kv(
                    all_idx[i], all_qjl[i], all_gamma[i],
                    self._Pi, self._S, self._c2, self._c3,
                    all_masks[i].to(self.device),
                    target_dtype=self.dtype,
                )
            else:
                xv = decompress_kv_python(
                    all_idx[i], all_qjl[i], all_gamma[i],
                    self._Pi, self._S, self._c2, self._c3,
                    all_masks[i].to(self.device),
                    target_dtype=self.dtype,
                )
            slices.append(xv)

        return torch.stack(slices, dim=0).reshape(batch, heads, seq, d)

    # ------------------------------------------------------------------
    # DynamicCache API
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by each attention layer on every forward pass.

        Compresses the incoming key/value states, appends to any existing
        compressed cache for this layer, then decompresses the full sequence
        for use in attention.

        Args:
            key_states:   (batch, heads, seq_new, head_dim)
            value_states: (batch, heads, seq_new, head_dim)
            layer_idx:    Which transformer layer is calling.

        Returns:
            (keys_all, values_all) each (batch, heads, seq_total, head_dim)
            in self.dtype, ready for attention.
        """
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)
            self._compressed_values.append(None)

        if self._compressed_keys[layer_idx] is None:
            # First call for this layer — compress directly
            self._compressed_keys[layer_idx]   = self._compress(key_states)
            self._compressed_values[layer_idx] = self._compress(value_states)
        else:
            # Subsequent calls — decompress, concatenate new tokens, recompress
            existing_k = self._decompress(self._compressed_keys[layer_idx])
            existing_v = self._decompress(self._compressed_values[layer_idx])
            full_k = torch.cat([existing_k, key_states.to(self.dtype)], dim=2)
            full_v = torch.cat([existing_v, value_states.to(self.dtype)], dim=2)
            self._compressed_keys[layer_idx]   = self._compress(full_k)
            self._compressed_values[layer_idx] = self._compress(full_v)

        out_k = self._decompress(self._compressed_keys[layer_idx])
        out_v = self._decompress(self._compressed_values[layer_idx])
        return out_k, out_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for the given layer."""
        if (layer_idx >= len(self._compressed_keys)
                or self._compressed_keys[layer_idx] is None):
            return 0
        return self._compressed_keys[layer_idx][6]  # seq is at index 6

    def get_max_length(self) -> Optional[int]:
        return None  # No hard limit

    def compressed_size_bytes(self) -> int:
        """
        Actual in-memory compressed storage size in bytes across all layers.

        Each int8 tensor (idx_all, qjl_bits) uses 1 byte per element.
        Each float32 tensor (gamma) uses 4 bytes per element.
        Each bool mask (all_masks) uses 1 byte per element.
        Note: with int8 storage the ratio is ~0.97× vs FP16. True ~4.4×
        compression requires future bit-packing (2-3 bits/index, 1 bit/sign).
        """
        total = 0
        for ck in self._compressed_keys + self._compressed_values:
            if ck is None:
                continue
            all_idx, all_qjl, all_gamma, all_masks = ck[0], ck[1], ck[2], ck[3]
            for i in range(len(all_idx)):
                total += all_idx[i].nelement()        # int8 = 1 byte each
                total += all_qjl[i].nelement()        # int8 = 1 byte each
                total += all_gamma[i].nelement() * 4  # float32 = 4 bytes each
                total += all_masks[i].nelement()      # bool = 1 byte each
        return total
