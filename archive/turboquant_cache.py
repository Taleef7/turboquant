"""
TurboQuant HuggingFace Cache Integration.

Subclasses transformers.DynamicCache to intercept key/value states
and compress them using TurboQuant before storage. Decompression
happens just-in-time when the cache is read for attention.

Key improvements over basic implementation (per reference @ 0xSero/turboquant):
  - Per-layer rotation matrices with distinct seeds
  - Proper Lloyd-Max codebooks from Beta distribution
  - Uncompressed buffer for recent tokens (buffer_size=128)
  - searchsorted quantization instead of argmin

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
from typing import Optional, Tuple, List, Dict
from transformers import DynamicCache

from utils.math_utils import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    get_codebook_tensors,
)
from kernels.compress_kv import (
    compress_kv_python,
    build_outlier_mask,
    compress_values_group_quant,
)
from kernels.decompress_kv import (
    decompress_kv_python,
    decompress_values_group_quant,
)

import os
import time

from kernels.bitpack_triton import (
    pack_4bit,
    unpack_4bit,
    pack_sign_bits,
    unpack_sign_bits,
)

# Environment variable to force Python fallback (useful when Triton kernel compilation hangs)
_FORCE_PYTHON_FALLBACK = os.environ.get("TURBOQUANT_FORCE_PYTHON", "0") == "1"

# Environment variable to enable profiling
_ENABLE_PROFILING = os.environ.get("TURBOQUANT_PROFILE", "0") == "1"

# Environment variable to enable bit-packing storage for eligible tensors.
_USE_BITPACKING = os.environ.get("TURBOQUANT_USE_BITPACKING", "0") == "1"

# Environment variable to optionally bypass key compression for quality debugging.
_COMPRESS_KEYS_DEFAULT = os.environ.get("TURBOQUANT_COMPRESS_KEYS", "1") != "0"

# Profiling statistics
_PROFILE_STATS = {
    "compress_memory_transfer": [],
    "compress_kernel": [],
    "compress_total": [],
    "decompress_kernel": [],
    "decompress_total": [],
    "update_total": [],
    "update_compress_k": [],
    "update_compress_v": [],
    "update_decompress_k": [],
    "update_decompress_v": [],
}

# Try to import hybrid functions (preferred for sm_120 compatibility)
try:
    from kernels.compress_kv import compress_kv_hybrid
    from kernels.decompress_kv import decompress_kv_hybrid

    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False

# Try to import original fused Triton kernels (may hang on sm_120)
try:
    from kernels.compress_kv import (
        triton_compress_kv,
        _TRITON_AVAILABLE as _TRITON_COMPRESS_AVAILABLE,
    )
    from kernels.decompress_kv import (
        triton_decompress_kv,
        _TRITON_AVAILABLE as _TRITON_DECOMPRESS_AVAILABLE,
    )

    _USE_FUSED_TRITON = (
        torch.cuda.is_available()
        and _TRITON_COMPRESS_AVAILABLE
        and _TRITON_DECOMPRESS_AVAILABLE
        and not _FORCE_PYTHON_FALLBACK
        and os.environ.get("TURBOQUANT_USE_FUSED", "0")
        == "1"  # Opt-in for fused kernels
    )
except (ImportError, AttributeError):
    _USE_FUSED_TRITON = False

# Determine which implementation to use (priority: hybrid > python)
# Note: We default to hybrid if available, as it works on sm_120
_USE_HYBRID = (
    _HYBRID_AVAILABLE and torch.cuda.is_available() and not _FORCE_PYTHON_FALLBACK
)

# Log which implementation is being used
import warnings

if _USE_FUSED_TRITON:
    warnings.warn(
        "TurboQuant: Using fused Triton kernels (TURBOQUANT_USE_FUSED=1)",
        UserWarning,
        stacklevel=2,
    )
elif _USE_HYBRID:
    # No warning needed - this is the expected fast path
    pass
elif _FORCE_PYTHON_FALLBACK:
    warnings.warn(
        "TurboQuant: Using PyTorch fallback (TURBOQUANT_FORCE_PYTHON=1)",
        UserWarning,
        stacklevel=2,
    )


class TurboQuantCache(DynamicCache):
    """
    KV Cache that compresses keys and values using TurboQuant on insert
    and decompresses just-in-time on read.

    Key improvements (per reference implementation):
      - Per-layer rotation matrices: seed = base_seed + layer_idx * 7
      - Proper Lloyd-Max codebooks from Beta distribution
      - Uncompressed buffer for recent tokens (quality improvement)

    Args:
        head_dim: Dimension of each attention head (128 for Qwen2.5-7B).
        n_qjl: Number of QJL projections k. Default 128.
        n_outliers: Number of higher-bit outlier channels. Default 32.
        device: Device for random matrices.
        dtype: Float dtype for decompressed output. Default float16.
        base_seed: Base random seed for reproducibility. Per-layer seed = base_seed + layer_idx * 7.
        use_qjl: Whether to use QJL correction (TurboQuant_prod). Default False.
                 Note: QJL improves inner product estimation but increases MSE.
                 For best reconstruction quality, keep this False (TurboQuant_mse).
        bit_width: Base quantization bit width (2, 3, or 4). Default 3.
                   - 2-bit: ~7x compression, ~95% cosine sim, may have quality issues
                   - 3-bit: ~5x compression, ~98% cosine sim, good quality (RECOMMENDED)
                   - 4-bit: ~4x compression, ~99.5% cosine sim, near-lossless
                   Outlier channels use bit_width+1 bits for improved quality.
        buffer_size: Number of recent tokens to keep uncompressed. Default 128.
                     Set to 0 to disable buffer (compress everything).
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_qjl: int = 128,
        n_outliers: int = 32,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        base_seed: int = 42,
        use_qjl: bool = False,
        bit_width: int = 3,
        buffer_size: int = 128,
        value_group_size: int = 32,
        compress_keys: Optional[bool] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_qjl = n_qjl
        self.n_outliers = n_outliers
        self.use_qjl = use_qjl
        self.bit_width = bit_width
        self.base_seed = base_seed
        self.buffer_size = buffer_size
        self.value_group_size = value_group_size
        self.compress_keys = (
            _COMPRESS_KEYS_DEFAULT if compress_keys is None else bool(compress_keys)
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        if bit_width not in (2, 3, 4):
            raise ValueError(f"bit_width must be 2, 3, or 4, got {bit_width}")
        if value_group_size <= 0 or head_dim % value_group_size != 0:
            raise ValueError(
                f"value_group_size must be a positive divisor of head_dim; got {value_group_size} for head_dim={head_dim}"
            )

        # Per-layer rotation matrices (lazy init)
        self._rotation_matrices: Dict[int, torch.Tensor] = {}
        self._qjl_matrices: Dict[int, torch.Tensor] = {}

        # Centroids from proper Lloyd-Max codebooks
        self._c_base, self._c_base_boundaries = get_codebook_tensors(
            head_dim, bits=bit_width, device=self.device
        )

        if bit_width < 4:
            self._c_outlier, self._c_outlier_boundaries = get_codebook_tensors(
                head_dim, bits=bit_width + 1, device=self.device
            )
        else:
            # No outliers at 4-bit (already high quality)
            self._c_outlier = self._c_base
            self._c_outlier_boundaries = self._c_base_boundaries

        # Aliases used by compress/decompress functions
        self._c2 = self._c_base
        self._c3 = self._c_outlier

        # Compressed storage: list per layer.
        # Entry format: (all_idx, all_qjl, all_gamma, all_norms, all_masks, batch, heads, seq, d)
        self._compressed_keys: List[Optional[tuple]] = []
        self._compressed_values: List[Optional[tuple]] = []

        # Uncompressed buffer for recent tokens (per layer)
        # Format: (keys_buffer, values_buffer) each (batch, heads, buffer_seq, head_dim)
        self._key_buffers: List[Optional[torch.Tensor]] = []
        self._value_buffers: List[Optional[torch.Tensor]] = []

    def _get_rotation_matrix(self, layer_idx: int) -> torch.Tensor:
        """Get or create rotation matrix for a specific layer."""
        if layer_idx not in self._rotation_matrices:
            # Per-layer seed: base_seed + layer_idx * 7 (per reference implementation)
            layer_seed = self.base_seed + layer_idx * 7
            self._rotation_matrices[layer_idx] = generate_rotation_matrix(
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
                seed=layer_seed,
            )
        return self._rotation_matrices[layer_idx]

    def _get_qjl_matrix(self, layer_idx: int) -> torch.Tensor:
        """Get or create QJL matrix for a specific layer."""
        if layer_idx not in self._qjl_matrices:
            # Use different seed offset for QJL
            qjl_seed = self.base_seed + 1000 + layer_idx * 7
            self._qjl_matrices[layer_idx] = generate_qjl_matrix(
                self.head_dim,
                self.n_qjl,
                dtype=torch.float32,
                device=self.device,
                seed=qjl_seed,
            )
        return self._qjl_matrices[layer_idx]

    # ------------------------------------------------------------------
    # Internal compress / decompress helpers
    # ------------------------------------------------------------------

    def _compress(self, x: torch.Tensor, layer_idx: int, kind: str = "k") -> tuple:
        """
        Compress (batch, heads, seq, head_dim) → compressed tuple.

        BATCHED IMPLEMENTATION: Processes all batch*heads slices in a single
        GPU kernel call by flattening to (batch*heads*seq, head_dim).

        UNIT NORM NORMALIZATION (per TurboQuant paper):
        The paper states: "the unit norm assumption, ∥x∥₂ = 1, is standard and not
        restrictive. For datasets that do not satisfy this assumption we can compute
        and store the L2 norms in floating-point precision and rescale the dequantized
        points using these stored norms."

        We normalize each input vector to unit norm before compression, store the
        original norms, and rescale during decompression.

        Returns:
            (idx_all, qjl_bits, gamma, norms, mask, batch, heads, seq, d)
            where idx_all is (batch*heads*seq, head_dim), norms is (batch*heads*seq,), etc.
        """
        if _ENABLE_PROFILING and torch.cuda.is_available():
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_mem = torch.cuda.Event(enable_timing=True)
            end_mem = torch.cuda.Event(enable_timing=True)
            start_kernel = torch.cuda.Event(enable_timing=True)
            end_kernel = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_total.record()

        batch, heads, seq, d = x.shape
        n_slices = batch * heads

        # Flatten to (batch*heads*seq, head_dim) for batched processing
        if _ENABLE_PROFILING and torch.cuda.is_available():
            start_mem.record()

        x_flat = x.reshape(n_slices * seq, d).to(torch.float32).to(self.device)

        # UNIT NORM NORMALIZATION: Store original norms and normalize to unit vectors
        # This is CRITICAL for TurboQuant correctness - centroids are designed for ||x||=1
        norms = torch.norm(x_flat, dim=-1, keepdim=True)  # (batch*heads*seq, 1)
        # Avoid division by zero for zero vectors (rare but possible)
        norms_safe = torch.clamp(norms, min=1e-8)
        x_normalized = x_flat / norms_safe  # (batch*heads*seq, head_dim) with ||x||=1
        norms = norms.squeeze(-1)  # (batch*heads*seq,) for storage

        if _ENABLE_PROFILING and torch.cuda.is_available():
            end_mem.record()

        # Choose implementation by tensor kind.
        if _ENABLE_PROFILING and torch.cuda.is_available():
            start_kernel.record()

        if kind == "k":
            if not self.compress_keys:
                # Store keys losslessly in FP16 to isolate value-path behavior.
                raw_keys = x.to(self.dtype).to(self.device)
                return (
                    raw_keys,
                    None,
                    None,
                    None,
                    None,
                    batch,
                    heads,
                    seq,
                    d,
                    "k_raw",
                    False,
                )

            # Get per-layer matrices
            Pi = self._get_rotation_matrix(layer_idx)
            S = self._get_qjl_matrix(layer_idx)

            # Build a single mask from NORMALIZED data (outliers are channel-wise across all tokens)
            mask = build_outlier_mask(x_normalized, self.n_outliers)

            # Triton quantize kernels currently support 4-level base and 8-level outlier tables.
            # For bit_width=3 we use a 16-level outlier table, so fall back to Python path.
            triton_tables_supported = self._c2.numel() == 4 and self._c3.numel() == 8

            # Choose implementation: hybrid (preferred) > fused Triton > Python
            if _USE_HYBRID and self.device.type == "cuda" and triton_tables_supported:
                idx_all, qjl_bits, gamma = compress_kv_hybrid(
                    x_normalized,
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                )
            elif (
                _USE_FUSED_TRITON
                and self.device.type == "cuda"
                and triton_tables_supported
            ):
                idx_all, qjl_bits, gamma = triton_compress_kv(
                    x_normalized.to(torch.float16),
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                )
            else:
                idx_all, qjl_bits, gamma = compress_kv_python(
                    x_normalized,
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                )

            if _USE_BITPACKING:
                # Key path packs qjl signs always; index packing only when values are <= 4-bit.
                # For current 3-bit base + 4-bit outlier configuration, pack_4bit is valid.
                packed_idx = pack_4bit(idx_all)
                packed_qjl = pack_sign_bits(qjl_bits)
                payload = (
                    packed_idx,
                    packed_qjl,
                    gamma,
                    norms,
                    mask,
                    batch,
                    heads,
                    seq,
                    d,
                    "k",
                    True,
                    qjl_bits.shape[-1],
                )
            else:
                payload = (
                    idx_all,
                    qjl_bits,
                    gamma,
                    norms,
                    mask,
                    batch,
                    heads,
                    seq,
                    d,
                    "k",
                    False,
                )
        elif kind == "v":
            # Values use per-group min/max quantization with no rotation and no QJL.
            idx_all, mins, scales = compress_values_group_quant(
                x_normalized, group_size=self.value_group_size
            )
            if _USE_BITPACKING:
                idx_payload = pack_4bit(idx_all)
                packed_flag = True
            else:
                idx_payload = idx_all
                packed_flag = False
            payload = (
                idx_payload,
                mins,
                scales,
                norms,
                self.value_group_size,
                batch,
                heads,
                seq,
                d,
                "v",
                packed_flag,
            )
        else:
            raise ValueError(f"Unknown compression kind: {kind}")

        if _ENABLE_PROFILING and torch.cuda.is_available():
            end_kernel.record()
            end_total.record()
            torch.cuda.synchronize()
            _PROFILE_STATS["compress_memory_transfer"].append(
                start_mem.elapsed_time(end_mem)
            )
            _PROFILE_STATS["compress_kernel"].append(
                start_kernel.elapsed_time(end_kernel)
            )
            _PROFILE_STATS["compress_total"].append(start_total.elapsed_time(end_total))

        return payload

    def _append_compressed(
        self,
        existing: tuple,
        new_tokens: torch.Tensor,
        layer_idx: int,
        kind: str = "k",
    ) -> tuple:
        """
        Incrementally append new tokens to existing compressed storage.

        BATCHED IMPLEMENTATION: Processes all batch*heads slices at once.

        UNIT NORM NORMALIZATION: New tokens are normalized before compression,
        with their original norms stored for decompression rescaling.

        Args:
            existing: Existing compressed tuple from _compress()
            new_tokens: (batch, heads, seq_new, head_dim) new tokens to append
            layer_idx: Layer index for per-layer matrices

        Returns:
            Updated compressed tuple with new tokens appended
        """
        if kind == "k":
            idx_all, qjl_bits, gamma, norms, mask, batch, heads, seq_old, d = existing[
                :9
            ]
            packed_existing = bool(existing[10]) if len(existing) >= 11 else False
            qjl_width = int(existing[11]) if len(existing) >= 12 else self.n_qjl
        elif kind == "v":
            idx_all, mins, scales, norms, group_size, batch, heads, seq_old, d = (
                existing[:9]
            )
            packed_existing = bool(existing[10]) if len(existing) >= 11 else False
        else:
            raise ValueError(f"Unknown append kind: {kind}")
        batch_new, heads_new, seq_new, d_new = new_tokens.shape

        assert batch == batch_new and heads == heads_new and d == d_new, (
            f"Shape mismatch: existing ({batch}, {heads}, {seq_old}, {d}) vs new ({batch_new}, {heads_new}, {seq_new}, {d_new})"
        )

        n_slices = batch * heads
        # Flatten new tokens to (batch*heads*seq_new, head_dim)
        new_flat = (
            new_tokens.reshape(n_slices * seq_new, d).to(torch.float32).to(self.device)
        )

        # UNIT NORM NORMALIZATION for new tokens
        norms_new = torch.norm(
            new_flat, dim=-1, keepdim=True
        )  # (batch*heads*seq_new, 1)
        norms_new_safe = torch.clamp(norms_new, min=1e-8)
        new_normalized = new_flat / norms_new_safe  # (batch*heads*seq_new, head_dim)
        norms_new = norms_new.squeeze(-1)  # (batch*heads*seq_new,)

        if kind == "k":
            if len(existing) >= 10 and existing[9] == "k_raw":
                raw_old = existing[0]
                raw_new = new_tokens.to(self.dtype).to(self.device)
                raw_cat = torch.cat([raw_old, raw_new], dim=-2)
                return (
                    raw_cat,
                    None,
                    None,
                    None,
                    None,
                    batch,
                    heads,
                    seq_old + seq_new,
                    d,
                    "k_raw",
                    False,
                )

            # Get per-layer matrices
            Pi = self._get_rotation_matrix(layer_idx)
            S = self._get_qjl_matrix(layer_idx)

            # Compress new tokens using the SAME mask as existing (for consistency)
            # Triton quantize kernels currently support 4-level base and 8-level outlier tables.
            triton_tables_supported = self._c2.numel() == 4 and self._c3.numel() == 8

            # Choose implementation: hybrid (preferred) > fused Triton > Python
            if _USE_HYBRID and self.device.type == "cuda" and triton_tables_supported:
                idx_new, qjl_new, gamma_new = compress_kv_hybrid(
                    new_normalized,
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                )
            elif (
                _USE_FUSED_TRITON
                and self.device.type == "cuda"
                and triton_tables_supported
            ):
                idx_new, qjl_new, gamma_new = triton_compress_kv(
                    new_normalized.to(torch.float16),
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                )
            else:
                idx_new, qjl_new, gamma_new = compress_kv_python(
                    new_normalized,
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                )
        else:
            idx_new, mins_new, scales_new = compress_values_group_quant(
                new_normalized, group_size=group_size
            )

        norms_old_r = norms.reshape(n_slices, seq_old)
        norms_new_r = norms_new.reshape(n_slices, seq_new)
        norms_cat = torch.cat([norms_old_r, norms_new_r], dim=1).reshape(-1)

        if kind == "k":
            gamma_old_r = gamma.reshape(n_slices, seq_old)
            gamma_new_r = gamma_new.reshape(n_slices, seq_new)
            gamma_cat = torch.cat([gamma_old_r, gamma_new_r], dim=1).reshape(-1)

            if packed_existing:
                idx_old_r = idx_all.reshape(n_slices, seq_old, -1)
                qjl_old_r = qjl_bits.reshape(n_slices, seq_old, -1)
                idx_new_packed = pack_4bit(idx_new)
                qjl_new_packed = pack_sign_bits(qjl_new)
                idx_new_r = idx_new_packed.reshape(n_slices, seq_new, -1)
                qjl_new_r = qjl_new_packed.reshape(n_slices, seq_new, -1)
                idx_cat = torch.cat([idx_old_r, idx_new_r], dim=1).reshape(
                    -1, idx_old_r.shape[-1]
                )
                qjl_cat = torch.cat([qjl_old_r, qjl_new_r], dim=1).reshape(
                    -1, qjl_old_r.shape[-1]
                )
                return (
                    idx_cat,
                    qjl_cat,
                    gamma_cat,
                    norms_cat,
                    mask,
                    batch,
                    heads,
                    seq_old + seq_new,
                    d,
                    "k",
                    True,
                    qjl_width,
                )

            idx_old_r = idx_all.reshape(n_slices, seq_old, d)
            qjl_old_r = qjl_bits.reshape(n_slices, seq_old, -1)
            idx_new_r = idx_new.reshape(n_slices, seq_new, d)
            qjl_new_r = qjl_new.reshape(n_slices, seq_new, -1)
            idx_cat = torch.cat([idx_old_r, idx_new_r], dim=1).reshape(-1, d)
            qjl_cat = torch.cat([qjl_old_r, qjl_new_r], dim=1).reshape(
                -1, qjl_bits.shape[-1]
            )

            return (
                pack_4bit(idx_cat) if _USE_BITPACKING else idx_cat,
                pack_sign_bits(qjl_cat) if _USE_BITPACKING else qjl_cat,
                gamma_cat,
                norms_cat,
                mask,
                batch,
                heads,
                seq_old + seq_new,
                d,
                "k",
                _USE_BITPACKING,
                qjl_bits.shape[-1],
            )

        mins_old_r = mins.reshape(n_slices, seq_old, -1)
        scales_old_r = scales.reshape(n_slices, seq_old, -1)
        mins_new_r = mins_new.reshape(n_slices, seq_new, -1)
        scales_new_r = scales_new.reshape(n_slices, seq_new, -1)
        mins_cat = torch.cat([mins_old_r, mins_new_r], dim=1).reshape(
            -1, mins.shape[-1]
        )
        scales_cat = torch.cat([scales_old_r, scales_new_r], dim=1).reshape(
            -1, scales.shape[-1]
        )

        if packed_existing:
            idx_old_r = idx_all.reshape(n_slices, seq_old, -1)
            idx_new_packed = pack_4bit(idx_new)
            idx_new_r = idx_new_packed.reshape(n_slices, seq_new, -1)
            idx_cat = torch.cat([idx_old_r, idx_new_r], dim=1).reshape(
                -1, idx_old_r.shape[-1]
            )
            return (
                idx_cat,
                mins_cat,
                scales_cat,
                norms_cat,
                group_size,
                batch,
                heads,
                seq_old + seq_new,
                d,
                "v",
                True,
            )

        idx_old_r = idx_all.reshape(n_slices, seq_old, d)
        idx_new_r = idx_new.reshape(n_slices, seq_new, d)
        idx_cat = torch.cat([idx_old_r, idx_new_r], dim=1).reshape(-1, d)
        return (
            pack_4bit(idx_cat) if _USE_BITPACKING else idx_cat,
            mins_cat,
            scales_cat,
            norms_cat,
            group_size,
            batch,
            heads,
            seq_old + seq_new,
            d,
            "v",
            _USE_BITPACKING,
        )

    def _decompress(
        self, compressed: tuple, layer_idx: int, kind: Optional[str] = None
    ) -> torch.Tensor:
        """
        Decompress stored tuple → (batch, heads, seq, head_dim) in self.dtype.

        BATCHED IMPLEMENTATION: Single GPU kernel call for all batch*heads.

        UNIT NORM RESCALING: After decompression, multiply by stored norms to
        recover the original scale of the vectors.
        """
        if _ENABLE_PROFILING and torch.cuda.is_available():
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_kernel = torch.cuda.Event(enable_timing=True)
            end_kernel = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_total.record()
            start_kernel.record()

        payload_kind = None
        if len(compressed) >= 10 and isinstance(compressed[9], str):
            payload_kind = compressed[9]

        # Payload metadata is authoritative; caller hint is fallback only.
        actual_kind = payload_kind or kind or "k"

        if actual_kind == "k":
            idx_all, qjl_bits, gamma, norms, mask, batch, heads, seq, d = compressed[:9]
            packed = bool(compressed[10]) if len(compressed) >= 11 else False
            qjl_width = int(compressed[11]) if len(compressed) >= 12 else self.n_qjl
            if packed:
                idx_all = unpack_4bit(idx_all, d)
                qjl_bits = unpack_sign_bits(qjl_bits, qjl_width)

            # Get per-layer matrices
            Pi = self._get_rotation_matrix(layer_idx)
            S = self._get_qjl_matrix(layer_idx)

            # Triton dequantize kernels currently support 4-level base and 8-level outlier tables.
            triton_tables_supported = self._c2.numel() == 4 and self._c3.numel() == 8

            # Choose implementation: hybrid (preferred) > fused Triton > Python
            if _USE_HYBRID and self.device.type == "cuda" and triton_tables_supported:
                x_flat = decompress_kv_hybrid(
                    idx_all,
                    qjl_bits,
                    gamma,
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                    target_dtype=torch.float32,  # Keep float32 for rescaling
                    use_qjl=self.use_qjl,
                )
            elif (
                _USE_FUSED_TRITON
                and self.device.type == "cuda"
                and triton_tables_supported
            ):
                x_flat = triton_decompress_kv(
                    idx_all,
                    qjl_bits,
                    gamma,
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                    target_dtype=torch.float32,  # Keep float32 for rescaling
                )
            else:
                x_flat = decompress_kv_python(
                    idx_all,
                    qjl_bits,
                    gamma,
                    Pi,
                    S,
                    self._c2,
                    self._c3,
                    mask,
                    target_dtype=torch.float32,  # Keep float32 for rescaling
                    use_qjl=self.use_qjl,
                )
        elif actual_kind == "k_raw":
            raw_keys, _, _, _, _, batch, heads, seq, d = compressed[:9]
            return raw_keys.reshape(batch, heads, seq, d).to(self.dtype)
        elif actual_kind == "v":
            idx_all, mins, scales, norms, group_size, batch, heads, seq, d = compressed[
                :9
            ]
            packed = bool(compressed[10]) if len(compressed) >= 11 else False
            if packed:
                idx_all = unpack_4bit(idx_all, d)
            x_flat = decompress_values_group_quant(
                idx_all,
                mins,
                scales,
                group_size=group_size,
                target_dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown decompress kind: {actual_kind}")

        # UNIT NORM RESCALING: Multiply by original norms to recover scale
        # x_flat: (batch*heads*seq, head_dim), norms: (batch*heads*seq,)
        x_flat = x_flat * norms.unsqueeze(-1)

        # Convert to target dtype
        x_flat = x_flat.to(self.dtype)

        if _ENABLE_PROFILING and torch.cuda.is_available():
            end_kernel.record()
            end_total.record()
            torch.cuda.synchronize()
            _PROFILE_STATS["decompress_kernel"].append(
                start_kernel.elapsed_time(end_kernel)
            )
            _PROFILE_STATS["decompress_total"].append(
                start_total.elapsed_time(end_total)
            )

        # Reshape back to (batch, heads, seq, head_dim)
        return x_flat.reshape(batch, heads, seq, d)

    def _flush_buffer_to_compressed(self, layer_idx: int, n_flush: int):
        """
        Move oldest n_flush tokens from buffer to compressed storage.

        This implements the buffer overflow logic from the reference implementation.
        """
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
            # First compression for this layer
            self._compressed_keys[layer_idx] = self._compress(
                keys_to_compress, layer_idx, kind="k"
            )
            self._compressed_values[layer_idx] = self._compress(
                values_to_compress, layer_idx, kind="v"
            )
        else:
            # Append to existing compressed storage
            self._compressed_keys[layer_idx] = self._append_compressed(
                self._compressed_keys[layer_idx], keys_to_compress, layer_idx, kind="k"
            )
            self._compressed_values[layer_idx] = self._append_compressed(
                self._compressed_values[layer_idx],
                values_to_compress,
                layer_idx,
                kind="v",
            )

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

        With buffer enabled (buffer_size > 0):
          - New tokens go to buffer first
          - When buffer exceeds buffer_size, oldest tokens are compressed
          - Returns concatenation of decompressed + buffer (full precision recent)

        Without buffer (buffer_size = 0):
          - All tokens are compressed immediately
          - Returns decompressed cache

        Args:
            key_states:   (batch, heads, seq_new, head_dim)
            value_states: (batch, heads, seq_new, head_dim)
            layer_idx:    Which transformer layer is calling.

        Returns:
            (keys_all, values_all) each (batch, heads, seq_total, head_dim)
            in self.dtype, ready for attention.
        """
        if _ENABLE_PROFILING and torch.cuda.is_available():
            start_update = torch.cuda.Event(enable_timing=True)
            end_update = torch.cuda.Event(enable_timing=True)
            start_compress_k = torch.cuda.Event(enable_timing=True)
            end_compress_k = torch.cuda.Event(enable_timing=True)
            start_compress_v = torch.cuda.Event(enable_timing=True)
            end_compress_v = torch.cuda.Event(enable_timing=True)
            start_decompress_k = torch.cuda.Event(enable_timing=True)
            end_decompress_k = torch.cuda.Event(enable_timing=True)
            start_decompress_v = torch.cuda.Event(enable_timing=True)
            end_decompress_v = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_update.record()

        # Ensure storage lists are long enough
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)
            self._compressed_values.append(None)
            self._key_buffers.append(None)
            self._value_buffers.append(None)

        # Convert to target dtype for buffer storage
        key_states = key_states.to(self.dtype)
        value_states = value_states.to(self.dtype)

        if self.buffer_size > 0:
            # ─── Buffer-enabled path ───
            # Add new tokens to buffer
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

            # Check if buffer exceeds size → flush oldest to compressed
            current_buffer_size = self._key_buffers[layer_idx].shape[-2]
            if current_buffer_size > self.buffer_size:
                n_flush = current_buffer_size - self.buffer_size
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    start_compress_k.record()
                self._flush_buffer_to_compressed(layer_idx, n_flush)
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    end_compress_k.record()
                    start_compress_v.record()
                    end_compress_v.record()  # Values flushed together with keys

            # Build output: compressed (decompressed) + buffer
            output_parts_k = []
            output_parts_v = []

            if self._compressed_keys[layer_idx] is not None:
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    start_decompress_k.record()
                out_k_compressed = self._decompress(
                    self._compressed_keys[layer_idx], layer_idx, kind="k"
                )
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    end_decompress_k.record()
                    start_decompress_v.record()
                out_v_compressed = self._decompress(
                    self._compressed_values[layer_idx], layer_idx, kind="v"
                )
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    end_decompress_v.record()
                output_parts_k.append(out_k_compressed)
                output_parts_v.append(out_v_compressed)

            # Add buffer (full precision)
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
            # ─── No-buffer path (original behavior) ───
            if self._compressed_keys[layer_idx] is None:
                # First call for this layer — compress directly
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    start_compress_k.record()
                self._compressed_keys[layer_idx] = self._compress(
                    key_states, layer_idx, kind="k"
                )
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    end_compress_k.record()
                    start_compress_v.record()
                self._compressed_values[layer_idx] = self._compress(
                    value_states, layer_idx, kind="v"
                )
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    end_compress_v.record()
            else:
                # Subsequent calls — incrementally append new tokens
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    start_compress_k.record()
                self._compressed_keys[layer_idx] = self._append_compressed(
                    self._compressed_keys[layer_idx], key_states, layer_idx, kind="k"
                )
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    end_compress_k.record()
                    start_compress_v.record()
                self._compressed_values[layer_idx] = self._append_compressed(
                    self._compressed_values[layer_idx],
                    value_states,
                    layer_idx,
                    kind="v",
                )
                if _ENABLE_PROFILING and torch.cuda.is_available():
                    end_compress_v.record()

            if _ENABLE_PROFILING and torch.cuda.is_available():
                start_decompress_k.record()
            out_k = self._decompress(
                self._compressed_keys[layer_idx], layer_idx, kind="k"
            )
            if _ENABLE_PROFILING and torch.cuda.is_available():
                end_decompress_k.record()
                start_decompress_v.record()
            out_v = self._decompress(
                self._compressed_values[layer_idx], layer_idx, kind="v"
            )
            if _ENABLE_PROFILING and torch.cuda.is_available():
                end_decompress_v.record()

        if _ENABLE_PROFILING and torch.cuda.is_available():
            end_update.record()
            torch.cuda.synchronize()
            # Only record if events were actually used
            try:
                _PROFILE_STATS["update_compress_k"].append(
                    start_compress_k.elapsed_time(end_compress_k)
                )
                _PROFILE_STATS["update_compress_v"].append(
                    start_compress_v.elapsed_time(end_compress_v)
                )
                _PROFILE_STATS["update_decompress_k"].append(
                    start_decompress_k.elapsed_time(end_decompress_k)
                )
                _PROFILE_STATS["update_decompress_v"].append(
                    start_decompress_v.elapsed_time(end_decompress_v)
                )
            except RuntimeError:
                pass  # Events not recorded in this path
            _PROFILE_STATS["update_total"].append(start_update.elapsed_time(end_update))

        return out_k, out_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for the given layer."""
        total = 0

        # Compressed tokens
        if (
            layer_idx < len(self._compressed_keys)
            and self._compressed_keys[layer_idx] is not None
        ):
            total += self._compressed_keys[layer_idx][7]  # seq is at index 7

        # Buffer tokens
        if (
            layer_idx < len(self._key_buffers)
            and self._key_buffers[layer_idx] is not None
        ):
            total += self._key_buffers[layer_idx].shape[-2]

        return total

    def get_max_length(self) -> Optional[int]:
        return None  # No hard limit

    def compressed_size_bytes(self) -> int:
        """
        Actual in-memory compressed storage size in bytes across all layers.

        Each int8 tensor (idx_all, qjl_bits) uses 1 byte per element.
        Each float32 tensor (gamma, norms) uses 4 bytes per element.
        Each bool mask uses 1 byte per element.
        Buffer tensors use 2 bytes per element (float16).

        Note: with int8 storage the ratio is ~0.97× vs FP16. True ~4.4×
        compression requires future bit-packing (2-3 bits/index, 1 bit/sign).

        Tuple formats:
          Key payload:   (idx_or_packed, qjl_or_packed, gamma, norms, mask, batch, heads, seq, d, "k", is_packed, qjl_width?)
                         (raw_keys, None, None, None, None, batch, heads, seq, d, "k_raw", False)
          Value payload: (idx_or_packed, mins, scales, norms, group_size, batch, heads, seq, d, "v", is_packed)
        """
        total = 0

        # Compressed storage
        for ck in self._compressed_keys + self._compressed_values:
            if ck is None:
                continue
            idx_all, second, third, norms, fifth = ck[0], ck[1], ck[2], ck[3], ck[4]
            kind = ck[9] if len(ck) >= 10 and isinstance(ck[9], str) else "k"
            if kind == "k_raw":
                total += idx_all.nelement() * 2  # fp16 keys
                continue

            total += idx_all.nelement()  # int8/uint8 = 1 byte each
            if kind == "k":
                # qjl_bits int8 + gamma float32 + mask bool
                total += second.nelement()
                total += third.nelement() * 4
                total += fifth.nelement()
            else:
                # mins/scales are float32, group_size is scalar metadata
                total += second.nelement() * 4
                total += third.nelement() * 4
            total += norms.nelement() * 4  # float32 = 4 bytes each

        # Buffer storage
        for kb in self._key_buffers + self._value_buffers:
            if kb is None:
                continue
            total += kb.nelement() * 2  # float16 = 2 bytes each

        return total


def print_profile_stats():
    """Print profiling statistics collected during execution."""
    if not _ENABLE_PROFILING:
        print("Profiling not enabled. Set TURBOQUANT_PROFILE=1 to collect stats.")
        return

    print("\n" + "=" * 70)
    print("TURBOQUANT PROFILING RESULTS")
    print("=" * 70)

    import numpy as np

    for key, times in _PROFILE_STATS.items():
        if len(times) == 0:
            continue
        times_arr = np.array(times)
        print(f"\n{key}:")
        print(f"  Calls:     {len(times)}")
        print(f"  Mean:      {times_arr.mean():.3f} ms")
        print(f"  Median:    {np.median(times_arr):.3f} ms")
        print(f"  Std:       {times_arr.std():.3f} ms")
        print(f"  Min:       {times_arr.min():.3f} ms")
        print(f"  Max:       {times_arr.max():.3f} ms")
        print(f"  Total:     {times_arr.sum():.3f} ms")

    # Calculate aggregate stats
    if len(_PROFILE_STATS["update_total"]) > 0:
        update_times = np.array(_PROFILE_STATS["update_total"])
        compress_k_times = np.array(_PROFILE_STATS["update_compress_k"])
        compress_v_times = np.array(_PROFILE_STATS["update_compress_v"])
        decompress_k_times = np.array(_PROFILE_STATS["update_decompress_k"])
        decompress_v_times = np.array(_PROFILE_STATS["update_decompress_v"])

        print("\n" + "-" * 70)
        print("AGGREGATED STATISTICS")
        print("-" * 70)
        print(f"\nAverage per-layer update cycle:")
        print(f"  Total:          {update_times.mean():.3f} ms")
        if len(compress_k_times) > 0:
            print(
                f"  Compress K:     {compress_k_times.mean():.3f} ms ({compress_k_times.mean() / update_times.mean() * 100:.1f}%)"
            )
            print(
                f"  Compress V:     {compress_v_times.mean():.3f} ms ({compress_v_times.mean() / update_times.mean() * 100:.1f}%)"
            )
        if len(decompress_k_times) > 0:
            print(
                f"  Decompress K:   {decompress_k_times.mean():.3f} ms ({decompress_k_times.mean() / update_times.mean() * 100:.1f}%)"
            )
            print(
                f"  Decompress V:   {decompress_v_times.mean():.3f} ms ({decompress_v_times.mean() / update_times.mean() * 100:.1f}%)"
            )

    if len(_PROFILE_STATS["compress_total"]) > 0:
        compress_total = np.array(_PROFILE_STATS["compress_total"])
        compress_mem = np.array(_PROFILE_STATS["compress_memory_transfer"])
        compress_kernel = np.array(_PROFILE_STATS["compress_kernel"])

        print(f"\nCompression breakdown:")
        print(f"  Total:            {compress_total.mean():.3f} ms")
        print(
            f"  Memory transfer:  {compress_mem.mean():.3f} ms ({compress_mem.mean() / compress_total.mean() * 100:.1f}%)"
        )
        print(
            f"  Kernel compute:   {compress_kernel.mean():.3f} ms ({compress_kernel.mean() / compress_total.mean() * 100:.1f}%)"
        )

    print("\n" + "=" * 70)
