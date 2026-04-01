"""
TurboQuant KV Cache Decompression Kernel — Fully Fused.

SRAM execution contract:
  - Compressed data (idx_all, qjl_bits, gamma) loaded from global memory ONCE.
  - All intermediates (y_hat, x_mse, correction) live in SRAM registers only.
  - Global memory write: only the reconstructed FP16 vector x_tilde.

Dequantization formula:
  x_tilde = Pi^T * y_hat  +  (sqrt(pi/2) / k) * gamma * S^T * qjl

Contents:
  decompress_kv_python  — CPU reference implementation
  decompress_kv_hybrid  — PyTorch matmuls + Triton dequantization (sm_120 compatible)
  _turboquant_decompress_fused — fully fused Triton kernel (may hang on sm_120)
"""

import math
import torch

# ---------------------------------------------------------------------------
# Optional Triton import (CPU-only environments will skip the kernel)
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# Import new vectorized Triton dequantization kernels
try:
    from kernels.quantize_triton import triton_dequantize

    _TRITON_QUANTIZE_AVAILABLE = _TRITON_AVAILABLE
except ImportError:
    _TRITON_QUANTIZE_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Python reference implementation
# ---------------------------------------------------------------------------


def decompress_kv_python(
    idx_all: torch.Tensor,  # (seq_len, head_dim) int8
    qjl_bits: torch.Tensor,  # (seq_len, k) int8 — values are +1 or -1
    gamma: torch.Tensor,  # (seq_len,) float32
    Pi: torch.Tensor,  # (head_dim, head_dim) float32
    S: torch.Tensor,  # (k, head_dim) float32
    centroids_2bit: torch.Tensor,  # (4,) float32
    centroids_3bit: torch.Tensor,  # (8,) float32
    is_outlier: torch.Tensor,  # (head_dim,) bool
    target_dtype: torch.dtype = torch.float16,
    use_qjl: bool = False,  # Whether to apply QJL correction
) -> torch.Tensor:
    """
    GPU-optimized vectorized decompression using pure PyTorch operations.
    NO Python for-loops - all operations run on GPU in parallel.

    Steps:
      1. Centroid lookup (vectorized by channel type)
      2. x_mse = y_hat @ Pi
      3. If use_qjl: correction = qjl_bits.float() @ S
      4. If use_qjl: scale = sqrt(pi/2) / k
      5. x_tilde = x_mse + (scale * gamma[:,None] * correction if use_qjl else 0)

    Args:
        use_qjl: If False (default), returns TurboQuant_mse (best MSE).
                 If True, returns TurboQuant_prod (better inner products, worse MSE).

    Returns (seq_len, head_dim) in target_dtype.
    """
    seq_len, head_dim = idx_all.shape
    k = S.shape[0]
    device = idx_all.device

    idx_long = idx_all.long()
    normal_mask = ~is_outlier  # (head_dim,) bool
    outlier_mask = is_outlier  # (head_dim,) bool

    # Step 1: Vectorized centroid lookup - NO PYTHON LOOPS
    y_hat = torch.zeros(seq_len, head_dim, dtype=torch.float32, device=device)

    # Normal channels (2-bit): use advanced indexing
    if normal_mask.any():
        idx_normal = idx_long[:, normal_mask]  # (seq_len, n_normal)
        y_hat[:, normal_mask] = centroids_2bit[idx_normal]

    # Outlier channels (3-bit): use advanced indexing
    if outlier_mask.any():
        idx_outlier = idx_long[:, outlier_mask]  # (seq_len, n_outlier)
        y_hat[:, outlier_mask] = centroids_3bit[idx_outlier]

    # Step 2: x_mse = y_hat @ Pi (GPU matrix multiply)
    x_mse = y_hat @ Pi

    # Steps 3-5: Optional QJL correction
    if use_qjl:
        qjl_f = qjl_bits.float()  # (seq_len, k)
        correction = qjl_f @ S  # (seq_len, head_dim)
        scale = math.sqrt(math.pi / 2.0) / k
        x_tilde = x_mse + scale * gamma.float().unsqueeze(-1) * correction
    else:
        x_tilde = x_mse

    return x_tilde.to(target_dtype)


# ---------------------------------------------------------------------------
# 1b. Hybrid implementation (PyTorch + Triton dequantize)
# ---------------------------------------------------------------------------


def decompress_kv_hybrid(
    idx_all: torch.Tensor,  # (seq_len, head_dim) int8
    qjl_bits: torch.Tensor,  # (seq_len, k) int8 — values are +1 or -1
    gamma: torch.Tensor,  # (seq_len,) float32
    Pi: torch.Tensor,  # (head_dim, head_dim) float32
    S: torch.Tensor,  # (k, head_dim) float32
    centroids_2bit: torch.Tensor,  # (4,) float32
    centroids_3bit: torch.Tensor,  # (8,) float32
    is_outlier: torch.Tensor,  # (head_dim,) bool
    target_dtype: torch.dtype = torch.float16,
    use_qjl: bool = False,  # Whether to apply QJL correction
) -> torch.Tensor:
    """
    Hybrid implementation: PyTorch cuBLAS for matmuls, Triton for dequantization.

    This avoids the tl.static_range(0, 128) loops that hang on sm_120.
    Uses Triton for the centroid lookup (embarrassingly parallel) and
    cuBLAS for the matrix operations (fast tensor cores).

    Pipeline:
      1. [Triton kernel]   y_hat = dequantize(idx)   (vectorized centroid lookup)
      2. [PyTorch cuBLAS]  x_mse = y_hat @ Pi        (back-rotation)
      3. [PyTorch cuBLAS]  If use_qjl: correction = qjl @ S (QJL correction)
      4. [PyTorch]         x_tilde = x_mse + (scale * gamma * correction if use_qjl)

    Args:
        use_qjl: If False (default), returns TurboQuant_mse (best MSE).
                 If True, returns TurboQuant_prod (better inner products, worse MSE).

    Returns (seq_len, head_dim) in target_dtype.
    """
    if not _TRITON_QUANTIZE_AVAILABLE:
        raise RuntimeError("Triton quantize kernels not available")

    seq_len, head_dim = idx_all.shape
    k = S.shape[0]

    # 1. Triton dequantization (vectorized centroid lookup)
    y_hat = triton_dequantize(idx_all, centroids_2bit, centroids_3bit, is_outlier)

    # 2. Back-rotation: x_mse = y_hat @ Pi (cuBLAS)
    x_mse = y_hat @ Pi  # (seq_len, head_dim)

    # 3-4. Optional QJL correction
    if use_qjl:
        qjl_f = qjl_bits.float()  # (seq_len, k)
        correction = qjl_f @ S  # (seq_len, head_dim)
        scale = math.sqrt(math.pi / 2.0) / k
        x_tilde = x_mse + scale * gamma.float().unsqueeze(-1) * correction
    else:
        x_tilde = x_mse

    return x_tilde.to(target_dtype)


def decompress_values_group_quant(
    idx_all: torch.Tensor,
    mins: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Decompress Value vectors from per-group 4-bit min/max quantization.

    Args:
        idx_all:     (seq_len, head_dim) int8 indices in [0, 15]
        mins:        (seq_len, n_groups) float32 group minima
        scales:      (seq_len, n_groups) float32 group scales
        group_size:  Number of channels per quantization group
        target_dtype:Output dtype

    Returns:
        x_hat: (seq_len, head_dim) tensor in target_dtype.
    """
    if idx_all.ndim != 2:
        raise ValueError(
            f"idx_all must be 2D (seq_len, head_dim), got shape {idx_all.shape}"
        )

    seq_len, head_dim = idx_all.shape
    if head_dim % group_size != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by group_size ({group_size})"
        )

    n_groups = head_dim // group_size
    if mins.shape != (seq_len, n_groups):
        raise ValueError(
            f"mins shape must be ({seq_len}, {n_groups}), got {mins.shape}"
        )
    if scales.shape != (seq_len, n_groups):
        raise ValueError(
            f"scales shape must be ({seq_len}, {n_groups}), got {scales.shape}"
        )

    idx_grouped = idx_all.to(torch.float32).reshape(seq_len, n_groups, group_size)
    x_hat = mins.unsqueeze(-1).float() + idx_grouped * scales.unsqueeze(-1).float()
    return x_hat.reshape(seq_len, head_dim).to(target_dtype)


# ---------------------------------------------------------------------------
# 2. Fully fused Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _turboquant_decompress_fused(
        # Compressed inputs — read once per token
        idx_all_ptr,  # (seq_len, head_dim) int8
        qjl_ptr,  # (seq_len, k) int8  (values +1 or -1)
        gamma_ptr,  # (seq_len,) float32
        # Random matrices and centroid tables
        Pi_ptr,  # (head_dim, head_dim) float32, row-major
        S_ptr,  # (k, head_dim) float32, row-major
        is_outlier_ptr,  # (head_dim,) int8  (1 = outlier, 0 = normal)
        c2_ptr,  # (4,) float32 — 2-bit centroids
        c3_ptr,  # (8,) float32 — 3-bit centroids
        # Output — written once per token
        out_ptr,  # (seq_len, head_dim) float16
        # Compile-time constants
        head_dim: tl.constexpr,
        k: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        One Triton program = one token (grid = [seq_len]).

        Global memory traffic:
          READ:  idx_all[token,:], qjl[token,:], gamma[token],
                 Pi[:,:], S[:,:], is_outlier[:], c2[:], c3[:]
          WRITE: out[token,:]  (float16)

        All intermediate tensors (y_hat, x_mse, correction) live only in
        SRAM registers — never written to global memory.
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < head_dim

        # ------------------------------------------------------------------
        # Load gamma scalar and compute scale
        # ------------------------------------------------------------------
        gamma = tl.load(gamma_ptr + pid)
        scale = gamma * tl.sqrt(tl.full([], math.pi / 2.0, dtype=tl.float32)) / k

        # ------------------------------------------------------------------
        # Load centroid tables into registers (12 floats total)
        # ------------------------------------------------------------------
        c2_0 = tl.load(c2_ptr + 0)
        c2_1 = tl.load(c2_ptr + 1)
        c2_2 = tl.load(c2_ptr + 2)
        c2_3 = tl.load(c2_ptr + 3)
        c3_0 = tl.load(c3_ptr + 0)
        c3_1 = tl.load(c3_ptr + 1)
        c3_2 = tl.load(c3_ptr + 2)
        c3_3 = tl.load(c3_ptr + 3)
        c3_4 = tl.load(c3_ptr + 4)
        c3_5 = tl.load(c3_ptr + 5)
        c3_6 = tl.load(c3_ptr + 6)
        c3_7 = tl.load(c3_ptr + 7)

        # ------------------------------------------------------------------
        # Step 1: reconstruct y_hat in SRAM via centroid lookup
        # For each channel j: look up idx_all[pid, j] in the appropriate table
        # ------------------------------------------------------------------
        y_hat = tl.zeros([BLOCK_D], dtype=tl.float32)

        for j in tl.static_range(0, BLOCK_D):
            if j < head_dim:
                idx_j = tl.load(idx_all_ptr + pid * head_dim + j).to(tl.int32)
                is_out = tl.load(is_outlier_ptr + j)

                # 2-bit lookup (4 levels)
                val2 = tl.where(
                    idx_j == 0,
                    c2_0,
                    tl.where(idx_j == 1, c2_1, tl.where(idx_j == 2, c2_2, c2_3)),
                )

                # 3-bit lookup (8 levels)
                val3 = tl.where(
                    idx_j == 0,
                    c3_0,
                    tl.where(
                        idx_j == 1,
                        c3_1,
                        tl.where(
                            idx_j == 2,
                            c3_2,
                            tl.where(
                                idx_j == 3,
                                c3_3,
                                tl.where(
                                    idx_j == 4,
                                    c3_4,
                                    tl.where(
                                        idx_j == 5,
                                        c3_5,
                                        tl.where(idx_j == 6, c3_6, c3_7),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )

                chosen = tl.where(is_out != 0, val3, val2)
                y_hat = tl.where(offs == j, chosen, y_hat)

        # ------------------------------------------------------------------
        # Step 2: x_mse[i] = sum_j Pi[j,i] * y_hat[j]  (Pi^T @ y_hat)
        # Pi column i: Pi_ptr + offs * head_dim + i
        # ------------------------------------------------------------------
        x_mse = tl.zeros([BLOCK_D], dtype=tl.float32)

        for i in tl.static_range(0, BLOCK_D):
            if i < head_dim:
                pi_col_i = tl.load(Pi_ptr + offs * head_dim + i, mask=mask, other=0.0)
                x_mse_i = tl.sum(y_hat * pi_col_i)
                x_mse = tl.where(offs == i, x_mse_i, x_mse)

        # ------------------------------------------------------------------
        # Step 3: correction[d] = sum_m qjl[m] * S[m,d] * scale
        # All in SRAM; scale already incorporates gamma
        # ------------------------------------------------------------------
        correction = tl.zeros([BLOCK_D], dtype=tl.float32)

        for m in tl.static_range(0, k):
            qjl_m = tl.load(qjl_ptr + pid * k + m).to(tl.float32)
            s_m = tl.load(S_ptr + m * head_dim + offs, mask=mask, other=0.0)
            correction += qjl_m * s_m

        correction = correction * scale

        # ------------------------------------------------------------------
        # Step 4: combine and write FP16 output — ONLY global memory write
        # ------------------------------------------------------------------
        x_out = (x_mse + correction).to(tl.float16)
        tl.store(out_ptr + pid * head_dim + offs, x_out, mask=mask)


# ---------------------------------------------------------------------------
# 3. Launcher
# ---------------------------------------------------------------------------


def triton_decompress_kv(
    idx_all: torch.Tensor,
    qjl_bits: torch.Tensor,
    gamma: torch.Tensor,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids_2bit: torch.Tensor,
    centroids_3bit: torch.Tensor,
    is_outlier: torch.Tensor,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Launch _turboquant_decompress_fused on the GPU.

    Global memory access pattern:
      READ:  idx_all, qjl_bits, gamma (once), Pi, S, is_outlier, c2, c3
      WRITE: out (float16, once per token)

    No intermediate tensors (y_hat, x_mse, correction) are ever written
    to global memory.

    Args:
        idx_all:        (seq_len, head_dim) int8
        qjl_bits:       (seq_len, k) int8
        gamma:          (seq_len,) float32
        Pi:             (head_dim, head_dim) float32 on CUDA
        S:              (k, head_dim) float32 on CUDA
        centroids_2bit: (4,) float32 on CUDA
        centroids_3bit: (8,) float32 on CUDA
        is_outlier:     (head_dim,) bool/int8 on CUDA

    Returns:
        x_tilde: (seq_len, head_dim) in target_dtype
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "triton_decompress_kv requires triton (pip install triton). "
            "Use decompress_kv_python for CPU testing."
        )

    if idx_all.device != Pi.device:
        raise ValueError(
            f"idx_all device ({idx_all.device}) must match Pi device ({Pi.device})"
        )
    if qjl_bits.device != Pi.device:
        raise ValueError(
            f"qjl_bits device ({qjl_bits.device}) must match Pi device ({Pi.device})"
        )
    if gamma.device != Pi.device:
        raise ValueError(
            f"gamma device ({gamma.device}) must match Pi device ({Pi.device})"
        )

    seq_len, head_dim = idx_all.shape
    k = S.shape[0]
    BLOCK_D = triton.next_power_of_2(head_dim)

    out = torch.empty(seq_len, head_dim, dtype=torch.float16, device=Pi.device)
    is_outlier_i8 = is_outlier.to(torch.int8)

    _turboquant_decompress_fused[(seq_len,)](
        idx_all.contiguous(),
        qjl_bits.contiguous(),
        gamma.contiguous(),
        Pi.contiguous(),
        S.contiguous(),
        is_outlier_i8,
        centroids_2bit.contiguous(),
        centroids_3bit.contiguous(),
        out,
        head_dim=head_dim,
        k=k,
        BLOCK_D=BLOCK_D,
    )

    return out.to(target_dtype)
