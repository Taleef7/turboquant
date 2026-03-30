"""
TurboQuant fully-fused KV compression kernel.

Design constraints (CRITICAL):
  - x is loaded from global memory EXACTLY ONCE per token.
  - All intermediate tensors (y, y_hat, r, projections) live in SRAM registers ONLY.
  - Only compressed outputs (idx_all, qjl_bits, gamma) are written to global memory.

Contents:
  compress_kv_python  — CPU reference implementation (mirrors the fused kernel)
  build_outlier_mask  — select top-variance channels for 3-bit quantization
  _turboquant_compress_fused — fully fused Triton kernel (requires CUDA)
  triton_compress_kv  — launcher for the Triton kernel
"""

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


# ---------------------------------------------------------------------------
# 1. Python reference implementation
# ---------------------------------------------------------------------------

def compress_kv_python(
    x: torch.Tensor,                # (seq_len, head_dim) float32
    Pi: torch.Tensor,               # (head_dim, head_dim) float32
    S: torch.Tensor,                # (k, head_dim) float32
    centroids_2bit: torch.Tensor,   # (4,) float32
    centroids_3bit: torch.Tensor,   # (8,) float32
    is_outlier: torch.Tensor,       # (head_dim,) bool — True for 3-bit channels
) -> tuple:
    """
    CPU reference implementation that mirrors the fused Triton kernel exactly.

    The pipeline per token:
      1. Rotate:          y      = Pi @ x          (SRAM only in kernel)
      2. Quantize:        y_hat  = nearest centroid  (SRAM only in kernel)
      3. Back-rotate:     x_mse  = Pi^T @ y_hat     (SRAM only in kernel)
      4. Residual + norm: r = x - x_mse, gamma = ||r||
      5. QJL projection:  qjl_bits = sign(S @ r)

    Returns:
        idx_all:   (seq_len, head_dim) int8  — centroid indices (0-based)
        qjl_bits:  (seq_len, k) int8         — ±1 sketch bits
        gamma:     (seq_len,) float32        — residual norms
    """
    x = x.float()
    seq_len, head_dim = x.shape
    k = S.shape[0]

    # 1. Rotate: y = Pi @ x  (each row: y_i = Pi @ x_i)
    y = x @ Pi.T                                     # (seq_len, head_dim)

    # 2. Quantize per channel — assign nearest centroid, record index + value
    y_hat = torch.zeros_like(y)
    idx_all = torch.zeros(seq_len, head_dim, dtype=torch.int8)

    for j in range(head_dim):
        yj = y[:, j]                                  # (seq_len,)
        centroids = centroids_3bit if is_outlier[j] else centroids_2bit
        dists = (yj.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # (seq_len, n_levels)
        idx = dists.argmin(dim=1)                     # (seq_len,)
        idx_all[:, j] = idx.to(torch.int8)
        y_hat[:, j] = centroids[idx]

    # 3. Back-rotate: x_mse = Pi^T @ y_hat
    x_mse = y_hat @ Pi                               # (seq_len, head_dim)

    # 4. Residual + norm
    r = x - x_mse                                    # (seq_len, head_dim)
    gamma = torch.norm(r, dim=-1)                    # (seq_len,)

    # 5. QJL sketch: sign(S @ r^T), result (seq_len, k)
    projections = r @ S.T                            # (seq_len, k)
    qjl_bits = torch.sign(projections).to(torch.int8)

    return idx_all, qjl_bits, gamma


# ---------------------------------------------------------------------------
# 2. Helper: build outlier channel mask
# ---------------------------------------------------------------------------

def build_outlier_mask(x: torch.Tensor, n_outliers: int = 32) -> torch.Tensor:
    """
    Returns a (head_dim,) bool tensor marking the n_outliers highest-variance
    channels as True (those channels use the 3-bit codebook).

    Args:
        x:          (seq_len, head_dim) tensor of token vectors.
        n_outliers: number of channels to mark as outliers.

    Returns:
        Boolean mask of shape (head_dim,).
    """
    var = x.float().var(dim=0)                       # (head_dim,)
    _, top_idx = torch.topk(var, n_outliers)
    mask = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    mask[top_idx] = True
    return mask


# ---------------------------------------------------------------------------
# 3. Fully fused Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _turboquant_compress_fused(
        x_ptr,             # (seq_len, head_dim) float16  [input]
        Pi_ptr,            # (head_dim, head_dim) float32 [input]
        S_ptr,             # (k, head_dim) float32        [input]
        is_outlier_ptr,    # (head_dim,) int8             [input]
        c2_ptr,            # (4,) float32                 [input]
        c3_ptr,            # (8,) float32                 [input]
        idx_all_ptr,       # (seq_len, head_dim) int8     [output]
        qjl_ptr,           # (seq_len, k) int8            [output]
        gamma_ptr,         # (seq_len,) float32           [output]
        head_dim: tl.constexpr,
        k: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        One Triton program = one token (grid = [seq_len]).

        Global memory traffic:
          READ:  x[token, :], Pi[:,:], S[:,:], is_outlier[:], c2[:], c3[:]
          WRITE: idx_all[token,:], qjl_bits[token,:], gamma[token]

        All intermediate tensors (y, y_hat, r, projections) live only in
        SRAM registers — never spill to global memory.
        """
        token_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)                 # [0 .. head_dim-1]

        # ------------------------------------------------------------------
        # Load x[token, :] from global memory — THE ONLY READ OF x
        # ------------------------------------------------------------------
        x_base = x_ptr + token_id * head_dim
        x_vec = tl.load(x_base + offs).to(tl.float32)  # (BLOCK_D,) in SRAM

        # ------------------------------------------------------------------
        # Load tiny centroid arrays into registers (12 floats total)
        # ------------------------------------------------------------------
        c2_offs = tl.arange(0, 4)
        c3_offs = tl.arange(0, 8)
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
        # Step 1+2: Rotate y = Pi @ x  AND  quantize — all in SRAM
        #
        # For each output channel j:
        #   y_j = dot(Pi[j, :], x)  (Pi row-major: Pi[j, i] = *(Pi_ptr + j*head_dim + i))
        #   find nearest centroid for y_j
        #   accumulate results into SRAM vectors y_hat_vec and idx_vec
        # ------------------------------------------------------------------
        y_hat_vec = tl.zeros([BLOCK_D], dtype=tl.float32)  # reconstructed rotated coords
        idx_vec   = tl.zeros([BLOCK_D], dtype=tl.int8)

        for j in tl.static_range(0, BLOCK_D):
            # Load Pi row j from global memory
            pi_row = tl.load(Pi_ptr + j * head_dim + offs)   # (BLOCK_D,)
            y_j = tl.sum(pi_row * x_vec)                      # scalar dot product

            # Check is_outlier[j]
            outlier_flag = tl.load(is_outlier_ptr + j)        # int8 scalar

            # ---- 2-bit nearest centroid (4 levels) ----
            # Compare absolute distances sequentially, keep running best
            d0_2 = tl.abs(y_j - c2_0)
            d1_2 = tl.abs(y_j - c2_1)
            d2_2 = tl.abs(y_j - c2_2)
            d3_2 = tl.abs(y_j - c2_3)

            best2_val = c2_0
            best2_idx = tl.zeros([], dtype=tl.int8)
            # Level 1
            cond1_2 = d1_2 < d0_2
            best2_val = tl.where(cond1_2, c2_1, best2_val)
            best2_idx = tl.where(cond1_2, tl.full([], 1, dtype=tl.int8), best2_idx)
            # Level 2
            cond2_2 = d2_2 < tl.where(cond1_2, d1_2, d0_2)
            best2_val = tl.where(cond2_2, c2_2, best2_val)
            best2_idx = tl.where(cond2_2, tl.full([], 2, dtype=tl.int8), best2_idx)
            # Level 3
            best_dist_so_far_2 = tl.where(cond2_2, d2_2, tl.where(cond1_2, d1_2, d0_2))
            cond3_2 = d3_2 < best_dist_so_far_2
            best2_val = tl.where(cond3_2, c2_3, best2_val)
            best2_idx = tl.where(cond3_2, tl.full([], 3, dtype=tl.int8), best2_idx)

            # ---- 3-bit nearest centroid (8 levels) ----
            d0_3 = tl.abs(y_j - c3_0)
            d1_3 = tl.abs(y_j - c3_1)
            d2_3 = tl.abs(y_j - c3_2)
            d3_3 = tl.abs(y_j - c3_3)
            d4_3 = tl.abs(y_j - c3_4)
            d5_3 = tl.abs(y_j - c3_5)
            d6_3 = tl.abs(y_j - c3_6)
            d7_3 = tl.abs(y_j - c3_7)

            best3_val = c3_0
            best3_idx = tl.zeros([], dtype=tl.int8)
            cond1_3 = d1_3 < d0_3
            best3_val = tl.where(cond1_3, c3_1, best3_val)
            best3_idx = tl.where(cond1_3, tl.full([], 1, dtype=tl.int8), best3_idx)
            best3_dist = tl.where(cond1_3, d1_3, d0_3)
            cond2_3 = d2_3 < best3_dist
            best3_val = tl.where(cond2_3, c3_2, best3_val)
            best3_idx = tl.where(cond2_3, tl.full([], 2, dtype=tl.int8), best3_idx)
            best3_dist = tl.where(cond2_3, d2_3, best3_dist)
            cond3_3 = d3_3 < best3_dist
            best3_val = tl.where(cond3_3, c3_3, best3_val)
            best3_idx = tl.where(cond3_3, tl.full([], 3, dtype=tl.int8), best3_idx)
            best3_dist = tl.where(cond3_3, d3_3, best3_dist)
            cond4_3 = d4_3 < best3_dist
            best3_val = tl.where(cond4_3, c3_4, best3_val)
            best3_idx = tl.where(cond4_3, tl.full([], 4, dtype=tl.int8), best3_idx)
            best3_dist = tl.where(cond4_3, d4_3, best3_dist)
            cond5_3 = d5_3 < best3_dist
            best3_val = tl.where(cond5_3, c3_5, best3_val)
            best3_idx = tl.where(cond5_3, tl.full([], 5, dtype=tl.int8), best3_idx)
            best3_dist = tl.where(cond5_3, d5_3, best3_dist)
            cond6_3 = d6_3 < best3_dist
            best3_val = tl.where(cond6_3, c3_6, best3_val)
            best3_idx = tl.where(cond6_3, tl.full([], 6, dtype=tl.int8), best3_idx)
            best3_dist = tl.where(cond6_3, d6_3, best3_dist)
            cond7_3 = d7_3 < best3_dist
            best3_val = tl.where(cond7_3, c3_7, best3_val)
            best3_idx = tl.where(cond7_3, tl.full([], 7, dtype=tl.int8), best3_idx)

            # ---- Select 2-bit or 3-bit result based on is_outlier[j] ----
            use_3bit = outlier_flag != 0
            chosen_val = tl.where(use_3bit, best3_val, best2_val)
            chosen_idx = tl.where(use_3bit, best3_idx, best2_idx)

            # ---- Scatter scalar into SRAM vectors at position j ----
            # tl.where(offs == j, scalar, vector) writes scalar at index j
            y_hat_vec = tl.where(offs == j, chosen_val, y_hat_vec)
            idx_vec   = tl.where(offs == j, chosen_idx, idx_vec)

        # ------------------------------------------------------------------
        # Step 3: Back-rotate: x_mse = Pi^T @ y_hat
        #
        # For output channel i:
        #   x_mse[i] = sum_j Pi[j, i] * y_hat[j]
        #             = dot(Pi[:, i], y_hat)
        # Pi column i: Pi_ptr + offs * head_dim + i  (stride head_dim between rows)
        # ------------------------------------------------------------------
        x_mse_vec = tl.zeros([BLOCK_D], dtype=tl.float32)

        for i in tl.static_range(0, BLOCK_D):
            # Load Pi column i — Pi[j, i] for j in 0..head_dim-1
            pi_col = tl.load(Pi_ptr + offs * head_dim + i)   # (BLOCK_D,)
            x_mse_i = tl.sum(pi_col * y_hat_vec)             # scalar
            x_mse_vec = tl.where(offs == i, x_mse_i, x_mse_vec)

        # ------------------------------------------------------------------
        # Step 4: Residual and norm
        # ------------------------------------------------------------------
        r_vec = x_vec - x_mse_vec                            # (BLOCK_D,) in SRAM
        gamma_val = tl.sqrt(tl.sum(r_vec * r_vec))          # scalar in SRAM

        # ------------------------------------------------------------------
        # Step 5: QJL — sign(S @ r)
        # ------------------------------------------------------------------
        qjl_base = qjl_ptr + token_id * k

        for m in tl.static_range(0, k):
            s_row = tl.load(S_ptr + m * head_dim + offs)    # (BLOCK_D,)
            proj = tl.sum(s_row * r_vec)                     # scalar
            bit = tl.where(proj >= 0.0, tl.full([], 1, dtype=tl.int8),
                           tl.full([], -1, dtype=tl.int8))
            tl.store(qjl_base + m, bit)

        # ------------------------------------------------------------------
        # Write outputs to global memory (ONLY writes in the kernel)
        # ------------------------------------------------------------------
        # idx_all[token, :]
        tl.store(idx_all_ptr + token_id * head_dim + offs, idx_vec)
        # gamma[token]
        tl.store(gamma_ptr + token_id, gamma_val)
        # (qjl_bits written inside the loop above)


# ---------------------------------------------------------------------------
# 4. Launcher
# ---------------------------------------------------------------------------

def triton_compress_kv(
    x: torch.Tensor,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids_2bit: torch.Tensor,
    centroids_3bit: torch.Tensor,
    is_outlier: torch.Tensor,
) -> tuple:
    """
    Launch _turboquant_compress_fused on the GPU.

    Global memory access pattern:
      READ:  x (once, float16), Pi, S, is_outlier, c2, c3
      WRITE: idx_all, qjl_bits, gamma

    No intermediate tensors (y, y_hat, r) are ever written to global memory.

    Args:
        x:              (seq_len, head_dim) float16 on CUDA
        Pi:             (head_dim, head_dim) float32 on CUDA
        S:              (k, head_dim) float32 on CUDA
        centroids_2bit: (4,) float32 on CUDA
        centroids_3bit: (8,) float32 on CUDA
        is_outlier:     (head_dim,) bool/int8 on CUDA

    Returns:
        idx_all:   (seq_len, head_dim) int8
        qjl_bits:  (seq_len, k) int8
        gamma:     (seq_len,) float32
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "triton_compress_kv requires triton (pip install triton). "
            "Use compress_kv_python for CPU testing."
        )

    seq_len, head_dim = x.shape
    k = S.shape[0]

    # Ensure correct dtypes and contiguity
    x = x.contiguous().to(torch.float16)
    Pi = Pi.contiguous().to(torch.float32)
    S = S.contiguous().to(torch.float32)
    centroids_2bit = centroids_2bit.contiguous().to(torch.float32)
    centroids_3bit = centroids_3bit.contiguous().to(torch.float32)
    is_outlier_i8 = is_outlier.to(torch.int8).contiguous()

    # Allocate outputs
    idx_all  = torch.empty(seq_len, head_dim, dtype=torch.int8, device=x.device)
    qjl_bits = torch.empty(seq_len, k,        dtype=torch.int8, device=x.device)
    gamma    = torch.empty(seq_len,           dtype=torch.float32, device=x.device)

    # BLOCK_D must be a power-of-two >= head_dim for tl.arange
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (seq_len,)
    _turboquant_compress_fused[grid](
        x, Pi, S,
        is_outlier_i8,
        centroids_2bit, centroids_3bit,
        idx_all, qjl_bits, gamma,
        head_dim=head_dim,
        k=k,
        BLOCK_D=BLOCK_D,
    )

    return idx_all, qjl_bits, gamma
