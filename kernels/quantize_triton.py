"""
TurboQuant Vectorized Quantization Kernels.

These kernels handle ONLY the quantization/dequantization step - mapping
continuous values to discrete centroid indices and back. No matrix multiplies.

Design principle: AVOID tl.static_range(0, large_number) loops.
Instead use fully vectorized operations that process all elements in parallel.

Contents:
  triton_quantize   - (seq, d) float -> (seq, d) int8 indices + (seq, d) float y_hat
  triton_dequantize - (seq, d) int8 indices -> (seq, d) float y_hat
"""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel: Vectorized Quantization
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _quantize_kernel(
        y_ptr,              # (seq_len, head_dim) float32 input
        is_outlier_ptr,     # (head_dim,) int8 mask
        c2_ptr,             # (4,) float32 2-bit centroids
        c3_ptr,             # (8,) float32 3-bit centroids
        idx_ptr,            # (seq_len, head_dim) int8 output
        y_hat_ptr,          # (seq_len, head_dim) float32 output
        seq_len,
        head_dim,
        BLOCK_SEQ: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Vectorized quantization kernel.
        
        Grid: (cdiv(seq_len, BLOCK_SEQ), cdiv(head_dim, BLOCK_D))
        Each program processes a (BLOCK_SEQ, BLOCK_D) tile.
        
        NO tl.static_range loops over head_dim!
        """
        pid_seq = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        # Compute offsets for this tile
        seq_offs = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        # Masks for bounds checking
        seq_mask = seq_offs < seq_len
        d_mask = d_offs < head_dim
        
        # Load outlier mask for this d-tile (broadcast across seq)
        # Shape: (BLOCK_D,)
        is_outlier = tl.load(is_outlier_ptr + d_offs, mask=d_mask, other=0).to(tl.int32)
        
        # Load centroids into registers
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
        
        # Load y values for this tile
        # Shape: (BLOCK_SEQ, BLOCK_D)
        y_ptrs = y_ptr + seq_offs[:, None] * head_dim + d_offs[None, :]
        tile_mask = seq_mask[:, None] & d_mask[None, :]
        y_vals = tl.load(y_ptrs, mask=tile_mask, other=0.0)
        
        # ---- 2-bit quantization (4 centroids) ----
        # Compute distances to all 4 centroids: shape (BLOCK_SEQ, BLOCK_D)
        d0_2 = tl.abs(y_vals - c2_0)
        d1_2 = tl.abs(y_vals - c2_1)
        d2_2 = tl.abs(y_vals - c2_2)
        d3_2 = tl.abs(y_vals - c2_3)
        
        # Find argmin using comparisons (fully vectorized)
        best2_idx = tl.zeros([BLOCK_SEQ, BLOCK_D], dtype=tl.int32)
        best2_dist = d0_2
        best2_val = tl.full([BLOCK_SEQ, BLOCK_D], c2_0, dtype=tl.float32)
        
        cond1 = d1_2 < best2_dist
        best2_idx = tl.where(cond1, 1, best2_idx)
        best2_dist = tl.where(cond1, d1_2, best2_dist)
        best2_val = tl.where(cond1, c2_1, best2_val)
        
        cond2 = d2_2 < best2_dist
        best2_idx = tl.where(cond2, 2, best2_idx)
        best2_dist = tl.where(cond2, d2_2, best2_dist)
        best2_val = tl.where(cond2, c2_2, best2_val)
        
        cond3 = d3_2 < best2_dist
        best2_idx = tl.where(cond3, 3, best2_idx)
        best2_val = tl.where(cond3, c2_3, best2_val)
        
        # ---- 3-bit quantization (8 centroids) ----
        d0_3 = tl.abs(y_vals - c3_0)
        d1_3 = tl.abs(y_vals - c3_1)
        d2_3 = tl.abs(y_vals - c3_2)
        d3_3 = tl.abs(y_vals - c3_3)
        d4_3 = tl.abs(y_vals - c3_4)
        d5_3 = tl.abs(y_vals - c3_5)
        d6_3 = tl.abs(y_vals - c3_6)
        d7_3 = tl.abs(y_vals - c3_7)
        
        best3_idx = tl.zeros([BLOCK_SEQ, BLOCK_D], dtype=tl.int32)
        best3_dist = d0_3
        best3_val = tl.full([BLOCK_SEQ, BLOCK_D], c3_0, dtype=tl.float32)
        
        c1 = d1_3 < best3_dist
        best3_idx = tl.where(c1, 1, best3_idx)
        best3_dist = tl.where(c1, d1_3, best3_dist)
        best3_val = tl.where(c1, c3_1, best3_val)
        
        c2 = d2_3 < best3_dist
        best3_idx = tl.where(c2, 2, best3_idx)
        best3_dist = tl.where(c2, d2_3, best3_dist)
        best3_val = tl.where(c2, c3_2, best3_val)
        
        c3 = d3_3 < best3_dist
        best3_idx = tl.where(c3, 3, best3_idx)
        best3_dist = tl.where(c3, d3_3, best3_dist)
        best3_val = tl.where(c3, c3_3, best3_val)
        
        c4 = d4_3 < best3_dist
        best3_idx = tl.where(c4, 4, best3_idx)
        best3_dist = tl.where(c4, d4_3, best3_dist)
        best3_val = tl.where(c4, c3_4, best3_val)
        
        c5 = d5_3 < best3_dist
        best3_idx = tl.where(c5, 5, best3_idx)
        best3_dist = tl.where(c5, d5_3, best3_dist)
        best3_val = tl.where(c5, c3_5, best3_val)
        
        c6 = d6_3 < best3_dist
        best3_idx = tl.where(c6, 6, best3_idx)
        best3_dist = tl.where(c6, d6_3, best3_dist)
        best3_val = tl.where(c6, c3_6, best3_val)
        
        c7 = d7_3 < best3_dist
        best3_idx = tl.where(c7, 7, best3_idx)
        best3_val = tl.where(c7, c3_7, best3_val)
        
        # ---- Select 2-bit or 3-bit based on is_outlier ----
        # Broadcast is_outlier (BLOCK_D,) to (BLOCK_SEQ, BLOCK_D)
        is_out = is_outlier[None, :] != 0
        final_idx = tl.where(is_out, best3_idx, best2_idx)
        final_val = tl.where(is_out, best3_val, best2_val)
        
        # ---- Store outputs ----
        idx_ptrs = idx_ptr + seq_offs[:, None] * head_dim + d_offs[None, :]
        tl.store(idx_ptrs, final_idx.to(tl.int8), mask=tile_mask)
        
        y_hat_ptrs = y_hat_ptr + seq_offs[:, None] * head_dim + d_offs[None, :]
        tl.store(y_hat_ptrs, final_val, mask=tile_mask)


    @triton.jit
    def _dequantize_kernel(
        idx_ptr,            # (seq_len, head_dim) int8 input
        is_outlier_ptr,     # (head_dim,) int8 mask
        c2_ptr,             # (4,) float32 2-bit centroids
        c3_ptr,             # (8,) float32 3-bit centroids
        y_hat_ptr,          # (seq_len, head_dim) float32 output
        seq_len,
        head_dim,
        BLOCK_SEQ: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Vectorized dequantization kernel (centroid lookup).
        
        Grid: (cdiv(seq_len, BLOCK_SEQ), cdiv(head_dim, BLOCK_D))
        Each program processes a (BLOCK_SEQ, BLOCK_D) tile.
        
        NO tl.static_range loops!
        """
        pid_seq = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        seq_offs = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        seq_mask = seq_offs < seq_len
        d_mask = d_offs < head_dim
        tile_mask = seq_mask[:, None] & d_mask[None, :]
        
        # Load outlier mask
        is_outlier = tl.load(is_outlier_ptr + d_offs, mask=d_mask, other=0).to(tl.int32)
        
        # Load centroids
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
        
        # Load indices
        idx_ptrs = idx_ptr + seq_offs[:, None] * head_dim + d_offs[None, :]
        idx_vals = tl.load(idx_ptrs, mask=tile_mask, other=0).to(tl.int32)
        
        # ---- 2-bit lookup using vectorized where ----
        val2 = tl.where(idx_vals == 0, c2_0,
               tl.where(idx_vals == 1, c2_1,
               tl.where(idx_vals == 2, c2_2, c2_3)))
        
        # ---- 3-bit lookup using vectorized where ----
        val3 = tl.where(idx_vals == 0, c3_0,
               tl.where(idx_vals == 1, c3_1,
               tl.where(idx_vals == 2, c3_2,
               tl.where(idx_vals == 3, c3_3,
               tl.where(idx_vals == 4, c3_4,
               tl.where(idx_vals == 5, c3_5,
               tl.where(idx_vals == 6, c3_6, c3_7)))))))
        
        # Select based on outlier mask
        is_out = is_outlier[None, :] != 0
        final_val = tl.where(is_out, val3, val2)
        
        # Store output
        y_hat_ptrs = y_hat_ptr + seq_offs[:, None] * head_dim + d_offs[None, :]
        tl.store(y_hat_ptrs, final_val, mask=tile_mask)


# ---------------------------------------------------------------------------
# Python launchers
# ---------------------------------------------------------------------------

def triton_quantize(
    y: torch.Tensor,                # (seq_len, head_dim) float32
    centroids_2bit: torch.Tensor,   # (4,) float32
    centroids_3bit: torch.Tensor,   # (8,) float32
    is_outlier: torch.Tensor,       # (head_dim,) bool
) -> tuple:
    """
    Launch vectorized quantization kernel.
    
    Args:
        y: Rotated values (seq_len, head_dim) float32
        centroids_2bit: 4 centroid values for normal channels
        centroids_3bit: 8 centroid values for outlier channels
        is_outlier: Boolean mask indicating outlier channels
        
    Returns:
        idx: (seq_len, head_dim) int8 centroid indices
        y_hat: (seq_len, head_dim) float32 quantized values
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    
    seq_len, head_dim = y.shape
    device = y.device
    
    # Allocate outputs
    idx = torch.empty(seq_len, head_dim, dtype=torch.int8, device=device)
    y_hat = torch.empty(seq_len, head_dim, dtype=torch.float32, device=device)
    
    # Choose block sizes - smaller to avoid compilation issues
    BLOCK_SEQ = 32
    BLOCK_D = 32  # Small tile size to avoid large unrolled code
    
    # Grid dimensions
    grid = (triton.cdiv(seq_len, BLOCK_SEQ), triton.cdiv(head_dim, BLOCK_D))
    
    _quantize_kernel[grid](
        y.contiguous(),
        is_outlier.to(torch.int8).contiguous(),
        centroids_2bit.contiguous(),
        centroids_3bit.contiguous(),
        idx,
        y_hat,
        seq_len,
        head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_D=BLOCK_D,
    )
    
    return idx, y_hat


def triton_dequantize(
    idx: torch.Tensor,              # (seq_len, head_dim) int8
    centroids_2bit: torch.Tensor,   # (4,) float32
    centroids_3bit: torch.Tensor,   # (8,) float32
    is_outlier: torch.Tensor,       # (head_dim,) bool
) -> torch.Tensor:
    """
    Launch vectorized dequantization kernel (centroid lookup).
    
    Args:
        idx: Centroid indices (seq_len, head_dim) int8
        centroids_2bit: 4 centroid values for normal channels
        centroids_3bit: 8 centroid values for outlier channels
        is_outlier: Boolean mask indicating outlier channels
        
    Returns:
        y_hat: (seq_len, head_dim) float32 quantized values
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    
    seq_len, head_dim = idx.shape
    device = idx.device
    
    y_hat = torch.empty(seq_len, head_dim, dtype=torch.float32, device=device)
    
    BLOCK_SEQ = 32
    BLOCK_D = 32
    
    grid = (triton.cdiv(seq_len, BLOCK_SEQ), triton.cdiv(head_dim, BLOCK_D))
    
    _dequantize_kernel[grid](
        idx.contiguous(),
        is_outlier.to(torch.int8).contiguous(),
        centroids_2bit.contiguous(),
        centroids_3bit.contiguous(),
        y_hat,
        seq_len,
        head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_D=BLOCK_D,
    )
    
    return y_hat
