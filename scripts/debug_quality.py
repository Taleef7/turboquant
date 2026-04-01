"""
Diagnostic script to trace where corruption occurs in TurboQuant compression.

This script captures actual KV states from the model and analyzes:
1. Distribution of raw KV values (mean, std, range)
2. Effect of normalization
3. Compression/decompression quality per layer
4. Which layers (if any) are most sensitive to compression

Usage:
    source venv312/bin/activate
    python scripts/debug_quality.py
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.turboquant_cache import TurboQuantCache
from utils.math_utils import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    get_centroids_2bit,
    get_centroids_3bit,
)
from kernels.compress_kv import compress_kv_python, build_outlier_mask
from kernels.decompress_kv import decompress_kv_python

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def analyze_tensor(name: str, t: torch.Tensor):
    """Print comprehensive statistics about a tensor."""
    t_flat = t.flatten().float().cpu().numpy()
    print(f"\n{name}:")
    print(f"  Shape: {t.shape}, dtype: {t.dtype}")
    print(f"  Mean: {t_flat.mean():.6f}, Std: {t_flat.std():.6f}")
    print(f"  Min: {t_flat.min():.6f}, Max: {t_flat.max():.6f}")
    print(f"  Abs Mean: {np.abs(t_flat).mean():.6f}")

    # L2 norms per row (if 2D or higher)
    if t.dim() >= 2:
        last_dim = t.shape[-1]
        t_2d = t.reshape(-1, last_dim)
        norms = torch.norm(t_2d.float(), dim=-1).cpu().numpy()
        print(
            f"  L2 Norms - Mean: {norms.mean():.6f}, Std: {norms.std():.6f}, Min: {norms.min():.6f}, Max: {norms.max():.6f}"
        )

    # Percentiles
    percs = np.percentile(t_flat, [1, 5, 25, 50, 75, 95, 99])
    print(f"  Percentiles [1,5,25,50,75,95,99]: {percs}")


def compute_quality_metrics(original: torch.Tensor, reconstructed: torch.Tensor):
    """Compute quality metrics between original and reconstructed tensors."""
    orig = original.float()
    recon = reconstructed.float()

    # MSE
    mse = torch.mean((orig - recon) ** 2).item()

    # NMSE (normalized by original variance)
    var_orig = torch.var(orig).item()
    nmse = mse / var_orig if var_orig > 1e-10 else float("inf")

    # Cosine similarity (per row, then average)
    orig_flat = orig.reshape(-1, orig.shape[-1])
    recon_flat = recon.reshape(-1, recon.shape[-1])

    cos_sim = torch.nn.functional.cosine_similarity(orig_flat, recon_flat, dim=-1)
    cos_mean = cos_sim.mean().item()
    cos_min = cos_sim.min().item()

    # Max absolute error
    max_abs_err = torch.max(torch.abs(orig - recon)).item()

    return {
        "mse": mse,
        "nmse": nmse,
        "cos_sim_mean": cos_mean,
        "cos_sim_min": cos_min,
        "max_abs_err": max_abs_err,
    }


class DiagnosticCache(TurboQuantCache):
    """
    TurboQuantCache with diagnostic instrumentation.

    Captures original and reconstructed KV states for analysis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_data = {
            "original_keys": {},
            "original_values": {},
            "reconstructed_keys": {},
            "reconstructed_values": {},
        }
        self.capture_diagnostics = True
        self.layers_to_capture = set()  # Empty = all layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Capture original states BEFORE compression
        if self.capture_diagnostics:
            if not self.layers_to_capture or layer_idx in self.layers_to_capture:
                if layer_idx not in self.diagnostic_data["original_keys"]:
                    self.diagnostic_data["original_keys"][layer_idx] = (
                        key_states.clone()
                    )
                    self.diagnostic_data["original_values"][layer_idx] = (
                        value_states.clone()
                    )

        # Call parent update (compress + decompress)
        out_k, out_v = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Capture reconstructed states
        if self.capture_diagnostics:
            if not self.layers_to_capture or layer_idx in self.layers_to_capture:
                # Only capture on first call (prefill) when we have full sequence
                if layer_idx not in self.diagnostic_data["reconstructed_keys"]:
                    self.diagnostic_data["reconstructed_keys"][layer_idx] = (
                        out_k.clone()
                    )
                    self.diagnostic_data["reconstructed_values"][layer_idx] = (
                        out_v.clone()
                    )

        return out_k, out_v


def test_compression_isolated():
    """
    Test compression on synthetic data vs real KV states to isolate the issue.
    """
    print("\n" + "=" * 70)
    print("TEST 1: SYNTHETIC DATA vs REAL KV STATES")
    print("=" * 70)

    device = torch.device("cuda")
    head_dim = 128
    n_qjl = 128
    n_outliers = 32
    seed = 42

    torch.manual_seed(seed)
    Pi = generate_rotation_matrix(head_dim, dtype=torch.float32, device=device)
    S = generate_qjl_matrix(head_dim, n_qjl, dtype=torch.float32, device=device)
    c2 = get_centroids_2bit(head_dim, device=device)
    c3 = get_centroids_3bit(head_dim, device=device)

    def test_roundtrip(x_input: torch.Tensor, name: str):
        """Test compression roundtrip on input tensor."""
        x = x_input.to(torch.float32).to(device)

        # Normalize to unit norm
        norms = torch.norm(x, dim=-1, keepdim=True)
        norms_safe = torch.clamp(norms, min=1e-8)
        x_normalized = x / norms_safe

        # Build outlier mask
        mask = build_outlier_mask(x_normalized, n_outliers)

        # Compress
        idx_all, qjl_bits, gamma = compress_kv_python(x_normalized, Pi, S, c2, c3, mask)

        # Decompress (without QJL for best MSE)
        x_hat_normalized = decompress_kv_python(
            idx_all,
            qjl_bits,
            gamma,
            Pi,
            S,
            c2,
            c3,
            mask,
            target_dtype=torch.float32,
            use_qjl=False,
        )

        # Rescale by original norms
        x_hat = x_hat_normalized * norms

        # Compute metrics
        metrics = compute_quality_metrics(x, x_hat)

        print(f"\n{name}:")
        print(f"  Input shape: {x.shape}")
        print(
            f"  Norms - Mean: {norms.mean().item():.4f}, Std: {norms.std().item():.4f}, Range: [{norms.min().item():.4f}, {norms.max().item():.4f}]"
        )
        print(f"  NMSE: {metrics['nmse']:.6f}")
        print(f"  Cosine Sim (mean): {metrics['cos_sim_mean']:.6f}")
        print(f"  Cosine Sim (min): {metrics['cos_sim_min']:.6f}")
        print(f"  Max Abs Error: {metrics['max_abs_err']:.6f}")

        return metrics

    # Test 1a: Synthetic Gaussian data (unit variance)
    print("\n--- Synthetic Data Tests ---")
    x_gauss = torch.randn(100, head_dim, dtype=torch.float32)
    test_roundtrip(x_gauss, "Gaussian (μ=0, σ=1)")

    # Test 1b: Scaled Gaussian (similar to real KV norms)
    x_scaled = torch.randn(100, head_dim, dtype=torch.float32) * 10
    test_roundtrip(x_scaled, "Scaled Gaussian (σ=10)")

    # Test 1c: Heavy-tailed distribution
    x_heavy = (
        torch.randn(100, head_dim, dtype=torch.float32) * torch.randn(100, 1).abs() * 50
    )
    test_roundtrip(x_heavy, "Heavy-tailed (variable norms)")


def test_real_kv_states():
    """
    Load model, run prefill, capture actual KV states, and analyze compression quality.
    """
    print("\n" + "=" * 70)
    print("TEST 2: REAL MODEL KV STATES")
    print("=" * 70)

    print(f"\nLoading {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # Use a short prompt for faster analysis
    prompt = "The quick brown fox jumps over the lazy dog. In the realm of artificial intelligence,"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    print(f"Prompt tokens: {input_len}")

    # Run with diagnostic cache
    diag_cache = DiagnosticCache(
        head_dim=128, n_qjl=128, device=torch.device("cuda"), dtype=torch.float16
    )
    # Only capture layers 0, 13, 27 (first, middle, last)
    diag_cache.layers_to_capture = {0, 13, 27}

    print("\nRunning prefill with TurboQuant...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,  # Just do prefill + 1 token
            do_sample=False,
            past_key_values=diag_cache,
        )

    # Analyze captured data
    print("\n--- Per-Layer Analysis ---")

    layer_results = []
    for layer_idx in sorted(diag_cache.diagnostic_data["original_keys"].keys()):
        orig_k = diag_cache.diagnostic_data["original_keys"][layer_idx]
        orig_v = diag_cache.diagnostic_data["original_values"][layer_idx]
        recon_k = diag_cache.diagnostic_data["reconstructed_keys"][layer_idx]
        recon_v = diag_cache.diagnostic_data["reconstructed_values"][layer_idx]

        print(f"\n=== Layer {layer_idx} ===")

        # Key analysis
        analyze_tensor(f"Original Keys (Layer {layer_idx})", orig_k)
        metrics_k = compute_quality_metrics(orig_k, recon_k)
        print(f"\nKey Reconstruction Quality:")
        print(f"  NMSE: {metrics_k['nmse']:.6f}")
        print(f"  Cosine Sim (mean): {metrics_k['cos_sim_mean']:.6f}")
        print(f"  Cosine Sim (min): {metrics_k['cos_sim_min']:.6f}")

        # Value analysis
        analyze_tensor(f"Original Values (Layer {layer_idx})", orig_v)
        metrics_v = compute_quality_metrics(orig_v, recon_v)
        print(f"\nValue Reconstruction Quality:")
        print(f"  NMSE: {metrics_v['nmse']:.6f}")
        print(f"  Cosine Sim (mean): {metrics_v['cos_sim_mean']:.6f}")
        print(f"  Cosine Sim (min): {metrics_v['cos_sim_min']:.6f}")

        layer_results.append(
            {
                "layer": layer_idx,
                "k_nmse": metrics_k["nmse"],
                "k_cos": metrics_k["cos_sim_mean"],
                "v_nmse": metrics_v["nmse"],
                "v_cos": metrics_v["cos_sim_mean"],
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Per-Layer Quality")
    print("=" * 70)
    print(
        f"{'Layer':>6} | {'K NMSE':>10} | {'K Cos':>8} | {'V NMSE':>10} | {'V Cos':>8}"
    )
    print("-" * 55)
    for r in layer_results:
        print(
            f"{r['layer']:>6} | {r['k_nmse']:>10.6f} | {r['k_cos']:>8.4f} | {r['v_nmse']:>10.6f} | {r['v_cos']:>8.4f}"
        )

    return diag_cache


def test_generation_comparison():
    """
    Compare baseline vs TurboQuant generation token-by-token.
    """
    print("\n" + "=" * 70)
    print("TEST 3: GENERATION COMPARISON")
    print("=" * 70)

    print(f"\nLoading {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # Simple prompt
    prompt = "Hello, I am a language model and I"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Baseline generation
    print("\n--- Baseline Generation ---")
    with torch.no_grad():
        baseline_out = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    print(f"Baseline: {baseline_text}")

    # TurboQuant generation
    print("\n--- TurboQuant Generation ---")
    tq_cache = TurboQuantCache(
        head_dim=128, n_qjl=128, device=torch.device("cuda"), dtype=torch.float16
    )
    with torch.no_grad():
        tq_out = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            past_key_values=tq_cache,
        )
    tq_text = tokenizer.decode(tq_out[0], skip_special_tokens=True)
    print(f"TurboQuant: {tq_text}")

    # Compare token-by-token
    print("\n--- Token Comparison ---")
    baseline_tokens = baseline_out[0].tolist()
    tq_tokens = tq_out[0].tolist()

    print(
        f"Baseline tokens: {len(baseline_tokens)}, TurboQuant tokens: {len(tq_tokens)}"
    )

    diverge_idx = None
    for i in range(min(len(baseline_tokens), len(tq_tokens))):
        if baseline_tokens[i] != tq_tokens[i]:
            diverge_idx = i
            break

    if diverge_idx is not None:
        print(f"\nFirst divergence at position {diverge_idx}:")
        print(
            f"  Baseline token: {baseline_tokens[diverge_idx]} = '{tokenizer.decode([baseline_tokens[diverge_idx]])}'"
        )
        print(
            f"  TurboQuant token: {tq_tokens[diverge_idx]} = '{tokenizer.decode([tq_tokens[diverge_idx]])}'"
        )
        print(f"  Context before: '{tokenizer.decode(baseline_tokens[:diverge_idx])}'")
    else:
        print("\nNo divergence found - outputs match!")


if __name__ == "__main__":
    test_compression_isolated()
    test_real_kv_states()
    test_generation_comparison()
