"""
patch_fungibility/joint_stream_geometry.py

Comprehensive implementation of the Fungibility Joint Stream Geometry Pipeline:
Maps the joint interaction between Feature-Space Direction v and Token-Space Pattern a:
    Delta P = alpha * a * v^T in R^{N x D}
under strict Frobenius norm matching: ||Delta P||_F = alpha * ||a||_2 * ||v||_2 = alpha.

1. Token-space pattern generators (Single, Global Coherent, Random Sign, Gaussian, Cluster, Checkerboard, Smooth).
2. Feature-space direction generators (Near-null, Sensitive, PC1, Random, Centroid).
3. Structured stream perturbation hook with exact Frobenius norm matching.
4. Full factorial condition evaluation across calibrated scales s in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0].
5. Token-feature two-way ANOVA and interaction variance decomposition.
6. Fraction-of-tokens sweep (1 token to 100% tokens) comparing coherence modes.
7. Confirmatory replication on DeiT-Tiny and DINOv2 ViT-S/14.
8. Publication figures generation (Figures A - E).
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

from patch_fungibility.dense_fraction_models import load_model_and_transform, forward_block_by_block
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits


# ==============================================================================
# 1. TOKEN-SPACE PATTERN GENERATORS (||a||_2 = 1.0)
# ==============================================================================

def construct_token_patterns(
    num_patches: int,
    seed: int = 42,
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Constructs standardized token-space distribution patterns a in R^N.
    Strictly verifies ||a||_2 = 1.0 for every pattern.
    """
    patterns = {}
    Hp = Wp = int(round(np.sqrt(num_patches)))

    # 1. Single Central Token
    a_sc = torch.zeros(num_patches, device=device)
    a_sc[num_patches // 2] = 1.0
    patterns["single_central"] = a_sc

    # 2. Single Corner Token
    a_co = torch.zeros(num_patches, device=device)
    a_co[0] = 1.0
    patterns["single_corner"] = a_co

    # 3. Global Coherent (all tokens move with identical positive sign)
    a_gc = torch.ones(num_patches, device=device) / np.sqrt(num_patches)
    patterns["global_coherent"] = a_gc

    # 4. Random Sign (incoherent matched-norm control, E[a_i] = 0)
    torch.manual_seed(seed)
    s = (torch.randint(0, 2, (num_patches,), device=device).float() * 2.0 - 1.0)
    a_rs = s / np.sqrt(num_patches)
    patterns["random_sign"] = a_rs

    # 5. Random Gaussian (continuous incoherent control)
    torch.manual_seed(seed + 1)
    z = torch.randn(num_patches, device=device)
    patterns["random_gaussian"] = z / z.norm()

    # 6. Spatial Cluster 25% (contiguous 2D block)
    grid = torch.zeros(Hp, Wp, device=device)
    r_start, r_len = Hp // 4, Hp // 2
    c_start, c_len = Wp // 4, Wp // 2
    grid[r_start:r_start + r_len, c_start:c_start + c_len] = 1.0
    k_cluster = grid.sum().item()
    patterns["spatial_cluster_25%"] = (grid / np.sqrt(k_cluster)).reshape(-1)

    # 7. Checkerboard (high-spatial-frequency alternating mode)
    r_idx = torch.arange(Hp, device=device).view(-1, 1)
    c_idx = torch.arange(Wp, device=device).view(1, -1)
    cb = ((-1.0) ** (r_idx + c_idx)).reshape(-1)
    patterns["checkerboard"] = cb / np.sqrt(num_patches)

    # 8. Smooth Spatial (low-spatial-frequency 2D cosine mode)
    r_f = torch.arange(Hp, device=device).float().view(-1, 1) / Hp
    c_f = torch.arange(Wp, device=device).float().view(1, -1) / Wp
    sm_grid = (torch.cos(np.pi * r_f) * torch.cos(np.pi * c_f)).reshape(-1)
    patterns["smooth_spatial"] = sm_grid / sm_grid.norm()

    # Strict norm verification
    for name, p in patterns.items():
        n_val = float(p.norm().item())
        assert abs(n_val - 1.0) < 1e-4, f"Token pattern {name} norm mismatch: {n_val}"

    return patterns


def construct_fraction_patterns(
    num_patches: int,
    fraction: float,
    seed: int = 42,
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Constructs token patterns for a specific participation fraction f in [1/N, 1.0].
    Strictly verifies ||a||_2 = 1.0 for every pattern.
    """
    k = max(1, int(round(fraction * num_patches)))
    Hp = Wp = int(round(np.sqrt(num_patches)))
    patterns = {}

    # Seed for consistent token selection across fraction calls
    torch.manual_seed(seed + int(fraction * 1000))
    perm = torch.randperm(num_patches, device=device)
    subset_indices = perm[:k]

    # 1. Coherent Fraction (k tokens move with identical positive sign)
    a_coh = torch.zeros(num_patches, device=device)
    a_coh[subset_indices] = 1.0 / np.sqrt(k)
    patterns["coherent_frac"] = a_coh

    # 2. Random Sign Fraction (k tokens move with random signs)
    s = (torch.randint(0, 2, (k,), device=device).float() * 2.0 - 1.0)
    a_rs = torch.zeros(num_patches, device=device)
    a_rs[subset_indices] = s / np.sqrt(k)
    patterns["random_sign_frac"] = a_rs

    # 3. Spatial Cluster Fraction (contiguous 2D block of approximately k tokens)
    grid = torch.zeros(Hp, Wp, device=device)
    side = max(1, int(round(np.sqrt(k))))
    side_r = min(side, Hp)
    side_c = min(max(1, int(round(k / side_r))), Wp)
    r_start = max(0, (Hp - side_r) // 2)
    c_start = max(0, (Wp - side_c) // 2)
    grid[r_start:r_start + side_r, c_start:c_start + side_c] = 1.0
    actual_k = grid.sum().item()
    patterns["cluster_frac"] = (grid / np.sqrt(actual_k)).reshape(-1)

    for name, p in patterns.items():
        n_val = float(p.norm().item())
        assert abs(n_val - 1.0) < 1e-4, f"Fraction pattern {name} norm mismatch: {n_val}"

    return patterns


# ==============================================================================
# 2. FEATURE-SPACE DIRECTION GENERATORS (||v||_2 = 1.0)
# ==============================================================================

def construct_joint_feature_directions(
    cov_matrix: torch.Tensor,      # (D, D)
    centroid: torch.Tensor,        # (D,)
    metric_matrix: torch.Tensor,   # (D, D)
    embed_dim: int,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Constructs unit feature probe directions:
    - jac_null: near-null eigenvector of M_l
    - jac_top: top sensitive eigenvector of M_l
    - pc1: leading PC of Sigma_l
    - rand_dir: isotropic random unit direction
    - centroid_dir: normalized centroid direction
    """
    directions = {}
    device = cov_matrix.device

    # Functional Metric Eigenspace
    evals_m, evecs_m = torch.linalg.eigh(metric_matrix)
    idx_m = torch.argsort(evals_m, descending=True)
    evecs_m = evecs_m[:, idx_m]

    directions["jac_top"] = evecs_m[:, 0]
    directions["jac_null"] = evecs_m[:, -1]

    # Covariance PC1
    evals_cov, evecs_cov = torch.linalg.eigh(cov_matrix)
    idx_cov = torch.argsort(evals_cov, descending=True)
    directions["pc1"] = evecs_cov[:, idx_cov[0]]

    # Isotropic Random Unit Vector
    torch.manual_seed(seed + 999)
    r_vec = torch.randn(embed_dim, device=device)
    directions["rand_dir"] = r_vec / r_vec.norm()

    # Centroid Direction
    if centroid.norm() > 1e-8:
        directions["centroid_dir"] = centroid / centroid.norm()
    else:
        directions["centroid_dir"] = directions["rand_dir"]

    for k, v in directions.items():
        directions[k] = v / (v.norm() + 1e-10)
        assert abs(float(directions[k].norm().item()) - 1.0) < 1e-4, f"Feature dir {k} not unit norm"

    return directions


def estimate_functional_metric(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Estimates the downstream functional metric matrix M_l = E[J^T J] on true-class margin.
    """
    embed_dim = 384 if "small" in model_key or "dinov2" in model_key else (192 if "tiny" in model_key else 768)
    M_accum = torch.zeros(embed_dim, embed_dim, dtype=torch.float64, device=device)
    n_samples = images.size(0)

    for i in range(n_samples):
        img_single = images[i:i+1]
        with torch.no_grad():
            _, coll = forward_block_by_block(model, model_key, x=img_single, collect_depths=(depth,))
            h_clean = coll[depth]

        h_param = nn.Parameter(h_clean.clone())
        out_logits, _ = forward_block_by_block(model, model_key, start_depth=depth, h_start=h_param)

        top_class = out_logits[0].argmax().item()
        logits_copy = out_logits[0].clone()
        logits_copy[top_class] = -1e9
        runner_up = logits_copy.argmax().item()

        margin = out_logits[0, top_class] - out_logits[0, runner_up]
        grad_h = torch.autograd.grad(margin, h_param, retain_graph=False)[0]

        # Average gradient across patch tokens
        mean_patch_grad = grad_h[0, 1:, :].mean(dim=0).double() # (D,)
        M_accum += torch.outer(mean_patch_grad, mean_patch_grad)

    return (M_accum / n_samples).float()


# ==============================================================================
# 3. STRUCTURED STREAM PERTURBATION HOOK (EXACT FROBENIUS NORM MATCHING)
# ==============================================================================

class StructuredStreamPerturbationHook:
    """
    Applies structured stream perturbation:
        Delta P = alpha * (a * v^T) in R^{N x D}
    where ||a||_2 = 1.0, ||v||_2 = 1.0.
    Guarantees ||Delta P||_F = alpha exactly.
    """
    def __init__(
        self,
        target_depth: int,
        token_pattern: torch.Tensor,   # (N,) on device
        feature_dir: torch.Tensor,     # (D,) on device
        amplitude: float               # scalar alpha = s * sigma_norm
    ):
        self.target_depth = target_depth
        self.a = token_pattern / (token_pattern.norm() + 1e-10)
        self.v = feature_dir / (feature_dir.norm() + 1e-10)
        self.amplitude = amplitude

    def __call__(self, depth: int, h: torch.Tensor) -> torch.Tensor:
        if depth != self.target_depth or self.amplitude == 0.0:
            return h

        # h: (B, 1 + N, D)
        h_mod = h.clone()
        # delta_P: (1, N, D)
        delta_P = (self.amplitude * torch.outer(self.a, self.v)).unsqueeze(0)
        h_mod[:, 1:, :] = h_mod[:, 1:, :] + delta_P
        return h_mod


# ==============================================================================
# 4. EVALUATION MODULES: CONDITIONS & FRACTION SWEEPS
# ==============================================================================

def evaluate_joint_condition_sweep(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    token_pattern_name: str,
    token_pattern: torch.Tensor,
    feature_dir_name: str,
    feature_dir: torch.Tensor,
    sigma_norm: float,
    scale_grid: List[float],
    device: torch.device,
    tolerance_threshold: float = 0.20
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Sweeps a single (token_pattern, feature_dir) pair across scale_grid.
    Returns:
    - list of records for each scale s
    - summary record with tolerance_radius_s
    """
    B = images.size(0)

    # Clean forward baseline
    with torch.no_grad():
        clean_logits, clean_coll = forward_block_by_block(model, model_key, x=images, collect_depths=(depth,))
        h_clean = clean_coll[depth]
        top_c = clean_logits.argmax(dim=-1)
        l_temp = clean_logits.clone()
        l_temp[torch.arange(B), top_c] = -1e9
        r_up = l_temp.argmax(dim=-1)
        clean_margin = clean_logits[torch.arange(B), top_c] - clean_logits[torch.arange(B), r_up]
        clean_probs = F.softmax(clean_logits, dim=-1)

    curve_records = []
    damages = []

    for s in scale_grid:
        alpha = float(s * sigma_norm)
        hook = StructuredStreamPerturbationHook(depth, token_pattern, feature_dir, alpha)
        h_pert = hook(depth, h_clean)

        with torch.no_grad():
            pert_logits, _ = forward_block_by_block(model, model_key, start_depth=depth, h_start=h_pert)

        pert_margin = pert_logits[torch.arange(B), top_c] - pert_logits[torch.arange(B), r_up]
        margin_damage = float((clean_margin - pert_margin).mean().item())
        damages.append(margin_damage)

        logit_l2 = float((pert_logits - clean_logits).norm(dim=-1).mean().item())
        pert_probs = F.softmax(pert_logits, dim=-1)
        kl = float(F.kl_div(pert_probs.log(), clean_probs, reduction="batchmean").item())
        top1_flips = float((pert_logits.argmax(dim=-1) != top_c).float().mean().item())
        top1_acc = float((pert_logits.argmax(dim=-1) == top_c).float().mean().item())

        curve_records.append({
            "model_key": model_key,
            "depth": depth,
            "token_pattern": token_pattern_name,
            "feature_dir": feature_dir_name,
            "scale_s": s,
            "alpha": alpha,
            "margin_damage": margin_damage,
            "logit_l2": logit_l2,
            "kl_divergence": kl,
            "top1_flip_rate": top1_flips,
            "top1_accuracy": top1_acc
        })

    # Estimate tolerance radius r_l(a, v; epsilon)
    r_tol = scale_grid[-1]
    for idx in range(len(scale_grid) - 1):
        if damages[idx] <= tolerance_threshold and damages[idx + 1] > tolerance_threshold:
            s_low, s_high = scale_grid[idx], scale_grid[idx + 1]
            d_low, d_high = damages[idx], damages[idx + 1]
            if d_high > d_low:
                r_tol = s_low + (tolerance_threshold - d_low) / (d_high - d_low) * (s_high - s_low)
            else:
                r_tol = s_low
            break
    if damages[0] > tolerance_threshold:
        r_tol = 0.0

    summary_record = {
        "model_key": model_key,
        "depth": depth,
        "token_pattern": token_pattern_name,
        "feature_dir": feature_dir_name,
        "tolerance_radius_s": float(r_tol),
        "tolerance_radius_alpha": float(r_tol * sigma_norm),
        "damage_at_s05": float(damages[scale_grid.index(0.5)] if 0.5 in scale_grid else damages[len(scale_grid)//2]),
        "damage_at_s10": float(damages[scale_grid.index(1.0)] if 1.0 in scale_grid else damages[-1]),
        "max_damage": float(max(damages))
    }

    return curve_records, summary_record


def run_fraction_sweep(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    feature_directions: Dict[str, torch.Tensor],
    sigma_norm: float,
    device: torch.device,
    fractions: List[float] = [0.005, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00],
    eval_scales: List[float] = [0.5, 1.0]
) -> pd.DataFrame:
    """
    Evaluates damage as a function of the fraction of participating tokens
    under Coherent vs. Random Sign vs. Spatial Cluster modes.
    """
    B = images.size(0)
    num_patches = 256 if model_key == "dinov2" else 196
    results = []

    with torch.no_grad():
        clean_logits, clean_coll = forward_block_by_block(model, model_key, x=images, collect_depths=(depth,))
        h_clean = clean_coll[depth]
        top_c = clean_logits.argmax(dim=-1)
        l_temp = clean_logits.clone()
        l_temp[torch.arange(B), top_c] = -1e9
        r_up = l_temp.argmax(dim=-1)
        clean_margin = clean_logits[torch.arange(B), top_c] - clean_logits[torch.arange(B), r_up]

    f_dirs_to_test = ["jac_null", "pc1", "jac_top", "rand_dir"]

    for frac in fractions:
        patterns = construct_fraction_patterns(num_patches, frac, seed=42, device=device)
        k_tokens = max(1, int(round(frac * num_patches)))

        for pat_mode, pat_tensor in patterns.items():
            for f_name in f_dirs_to_test:
                v_tensor = feature_directions[f_name]
                for s in eval_scales:
                    alpha = float(s * sigma_norm)
                    hook = StructuredStreamPerturbationHook(depth, pat_tensor, v_tensor, alpha)
                    h_pert = hook(depth, h_clean)

                    with torch.no_grad():
                        pert_logits, _ = forward_block_by_block(model, model_key, start_depth=depth, h_start=h_pert)

                    pert_margin = pert_logits[torch.arange(B), top_c] - pert_logits[torch.arange(B), r_up]
                    margin_damage = float((clean_margin - pert_margin).mean().item())
                    logit_l2 = float((pert_logits - clean_logits).norm(dim=-1).mean().item())
                    top1_flips = float((pert_logits.argmax(dim=-1) != top_c).float().mean().item())

                    results.append({
                        "model_key": model_key,
                        "depth": depth,
                        "fraction": frac,
                        "num_tokens": k_tokens,
                        "pattern_mode": pat_mode,
                        "feature_dir": f_name,
                        "scale_s": s,
                        "margin_damage": margin_damage,
                        "logit_l2": logit_l2,
                        "top1_flip_rate": top1_flips
                    })

    return pd.DataFrame(results)


# ==============================================================================
# 5. STATISTICAL INTERACTION ANALYSIS (TWO-WAY ANOVA & VARIANCE DECOMPOSITION)
# ==============================================================================

def analyze_token_feature_interaction(df_condition_results: pd.DataFrame) -> pd.DataFrame:
    """
    Fits Additive vs. Interaction linear models to quantify:
        Damage ~ TokenPattern + FeatureDir  (Additive)
        Damage ~ TokenPattern * FeatureDir  (Interaction)
    Calculates R^2_additive, R^2_interaction, and Delta R^2 for each model and depth.
    """
    interaction_stats = []

    for (m_key, depth), group in df_condition_results.groupby(["model_key", "depth"]):
        # Evaluate on damage_at_s10 (or damage_at_s05)
        df_sub = group.copy()

        # Fit Additive OLS
        model_add = ols("damage_at_s10 ~ C(token_pattern) + C(feature_dir)", data=df_sub).fit()
        r2_add = float(model_add.rsquared)

        # Fit Interaction OLS
        model_int = ols("damage_at_s10 ~ C(token_pattern) * C(feature_dir)", data=df_sub).fit()
        r2_int = float(model_int.rsquared)
        delta_r2 = float(r2_int - r2_add)

        # ANOVA comparison
        anova_table = sm.stats.anova_lm(model_add, model_int)
        f_stat = float(anova_table["F"].iloc[1]) if len(anova_table) > 1 and pd.notna(anova_table["F"].iloc[1]) else 0.0
        p_val = float(anova_table["Pr(>F)"].iloc[1]) if len(anova_table) > 1 and pd.notna(anova_table["Pr(>F)"].iloc[1]) else 1.0

        # Calculate Coherence Ratio: Global Coherent / Random Sign across directions
        gc_sub = df_sub[df_sub["token_pattern"] == "global_coherent"].set_index("feature_dir")["damage_at_s10"]
        rs_sub = df_sub[df_sub["token_pattern"] == "random_sign"].set_index("feature_dir")["damage_at_s10"]
        ratio_series = (gc_sub / (rs_sub + 1e-6))
        mean_coherence_ratio = float(ratio_series.mean())

        interaction_stats.append({
            "model_key": m_key,
            "depth": depth,
            "r2_additive": r2_add,
            "r2_interaction": r2_int,
            "delta_r2_interaction": delta_r2,
            "f_statistic": f_stat,
            "p_value": p_val,
            "mean_coherence_damage_ratio_gc_vs_rs": mean_coherence_ratio
        })

    return pd.DataFrame(interaction_stats)


# ==============================================================================
# 6. CONFIRMATORY REPLICATION ON DEIT-TINY & DINOV2
# ==============================================================================

def run_confirmatory_replication(
    images_100: torch.Tensor,
    device: torch.device
) -> pd.DataFrame:
    """
    Replicates the decisive contrast (Coherent vs. Random Sign vs. Checkerboard vs. Single)
    across Near-Null vs. Sensitive vs. PC1 on DeiT-Tiny and DINOv2 at their peak fungible depth (Depth 8).
    """
    rep_records = []
    models_to_rep = [("deit_tiny", 8), ("dinov2", 8)]
    scale_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

    for m_key, depth in models_to_rep:
        print(f"\nReplicating decisive contrast on {m_key} at Depth {depth}...")
        model, transform, meta = load_model_and_transform(m_key, device)
        embed_dim = meta["embed_dim"]
        num_patches = meta["num_patches"]

        calib_ds, _, _, _ = get_disjoint_imagenet_splits()
        calib_ds.transform = transform
        calib_subset = torch.utils.data.Subset(calib_ds, list(range(len(images_100))))
        loader = DataLoader(calib_subset, batch_size=25, shuffle=False)
        m_imgs = torch.cat([x for x, _ in loader], dim=0).to(device)

        # 1. Compute Sigma_l and Centroid
        with torch.no_grad():
            _, coll = forward_block_by_block(model, m_key, x=m_imgs, collect_depths=(depth,))
            h_depth = coll[depth]
            patches = h_depth[:, 1:, :].reshape(-1, embed_dim).to(torch.float64)
            centroid = patches.mean(dim=0).float()
            centered = patches - centroid.double()
            cov_matrix = ((centered.t() @ centered) / (patches.size(0) - 1)).float()
            sigma_norm = float(torch.sqrt(torch.trace(cov_matrix)).item())

        # 2. Metric M_l
        metric_matrix = estimate_functional_metric(model, m_key, depth, m_imgs, device)

        # 3. Directions & Patterns
        f_dirs = construct_joint_feature_directions(cov_matrix, centroid, metric_matrix, embed_dim)
        t_pats = construct_token_patterns(num_patches, seed=42, device=device)

        pats_to_rep = ["single_central", "global_coherent", "random_sign", "checkerboard"]
        dirs_to_rep = ["jac_null", "pc1", "jac_top"]

        for p_name in pats_to_rep:
            for d_name in dirs_to_rep:
                _, summary = evaluate_joint_condition_sweep(
                    model, m_key, depth, m_imgs,
                    p_name, t_pats[p_name],
                    d_name, f_dirs[d_name],
                    sigma_norm, scale_grid, device
                )
                rep_records.append(summary)

    return pd.DataFrame(rep_records)


# ==============================================================================
# 7. PUBLICATION FIGURES GENERATION (FIGURES A - E)
# ==============================================================================

def generate_joint_stream_figures(
    df_conditions: pd.DataFrame,
    df_curves: pd.DataFrame,
    df_fractions: pd.DataFrame,
    df_replication: pd.DataFrame,
    output_dir: str
):
    """
    Generates publication figures A through E.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # -------------------------------------------------------------------------
    # FIGURE A: Heatmap of Token Mode x Feature Mode (Tolerance Radius)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    for ax, (m_key, depth, title) in zip(axes, [("deit_small", 8, "DeiT-Small (Depth 8 - Peak Fungible)"),
                                                 ("vit_base", 7, "ViT-Base (Depth 7 - Peak Fungible)")]):
        sub = df_conditions[(df_conditions["model_key"] == m_key) & (df_conditions["depth"] == depth)]
        pivot_r = sub.pivot(index="token_pattern", columns="feature_dir", values="tolerance_radius_s")
        # Reorder rows and columns logically
        row_order = ["single_central", "single_corner", "checkerboard", "random_sign",
                     "random_gaussian", "spatial_cluster_25%", "smooth_spatial", "global_coherent"]
        col_order = ["jac_null", "pc1", "centroid_dir", "rand_dir", "jac_top"]
        row_order = [r for r in row_order if r in pivot_r.index]
        col_order = [c for c in col_order if c in pivot_r.columns]
        pivot_r = pivot_r.reindex(index=row_order, columns=col_order)

        sns.heatmap(pivot_r, annot=True, fmt=".2f", cmap="magma_r", cbar=True, ax=ax, vmin=0.0, vmax=4.0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature-Space Direction $v$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Token-Space Pattern $a$", fontsize=11, fontweight="bold")

    fig.suptitle("Figure A: Joint Fungibility Heatmap: Token-Space Pattern × Feature-Space Direction\n(Color: Directional Tolerance Radius $r_l(a, v)$ under strict $\|\Delta P\|_F$ matching)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_a_token_feature_heatmap.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE B: Fraction-of-Tokens Sweep (Damage vs. Fraction Perturbed)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    for ax, (m_key, depth, title) in zip(axes, [("deit_small", 8, "DeiT-Small (Depth 8, $s=1.0$)"),
                                                 ("vit_base", 7, "ViT-Base (Depth 7, $s=1.0$)")]):
        sub_frac = df_fractions[(df_fractions["model_key"] == m_key) &
                                (df_fractions["depth"] == depth) &
                                (df_fractions["scale_s"] == 1.0)]

        for pat_mode, color, marker in zip(["coherent_frac", "random_sign_frac", "cluster_frac"],
                                           ["#d95f02", "#2b5c8f", "#7570b3"],
                                           ["o", "s", "^"]):
            p_sub = sub_frac[(sub_frac["pattern_mode"] == pat_mode) & (sub_frac["feature_dir"] == "jac_top")]
            ax.plot(p_sub["fraction"], p_sub["margin_damage"],
                    marker=marker, linewidth=2.5, label=f"Sensitive ($J_{{top}}$) - {pat_mode}", color=color)

        for pat_mode, color, marker in zip(["coherent_frac", "random_sign_frac"],
                                           ["#fc8d62", "#8da0cb"],
                                           ["--o", "--s"]):
            p_sub = sub_frac[(sub_frac["pattern_mode"] == pat_mode.replace("--", "")) & (sub_frac["feature_dir"] == "jac_null")]
            ax.plot(p_sub["fraction"], p_sub["margin_damage"],
                    linestyle="--", marker="x", linewidth=2.0, label=f"Near-Null ($J_{{null}}$) - {pat_mode.replace('--', '')}", color=color)

        ax.set_xlabel("Fraction of Tokens Participating ($f = k / N$)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Margin Damage (Matched $\|\Delta P\|_F$)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Figure B: Fraction-of-Tokens Sweep: Collective Coherence vs. Incoherent Cancellation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_b_fraction_sweep.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE C: Finite-Radius Response Curves for Key Combinations
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    key_combos = [
        ("jac_null", "global_coherent", "#2ca02c", "o", "Null × Coherent"),
        ("jac_null", "random_sign", "#98df8a", "s", "Null × Random Sign"),
        ("jac_top", "global_coherent", "#d62728", "^", "Sensitive × Coherent"),
        ("jac_top", "random_sign", "#ff9896", "v", "Sensitive × Random Sign"),
        ("pc1", "global_coherent", "#1f77b4", "D", "PC1 × Coherent"),
        ("pc1", "random_sign", "#aec7e8", "d", "PC1 × Random Sign"),
    ]

    for ax, (m_key, depth, title) in zip(axes, [("deit_small", 8, "DeiT-Small (Depth 8)"),
                                                 ("vit_base", 7, "ViT-Base (Depth 7)")]):
        sub_c = df_curves[(df_curves["model_key"] == m_key) & (df_curves["depth"] == depth)]

        for f_name, t_name, color, marker, label in key_combos:
            curve = sub_c[(sub_c["feature_dir"] == f_name) & (sub_c["token_pattern"] == t_name)].sort_values("scale_s")
            ax.plot(curve["scale_s"], curve["margin_damage"],
                    marker=marker, color=color, linewidth=2.0, label=label)

        ax.axhline(0.20, color="black", linestyle="--", alpha=0.7, label="Tolerance (0.20)")
        ax.set_xlabel("Perturbation Scale $s = \\alpha / \\sigma_{\\text{norm}}$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Margin Damage", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Figure C: Finite-Radius Perturbation Curves for Key Feature × Token Combinations",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_c_key_combination_curves.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE D: Spatial Pattern Comparison across Feature Directions
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6), dpi=300)
    sub_d = df_conditions[(df_conditions["model_key"] == "deit_small") & (df_conditions["depth"] == 8)].copy()
    patterns_spatial = ["single_central", "spatial_cluster_25%", "smooth_spatial", "global_coherent", "checkerboard", "random_sign"]
    sub_d = sub_d[sub_d["token_pattern"].isin(patterns_spatial)]

    sns.barplot(
        data=sub_d, x="token_pattern", y="damage_at_s10", hue="feature_dir",
        palette="tab10", order=patterns_spatial
    )
    plt.xlabel("Spatial Token Pattern", fontsize=12, fontweight="bold")
    plt.ylabel("Margin Damage at Scale $s = 1.0$ (Matched $\|\Delta P\|_F$)", fontsize=12, fontweight="bold")
    plt.title("Figure D: Spatial Organization of Token Patterns across Feature Directions (DeiT-Small, Depth 8)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Feature Direction", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_d_spatial_pattern_comparison.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE E: Replication Summary across Architectures
    # -------------------------------------------------------------------------
    plt.figure(figsize=(13, 6), dpi=300)
    # Combine exploratory peak-depth data with replication summary
    sub_exp = df_conditions[
        ((df_conditions["model_key"] == "deit_small") & (df_conditions["depth"] == 8)) |
        ((df_conditions["model_key"] == "vit_base") & (df_conditions["depth"] == 7))
    ][["model_key", "token_pattern", "feature_dir", "tolerance_radius_s"]].copy()
    sub_rep = df_replication[["model_key", "token_pattern", "feature_dir", "tolerance_radius_s"]].copy()
    full_rep_df = pd.concat([sub_exp, sub_rep], ignore_index=True)
    full_rep_df = full_rep_df[full_rep_df["token_pattern"].isin(["global_coherent", "random_sign"])]

    sns.barplot(
        data=full_rep_df, x="model_key", y="tolerance_radius_s", hue="token_pattern",
        palette=["#d95f02", "#2b5c8f"]
    )
    plt.xlabel("Vision Transformer Architecture (Depth 8)", fontsize=12, fontweight="bold")
    plt.ylabel("Tolerance Radius $r_l(a, v)$ [units of $\sigma_{\\text{norm}}$]", fontsize=12, fontweight="bold")
    plt.title("Figure E: Cross-Architecture Replication of Coherence Bottleneck (Depth 8)", fontsize=14, fontweight="bold")
    plt.legend(title="Token Pattern", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_e_replication_summary.png"), bbox_inches="tight")
    plt.close()
    print("All publication figures (A - E) successfully generated.")


# ==============================================================================
# 8. MASTER EXECUTION PIPELINE
# ==============================================================================

def run_joint_stream_geometry_experiment(
    n_images: int = 100,
    models_to_evaluate: Tuple[str, ...] = ("deit_small", "vit_base"),
    depths_map: Dict[str, Tuple[int, ...]] = {
        "deit_small": (5, 8, 10),
        "vit_base": (5, 7, 10)
    },
    output_dir: str = os.path.join("outputs", "fungibility_joint_stream_geometry"),
    figures_dir: str = os.path.join("figures", "fungibility_joint_stream_geometry")
) -> Dict[str, Any]:
    """
    Executes the comprehensive joint stream geometry mapping:
    1. Condition results table (8 token patterns x 5 feature directions x depths x models).
    2. Finite-radius curves.
    3. Fraction-of-tokens sweeps.
    4. Two-way ANOVA and interaction decomposition.
    5. Confirmatory replication on DeiT-Tiny and DINOv2.
    6. Figures A through E.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Fungibility Joint Stream Geometry Experiment on {device} (N={n_images} images)")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    scale_grid = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

    all_curves = []
    all_summaries = []
    all_fractions = []

    for m_key in models_to_evaluate:
        depths = depths_map[m_key]
        print(f"\n=======================================================")
        print(f"Mapping Joint Stream Geometry for: {m_key} (Depths: {depths})")
        print(f"=======================================================")

        model, transform, meta = load_model_and_transform(m_key, device)
        embed_dim = meta["embed_dim"]
        num_patches = meta["num_patches"]

        calib_ds, _, _, _ = get_disjoint_imagenet_splits()
        calib_ds.transform = transform
        calib_subset = torch.utils.data.Subset(calib_ds, list(range(n_images)))
        loader = DataLoader(calib_subset, batch_size=25, shuffle=False)

        img_batches = []
        for x_b, _ in loader:
            img_batches.append(x_b)
        all_imgs = torch.cat(img_batches, dim=0).to(device)
        print(f"Loaded {all_imgs.size(0)} evaluation images for {m_key}.")

        # Generate Token Patterns (N)
        token_patterns = construct_token_patterns(num_patches, seed=42, device=device)

        for depth in depths:
            print(f"\n--- Depth {depth} ({m_key}) ---")

            # 1. Compute covariance matrix Sigma_l and centroid
            with torch.no_grad():
                _, coll = forward_block_by_block(model, m_key, x=all_imgs, collect_depths=(depth,))
                h_depth = coll[depth]
                patches = h_depth[:, 1:, :].reshape(-1, embed_dim).to(torch.float64)
                centroid = patches.mean(dim=0).float()
                centered = patches - centroid.double()
                cov_matrix = ((centered.t() @ centered) / (patches.size(0) - 1)).float()
                sigma_norm = float(torch.sqrt(torch.trace(cov_matrix)).item())

            # 2. Estimate Functional Metric M_l
            metric_matrix = estimate_functional_metric(model, m_key, depth, all_imgs, device)

            # 3. Construct Feature Directions
            feature_dirs = construct_joint_feature_directions(
                cov_matrix, centroid, metric_matrix, embed_dim
            )

            # 4. Full Factorial Grid Sweep: Token Patterns x Feature Directions
            print(f"  Evaluating {len(token_patterns)} token patterns x {len(feature_dirs)} feature directions...")
            for t_name, t_vec in token_patterns.items():
                for f_name, f_vec in feature_dirs.items():
                    curves, summary = evaluate_joint_condition_sweep(
                        model, m_key, depth, all_imgs,
                        t_name, t_vec, f_name, f_vec,
                        sigma_norm, scale_grid, device
                    )
                    all_curves.extend(curves)
                    all_summaries.append(summary)

            # 5. Fraction Sweep at Peak Depth
            if depth in [8, 7]:
                print(f"  Evaluating fraction-of-tokens sweep at depth {depth}...")
                frac_df = run_fraction_sweep(
                    model, m_key, depth, all_imgs, feature_dirs, sigma_norm, device
                )
                all_fractions.append(frac_df)

    # Compile dataframes
    full_curves_df = pd.DataFrame(all_curves)
    full_conditions_df = pd.DataFrame(all_summaries)
    full_fractions_df = pd.concat(all_fractions, ignore_index=True)

    # 6. Statistical Interaction Analysis (Two-way ANOVA)
    print("\nRunning statistical interaction analysis (Two-way ANOVA)...")
    df_interaction = analyze_token_feature_interaction(full_conditions_df)

    # 7. Confirmatory Replication on DeiT-Tiny & DINOv2
    print("\nRunning confirmatory replication on DeiT-Tiny and DINOv2...")
    df_replication = run_confirmatory_replication(all_imgs, device)

    # Save CSV outputs
    full_conditions_df.to_csv(os.path.join(output_dir, "condition_results.csv"), index=False)
    full_curves_df.to_csv(os.path.join(output_dir, "directional_curves.csv"), index=False)
    full_fractions_df.to_csv(os.path.join(output_dir, "fraction_sweep.csv"), index=False)
    df_interaction.to_csv(os.path.join(output_dir, "token_feature_interaction.csv"), index=False)
    df_replication.to_csv(os.path.join(output_dir, "replication_summary.csv"), index=False)
    print(f"\nAll machine-readable CSV outputs saved to: {output_dir}")

    # Generate Figures A - E
    generate_joint_stream_figures(
        full_conditions_df, full_curves_df, full_fractions_df, df_replication, figures_dir
    )

    # Validation Manifest
    manifest = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "device": str(device),
        "models_evaluated": list(models_to_evaluate),
        "depths_evaluated": depths_map,
        "num_images": n_images,
        "mean_coherence_damage_ratio": {
            f"{row['model_key']}_depth_{row['depth']}": float(row["mean_coherence_damage_ratio_gc_vs_rs"])
            for _, row in df_interaction.iterrows()
        },
        "interaction_delta_r2": {
            f"{row['model_key']}_depth_{row['depth']}": float(row["delta_r2_interaction"])
            for _, row in df_interaction.iterrows()
        }
    }
    with open(os.path.join(output_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Validation manifest saved to: {os.path.join(output_dir, 'validation_manifest.json')}")

    return manifest


if __name__ == "__main__":
    run_joint_stream_geometry_experiment()
