"""
patch_fungibility/predictive_operator.py

Implementation of the Token-Transmission Predictive Operator (T_v):
A linear operator mapping token-space perturbation patterns a in R^N
to predicted downstream readout disturbances in R^{D_out}:

    T_v a = sum_h W_O^h [ (w_h^T a) (v^T W_{V,h}) ]

Provides:
- Exact construction of T_v for CLS-readout models and dual-pooled DINOv2
- SVD spectral decomposition T_v = U Sigma V^T and singular modes q_k
- Direct mode prediction tests on real unconstrained models
- Finite-radius validation sweeps
- Out-of-basis pattern transmission predictions
- Random held-out pattern predictions
- Depth spectrum evolution and null space dimension tracking
- Bilinear feature x token interaction matrix
- Spatial mode structure analysis
"""

import os
import math
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.joint_stream_geometry import (
    construct_token_patterns,
    construct_joint_feature_directions,
    estimate_functional_metric
)


# ==============================================================================
# 1. Operator Construction and SVD
# ==============================================================================

def construct_token_transmission_operator(
    model: nn.Module,
    model_key: str,
    depth: int,
    h_clean_depth: torch.Tensor,
    v: torch.Tensor,
    device: torch.device = torch.device("cuda")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs the linear operator T_v in R^{D_out x N} and computes its SVD:
        T_v = U Sigma Vh
    where:
        Vh in R^{N x N} contains the right singular modes q_k (rows of Vh).
    """
    blk_idx = depth  # block takes h_depth to h_{depth+1}
    B, total_tokens, D = h_clean_depth.shape
    N = total_tokens - 1

    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        blk = model.blocks[blk_idx]
        norm_c = blk.norm1(h_clean_depth)
        num_heads = blk.attn.num_heads
        head_dim = blk.attn.head_dim
        scale = blk.attn.scale

        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qc, kc, _ = qkv_c.unbind(0)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
        mean_w = A_clean[:, :, 0, 1:].mean(dim=0)  # (H, N)

        W_V = blk.attn.qkv.weight[2 * D : 3 * D, :]  # (D, D)
        W_O = blk.attn.proj.weight                   # (D, D)

        T_v = torch.zeros(D, N, device=device)
        for h in range(num_heads):
            W_V_h = W_V[h * head_dim : (h + 1) * head_dim, :]
            v_h = W_V_h @ v
            W_O_h = W_O[:, h * head_dim : (h + 1) * head_dim]
            head_vec = W_O_h @ v_h
            T_v += torch.outer(head_vec, mean_w[h, :])

    elif model_key == "dinov2":
        blk = model.backbone.blocks[blk_idx]
        norm_c = blk.norm1(h_clean_depth)
        num_heads = blk.attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5

        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim)
        qc, kc, _ = torch.unbind(qkv_c, 2)
        qc = qc.transpose(1, 2)
        kc = kc.transpose(1, 2)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)

        w_cls = A_clean[:, :, 0, 1:].mean(dim=0)                # (H, N)
        w_patch = A_clean[:, :, 1:, 1:].mean(dim=2).mean(dim=0)  # (H, N)

        W_V = blk.attn.qkv.weight[2 * D : 3 * D, :]
        W_O = blk.attn.proj.weight

        T_cls = torch.zeros(D, N, device=device)
        T_patch = torch.zeros(D, N, device=device)
        for h in range(num_heads):
            W_V_h = W_V[h * head_dim : (h + 1) * head_dim, :]
            v_h = W_V_h @ v
            W_O_h = W_O[:, h * head_dim : (h + 1) * head_dim]
            head_vec = W_O_h @ v_h
            T_cls += torch.outer(head_vec, w_cls[h, :])
            T_patch += torch.outer(head_vec, w_patch[h, :])

        T_v = torch.cat([T_cls, T_patch], dim=0)  # (2D, N)

    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    U, S, Vh = torch.linalg.svd(T_v, full_matrices=True)
    return T_v, U, S, Vh


# ==============================================================================
# 2. Direct Mode Prediction Test (Real Unconstrained Forward)
# ==============================================================================

def evaluate_singular_modes(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor,
    feature_dirs: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_s: float = 1.0,
    device: torch.device = torch.device("cuda")
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluates singular modes q_k of T_v in the REAL UNCONSTRAINED MODEL:
    - Records singular value sigma_k, predicted transmission tau = ||T_v q_k||,
      real true-class margin drop, top-1 flip rate, logit L2 distance, KL divergence.
    """
    records_spectrum = []
    records_validation = []
    B = images.shape[0]
    alpha = scale_s * sigma_P
    clean_target_logits = clean_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    clean_top1 = clean_logits.argmax(dim=-1)
    clean_probs = F.softmax(clean_logits, dim=-1)

    with torch.no_grad():
        _, clean_coll = forward_block_by_block(model, model_key, x=images, collect_depths=(depth,))
        h_clean = clean_coll[depth]

    N = h_clean.shape[1] - 1

    for f_name in ["jac_top", "jac_null"]:
        v = feature_dirs[f_name].to(device)
        T_v, U, S, Vh = construct_token_transmission_operator(
            model, model_key, depth, h_clean, v, device=device
        )

        # 1. Record full singular spectrum
        for k in range(N):
            records_spectrum.append({
                "model_key": model_key,
                "depth": depth,
                "feature_dir": f_name,
                "mode_idx": k,
                "singular_value": S[k].item() if k < len(S) else 0.0,
                "relative_singular_value": (S[k] / S[0]).item() if k < len(S) else 0.0
            })

        # 2. Select representative modes across spectrum
        # Indices: top (0), second (1), top 5% (~10), mid (N//2), bottom 5% (N - 10), bottom (N - 1)
        idx_top = 0
        idx_second = 1
        idx_top5 = max(2, int(N * 0.05))
        idx_mid = N // 2
        idx_bot5 = int(N * 0.95)
        idx_bottom = N - 1

        chosen_indices = [
            ("top_mode", idx_top),
            ("second_mode", idx_second),
            ("top_5pct_mode", idx_top5),
            ("mid_mode", idx_mid),
            ("bottom_5pct_mode", idx_bot5),
            ("bottom_mode", idx_bottom)
        ]

        for mode_label, k_idx in chosen_indices:
            q_k = Vh[k_idx, :]  # (N,)
            sig_k = S[k_idx].item() if k_idx < len(S) else 0.0
            tau_pred = torch.norm(T_v @ q_k).item()

            delta_P = alpha * torch.einsum('n,d->nd', q_k, v).unsqueeze(0).expand(B, -1, -1)

            def intervention_fn(d, h):
                if d == depth:
                    h_out = h.clone()
                    h_out[:, 1:, :] += delta_P
                    return h_out
                return h

            with torch.no_grad():
                logits_pert, _ = forward_block_by_block(
                    model, model_key, x=images, start_depth=0, intervention_fn=intervention_fn
                )

            pert_target = logits_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
            margin_drop = (clean_target_logits - pert_target).mean().item()
            flips = (logits_pert.argmax(dim=-1) != clean_top1).float().mean().item()
            l2 = (logits_pert - clean_logits).norm(dim=-1).mean().item()

            pert_probs = F.softmax(logits_pert, dim=-1)
            kl = (clean_probs * ((clean_probs + 1e-12).log() - (pert_probs + 1e-12).log())).sum(dim=-1).mean().item()

            records_validation.append({
                "model_key": model_key,
                "depth": depth,
                "feature_dir": f_name,
                "mode_label": mode_label,
                "mode_idx": k_idx,
                "singular_value": sig_k,
                "predicted_transmission": tau_pred,
                "scale_s": scale_s,
                "logit_margin_drop": margin_drop,
                "top1_flip_rate": flips,
                "logit_l2": l2,
                "kl_divergence": kl
            })

    return records_spectrum, records_validation


# ==============================================================================
# 3. Finite-Radius Sweep for Key Modes
# ==============================================================================

def evaluate_finite_radius_curves(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor,
    fdirs: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_grid: List[float] = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0],
    device: torch.device = torch.device("cuda")
) -> List[Dict[str, Any]]:
    """Evaluates nonlinear damage curves across scales for top, mid, and null modes."""
    records = []
    B = images.shape[0]
    clean_target_logits = clean_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    clean_top1 = clean_logits.argmax(dim=-1)

    with torch.no_grad():
        _, clean_coll = forward_block_by_block(model, model_key, x=images, collect_depths=(depth,))
        h_clean = clean_coll[depth]

    N = h_clean.shape[1] - 1
    v = fdirs["jac_top"].to(device)
    T_v, U, S, Vh = construct_token_transmission_operator(
        model, model_key, depth, h_clean, v, device=device
    )

    test_modes = [
        ("top_mode", 0),
        ("mid_mode", N // 2),
        ("null_mode", N - 1)
    ]

    for mode_label, k_idx in test_modes:
        q_k = Vh[k_idx, :]
        sig_k = S[k_idx].item() if k_idx < len(S) else 0.0

        for s in scale_grid:
            alpha = s * sigma_P
            delta_P = alpha * torch.einsum('n,d->nd', q_k, v).unsqueeze(0).expand(B, -1, -1)

            def intervention_fn(d, h):
                if d == depth:
                    h_out = h.clone()
                    h_out[:, 1:, :] += delta_P
                    return h_out
                return h

            with torch.no_grad():
                logits_pert, _ = forward_block_by_block(
                    model, model_key, x=images, start_depth=0, intervention_fn=intervention_fn
                )

            pert_target = logits_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
            margin_drop = (clean_target_logits - pert_target).mean().item()
            flips = (logits_pert.argmax(dim=-1) != clean_top1).float().mean().item()
            l2 = (logits_pert - clean_logits).norm(dim=-1).mean().item()

            records.append({
                "model_key": model_key,
                "depth": depth,
                "mode_label": mode_label,
                "mode_idx": k_idx,
                "singular_value": sig_k,
                "scale_s": s,
                "logit_margin_drop": margin_drop,
                "top1_flip_rate": flips,
                "logit_l2": l2
            })

    return records


# ==============================================================================
# 4. Out-of-Basis Patterns & Subspace Projection Energy
# ==============================================================================

def evaluate_out_of_basis_patterns(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor,
    fdirs: Dict[str, torch.Tensor],
    token_patterns: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_s: float = 1.0,
    device: torch.device = torch.device("cuda")
) -> List[Dict[str, Any]]:
    """
    Tests whether tau_v(a) = ||T_v a|| predicts real damage for previously evaluated
    out-of-basis patterns (global_coherent, random_sign, checkerboard, etc.).
    """
    records = []
    B = images.shape[0]
    alpha = scale_s * sigma_P
    clean_target_logits = clean_logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        _, clean_coll = forward_block_by_block(model, model_key, x=images, collect_depths=(depth,))
        h_clean = clean_coll[depth]

    num_heads = model.blocks[depth].attn.num_heads if model_key in ("deit_tiny", "deit_small", "vit_base") else model.backbone.blocks[depth].attn.num_heads
    v = fdirs["jac_top"].to(device)
    T_v, U, S, Vh = construct_token_transmission_operator(
        model, model_key, depth, h_clean, v, device=device
    )

    # Subspace projection operators:
    # Top subspace: first H singular modes (Vh[:H, :])
    # Null subspace: remaining singular modes (Vh[H:, :])
    V_top = Vh[:num_heads, :]  # (H, N)
    V_null = Vh[num_heads:, :] # (N - H, N)

    patterns_to_test = [
        "global_coherent", "random_sign", "checkerboard",
        "random_gaussian", "spatial_cluster_25%", "smooth_spatial"
    ]

    for p_name in patterns_to_test:
        if p_name not in token_patterns:
            continue
        a = token_patterns[p_name].to(device)

        # Transmission energy
        tau_pred = torch.norm(T_v @ a).item()

        # Projection energy onto top and null subspaces
        top_energy = torch.sum((V_top @ a) ** 2).item()
        null_energy = torch.sum((V_null @ a) ** 2).item()

        # Real model execution
        delta_P = alpha * torch.einsum('n,d->nd', a, v).unsqueeze(0).expand(B, -1, -1)

        def intervention_fn(d, h):
            if d == depth:
                h_out = h.clone()
                h_out[:, 1:, :] += delta_P
                return h_out
            return h

        with torch.no_grad():
            logits_pert, _ = forward_block_by_block(
                model, model_key, x=images, start_depth=0, intervention_fn=intervention_fn
            )

        pert_target = logits_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
        margin_drop = (clean_target_logits - pert_target).mean().item()
        l2 = (logits_pert - clean_logits).norm(dim=-1).mean().item()

        records.append({
            "model_key": model_key,
            "depth": depth,
            "pattern_name": p_name,
            "predicted_transmission": tau_pred,
            "top_subspace_energy": top_energy,
            "null_subspace_energy": null_energy,
            "logit_margin_drop": margin_drop,
            "logit_l2": l2
        })

    return records


# ==============================================================================
# 5. Random Held-Out Pattern Predictions
# ==============================================================================

def evaluate_random_held_out_patterns(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor,
    fdirs: Dict[str, torch.Tensor],
    sigma_P: float,
    num_patterns: int = 50,
    scale_s: float = 1.0,
    device: torch.device = torch.device("cuda")
) -> List[Dict[str, Any]]:
    """Evaluates prediction quality on M=50 random unseen Gaussian patterns a_j ~ N(0, I)."""
    records = []
    B = images.shape[0]
    alpha = scale_s * sigma_P
    clean_target_logits = clean_logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        _, clean_coll = forward_block_by_block(model, model_key, x=images, collect_depths=(depth,))
        h_clean = clean_coll[depth]

    N = h_clean.shape[1] - 1
    v = fdirs["jac_top"].to(device)
    T_v, _, _, _ = construct_token_transmission_operator(
        model, model_key, depth, h_clean, v, device=device
    )

    torch.manual_seed(12345)
    for j in range(num_patterns):
        a_raw = torch.randn(N, device=device)
        a_j = a_raw / a_raw.norm()

        tau_j = torch.norm(T_v @ a_j).item()

        delta_P = alpha * torch.einsum('n,d->nd', a_j, v).unsqueeze(0).expand(B, -1, -1)

        def intervention_fn(d, h):
            if d == depth:
                h_out = h.clone()
                h_out[:, 1:, :] += delta_P
                return h_out
            return h

        with torch.no_grad():
            logits_pert, _ = forward_block_by_block(
                model, model_key, x=images, start_depth=0, intervention_fn=intervention_fn
            )

        pert_target = logits_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
        margin_drop = (clean_target_logits - pert_target).mean().item()
        l2 = (logits_pert - clean_logits).norm(dim=-1).mean().item()

        records.append({
            "model_key": model_key,
            "depth": depth,
            "pattern_idx": j,
            "predicted_transmission": tau_j,
            "logit_margin_drop": margin_drop,
            "logit_l2": l2
        })

    return records


# ==============================================================================
# 6. Depth Spectrum Evolution & Null Dimension
# ==============================================================================

def evaluate_depth_spectrum_evolution(
    model: nn.Module,
    model_key: str,
    depths: List[int],
    images: torch.Tensor,
    fdirs_by_depth: Dict[int, Dict[str, torch.Tensor]],
    device: torch.device = torch.device("cuda")
) -> List[Dict[str, Any]]:
    """Analyzes how the spectrum and approximate null space dimension evolve across depth."""
    records = []

    with torch.no_grad():
        _, clean_coll = forward_block_by_block(
            model, model_key, x=images, collect_depths=tuple(depths)
        )

    for dep in depths:
        h_dep = clean_coll[dep]
        N = h_dep.shape[1] - 1
        v = fdirs_by_depth[dep]["jac_top"].to(device)

        T_v, _, S, _ = construct_token_transmission_operator(
            model, model_key, dep, h_dep, v, device=device
        )

        sig1 = S[0].item()
        cond_num = (S[0] / (S[-1] + 1e-12)).item()
        eff_rank = (-torch.sum((S / S.sum()) * torch.log(S / S.sum() + 1e-12))).exp().item()

        # Approximate null dimensions: # { k : sigma_k / sigma_1 <= epsilon }
        null_dim_1e2 = int((S / sig1 <= 1e-2).sum().item())
        null_dim_1e3 = int((S / sig1 <= 1e-3).sum().item())
        null_dim_1e4 = int((S / sig1 <= 1e-4).sum().item())

        records.append({
            "model_key": model_key,
            "depth": dep,
            "top_singular_value": sig1,
            "effective_rank": eff_rank,
            "condition_number": cond_num,
            "null_dim_eps_1e2": null_dim_1e2,
            "null_dim_eps_1e3": null_dim_1e3,
            "null_dim_eps_1e4": null_dim_1e4,
            "total_tokens": N
        })

    return records


# ==============================================================================
# 7. Bilinear Feature x Token Matrix
# ==============================================================================

def evaluate_feature_token_matrix(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor,
    fdirs: Dict[str, torch.Tensor],
    token_patterns: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_s: float = 1.0,
    device: torch.device = torch.device("cuda")
) -> List[Dict[str, Any]]:
    """Compares predicted transmission matrix tau(v_j, a_k) against real damage matrix."""
    records = []
    B = images.shape[0]
    alpha = scale_s * sigma_P
    clean_target_logits = clean_logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        _, clean_coll = forward_block_by_block(model, model_key, x=images, collect_depths=(depth,))
        h_clean = clean_coll[depth]

    f_list = ["jac_top", "pc1", "rand_dir", "jac_null"]
    t_list = ["global_coherent", "random_sign", "checkerboard", "spatial_cluster_25%"]

    for f_name in f_list:
        v = fdirs[f_name].to(device)
        T_v, _, _, _ = construct_token_transmission_operator(
            model, model_key, depth, h_clean, v, device=device
        )

        for t_name in t_list:
            a = token_patterns[t_name].to(device)
            tau = torch.norm(T_v @ a).item()

            delta_P = alpha * torch.einsum('n,d->nd', a, v).unsqueeze(0).expand(B, -1, -1)

            def intervention_fn(d, h):
                if d == depth:
                    h_out = h.clone()
                    h_out[:, 1:, :] += delta_P
                    return h_out
                return h

            with torch.no_grad():
                logits_pert, _ = forward_block_by_block(
                    model, model_key, x=images, start_depth=0, intervention_fn=intervention_fn
                )

            pert_target = logits_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
            margin_drop = (clean_target_logits - pert_target).mean().item()
            l2 = (logits_pert - clean_logits).norm(dim=-1).mean().item()

            records.append({
                "model_key": model_key,
                "depth": depth,
                "feature_dir": f_name,
                "token_pattern": t_name,
                "predicted_transmission": tau,
                "real_margin_drop": margin_drop,
                "real_logit_l2": l2
            })

    return records


# ==============================================================================
# 8. Spatial Mode Structure Analysis
# ==============================================================================

def analyze_spatial_mode_structure(
    Vh: torch.Tensor,
    N: int,
    model_key: str,
    depth: int
) -> List[Dict[str, Any]]:
    """Analyzes 2D spatial frequency content and total variation of singular modes q_k."""
    records = []
    grid_side = int(math.isqrt(N))
    if grid_side * grid_side != N:
        return records

    for k in range(N):
        q_k = Vh[k, :].view(grid_side, grid_side)
        fft_q = torch.fft.fft2(q_k)
        power = (fft_q.abs() ** 2) / (grid_side * grid_side)

        low_freq_power = power[:3, :3].sum().item()
        total_power = power.sum().item() + 1e-10
        lfpr = low_freq_power / total_power

        tv = ((q_k[1:, :] - q_k[:-1, :]).abs().mean() + (q_k[:, 1:] - q_k[:, :-1]).abs().mean()).item()

        records.append({
            "model_key": model_key,
            "depth": depth,
            "mode_idx": k,
            "low_frequency_power_ratio": lfpr,
            "total_variation": tv
        })

    return records


# ==============================================================================
# 9. Publication Figures Generation
# ==============================================================================

def generate_predictive_operator_figures(
    df_spectrum: pd.DataFrame,
    df_val: pd.DataFrame,
    df_rand: pd.DataFrame,
    df_curves: pd.DataFrame,
    df_depth: pd.DataFrame,
    df_rep: pd.DataFrame,
    df_spatial: Optional[pd.DataFrame],
    output_dir: str
):
    """Generates publication figures A through G."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ----------------------------------------------------
    # FIGURE A: Operator Spectrum (Scree Plot)
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    df_spec_top = df_spectrum[df_spectrum["feature_dir"] == "jac_top"].copy()
    if not df_spec_top.empty:
        # Plot singular values on log scale
        sns.lineplot(
            data=df_spec_top[df_spec_top["mode_idx"] < 30],
            x="mode_idx",
            y="singular_value",
            hue="model_key",
            marker="o",
            linewidth=2.5
        )
        plt.yscale("log")
        plt.title(r"Figure A: Singular Spectrum of Token-Transmission Operator $T_v$ (Log Scale)")
        plt.xlabel("Singular Mode Index k")
        plt.ylabel(r"Singular Value $\sigma_k$")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_a_operator_spectrum.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE B: Predicted vs Observed Damage (Random Patterns)
    # ----------------------------------------------------
    plt.figure(figsize=(7, 6))
    if not df_rand.empty:
        sns.scatterplot(
            data=df_rand,
            x="predicted_transmission",
            y="logit_l2",
            hue="model_key",
            s=70,
            alpha=0.85
        )
        # Compute Pearson r
        r_val = np.corrcoef(df_rand["predicted_transmission"], df_rand["logit_l2"])[0, 1]
        plt.title(f"Figure B: Predicted Transmission ||T_v a|| vs Real Logit L2 (r = {r_val:.3f})")
        plt.xlabel(r"Predicted Transmission $\tau(a) = \|T_v a\|_2$")
        plt.ylabel("Observed Real Logit L2 Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_b_predicted_vs_observed_damage.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE C: Top vs Null Modes Bar Contrast
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    if not df_val.empty:
        df_v_top = df_val[df_val["feature_dir"] == "jac_top"]
        sns.barplot(
            data=df_v_top,
            x="mode_label",
            y="logit_l2",
            hue="model_key",
            palette="magma"
        )
        plt.title("Figure C: Finite Perturbation Damage Across Singular Modes (Matched Norm)")
        plt.xlabel("Singular Mode of T_v")
        plt.ylabel("Real Logit L2 Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_c_top_vs_null_modes.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE D: Finite-Radius Validation Curves
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    if not df_curves.empty:
        sns.lineplot(
            data=df_curves,
            x="scale_s",
            y="logit_l2",
            hue="mode_label",
            marker="o",
            linewidth=2.5,
            palette="Set1"
        )
        plt.title("Figure D: Finite-Radius Response Curves for Top vs Mid vs Null Modes")
        plt.xlabel(r"Perturbation Scale $s = \alpha / \sigma_P$")
        plt.ylabel("Real Logit L2 Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_d_finite_radius_validation.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE E: Depth Evolution of Spectrum & Null Dimension
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    if not df_depth.empty:
        sns.barplot(
            data=df_depth,
            x="depth",
            y="null_dim_eps_1e3",
            hue="model_key",
            palette="Blues_d"
        )
        plt.title(r"Figure E: Approximate Null Space Dimension ($\sigma_k / \sigma_1 \leq 10^{-3}$) Across Depth")
        plt.xlabel("Layer Depth")
        plt.ylabel("Dimension of Approximate Null Space")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_e_depth_evolution.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE F: Cross-Architecture Replication Summary
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    if not df_rep.empty:
        sns.barplot(
            data=df_rep,
            x="mode_label",
            y="logit_l2",
            hue="model_key",
            palette="viridis"
        )
        plt.title("Figure F: Confirmatory Replication Across Architectures")
        plt.xlabel("Singular Mode")
        plt.ylabel("Real Logit L2 Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_f_cross_architecture_replication.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE G: Spatial Structure of Modes (Optional)
    # ----------------------------------------------------
    if df_spatial is not None and not df_spatial.empty:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=df_spatial[df_spatial["mode_idx"] < 30],
            x="mode_idx",
            y="low_frequency_power_ratio",
            color="purple",
            s=80
        )
        plt.title("Figure G: Spatial Low-Frequency Power Fraction Across Singular Modes")
        plt.xlabel("Mode Index k")
        plt.ylabel("Low-Frequency Power Ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_g_token_mode_spatial_structure.png"), dpi=300)
        plt.close()

    print(f"Generated all figures in {output_dir}")
