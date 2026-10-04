"""
patch_fungibility/predictive_principle.py

Full implementation of the Fungibility Predictive Principle Pipeline:
1. Assembles ground-truth continuous fungibility scores (F_l) and secondary targets from validated outputs.
2. Computes clean-state representation geometry, token mixing, downstream sensitivity, and sensitivity-weighted geometry.
3. Performs baseline control against normalized depth (l / L).
4. Conducts statistical analysis (Pearson, Spearman, bootstrap CI, incremental R^2).
5. Executes Leave-One-Architecture-Out (LOAO) cross-validation across the four architectures.
6. Evaluates transition layer prediction (|l_pred - l_true|) for the pre-registered threshold F_l >= 0.80.
7. Produces all machine-readable output CSVs, validation manifest, and figures.
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
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from patch_fungibility.dense_fraction_models import load_model_and_transform, forward_block_by_block
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits


# ==============================================================================
# 1. GROUND TRUTH TARGET ASSEMBLY
# ==============================================================================

def assemble_ground_truth_targets() -> pd.DataFrame:
    """
    Assembles the exact ground-truth dependent variable F_l = 1 - D_centroid / D_zero
    and secondary targets across all 4 architectures and validated depths.
    """
    records = []

    # 1. DeiT-Tiny & DeiT-Small from outputs/fungibility_v0_6/summary_all_models.csv (25% budget)
    v06_path = os.path.join("outputs", "fungibility_v0_6", "summary_all_models.csv")
    if not os.path.exists(v06_path):
        raise FileNotFoundError(f"Missing {v06_path}")
    df_v06 = pd.read_csv(v06_path)
    df_v06_25 = df_v06[df_v06["fraction"] == "25%"].copy()

    for _, r in df_v06_25.iterrows():
        m_id = str(r["model"])
        depth = int(r["depth"])
        z_dmg = float(r["zero_damage"])
        c_dmg = float(r["global_mean_damage"])  # static centroid
        g_dmg = float(r["gaussian_damage"])
        clean_acc = float(r["clean_acc"])
        c_acc = float(r["global_mean_acc"])
        g_acc = float(r["gaussian_acc"])

        f_l = 1.0 - (c_dmg / z_dmg) if z_dmg != 0 else np.nan
        rec_gauss = (z_dmg - g_dmg) / z_dmg if z_dmg != 0 else np.nan
        c_ret = c_acc / clean_acc if clean_acc > 0 else np.nan
        g_ret = g_acc / clean_acc if clean_acc > 0 else np.nan

        model_key = "deit_tiny" if "tiny" in m_id else "deit_small"

        records.append({
            "model_key": model_key,
            "model_name": m_id,
            "depth": depth,
            "norm_depth": depth / 12.0,
            "clean_acc": clean_acc,
            "zero_damage": z_dmg,
            "centroid_damage": c_dmg,
            "gaussian_damage": g_dmg,
            "F_l": f_l,
            "F_l_clip": float(np.clip(f_l, 0.0, 1.0)) if not np.isnan(f_l) else np.nan,
            "recovery_gaussian": rec_gauss,
            "centroid_acc": c_acc,
            "gaussian_acc": g_acc,
            "centroid_top1_retention": c_ret,
            "gaussian_top1_retention": g_ret,
            "is_fungible_binary": int(f_l >= 0.80) if not np.isnan(f_l) else 0
        })

    # 2. ViT-B from outputs/fungibility_v1/vitb_depth_results.csv (25% budget)
    vitb_path = os.path.join("outputs", "fungibility_v1", "vitb_depth_results.csv")
    df_vitb = pd.read_csv(vitb_path)
    clean_vitb = df_vitb[df_vitb["depth"] == 0].iloc[0]
    clean_acc_vitb = float(clean_vitb["top1_accuracy"])

    for d in [5, 7, 8, 9, 10]:
        sub = df_vitb[df_vitb["depth"] == d]
        z = sub[sub["condition"] == "ZERO"].iloc[0]
        c = sub[sub["condition"] == "CENTROID"].iloc[0]
        g = sub[sub["condition"] == "DIAGONAL_GAUSSIAN"]

        z_dmg = float(z["margin_damage"])
        c_dmg = float(c["margin_damage"])
        g_dmg = float(g["margin_damage"].mean())
        c_acc = float(c["top1_accuracy"])
        g_acc = float(g["top1_accuracy"].mean())

        f_l = 1.0 - (c_dmg / z_dmg) if z_dmg != 0 else np.nan
        rec_gauss = (z_dmg - g_dmg) / z_dmg if z_dmg != 0 else np.nan

        records.append({
            "model_key": "vit_base",
            "model_name": "vit_base_patch16_224.augreg_in1k",
            "depth": d,
            "norm_depth": d / 12.0,
            "clean_acc": clean_acc_vitb,
            "zero_damage": z_dmg,
            "centroid_damage": c_dmg,
            "gaussian_damage": g_dmg,
            "F_l": f_l,
            "F_l_clip": float(np.clip(f_l, 0.0, 1.0)) if not np.isnan(f_l) else np.nan,
            "recovery_gaussian": rec_gauss,
            "centroid_acc": c_acc,
            "gaussian_acc": g_acc,
            "centroid_top1_retention": c_acc / clean_acc_vitb,
            "gaussian_top1_retention": g_acc / clean_acc_vitb,
            "is_fungible_binary": int(f_l >= 0.80) if not np.isnan(f_l) else 0
        })

    # 3. DINOv2 from outputs/fungibility_v1/dinov2_depth_results.csv (25% budget)
    dino_path = os.path.join("outputs", "fungibility_v1", "dinov2_depth_results.csv")
    df_dino = pd.read_csv(dino_path)
    clean_dino = df_dino[df_dino["depth"] == 0].iloc[0]
    clean_acc_dino = float(clean_dino["top1_accuracy"])

    for d in [5, 7, 8, 9, 10]:
        sub = df_dino[df_dino["depth"] == d]
        z = sub[sub["condition"] == "ZERO"].iloc[0]
        c = sub[sub["condition"] == "CENTROID"].iloc[0]
        g = sub[sub["condition"] == "DIAGONAL_GAUSSIAN"]

        z_dmg = float(z["margin_damage"])
        c_dmg = float(c["margin_damage"])
        g_dmg = float(g["margin_damage"].mean())
        c_acc = float(c["top1_accuracy"])
        g_acc = float(g["top1_accuracy"].mean())

        f_l = 1.0 - (c_dmg / z_dmg) if z_dmg != 0 else np.nan
        rec_gauss = (z_dmg - g_dmg) / z_dmg if z_dmg != 0 else np.nan

        records.append({
            "model_key": "dinov2",
            "model_name": "dinov2_vits14_lc",
            "depth": d,
            "norm_depth": d / 12.0,
            "clean_acc": clean_acc_dino,
            "zero_damage": z_dmg,
            "centroid_damage": c_dmg,
            "gaussian_damage": g_dmg,
            "F_l": f_l,
            "F_l_clip": float(np.clip(f_l, 0.0, 1.0)) if not np.isnan(f_l) else np.nan,
            "recovery_gaussian": rec_gauss,
            "centroid_acc": c_acc,
            "gaussian_acc": g_acc,
            "centroid_top1_retention": c_acc / clean_acc_dino,
            "gaussian_top1_retention": g_acc / clean_acc_dino,
            "is_fungible_binary": int(f_l >= 0.80) if not np.isnan(f_l) else 0
        })

    targets_df = pd.DataFrame(records)
    return targets_df


# ==============================================================================
# 2. CLEAN-STATE PREDICTOR COMPUTATION PIPELINE
# ==============================================================================

def compute_clean_layer_metrics(
    model_key: str,
    device: torch.device,
    batch_size: int = 25,
    max_calib_samples: int = 1000
) -> pd.DataFrame:
    """
    Computes all candidate clean-state predictors across blocks 1..12 for a frozen model
    using ONLY the calibration split (N_calib=1,000, seed 9101).
    """
    print(f"\n=======================================================")
    print(f"Computing Clean-State Predictors for Model: {model_key}")
    print(f"=======================================================")

    model, transform, meta = load_model_and_transform(model_key, device)
    model.eval()

    # Verify frozen weights
    for p in model.parameters():
        assert not p.requires_grad, f"Model parameter is not frozen in {model_key}"

    calib_ds, _, _, _ = get_disjoint_imagenet_splits()
    calib_ds.transform = transform
    if len(calib_ds) > max_calib_samples:
        indices = list(range(max_calib_samples))
        calib_ds = torch.utils.data.Subset(calib_ds, indices)

    loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_blocks = meta["depth"]
    num_patches = meta["num_patches"]
    embed_dim = meta["embed_dim"]
    all_depths = tuple(range(1, num_blocks + 1))

    # ---------------------------------------------------------
    # PASS 1: Clean Activations, Covariance & Geometry / Mixing
    # ---------------------------------------------------------
    print("Pass 1: Extracting clean activations and accumulating representation geometry...")
    
    # Online covariance statistics per layer: sum_p, sum_outer, total_tokens
    sum_tokens = {d: torch.zeros(embed_dim, device=device, dtype=torch.float64) for d in all_depths}
    sum_outer = {d: torch.zeros((embed_dim, embed_dim), device=device, dtype=torch.float64) for d in all_depths}
    total_token_count = {d: 0 for d in all_depths}

    # Token mixing statistics accumulators
    mixing_stats = {
        d: {
            "pairwise_cos": 0.0,
            "dist_to_mean": 0.0,
            "patch_cls_cos": 0.0,
            "image_count": 0
        }
        for d in all_depths
    }

    t0 = time.time()
    with torch.no_grad():
        for b_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            B = images.size(0)

            _, collected = forward_block_by_block(
                model, model_key, x=images, collect_depths=all_depths
            )

            for d in all_depths:
                h = collected[d]  # (B, 1 + N, D)
                cls_tok = h[:, 0, :]  # (B, D)
                patches = h[:, 1:, :]  # (B, N, D)

                # Flatten spatial patches for covariance
                patches_flat = patches.reshape(-1, embed_dim).to(torch.float64)
                sum_tokens[d] += patches_flat.sum(dim=0)
                sum_outer[d] += patches_flat.t() @ patches_flat
                total_token_count[d] += patches_flat.size(0)

                # Token mixing / homogeneity
                # 1. Pairwise cosine similarity among patches
                p_norm = patches / (patches.norm(dim=-1, keepdim=True) + 1e-10)
                sum_p_norm = p_norm.sum(dim=1)  # (B, D)
                norm_sq = (sum_p_norm ** 2).sum(dim=-1)  # (B,)
                # sum_{j<k} cos(j,k) = (||sum||^2 - N) / 2
                pairwise_cos = (norm_sq - num_patches) / (num_patches * (num_patches - 1))
                mixing_stats[d]["pairwise_cos"] += pairwise_cos.sum().item()

                # 2. Distance to image-wise patch mean
                p_mean = patches.mean(dim=1, keepdim=True)  # (B, 1, D)
                dist_mean = (patches - p_mean).norm(dim=-1).mean(dim=1)  # (B,)
                mixing_stats[d]["dist_to_mean"] += dist_mean.sum().item()

                # 3. Patch-to-CLS cosine similarity
                cls_norm = cls_tok / (cls_tok.norm(dim=-1, keepdim=True) + 1e-10)  # (B, D)
                p_cls_cos = (p_norm * cls_norm.unsqueeze(1)).sum(dim=-1).mean(dim=1)  # (B,)
                mixing_stats[d]["patch_cls_cos"] += p_cls_cos.sum().item()

                mixing_stats[d]["image_count"] += B

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(loader):
                print(f"  Processed {min((b_idx + 1) * batch_size, len(calib_ds))}/{len(calib_ds)} images ({time.time() - t0:.1f}s)")

    # Compute covariance matrices, centroids, and eigenspectra
    cov_matrices = {}
    centroids = {}
    spectral_metrics = {}

    for d in all_depths:
        M = total_token_count[d]
        mu = sum_tokens[d] / M
        centroids[d] = mu.float()  # (D,) on device
        # Sample covariance: (sum_outer - M * mu * mu^T) / (M - 1)
        cov = (sum_outer[d] - M * torch.outer(mu, mu)) / (M - 1)
        cov_matrices[d] = cov.float()  # (D, D) on device

        # Eigenspectrum
        evals, _ = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=0.0)
        # Sort descending
        evals = torch.flip(evals, dims=[0])
        total_var = evals.sum().item()

        # Entropy effective rank
        if total_var > 0:
            q = evals / total_var
            # filter zeros for log
            q_nonzero = q[q > 1e-12]
            ent = -(q_nonzero * torch.log(q_nonzero)).sum().item()
            r_eff = float(np.exp(ent))
            c1 = float(evals[0].item() / total_var)
            c4 = float(evals[:4].sum().item() / total_var)
            c16 = float(evals[:min(16, embed_dim)].sum().item() / total_var)
        else:
            r_eff, c1, c4, c16 = 1.0, 1.0, 1.0, 1.0

        spectral_metrics[d] = {
            "cov_trace": float(total_var),
            "var_token": float(total_var / embed_dim),
            "effective_rank": r_eff,
            "c1_concentration": c1,
            "c4_concentration": c4,
            "c16_concentration": c16
        }

    # Finalize mixing metrics and compute dist to centroid
    token_mixing_metrics = {}
    for d in all_depths:
        cnt = mixing_stats[d]["image_count"]
        token_mixing_metrics[d] = {
            "patch_pairwise_cos_sim": float(mixing_stats[d]["pairwise_cos"] / cnt),
            "dist_to_patch_mean": float(mixing_stats[d]["dist_to_mean"] / cnt),
            "patch_to_cls_cos_sim": float(mixing_stats[d]["patch_cls_cos"] / cnt),
            # Theoretical mean distance to calibration centroid = sqrt(tr(Sigma))
            "dist_to_centroid": float(np.sqrt(spectral_metrics[d]["cov_trace"]))
        }

    # ---------------------------------------------------------
    # PASS 2: Downstream Sensitivity & Sensitivity-Weighted Geometry
    # ---------------------------------------------------------
    print("Pass 2: Computing downstream sensitivity and sensitivity-weighted geometry...")
    
    sens_stats = {
        d: {
            "margin_grad_fro_sq": 0.0,
            "mean_token_grad_norm": 0.0,
            "s_margin_unnorm": 0.0,
            "logit_jac_fro_sq": 0.0,
            "s_logit_unnorm": 0.0,
            "sample_count": 0
        }
        for d in all_depths
    }

    t1 = time.time()
    for b_idx, (images, _) in enumerate(loader):
        images = images.to(device)
        B = images.size(0)

        # First collect activations after all blocks
        with torch.no_grad():
            clean_logits, collected = forward_block_by_block(
                model, model_key, x=images, collect_depths=all_depths
            )
            # Determine top-1 and runner-up classes
            top_class = clean_logits.argmax(dim=-1)
            logits_temp = clean_logits.clone()
            logits_temp[torch.arange(B), top_class] = -1e9
            runner_up = logits_temp.argmax(dim=-1)

        # For each depth, compute gradient through downstream network
        for d in all_depths:
            h_d = collected[d]  # (B, 1 + N, D)
            h_leaf = h_d.clone().detach().requires_grad_(True)

            # Downstream forward
            logits_down, _ = forward_block_by_block(
                model, model_key, start_depth=d, h_start=h_leaf
            )

            # 1. Margin sensitivity
            margins = logits_down[torch.arange(B), top_class] - logits_down[torch.arange(B), runner_up]
            sum_m = margins.sum()

            grad_h = torch.autograd.grad(sum_m, h_leaf, retain_graph=True)[0]
            grad_p = grad_h[:, 1:, :]  # (B, N, D)

            # Frobenius norm squared of margin gradient per image
            grad_fro_sq = (grad_p ** 2).sum(dim=(1, 2))  # (B,)
            token_grad_norm = grad_p.norm(dim=-1).mean(dim=1)  # (B,)

            # Sensitivity-weighted geometry: S_margin = (1/N) * sum_k g_k^T Sigma g_k
            # (B, N, D) @ (D, D) -> (B, N, D)
            cov_d = cov_matrices[d]
            q_margin = torch.matmul(grad_p, cov_d)  # (B, N, D)
            s_margin_per_img = (q_margin * grad_p).sum(dim=(1, 2)) / num_patches  # (B,)

            # 2. Hutchinson Logit Jacobian Sensitivity (Rademacher projection, K=1 vector)
            v = torch.randint(0, 2, logits_down.shape, device=device).float() * 2.0 - 1.0
            sum_v = (logits_down * v).sum()
            grad_v = torch.autograd.grad(sum_v, h_leaf)[0][:, 1:, :]  # (B, N, D)

            logit_fro_sq = (grad_v ** 2).sum(dim=(1, 2))  # (B,)
            q_logit = torch.matmul(grad_v, cov_d)
            s_logit_per_img = (q_logit * grad_v).sum(dim=(1, 2)) / num_patches  # (B,)

            sens_stats[d]["margin_grad_fro_sq"] += grad_fro_sq.sum().item()
            sens_stats[d]["mean_token_grad_norm"] += token_grad_norm.sum().item()
            sens_stats[d]["s_margin_unnorm"] += s_margin_per_img.sum().item()
            sens_stats[d]["logit_jac_fro_sq"] += logit_fro_sq.sum().item()
            sens_stats[d]["s_logit_unnorm"] += s_logit_per_img.sum().item()
            sens_stats[d]["sample_count"] += B

        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(loader):
            print(f"  Sensitivity evaluated {min((b_idx + 1) * batch_size, len(calib_ds))}/{len(calib_ds)} images ({time.time() - t1:.1f}s)")

    # Assemble all layer metrics
    rows = []
    for d in all_depths:
        cnt = sens_stats[d]["sample_count"]
        mean_fro_sq = sens_stats[d]["margin_grad_fro_sq"] / cnt
        mean_fro = np.sqrt(max(mean_fro_sq, 0.0))
        mean_tok_grad = sens_stats[d]["mean_token_grad_norm"] / cnt
        s_margin = sens_stats[d]["s_margin_unnorm"] / cnt

        logit_fro_sq = sens_stats[d]["logit_jac_fro_sq"] / cnt
        logit_fro = np.sqrt(max(logit_fro_sq, 0.0))
        s_logit = sens_stats[d]["s_logit_unnorm"] / cnt

        tr_sig = spectral_metrics[d]["cov_trace"]

        # Normalized variants
        s_margin_trace_norm = s_margin / (tr_sig + 1e-10)
        s_margin_grad_norm = s_margin / (mean_fro_sq + 1e-10)
        s_logit_trace_norm = s_logit / (tr_sig + 1e-10)
        sens_per_unit_var = mean_fro / (np.sqrt(tr_sig) + 1e-10)

        # Inverse and log-scale variants for positive correlation with fungibility
        inv_s_margin = 1.0 / (s_margin + 1e-10)
        neg_log_s_margin = -float(np.log(max(s_margin, 1e-10)))

        row = {
            "model_key": model_key,
            "depth": d,
            "norm_depth": d / float(num_blocks),
            # Family A
            "cov_trace": tr_sig,
            "var_token": spectral_metrics[d]["var_token"],
            "effective_rank": spectral_metrics[d]["effective_rank"],
            "c1_concentration": spectral_metrics[d]["c1_concentration"],
            "c4_concentration": spectral_metrics[d]["c4_concentration"],
            "c16_concentration": spectral_metrics[d]["c16_concentration"],
            # Family B
            "patch_pairwise_cos_sim": token_mixing_metrics[d]["patch_pairwise_cos_sim"],
            "dist_to_patch_mean": token_mixing_metrics[d]["dist_to_patch_mean"],
            "dist_to_centroid": token_mixing_metrics[d]["dist_to_centroid"],
            "patch_to_cls_cos_sim": token_mixing_metrics[d]["patch_to_cls_cos_sim"],
            # Family C
            "margin_grad_norm": float(mean_fro),
            "mean_token_grad_norm": float(mean_tok_grad),
            "logit_jac_norm": float(logit_fro),
            "sens_per_unit_var": float(sens_per_unit_var),
            # Family D
            "s_margin": float(s_margin),
            "s_margin_trace_norm": float(s_margin_trace_norm),
            "s_margin_grad_norm": float(s_margin_grad_norm),
            "s_logit": float(s_logit),
            "s_logit_trace_norm": float(s_logit_trace_norm),
            "inv_s_margin": float(inv_s_margin),
            "neg_log_s_margin": float(neg_log_s_margin)
        }
        rows.append(row)

    model_df = pd.DataFrame(rows)
    return model_df


# ==============================================================================
# 3. STATISTICAL REGRESSION & LEAVE-ONE-ARCHITECTURE-OUT (LOAO)
# ==============================================================================

def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float]:
    """Computes two-sided bootstrap confidence interval on Spearman rho."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rx = pd.Series(x[idx]).rank().values
        ry = pd.Series(y[idx]).rank().values
        if np.std(rx) > 0 and np.std(ry) > 0:
            r = np.corrcoef(rx, ry)[0, 1]
            rhos.append(r)
    if not rhos:
        return np.nan, np.nan
    low = float(np.percentile(rhos, 100 * (alpha / 2)))
    high = float(np.percentile(rhos, 100 * (1 - alpha / 2)))
    return low, high


def run_statistical_analysis(
    layer_df: pd.DataFrame,
    targets_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluates correlation, linear regression, incremental R^2 beyond normalized depth,
    Leave-One-Architecture-Out (LOAO) cross-validation, and transition prediction.
    """
    # Merge predictors with ground truth targets
    # Only keep layers present in targets_df
    merged = pd.merge(
        targets_df,
        layer_df,
        on=["model_key", "depth", "norm_depth"],
        how="inner"
    )

    print(f"\nMerged {len(merged)} validated model-depth target points across {merged['model_key'].nunique()} architectures.")
    assert len(merged) == 22, f"Expected 22 model-depth points, got {len(merged)}"

    # Exclude non-predictor columns
    meta_cols = {
        "model_key", "model_name", "depth", "clean_acc", "zero_damage",
        "centroid_damage", "gaussian_damage", "F_l", "F_l_clip", "recovery_gaussian",
        "centroid_acc", "gaussian_acc", "centroid_top1_retention",
        "gaussian_top1_retention", "is_fungible_binary"
    }
    predictor_cols = [c for c in merged.columns if c not in meta_cols]

    # Target dependent variable for continuous analysis
    y_raw = merged["F_l"].values
    y_clip = merged["F_l_clip"].values
    y_target = y_clip  # Bounded F_l avoids singularity from depth 10 zero damage

    depth_vec = merged["norm_depth"].values

    # Baseline Model: F_l ~ norm_depth
    A_depth = np.column_stack([np.ones_like(depth_vec), depth_vec])
    beta_depth, _, _, _ = np.linalg.lstsq(A_depth, y_target, rcond=None)
    y_pred_depth = A_depth @ beta_depth
    ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
    ss_res_depth = np.sum((y_target - y_pred_depth) ** 2)
    r2_depth_baseline = float(1.0 - ss_res_depth / ss_tot)
    print(f"Baseline Control: F_l ~ normalized_depth yields R^2 = {r2_depth_baseline:.4f}")

    # ---------------------------------------------------------
    # 1. Predictor vs Fungibility Table
    # ---------------------------------------------------------
    pred_records = []
    models = sorted(merged["model_key"].unique())

    for pred in predictor_cols:
        x_vec = merged[pred].values

        # Pooled Pearson & Spearman
        if np.std(x_vec) > 0 and np.std(y_target) > 0:
            pearson_r = float(np.corrcoef(x_vec, y_target)[0, 1])
            spearman_rho = float(pd.Series(x_vec).corr(pd.Series(y_target), method="spearman"))
            ci_low, ci_high = bootstrap_spearman_ci(x_vec, y_target)
        else:
            pearson_r, spearman_rho, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan

        # Per-architecture Spearman rho
        arch_rhos = {}
        for m in models:
            sub = merged[merged["model_key"] == m]
            if len(sub) >= 4 and np.std(sub[pred]) > 0:
                arch_rhos[m] = float(pd.Series(sub[pred].values).corr(pd.Series(sub["F_l_clip"].values), method="spearman"))
            else:
                arch_rhos[m] = np.nan

        # Univariate Regression: F_l ~ X
        A_x = np.column_stack([np.ones_like(x_vec), x_vec])
        beta_x, _, _, _ = np.linalg.lstsq(A_x, y_target, rcond=None)
        y_pred_x = A_x @ beta_x
        ss_res_x = np.sum((y_target - y_pred_x) ** 2)
        r2_x = float(max(0.0, 1.0 - ss_res_x / ss_tot))

        # Joint Regression: F_l ~ depth + X
        A_joint = np.column_stack([np.ones_like(x_vec), depth_vec, x_vec])
        beta_joint, _, _, _ = np.linalg.lstsq(A_joint, y_target, rcond=None)
        y_pred_joint = A_joint @ beta_joint
        ss_res_joint = np.sum((y_target - y_pred_joint) ** 2)
        r2_joint = float(max(0.0, 1.0 - ss_res_joint / ss_tot))

        delta_r2 = float(max(0.0, r2_joint - r2_depth_baseline))

        # Partial correlation r(Y, X | depth)
        # Residual of Y on depth
        res_y_depth = y_target - y_pred_depth
        # Residual of X on depth
        beta_xd, _, _, _ = np.linalg.lstsq(A_depth, x_vec, rcond=None)
        res_x_depth = x_vec - (A_depth @ beta_xd)
        if np.std(res_x_depth) > 0 and np.std(res_y_depth) > 0:
            partial_r = float(np.corrcoef(res_x_depth, res_y_depth)[0, 1])
        else:
            partial_r = 0.0

        rec = {
            "predictor": pred,
            "pooled_pearson_r": pearson_r,
            "pooled_spearman_rho": spearman_rho,
            "rho_ci_95_lower": ci_low,
            "rho_ci_95_upper": ci_high,
            "r2_univariate": r2_x,
            "r2_depth_baseline": r2_depth_baseline,
            "r2_joint_with_depth": r2_joint,
            "incremental_delta_r2": delta_r2,
            "partial_corr_given_depth": partial_r,
            "spearman_deit_tiny": arch_rhos.get("deit_tiny", np.nan),
            "spearman_deit_small": arch_rhos.get("deit_small", np.nan),
            "spearman_vit_base": arch_rhos.get("vit_base", np.nan),
            "spearman_dinov2": arch_rhos.get("dinov2", np.nan),
        }
        pred_records.append(rec)

    pred_summary_df = pd.DataFrame(pred_records)
    # Sort by absolute Spearman rho descending
    pred_summary_df["abs_rho"] = pred_summary_df["pooled_spearman_rho"].abs()
    pred_summary_df = pred_summary_df.sort_values(by="abs_rho", ascending=False).drop(columns=["abs_rho"])

    # ---------------------------------------------------------
    # 2. Leave-One-Architecture-Out (LOAO) Cross-Validation
    # ---------------------------------------------------------
    loao_records = []
    
    # We evaluate all predictors across 4 folds
    for pred in predictor_cols:
        for heldout_m in models:
            train_mask = (merged["model_key"] != heldout_m)
            test_mask = (merged["model_key"] == heldout_m)

            train_df = merged[train_mask]
            test_df = merged[test_mask]

            x_train = train_df[pred].values
            y_train = train_df["F_l_clip"].values
            x_test = test_df[pred].values
            y_test = test_df["F_l_clip"].values

            # Fit OLS on train
            A_tr = np.column_stack([np.ones_like(x_train), x_train])
            beta, _, _, _ = np.linalg.lstsq(A_tr, y_train, rcond=None)

            # Predict on heldout
            A_te = np.column_stack([np.ones_like(x_test), x_test])
            y_hat = A_te @ beta

            mae = float(np.mean(np.abs(y_test - y_hat)))
            ss_tot_te = np.sum((y_test - np.mean(y_test)) ** 2)
            ss_res_te = np.sum((y_test - y_hat) ** 2)
            r2_loao = float(1.0 - (ss_res_te / ss_tot_te)) if ss_tot_te > 0 else np.nan

            if np.std(x_test) > 0 and np.std(y_test) > 0:
                spearman_test = float(pd.Series(x_test).corr(pd.Series(y_test), method="spearman"))
            else:
                spearman_test = np.nan

            loao_records.append({
                "predictor": pred,
                "heldout_model": heldout_m,
                "n_train_points": len(train_df),
                "n_test_points": len(test_df),
                "slope_beta": float(beta[1]),
                "intercept_alpha": float(beta[0]),
                "test_mae": mae,
                "test_r2_loao": r2_loao,
                "test_spearman_rho": spearman_test
            })

    loao_df = pd.DataFrame(loao_records)

    # ---------------------------------------------------------
    # 3. Transition Layer Prediction (|l_pred - l_true|)
    # ---------------------------------------------------------
    # Target definition: First layer index where F_l >= 0.80
    transition_targets = {}
    for m in models:
        sub = merged[merged["model_key"] == m].sort_values(by="depth")
        fungible_sub = sub[sub["F_l_clip"] >= 0.80]
        if len(fungible_sub) > 0:
            transition_targets[m] = int(fungible_sub.iloc[0]["depth"])
        else:
            transition_targets[m] = int(sub.iloc[-1]["depth"])
    print(f"True Transition Layers (F_l >= 0.80): {transition_targets}")

    trans_records = []
    # Test transition prediction for all predictors
    for pred in predictor_cols:
        model_errors = []
        for heldout_m in models:
            train_mask = (merged["model_key"] != heldout_m)
            test_mask = (merged["model_key"] == heldout_m)
            train_df = merged[train_mask]
            test_df = merged[test_mask].sort_values(by="depth")

            # Fit linear model on training set: F_l = a + b * X
            x_train = train_df[pred].values
            y_train = train_df["F_l_clip"].values
            A_tr = np.column_stack([np.ones_like(x_train), x_train])
            beta, _, _, _ = np.linalg.lstsq(A_tr, y_train, rcond=None)
            a, b = beta[0], beta[1]

            # Solve for predictor threshold where predicted F_l == 0.80:
            # 0.80 = a + b * T => T = (0.80 - a) / b
            if abs(b) > 1e-9:
                threshold = (0.80 - a) / b
            else:
                threshold = np.mean(x_train)

            # Find held-out predicted transition layer
            # If b > 0: higher X means higher fungibility (cross when X >= T)
            # If b < 0: lower X means higher fungibility (cross when X <= T)
            pred_layer = None
            for _, r in test_df.iterrows():
                val = r[pred]
                if (b > 0 and val >= threshold) or (b < 0 and val <= threshold):
                    pred_layer = int(r["depth"])
                    break
            if pred_layer is None:
                # If never crosses threshold, predict the last block
                pred_layer = int(test_df.iloc[-1]["depth"])

            true_layer = transition_targets[heldout_m]
            err = abs(pred_layer - true_layer)
            model_errors.append(err)

            trans_records.append({
                "predictor": pred,
                "heldout_model": heldout_m,
                "true_transition_depth": true_layer,
                "predicted_transition_depth": pred_layer,
                "transition_threshold_T": float(threshold),
                "fit_slope_b": float(b),
                "absolute_depth_error": err
            })

    trans_df = pd.DataFrame(trans_records)

    return pred_summary_df, loao_df, trans_df


# ==============================================================================
# 4. FIGURE GENERATION
# ==============================================================================

def generate_publication_figures(
    layer_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    pred_summary_df: pd.DataFrame,
    loao_df: pd.DataFrame,
    output_dir: str
):
    """
    Generates high-resolution publication figures:
    Figure A: Depth alignment (Fungibility F_l vs Best Predictor per architecture)
    Figure B: Predictor vs Fungibility Scatter Plot
    Figure C: Leave-One-Architecture-Out (Predicted vs Observed)
    Figure D: Spectral & Sensitivity Decomposition across depths
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    merged = pd.merge(
        targets_df,
        layer_df,
        on=["model_key", "depth", "norm_depth"],
        how="inner"
    )

    # Determine best predictor from pred_summary_df (highest pooled absolute rho)
    best_pred = pred_summary_df.iloc[0]["predictor"]
    best_rho = pred_summary_df.iloc[0]["pooled_spearman_rho"]
    print(f"Generating figures with top-ranked predictor: '{best_pred}' (rho = {best_rho:.4f})")

    model_display_names = {
        "deit_tiny": "DeiT-Tiny (D=192)",
        "deit_small": "DeiT-Small (D=384)",
        "vit_base": "ViT-B/16 AugReg (D=768)",
        "dinov2": "DINOv2 ViT-S/14 (D=384)"
    }
    model_colors = {
        "deit_tiny": "#2b5c8f",
        "deit_small": "#3a927a",
        "vit_base": "#d95f02",
        "dinov2": "#7570b3"
    }

    # ---------------------------------------------------------
    # FIGURE A: Depth Alignment (4 Panels)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    axes = axes.flatten()

    for idx, (m_key, ax1) in enumerate(zip(["deit_tiny", "deit_small", "vit_base", "dinov2"], axes)):
        sub_gt = merged[merged["model_key"] == m_key].sort_values(by="depth")
        sub_all = layer_df[layer_df["model_key"] == m_key].sort_values(by="depth")

        # Left axis: Fungibility Score F_l
        color_fl = "#1f77b4"
        ax1.set_xlabel("Transformer Block Depth ($l$)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Fungibility Score ($F_l$)", color=color_fl, fontsize=11, fontweight="bold")
        line1 = ax1.plot(
            sub_gt["depth"], sub_gt["F_l_clip"],
            marker="o", linewidth=2.5, markersize=8, color=color_fl, label="Observed $F_l$"
        )
        ax1.axhline(0.80, color=color_fl, linestyle="--", alpha=0.5, label="Fungible Threshold (0.80)")
        ax1.tick_params(axis="y", labelcolor=color_fl)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xticks(range(1, 13))

        # Right axis: Best Clean Predictor
        ax2 = ax1.twinx()
        color_pred = "#d95f02"
        ax2.set_ylabel(f"Predictor: {best_pred}", color=color_pred, fontsize=11, fontweight="bold")
        line2 = ax2.plot(
            sub_all["depth"], sub_all[best_pred],
            marker="s", linewidth=2.0, linestyle="-.", markersize=6, color=color_pred, label=f"Clean {best_pred}"
        )
        ax2.tick_params(axis="y", labelcolor=color_pred)

        ax1.set_title(model_display_names[m_key], fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Figure A: Clean-State Predictor vs. Observed Fungibility Transition\n(Predictor: {best_pred})",
        fontsize=15, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    fig_a_path = os.path.join(output_dir, "figure_a_depth_alignment.png")
    plt.savefig(fig_a_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_a_path}")

    # ---------------------------------------------------------
    # FIGURE B: Predictor vs Fungibility Scatter Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 7), dpi=300)
    for m_key in ["deit_tiny", "deit_small", "vit_base", "dinov2"]:
        sub = merged[merged["model_key"] == m_key]
        plt.scatter(
            sub[best_pred], sub["F_l_clip"],
            s=100, color=model_colors[m_key], label=model_display_names[m_key], edgecolors="black", linewidths=1.2
        )

    # Fit pooled regression line
    x_all = merged[best_pred].values
    y_all = merged["F_l_clip"].values
    if np.std(x_all) > 0:
        p_coef = np.polyfit(x_all, y_all, 1)
        x_grid = np.linspace(x_all.min(), x_all.max(), 100)
        plt.plot(x_grid, np.polyval(p_coef, x_grid), "k--", linewidth=2.0,
                 label=f"Linear Fit (Pooled $r={np.corrcoef(x_all, y_all)[0,1]:.3f}$, $\\rho={best_rho:.3f}$)")

    plt.axhline(0.80, color="gray", linestyle=":", label="Fungible Threshold ($F=0.80$)")
    plt.xlabel(f"Clean Predictor: {best_pred}", fontsize=12, fontweight="bold")
    plt.ylabel("Observed Fungibility Score ($F_l$)", fontsize=12, fontweight="bold")
    plt.title(
        f"Figure B: Clean-State Metric vs. Patch-Content Fungibility Score\n(N=22 Validated Model-Depth Points)",
        fontsize=13, fontweight="bold"
    )
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    fig_b_path = os.path.join(output_dir, "figure_b_predictor_scatter.png")
    plt.savefig(fig_b_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_b_path}")

    # ---------------------------------------------------------
    # FIGURE C: Leave-One-Architecture-Out (Predicted vs Observed)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 7), dpi=300)
    
    # Sub-table for best predictor in LOAO
    best_loao = loao_df[loao_df["predictor"] == best_pred]

    for m_key in ["deit_tiny", "deit_small", "vit_base", "dinov2"]:
        m_row = best_loao[best_loao["heldout_model"] == m_key].iloc[0]
        alpha = m_row["intercept_alpha"]
        beta = m_row["slope_beta"]
        sub = merged[merged["model_key"] == m_key]
        x_te = sub[best_pred].values
        y_obs = sub["F_l_clip"].values
        y_pred = alpha + beta * x_te

        plt.scatter(
            y_obs, y_pred,
            s=110, color=model_colors[m_key], label=f"{model_display_names[m_key]} (MAE: {m_row['test_mae']:.3f})",
            edgecolors="black", linewidths=1.2
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.8, label="Ideal Calibration (1:1)")
    plt.xlabel("Observed Fungibility Score ($F_l$)", fontsize=12, fontweight="bold")
    plt.ylabel(f"Held-Out Predicted Fungibility ($\\hat{{F}}_l$)", fontsize=12, fontweight="bold")
    plt.title(
        f"Figure C: Leave-One-Architecture-Out Generalization\n(Cross-Architecture Transfer of '{best_pred}')",
        fontsize=13, fontweight="bold"
    )
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    fig_c_path = os.path.join(output_dir, "figure_c_loao_predicted_vs_observed.png")
    plt.savefig(fig_c_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_c_path}")

    # ---------------------------------------------------------
    # FIGURE D: Spectral & Sensitivity Decomposition across Depths
    # ---------------------------------------------------------
    fig, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    for m_key in ["deit_tiny", "deit_small", "vit_base", "dinov2"]:
        sub_all = layer_df[layer_df["model_key"] == m_key].sort_values(by="depth")
        ax_d1.plot(
            sub_all["depth"], sub_all["effective_rank"],
            marker="o", linewidth=2.0, color=model_colors[m_key], label=model_display_names[m_key]
        )
        ax_d2.plot(
            sub_all["depth"], sub_all["margin_grad_norm"],
            marker="s", linewidth=2.0, color=model_colors[m_key], label=model_display_names[m_key]
        )

    ax_d1.set_title("Effective Representation Rank ($r_{\\text{eff}}$)", fontsize=12, fontweight="bold")
    ax_d1.set_xlabel("Block Depth ($l$)", fontsize=11, fontweight="bold")
    ax_d1.set_ylabel("Entropy Effective Rank", fontsize=11, fontweight="bold")
    ax_d1.set_xticks(range(1, 13))
    ax_d1.legend()

    ax_d2.set_title("Downstream Margin Sensitivity ($||J_l^{\\text{margin}}||_F$)", fontsize=12, fontweight="bold")
    ax_d2.set_xlabel("Block Depth ($l$)", fontsize=11, fontweight="bold")
    ax_d2.set_ylabel("Frobenius Gradient Norm", fontsize=11, fontweight="bold")
    ax_d2.set_xticks(range(1, 13))
    ax_d2.legend()

    fig.suptitle(
        "Figure D: Decomposition of Representation Geometry and Downstream Sensitivity Across Layers",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    fig_d_path = os.path.join(output_dir, "figure_d_spectral_sensitivity_decomposition.png")
    plt.savefig(fig_d_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_d_path}")


# ==============================================================================
# 5. MASTER EXECUTION FUNCTION
# ==============================================================================

def run_predictive_principle_experiment():
    """
    Executes the entire end-to-end predictive principle experiment.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Predictive Principle Experiment on {device} (PyTorch {torch.__version__})")

    out_dir = os.path.join("outputs", "fungibility_predictive_principle")
    fig_dir = os.path.join("figures", "fungibility_predictive_principle")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Assemble ground truth fungibility targets
    targets_df = assemble_ground_truth_targets()
    print(f"Loaded ground-truth targets ({len(targets_df)} points):")
    print(targets_df[["model_key", "depth", "zero_damage", "centroid_damage", "F_l", "F_l_clip"]].to_string())

    # 2. Compute clean-state layer metrics across all 4 models
    models_to_run = ["deit_tiny", "deit_small", "vit_base", "dinov2"]
    all_layer_dfs = []

    for m_key in models_to_run:
        m_df = compute_clean_layer_metrics(m_key, device, batch_size=25, max_calib_samples=1000)
        all_layer_dfs.append(m_df)

    layer_df = pd.concat(all_layer_dfs, ignore_index=True)
    layer_csv_path = os.path.join(out_dir, "layer_metrics.csv")
    layer_df.to_csv(layer_csv_path, index=False)
    print(f"\nSaved layer metrics to: {layer_csv_path}")

    # 3. Run Statistical Analysis, Baseline Control, and LOAO
    pred_summary_df, loao_df, trans_df = run_statistical_analysis(layer_df, targets_df)

    pred_csv_path = os.path.join(out_dir, "predictor_vs_fungibility.csv")
    pred_summary_df.to_csv(pred_csv_path, index=False)
    print(f"Saved predictor summary to: {pred_csv_path}")

    loao_csv_path = os.path.join(out_dir, "leave_one_architecture_out.csv")
    loao_df.to_csv(loao_csv_path, index=False)
    print(f"Saved LOAO results to: {loao_csv_path}")

    trans_csv_path = os.path.join(out_dir, "transition_predictions.csv")
    trans_df.to_csv(trans_csv_path, index=False)
    print(f"Saved transition predictions to: {trans_csv_path}")

    # 4. Generate Figures
    generate_publication_figures(layer_df, targets_df, pred_summary_df, loao_df, fig_dir)

    # 5. Build and save validation manifest
    best_pred_row = pred_summary_df.iloc[0]
    best_pred_name = best_pred_row["predictor"]

    # Calculate mean LOAO metrics for best predictor
    best_loao_sub = loao_df[loao_df["predictor"] == best_pred_name]
    mean_loao_mae = float(best_loao_sub["test_mae"].mean())
    mean_loao_r2 = float(best_loao_sub["test_r2_loao"].mean())

    best_trans_sub = trans_df[trans_df["predictor"] == best_pred_name]
    mean_trans_error = float(best_trans_sub["absolute_depth_error"].mean())

    # Pre-registered outcome determination
    # Criteria:
    # Outcome A: |rho| >= 0.70, delta_r2 >= 0.15, mean_loao_r2 > 0, mean_trans_error <= 1.0
    # Outcome B: High correlation (|rho| >= 0.70) but fails LOAO
    # Outcome C: Depth alone explains as much (delta_r2 < 0.05)
    # Outcome D: Kill
    pooled_rho = abs(float(best_pred_row["pooled_spearman_rho"]))
    delta_r2 = float(best_pred_row["incremental_delta_r2"])

    if delta_r2 < 0.05:
        verdict = "OUTCOME C — DEPTH REMAINS THE BEST PREDICTOR"
        explanation = f"Top predictor '{best_pred_name}' provides negligible explanatory gain beyond normalized depth (delta_R^2 = {delta_r2:.4f} < 0.05)."
    elif pooled_rho >= 0.70 and mean_loao_r2 > 0 and mean_trans_error <= 1.0 and delta_r2 >= 0.15:
        verdict = "OUTCOME A — STRONG PREDICTIVE PRINCIPLE"
        explanation = f"Top predictor '{best_pred_name}' achieves strong correlation (|rho|={pooled_rho:.4f}), incremental R^2 = {delta_r2:.4f}, passes LOAO transfer, and predicts held-out transition depth with mean error {mean_trans_error:.2f} blocks."
    elif pooled_rho >= 0.60:
        verdict = "OUTCOME B — DESCRIPTIVE CORRELATE ONLY"
        explanation = f"Top predictor '{best_pred_name}' correlates with fungibility (|rho|={pooled_rho:.4f}, delta_R^2 = {delta_r2:.4f}), but fails cross-architecture generalization or transition depth precision."
    else:
        verdict = "OUTCOME D — KILL PREDICTIVE-PRINCIPLE HYPOTHESIS"
        explanation = f"No candidate clean-state metric achieved robust correlation (|rho|={pooled_rho:.4f} < 0.60)."

    print(f"\n=======================================================")
    print(f"PRE-REGISTERED SCIENTIFIC VERDICT: {verdict}")
    print(f"Explanation: {explanation}")
    print(f"=======================================================")

    manifest = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "device": str(device),
        "num_models": len(models_to_run),
        "num_model_depth_points": len(targets_df),
        "calibration_sample_size": 1000,
        "pre_registered_threshold": 0.80,
        "best_predictor": best_pred_name,
        "best_predictor_metrics": {
            "pooled_spearman_rho": float(best_pred_row["pooled_spearman_rho"]),
            "pooled_pearson_r": float(best_pred_row["pooled_pearson_r"]),
            "r2_univariate": float(best_pred_row["r2_univariate"]),
            "r2_depth_baseline": float(best_pred_row["r2_depth_baseline"]),
            "r2_joint_with_depth": float(best_pred_row["r2_joint_with_depth"]),
            "incremental_delta_r2": delta_r2,
            "partial_corr_given_depth": float(best_pred_row["partial_corr_given_depth"]),
            "mean_loao_mae": mean_loao_mae,
            "mean_loao_r2": mean_loao_r2,
            "mean_transition_depth_error": mean_trans_error
        },
        "verdict": verdict,
        "verdict_explanation": explanation
    }

    manifest_path = os.path.join(out_dir, "validation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved validation manifest to: {manifest_path}")

    return manifest


if __name__ == "__main__":
    run_predictive_principle_experiment()
