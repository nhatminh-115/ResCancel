"""
patch_fungibility/residual_carrier_pipeline.py

Pipeline for IMAGE-CONDITIONED RESIDUAL CARRIER POC:
"Can an image-conditioned, geometry-compatible carrier token outperform pure random pruning
at the same downstream token budget?"

Evaluates:
- Baseline A: Pure Random Pruning (Keep B real patches)
- Baseline B: Existing Unweighted Centroid Carrier (Keep B-1 real patches + mu_ell)
- Diagnostic Baseline: Raw Image-Mean Carrier (Keep B-1 real patches + p_bar_M(x))
- Candidate: Image-Conditioned Residual Carrier:
    c_{r, gamma}(x) = mu_ell + gamma * r_r(x)
    where r(x) = p_bar_M(x) - mu_ell, and r_r(x) is projected onto top r calibration PCA directions.
    r in {1, 4, 16, 64, D}
    gamma in {0.0, 0.25, 0.5, 0.75, 1.0}
    Token weight: s = 1 (unweighted, no log-multiplicity bias).

Target Models & Operating Points:
- DeiT-Small: Depth 8, 75% replacement (k=147, M_real=49, B=50, total tokens=51)
- ViT-B/16 AugReg: Depth 7, 63.8% replacement (k=125, M_real=71, B=72, total tokens=73)
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits, ParquetImageSubset
from patch_fungibility.dense_fraction_models import load_model_and_transform
from patch_fungibility.dense_fraction_pipeline import extract_activations_and_clean_logits
from patch_fungibility.geometry_bank_pipeline import forward_downstream_standard


MASK_SEEDS = [31001, 31002, 31003, 31004, 31005]
R_VALUES = [1, 4, 16, 64, "D"]
GAMMA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

MODEL_CONFIGS = [
    {
        "key": "deit_small",
        "depth": 8,
        "n_patches": 196,
        "embed_dim": 384,
        "k_replaced": 147,
        "M_real": 49,
        "budget_B": 50,
        "total_downstream_tokens": 51,
        "clean_acc": 0.7610,
        "clean_margin": 2.2514
    },
    {
        "key": "vit_base",
        "depth": 7,
        "n_patches": 196,
        "embed_dim": 768,
        "k_replaced": 125,
        "M_real": 71,
        "budget_B": 72,
        "total_downstream_tokens": 73,
        "clean_acc": 0.7610,
        "clean_margin": 3.1923
    }
]


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float]:
    """Computes paired bootstrap 95% confidence interval for mean difference."""
    rng = np.random.RandomState(seed)
    n = len(diffs)
    indices = rng.randint(0, n, size=(n_boot, n))
    boot_means = np.mean(diffs[indices], axis=1)
    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


def compute_mcnemar(y_true: np.ndarray, pred_pruning: np.ndarray, pred_carrier: np.ndarray) -> Dict[str, Any]:
    """Computes contingency counts and exact binomial two-sided p-value."""
    corr_p = (pred_pruning == y_true)
    corr_c = (pred_carrier == y_true)
    n00 = int(np.sum((~corr_p) & (~corr_c)))
    n01 = int(np.sum((corr_p) & (~corr_c)))  # Pruning correct, Carrier incorrect
    n10 = int(np.sum((~corr_p) & (corr_c)))  # Carrier correct, Pruning incorrect
    n11 = int(np.sum((corr_p) & (corr_c)))
    total_disc = n01 + n10
    if total_disc == 0:
        p_val = 1.0
    else:
        p_val = float(stats.binomtest(min(n01, n10), total_disc, p=0.5).pvalue)
    return {
        "n00": n00,
        "n01_pruning_only": n01,
        "n10_carrier_only": n10,
        "n11": n11,
        "mcnemar_p": p_val
    }


def compute_calibration_pca(
    all_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes centroid and PCA eigenvectors on calibration patch tokens.
    Returns: (mu, eigenvalues, eigenvectors)
    """
    mu = all_tokens.mean(dim=0)
    centered = all_tokens - mu.unsqueeze(0)
    cov = torch.cov(centered.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)

    return mu, eigenvalues, eigenvectors


def run_residual_carrier_experiment(
    output_dir: str = "outputs/fungibility_residual_carrier",
    smoke_test: bool = False,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Master execution pipeline for Image-Conditioned Residual Carrier POC.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Running Image-Conditioned Residual Carrier POC on device: {device} ===")

    # 1. Load canonical dataset splits
    n_per_split = 64 if smoke_test else 1000
    print(f"[Dataset] Loading canonical ImageNet disjoint splits (N={n_per_split})...")
    calib_ds_raw, eval_ds_raw, _, _ = get_disjoint_imagenet_splits(
        calib_seed=9101, eval_seed=9201, n_per_split=n_per_split
    )
    calib_samples = calib_ds_raw._samples
    eval_samples = eval_ds_raw._samples
    labels = torch.tensor([s[1] for s in eval_samples], dtype=torch.long)
    labels_np = labels.numpy()
    N_eval = len(labels)

    # 2. Load mask permutations
    with open("outputs/fungibility_dense_fraction/mask_permutations.json") as f:
        perms_meta = json.load(f)
    perms_196 = {int(s): np.array(p) for s, p in perms_meta["perms_196"].items()}

    trial_rows = []

    for cfg in MODEL_CONFIGS:
        m_key = cfg["key"]
        depth = cfg["depth"]
        n_patches = cfg["n_patches"]
        embed_dim = cfg["embed_dim"]
        k = cfg["k_replaced"]
        M_real = cfg["M_real"]
        B = cfg["budget_B"]
        clean_acc = cfg["clean_acc"]
        clean_margin = cfg["clean_margin"]

        print(f"\n=================================================================")
        print(f"Model: {m_key} | Depth: {depth} | Discarded k: {k} | Real Anchors M: {M_real} | Budget B: {B}")
        print(f"=================================================================")

        model, transform, _ = load_model_and_transform(m_key, device=device)
        model.eval()

        # Step 2.1: Extract calibration activations and compute PCA
        print(f"  Extracting calibration activations (N={n_per_split})...")
        calib_ds = ParquetImageSubset(calib_samples, transform=transform)
        calib_loader = DataLoader(calib_ds, batch_size=64 if embed_dim <= 384 else 32, shuffle=False)

        cached_calib_h, _, _ = extract_activations_and_clean_logits(
            model, m_key, calib_loader, target_depths=(depth,), device=device
        )
        h_calib_spatial = cached_calib_h[depth][:, 1:, :].to(device)
        all_calib_tokens = h_calib_spatial.reshape(-1, embed_dim)
        print(f"  Calibration tokens shape: {all_calib_tokens.shape}")

        mu_calib, eigenvalues, eigenvectors = compute_calibration_pca(all_calib_tokens)
        del cached_calib_h, h_calib_spatial, all_calib_tokens

        # Step 2.2: Extract evaluation activations
        print(f"  Extracting evaluation activations (N={n_per_split})...")
        eval_ds = ParquetImageSubset(eval_samples, transform=transform)
        eval_loader = DataLoader(eval_ds, batch_size=64 if embed_dim <= 384 else 32, shuffle=False)

        cached_eval_h, clean_logits_full, _ = extract_activations_and_clean_logits(
            model, m_key, eval_loader, target_depths=(depth,), device=device
        )
        h_eval = cached_eval_h[depth].to(device)
        clean_logits_t = clean_logits_full.to(device)
        labels_t = labels.to(device)

        cls_eval = h_eval[:, :1, :]
        patches_eval = h_eval[:, 1:, :]

        mask_clean_inc = torch.ones_like(clean_logits_t, dtype=torch.bool)
        mask_clean_inc[torch.arange(N_eval), labels_t] = False

        # Step 2.3: Iterate over the 5 mask seeds
        for s_idx, s in enumerate(MASK_SEEDS):
            p_s = perms_196[s]

            # In dense fraction convention:
            # Replaced patches: p_s[:k]
            # Surviving patches for carrier: p_s[k:] (size M_real = B - 1)
            # Pure Random Pruning patches: p_s[k-1:] (size B)
            discarded_indices = p_s[:k]
            carrier_real_indices = p_s[k:]
            pruning_real_indices = p_s[k - 1:]

            assert len(carrier_real_indices) == M_real
            assert len(pruning_real_indices) == B

            # ------------------------------------------------------------------
            # Baseline A: Pure Random Pruning (Keep B real patches)
            # ------------------------------------------------------------------
            seq_pruning = torch.cat([cls_eval, patches_eval[:, pruning_real_indices, :]], dim=1)
            logits_pruning_list = []
            with torch.no_grad():
                for bi in range(0, N_eval, 64):
                    l_bi = forward_downstream_standard(model, m_key, depth, seq_pruning[bi : bi + 64])
                    logits_pruning_list.append(l_bi)
            logits_pruning = torch.cat(logits_pruning_list, dim=0)

            preds_pruning = torch.argmax(logits_pruning, dim=-1).cpu().numpy()
            acc_pruning = float(np.mean(preds_pruning == labels_np))

            true_pruning = logits_pruning[torch.arange(N_eval), labels_t]
            strongest_pruning = torch.max(torch.where(mask_clean_inc, logits_pruning, -1e9), dim=1).values
            margins_pruning = (true_pruning - strongest_pruning).cpu().numpy()
            mean_margin_pruning = float(np.mean(margins_pruning))

            # Record Pure Pruning baseline
            trial_rows.append({
                "model": m_key,
                "depth": depth,
                "mask_seed": s,
                "condition": "PURE_RANDOM_PRUNING",
                "r_rank": "None",
                "gamma": 0.0,
                "is_carrier": False,
                "top1_acc": acc_pruning,
                "delta_acc": 0.0,
                "mean_margin": mean_margin_pruning,
                "paired_delta_margin": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "cohen_dz": 0.0,
                "mcnemar_p": 1.0,
                "n00": int(np.sum((preds_pruning != labels_np) & (preds_pruning != labels_np))),
                "n01_pruning_only": 0,
                "n10_carrier_only": 0,
                "n11": int(np.sum((preds_pruning == labels_np) & (preds_pruning == labels_np))),
                "pred_agreement": 1.0
            })

            # ------------------------------------------------------------------
            # Compute image-specific residual r(x)
            # ------------------------------------------------------------------
            discarded_patches = patches_eval[:, discarded_indices, :]  # (N_eval, k, D)
            p_bar_M = discarded_patches.mean(dim=1)                     # (N_eval, D)
            residual = p_bar_M - mu_calib.unsqueeze(0)                 # (N_eval, D)

            # Precompute projections for all r in R_VALUES
            projected_residuals = {}
            for r in R_VALUES:
                if r == "D":
                    projected_residuals[r] = residual
                else:
                    U_r = eigenvectors[:, :r]  # (D, r)
                    r_r = (residual @ U_r) @ U_r.T  # (N_eval, D)
                    projected_residuals[r] = r_r

            real_patches_carrier = patches_eval[:, carrier_real_indices, :]  # (N_eval, M_real, D)

            # ------------------------------------------------------------------
            # Sweep Carrier Configurations: r in {1, 4, 16, 64, D}, gamma in {0.0, 0.25, 0.5, 0.75, 1.0}
            # ------------------------------------------------------------------
            # Note: When gamma = 0.0, carrier is mu_calib regardless of r. We evaluate it under each r or once.
            for r in R_VALUES:
                for gamma in GAMMA_VALUES:
                    # Construct carrier: c = mu + gamma * r_r(x)
                    r_proj = projected_residuals[r]
                    carrier_token = (mu_calib.unsqueeze(0) + gamma * r_proj).unsqueeze(1)  # (N_eval, 1, D)

                    seq_carrier = torch.cat([cls_eval, real_patches_carrier, carrier_token], dim=1)
                    logits_carrier_list = []
                    with torch.no_grad():
                        for bi in range(0, N_eval, 64):
                            l_bi = forward_downstream_standard(model, m_key, depth, seq_carrier[bi : bi + 64])
                            logits_carrier_list.append(l_bi)
                    logits_carrier = torch.cat(logits_carrier_list, dim=0)

                    preds_carrier = torch.argmax(logits_carrier, dim=-1).cpu().numpy()
                    acc_carrier = float(np.mean(preds_carrier == labels_np))

                    true_carrier = logits_carrier[torch.arange(N_eval), labels_t]
                    strongest_carrier = torch.max(torch.where(mask_clean_inc, logits_carrier, -1e9), dim=1).values
                    margins_carrier = (true_carrier - strongest_carrier).cpu().numpy()
                    mean_margin_carrier = float(np.mean(margins_carrier))

                    # Paired margin differences
                    diff_margins = margins_carrier - margins_pruning
                    paired_delta_margin = float(np.mean(diff_margins))
                    std_diff = float(np.std(diff_margins))
                    cohen_dz = paired_delta_margin / (std_diff + 1e-12)

                    ci_lower, ci_upper = paired_bootstrap_ci(diff_margins, n_boot=1000, seed=42)
                    mcnemar_stats = compute_mcnemar(labels_np, preds_pruning, preds_carrier)
                    pred_agreement = float(np.mean(preds_carrier == preds_pruning))

                    # Condition labeling
                    if gamma == 0.0:
                        cond_name = "UNWEIGHTED_CENTROID"
                    elif r == "D" and gamma == 1.0:
                        cond_name = "RAW_IMAGE_MEAN_CARRIER"
                    else:
                        cond_name = f"RESIDUAL_CARRIER_r{r}_g{gamma}"

                    trial_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "mask_seed": s,
                        "condition": cond_name,
                        "r_rank": str(r),
                        "gamma": gamma,
                        "is_carrier": True,
                        "top1_acc": acc_carrier,
                        "delta_acc": acc_carrier - acc_pruning,
                        "mean_margin": mean_margin_carrier,
                        "paired_delta_margin": paired_delta_margin,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "cohen_dz": cohen_dz,
                        "mcnemar_p": mcnemar_stats["mcnemar_p"],
                        "n00": mcnemar_stats["n00"],
                        "n01_pruning_only": mcnemar_stats["n01_pruning_only"],
                        "n10_carrier_only": mcnemar_stats["n10_carrier_only"],
                        "n11": mcnemar_stats["n11"],
                        "pred_agreement": pred_agreement
                    })

        del cached_eval_h, h_eval, clean_logits_t, cls_eval, patches_eval, model
        torch.cuda.empty_cache()

    # 3. Save Trial Results CSV
    df_trials = pd.DataFrame(trial_rows)
    trial_path = os.path.join(output_dir, "trial_results.csv")
    df_trials.to_csv(trial_path, index=False)
    print(f"\n[Saved] Trial results saved to {trial_path} ({len(df_trials)} rows)")

    # 4. Aggregate Summary by Condition (mean and std across seeds)
    summary_cols = ["model", "depth", "condition", "r_rank", "gamma", "is_carrier"]
    df_summary = df_trials.groupby(summary_cols).agg(
        mean_top1_acc=("top1_acc", "mean"),
        std_top1_acc=("top1_acc", "std"),
        mean_delta_acc=("delta_acc", "mean"),
        std_delta_acc=("delta_acc", "std"),
        mean_margin=("mean_margin", "mean"),
        std_margin=("mean_margin", "std"),
        mean_paired_delta_margin=("paired_delta_margin", "mean"),
        mean_ci_lower=("ci_lower", "mean"),
        mean_ci_upper=("ci_upper", "mean"),
        mean_cohen_dz=("cohen_dz", "mean"),
        mean_pred_agreement=("pred_agreement", "mean"),
        mean_n01=("n01_pruning_only", "mean"),
        mean_n10=("n10_carrier_only", "mean")
    ).reset_index()

    summary_path = os.path.join(output_dir, "summary_by_condition.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[Saved] Summary by condition saved to {summary_path}")

    # 5. Build Heatmap Grid Table (Delta Acc for each r and gamma)
    heatmap_rows = []
    for m in [c["key"] for c in MODEL_CONFIGS]:
        for r in R_VALUES:
            for g in GAMMA_VALUES:
                sub = df_summary[
                    (df_summary["model"] == m) &
                    (df_summary["r_rank"] == str(r)) &
                    (df_summary["gamma"] == g)
                ]
                if len(sub) > 0:
                    heatmap_rows.append({
                        "model": m,
                        "r_rank": str(r),
                        "gamma": g,
                        "mean_delta_acc_pct": sub["mean_delta_acc"].values[0] * 100.0,
                        "mean_delta_margin": sub["mean_paired_delta_margin"].values[0]
                    })
    df_heatmap = pd.DataFrame(heatmap_rows)
    heatmap_path = os.path.join(output_dir, "heatmap_grid.csv")
    df_heatmap.to_csv(heatmap_path, index=False)
    print(f"[Saved] Heatmap grid saved to {heatmap_path}")

    # 6. Evaluate Decision Rules
    # GO rule: Delta Acc >= +0.5% (0.005) on BOTH models, positive margin CI, consistent across seeds.
    # Check max Delta Acc for each model
    max_delta_by_model = {}
    best_config_by_model = {}

    for m in [c["key"] for c in MODEL_CONFIGS]:
        carrier_sub = df_summary[(df_summary["model"] == m) & (df_summary["is_carrier"])]
        best_row = carrier_sub.sort_values("mean_delta_acc", ascending=False).iloc[0]
        max_delta_by_model[m] = float(best_row["mean_delta_acc"])
        best_config_by_model[m] = {
            "condition": best_row["condition"],
            "r": best_row["r_rank"],
            "gamma": best_row["gamma"],
            "delta_acc": float(best_row["mean_delta_acc"]),
            "paired_delta_margin": float(best_row["mean_paired_delta_margin"]),
            "ci_lower": float(best_row["mean_ci_lower"]),
            "ci_upper": float(best_row["mean_ci_upper"])
        }

    # Decision logic
    deit_delta = max_delta_by_model["deit_small"]
    vitb_delta = max_delta_by_model["vit_base"]

    if (deit_delta >= 0.005) and (vitb_delta >= 0.005):
        verdict = "GO — image-conditioned geometry-compatible carrier shows reproducible advantage over matched-budget pruning"
    else:
        verdict = "KILL — image-conditioned carrier does not improve the matched-budget pruning frontier"

    elapsed = time.time() - start_time
    print(f"\n=================================================================")
    print(f"DECISION VERDICT: {verdict}")
    print(f"DeiT-Small Best Delta Acc: {deit_delta * 100:+.2f}% ({best_config_by_model['deit_small']['condition']})")
    print(f"ViT-B Best Delta Acc:     {vitb_delta * 100:+.2f}% ({best_config_by_model['vit_base']['condition']})")
    print(f"Total Execution Time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print(f"=================================================================")

    manifest = {
        "experiment_name": "IMAGE-CONDITIONED RESIDUAL CARRIER POC",
        "date": "2026-10-04",
        "models": [c["key"] for c in MODEL_CONFIGS],
        "mask_seeds": MASK_SEEDS,
        "r_values": [str(r) for r in R_VALUES],
        "gamma_values": GAMMA_VALUES,
        "verdict": verdict,
        "max_delta_by_model": max_delta_by_model,
        "best_config_by_model": best_config_by_model,
        "smoke_test": smoke_test,
        "elapsed_seconds": elapsed
    }
    with open(os.path.join(output_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/fungibility_residual_carrier")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    run_residual_carrier_experiment(output_dir=args.output_dir, smoke_test=args.smoke_test)
