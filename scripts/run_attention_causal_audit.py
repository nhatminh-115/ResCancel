"""
scripts/run_attention_causal_audit.py

Main runner script for the Causal Decomposition of the Coherence Effect
in Vision Transformers.

Executes:
1. Causal condition sweeps (full vs frozen-attention value-only vs rerouting-only)
2. Q/K/V sub-projection pathway decomposition
3. Head-wise Gamma scalar calculation and linear head disturbance prediction
4. Softmax attention matrix movement measurements (Frobenius, KL, entropy)
5. Block-by-block disturbance propagation and residual pathway decomposition
6. Spatial smoothness analysis & 2D Fourier spatial frequency response mapping
7. Confirmatory replication on DeiT-Tiny and DINOv2
8. Generation of publication figures and validation manifest
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.joint_stream_geometry import (
    construct_token_patterns,
    construct_joint_feature_directions,
    estimate_functional_metric
)
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.attention_causal_audit import (
    evaluate_causal_conditions,
    evaluate_qkv_decomposition,
    compute_headwise_gamma_and_predictions,
    measure_attention_movement,
    trace_blockwise_propagation_and_residuals,
    analyze_spatial_smoothness_and_fourier_response,
    generate_causal_audit_figures
)


def run_full_attention_causal_audit(
    device_str: str = "cuda",
    num_eval_images: int = 100,
    output_dir: str = "outputs/fungibility_attention_causal_audit",
    figures_dir: str = "figures/fungibility_attention_causal_audit"
):
    start_time = time.time()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"====================================================================")
    print(f"RUNNING ATTENTION CAUSAL AUDIT ON DEVICE: {device}")
    print(f"Number of evaluation images: {num_eval_images}")
    print(f"====================================================================")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    all_causal_records = []
    all_qkv_records = []
    all_gamma_records = []
    all_pred_records = []
    all_shift_records = []
    all_prop_records = []
    all_res_records = []
    all_fourier_records = []
    smoothness_results = {}

    # Exploratory models and depths
    exploratory_models = [
        ("deit_small", 8),
        ("vit_base", 7)
    ]

    for model_key, depth in exploratory_models:
        print(f"\n>>> Running Exploratory Model: {model_key} at Depth {depth}...")
        model, transform, meta = load_model_and_transform(model_key, device)
        embed_dim = meta["embed_dim"]
        num_patches = meta["num_patches"]

        # Load 100 validation images
        calib_ds, _, _, _ = get_disjoint_imagenet_splits()
        calib_ds.transform = transform
        calib_subset = Subset(calib_ds, list(range(num_eval_images)))
        loader = DataLoader(calib_subset, batch_size=25, shuffle=False)

        eval_images_list, eval_targets_list = [], []
        for x_b, y_b in loader:
            eval_images_list.append(x_b)
            eval_targets_list.append(y_b)
        eval_images = torch.cat(eval_images_list, dim=0).to(device)
        eval_targets = torch.cat(eval_targets_list, dim=0).to(device)

        # 1. Clean forward
        with torch.no_grad():
            clean_logits, clean_collected = forward_block_by_block(
                model, model_key, x=eval_images, start_depth=0, collect_depths=tuple(range(1, 13))
            )
            h_depth = clean_collected[depth]
            patches = h_depth[:, 1:, :].reshape(-1, embed_dim).to(torch.float64)
            centroid = patches.mean(dim=0).float()
            diff = patches - centroid.unsqueeze(0).to(torch.float64)
            cov_matrix = (diff.T @ diff) / (patches.shape[0] - 1)
            cov_matrix = cov_matrix.float()
            sigma_P = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()

        # 2. Metric matrix and feature directions
        print(f"Estimating functional metric matrix for {model_key} depth {depth}...")
        metric_matrix = estimate_functional_metric(
            model, model_key, depth, eval_images[:4], device=device
        )
        feature_dirs = construct_joint_feature_directions(
            cov_matrix, centroid, metric_matrix, embed_dim
        )
        token_patterns = construct_token_patterns(num_patches, device=device)

        # 3. Controlled Causal Conditions
        print("Evaluating controlled causal conditions...")
        causal_rec = evaluate_causal_conditions(
            model, model_key, depth, eval_images, eval_targets,
            clean_logits, h_depth, feature_dirs, token_patterns,
            sigma_P=sigma_P, scale_grid=[0.5, 1.0, 2.0], device=device
        )
        all_causal_records.extend(causal_rec)

        # 4. QKV Pathway Decomposition
        print("Evaluating Q/K/V sub-projection pathways...")
        qkv_rec = evaluate_qkv_decomposition(
            model, model_key, depth, eval_images, eval_targets,
            clean_logits, h_depth, feature_dirs, token_patterns,
            sigma_P=sigma_P, scale_s=1.0, device=device
        )
        all_qkv_records.extend(qkv_rec)

        # 5. Head-wise Gamma and Linear Prediction
        print("Computing head-wise Gamma and linear disturbance prediction...")
        gamma_rec, pred_rec = compute_headwise_gamma_and_predictions(
            model, model_key, depth, h_depth, feature_dirs, token_patterns,
            sigma_P=sigma_P, scale_s=1.0, device=device
        )
        all_gamma_records.extend(gamma_rec)
        all_pred_records.extend(pred_rec)

        # 6. Attention Movement Metrics
        print("Measuring attention movement...")
        shift_rec = measure_attention_movement(
            model, model_key, depth, h_depth, feature_dirs, token_patterns,
            sigma_P=sigma_P, scale_grid=[0.5, 1.0, 2.0], device=device
        )
        all_shift_records.extend(shift_rec)

        # 7. Block-by-Block Disturbance Propagation & Residual Decomposition
        print("Tracing blockwise disturbance propagation and residual paths...")
        prop_rec, res_rec = trace_blockwise_propagation_and_residuals(
            model, model_key, depth, eval_images, clean_collected,
            feature_dirs, token_patterns, sigma_P=sigma_P, scale_s=1.0, device=device
        )
        all_prop_records.extend(prop_rec)
        all_res_records.extend(res_rec)

        # 8. Spatial Smoothness & 2D Fourier Response (on DeiT-Small)
        if model_key == "deit_small":
            print("Analyzing spatial smoothness and 2D Fourier response map...")
            smoothness_info, fourier_rec = analyze_spatial_smoothness_and_fourier_response(
                model, model_key, depth, eval_images, eval_targets,
                clean_logits, h_depth, feature_dirs, sigma_P=sigma_P, scale_s=1.0, device=device
            )
            smoothness_results[model_key] = smoothness_info
            all_fourier_records.extend(fourier_rec)

    # -------------------------------------------------------------------------
    # Confirmatory Replication on DeiT-Tiny and DINOv2
    # -------------------------------------------------------------------------
    print(f"\n====================================================================")
    print(f"RUNNING CONFIRMATORY REPLICATION ON DEIT-TINY AND DINOV2")
    print(f"====================================================================")
    replication_models = [
        ("deit_tiny", 8),
        ("dinov2", 8)
    ]
    all_rep_records = []

    for model_key, depth in replication_models:
        print(f"\n>>> Replicating decisive conditions on {model_key} at Depth {depth}...")
        model, transform, meta = load_model_and_transform(model_key, device)
        embed_dim = meta["embed_dim"]
        num_patches = meta["num_patches"]

        calib_ds, _, _, _ = get_disjoint_imagenet_splits()
        calib_ds.transform = transform
        calib_subset = Subset(calib_ds, list(range(num_eval_images)))
        loader = DataLoader(calib_subset, batch_size=25, shuffle=False)

        eval_images_list, eval_targets_list = [], []
        for x_b, y_b in loader:
            eval_images_list.append(x_b)
            eval_targets_list.append(y_b)
        eval_images = torch.cat(eval_images_list, dim=0).to(device)
        eval_targets = torch.cat(eval_targets_list, dim=0).to(device)

        with torch.no_grad():
            clean_logits, clean_collected = forward_block_by_block(
                model, model_key, x=eval_images, start_depth=0, collect_depths=(depth, depth + 1)
            )
            h_depth = clean_collected[depth]
            patches = h_depth[:, 1:, :].reshape(-1, embed_dim).to(torch.float64)
            centroid = patches.mean(dim=0).float()
            diff = patches - centroid.unsqueeze(0).to(torch.float64)
            cov_matrix = (diff.T @ diff) / (patches.shape[0] - 1)
            cov_matrix = cov_matrix.float()
            sigma_P = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()

        metric_matrix = estimate_functional_metric(
            model, model_key, depth, eval_images[:4], device=device
        )
        feature_dirs = construct_joint_feature_directions(
            cov_matrix, centroid, metric_matrix, embed_dim
        )
        token_patterns = construct_token_patterns(num_patches, device=device)

        # Evaluate decisive conditions along jac_top at scale s=1.0
        rep_rec = evaluate_causal_conditions(
            model, model_key, depth, eval_images, eval_targets,
            clean_logits, h_depth, feature_dirs, token_patterns,
            sigma_P=sigma_P, scale_grid=[1.0], device=device
        )
        # Filter for jac_top
        rep_rec_filtered = [r for r in rep_rec if r["feature_dir"] in ("jac_top", "none")]
        all_rep_records.extend(rep_rec_filtered)

    # -------------------------------------------------------------------------
    # Persist Machine-Readable Outputs
    # -------------------------------------------------------------------------
    print(f"\nPersisting machine-readable datasets to {output_dir}...")
    df_causal = pd.DataFrame(all_causal_records)
    df_qkv = pd.DataFrame(all_qkv_records)
    df_gamma = pd.DataFrame(all_gamma_records)
    df_pred = pd.DataFrame(all_pred_records)
    df_shift = pd.DataFrame(all_shift_records)
    df_prop = pd.DataFrame(all_prop_records)
    df_res = pd.DataFrame(all_res_records)
    df_rep = pd.DataFrame(all_rep_records)
    df_fourier = pd.DataFrame(all_fourier_records) if all_fourier_records else None

    df_causal.to_csv(os.path.join(output_dir, "causal_conditions.csv"), index=False)
    df_qkv.to_csv(os.path.join(output_dir, "qkv_decomposition.csv"), index=False)
    df_gamma.to_csv(os.path.join(output_dir, "headwise_gamma.csv"), index=False)
    df_pred.to_csv(os.path.join(output_dir, "headwise_prediction.csv"), index=False)
    df_shift.to_csv(os.path.join(output_dir, "attention_shift.csv"), index=False)
    df_prop.to_csv(os.path.join(output_dir, "blockwise_propagation.csv"), index=False)
    df_res.to_csv(os.path.join(output_dir, "residual_path.csv"), index=False)
    df_rep.to_csv(os.path.join(output_dir, "replication_summary.csv"), index=False)

    # Generate Publication Figures
    print(f"Generating publication figures in {figures_dir}...")
    generate_causal_audit_figures(
        df_causal, df_pred, df_prop, df_shift, df_rep, df_fourier, figures_dir
    )

    # Compute Summary Attribution Ratios
    # Gap_full = Damage(coherent, full) - Damage(random_sign, full)
    # Gap_V = Damage(coherent, V_only) - Damage(random_sign, V_only)
    # Gap_route = Damage(coherent, reroute) - Damage(random_sign, reroute)
    attribution_summary = {}
    for mk, dep in exploratory_models:
        sub = df_causal[
            (df_causal["model_key"] == mk) &
            (df_causal["depth"] == dep) &
            (df_causal["feature_dir"] == "jac_top") &
            (df_causal["scale_s"] == 1.0)
        ]
        if not sub.empty:
            dmg_full_gc = sub[(sub["token_pattern"] == "global_coherent") & (sub["condition"] == "full_perturbation")]["logit_margin_drop"].values[0]
            dmg_full_rs = sub[(sub["token_pattern"] == "random_sign") & (sub["condition"] == "full_perturbation")]["logit_margin_drop"].values[0]
            gap_full = dmg_full_gc - dmg_full_rs

            dmg_v_gc = sub[(sub["token_pattern"] == "global_coherent") & (sub["condition"] == "frozen_attn_v_only_pert_res")]["logit_margin_drop"].values[0]
            dmg_v_rs = sub[(sub["token_pattern"] == "random_sign") & (sub["condition"] == "frozen_attn_v_only_pert_res")]["logit_margin_drop"].values[0]
            gap_v = dmg_v_gc - dmg_v_rs

            dmg_rt_gc = sub[(sub["token_pattern"] == "global_coherent") & (sub["condition"] == "reroute_clean_res")]["logit_margin_drop"].values[0]
            dmg_rt_rs = sub[(sub["token_pattern"] == "random_sign") & (sub["condition"] == "reroute_clean_res")]["logit_margin_drop"].values[0]
            gap_rt = dmg_rt_gc - dmg_rt_rs

            ratio_v = float(gap_v / (gap_full + 1e-10))
            ratio_rt = float(gap_rt / (gap_full + 1e-10))

            attribution_summary[f"{mk}_depth_{dep}"] = {
                "gap_full": float(gap_full),
                "gap_v_only": float(gap_v),
                "gap_reroute": float(gap_rt),
                "v_path_attribution_ratio": ratio_v,
                "reroute_attribution_ratio": ratio_rt
            }

    elapsed = time.time() - start_time
    manifest = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
        "device": device_str,
        "exploratory_models": exploratory_models,
        "replication_models": replication_models,
        "num_eval_images": num_eval_images,
        "attribution_summary": attribution_summary,
        "smoothness_results": smoothness_results
    }

    with open(os.path.join(output_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n====================================================================")
    print(f"ATTENTION CAUSAL AUDIT COMPLETED IN {elapsed:.2f}s")
    print(f"Manifest saved to: {os.path.join(output_dir, 'validation_manifest.json')}")
    print(f"====================================================================")


if __name__ == "__main__":
    run_full_attention_causal_audit()
