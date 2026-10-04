"""
scripts/run_predictive_operator.py

Main runner script for the Token-Transmission Predictive Operator (T_v) experiment.

Executes:
1. Spectral SVD of clean operator T_v across feature directions and depths
2. Direct singular mode prediction tests on real unconstrained models
3. Finite-radius scale sweeps
4. Out-of-basis pattern transmission predictions and subspace energy analysis
5. Random held-out Gaussian pattern predictions (M=50)
6. Depth spectrum evolution and approximate null space dimension tracking
7. Bilinear feature x token interaction matrix
8. Spatial mode structure analysis
9. Confirmatory replication on DeiT-Tiny and DINOv2
10. Generation of publication figures and validation manifest
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
from patch_fungibility.predictive_operator import (
    construct_token_transmission_operator,
    evaluate_singular_modes,
    evaluate_finite_radius_curves,
    evaluate_out_of_basis_patterns,
    evaluate_random_held_out_patterns,
    evaluate_depth_spectrum_evolution,
    evaluate_feature_token_matrix,
    analyze_spatial_mode_structure,
    generate_predictive_operator_figures
)


def run_full_predictive_operator_experiment(
    device_str: str = "cuda",
    num_eval_images: int = 100,
    output_dir: str = "outputs/fungibility_predictive_operator",
    figures_dir: str = "figures/fungibility_predictive_operator"
):
    start_time = time.time()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"====================================================================")
    print(f"RUNNING TOKEN-TRANSMISSION PREDICTIVE OPERATOR ON DEVICE: {device}")
    print(f"Number of evaluation images: {num_eval_images}")
    print(f"====================================================================")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    all_spectrum_records = []
    all_validation_records = []
    all_curves_records = []
    all_out_of_basis_records = []
    all_random_records = []
    all_depth_records = []
    all_feat_token_records = []
    all_spatial_records = []

    # Exploratory models and their canonical depths
    exploratory_configs = [
        ("deit_small", 8, [5, 8, 10]),
        ("vit_base", 7, [5, 7, 10])
    ]

    for model_key, peak_depth, all_depths in exploratory_configs:
        print(f"\n====================================================================")
        print(f">>> Running Exploratory Model: {model_key} (Peak Depth {peak_depth})...")
        print(f"====================================================================")
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

        fdirs_by_depth = {}
        sig_P_by_depth = {}

        # Precompute metrics and directions for all depths
        for d in all_depths:
            h_d = clean_collected[d]
            patches = h_d[:, 1:, :].reshape(-1, embed_dim).to(torch.float64)
            centroid = patches.mean(dim=0).float()
            diff = patches - centroid.unsqueeze(0).to(torch.float64)
            cov_matrix = (diff.T @ diff) / (patches.shape[0] - 1)
            cov_matrix = cov_matrix.float()
            sig_P_by_depth[d] = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()

            print(f"Estimating metric for {model_key} at depth {d}...")
            metric_M = estimate_functional_metric(
                model, model_key, d, eval_images[:4], device=device
            )
            fdirs_by_depth[d] = construct_joint_feature_directions(
                cov_matrix, centroid, metric_M, embed_dim
            )

        fdirs_peak = fdirs_by_depth[peak_depth]
        sigma_P_peak = sig_P_by_depth[peak_depth]
        token_patterns = construct_token_patterns(num_patches, device=device)

        # 2. Singular mode evaluation (spectrum and real model validation)
        print("Evaluating singular modes on real model...")
        spec_rec, val_rec = evaluate_singular_modes(
            model, model_key, peak_depth, eval_images, eval_targets,
            clean_logits, fdirs_peak, sigma_P=sigma_P_peak, scale_s=1.0, device=device
        )
        all_spectrum_records.extend(spec_rec)
        all_validation_records.extend(val_rec)

        # 3. Finite-radius curves
        print("Tracing finite-radius response curves...")
        curves_rec = evaluate_finite_radius_curves(
            model, model_key, peak_depth, eval_images, eval_targets,
            clean_logits, fdirs_peak, sigma_P=sigma_P_peak, device=device
        )
        all_curves_records.extend(curves_rec)

        # 4. Out-of-basis patterns
        print("Evaluating out-of-basis patterns...")
        oob_rec = evaluate_out_of_basis_patterns(
            model, model_key, peak_depth, eval_images, eval_targets,
            clean_logits, fdirs_peak, token_patterns, sigma_P=sigma_P_peak, device=device
        )
        all_out_of_basis_records.extend(oob_rec)

        # 5. Random held-out patterns (M=50)
        print("Evaluating random held-out patterns (M=50)...")
        rand_rec = evaluate_random_held_out_patterns(
            model, model_key, peak_depth, eval_images, eval_targets,
            clean_logits, fdirs_peak, sigma_P=sigma_P_peak, num_patterns=50, device=device
        )
        all_random_records.extend(rand_rec)

        # 6. Bilinear Feature x Token Matrix
        print("Evaluating bilinear feature x token matrix...")
        ft_rec = evaluate_feature_token_matrix(
            model, model_key, peak_depth, eval_images, eval_targets,
            clean_logits, fdirs_peak, token_patterns, sigma_P=sigma_P_peak, device=device
        )
        all_feat_token_records.extend(ft_rec)

        # 7. Depth Spectrum Evolution
        print("Tracking depth spectrum evolution...")
        depth_rec = evaluate_depth_spectrum_evolution(
            model, model_key, all_depths, eval_images, fdirs_by_depth, device=device
        )
        all_depth_records.extend(depth_rec)

        # 8. Spatial Mode Structure (on DeiT-Small)
        if model_key == "deit_small":
            print("Analyzing spatial mode structure...")
            h_peak = clean_collected[peak_depth]
            v_top = fdirs_peak["jac_top"].to(device)
            _, _, _, Vh_peak = construct_token_transmission_operator(
                model, model_key, peak_depth, h_peak, v_top, device=device
            )
            spatial_rec = analyze_spatial_mode_structure(
                Vh_peak, num_patches, model_key, peak_depth
            )
            all_spatial_records.extend(spatial_rec)

    # -------------------------------------------------------------------------
    # Confirmatory Replication on DeiT-Tiny and DINOv2
    # -------------------------------------------------------------------------
    print(f"\n====================================================================")
    print(f"RUNNING CONFIRMATORY REPLICATION ON DEIT-TINY AND DINOV2")
    print(f"====================================================================")
    rep_configs = [
        ("deit_tiny", 8),
        ("dinov2", 8)
    ]
    all_rep_records = []

    for model_key, depth in rep_configs:
        print(f"\n>>> Replicating singular mode predictions on {model_key} at Depth {depth}...")
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
                model, model_key, x=eval_images, start_depth=0, collect_depths=(depth,)
            )
            h_d = clean_collected[depth]
            patches = h_d[:, 1:, :].reshape(-1, embed_dim).to(torch.float64)
            centroid = patches.mean(dim=0).float()
            diff = patches - centroid.unsqueeze(0).to(torch.float64)
            cov_matrix = (diff.T @ diff) / (patches.shape[0] - 1)
            cov_matrix = cov_matrix.float()
            sigma_P = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()

        metric_M = estimate_functional_metric(
            model, model_key, depth, eval_images[:4], device=device
        )
        fdirs = construct_joint_feature_directions(
            cov_matrix, centroid, metric_M, embed_dim
        )

        _, rep_val = evaluate_singular_modes(
            model, model_key, depth, eval_images, eval_targets,
            clean_logits, fdirs, sigma_P=sigma_P, scale_s=1.0, device=device
        )
        # Filter for jac_top
        rep_val_filtered = [r for r in rep_val if r["feature_dir"] == "jac_top"]
        all_rep_records.extend(rep_val_filtered)

    # -------------------------------------------------------------------------
    # Persist Machine-Readable Datasets
    # -------------------------------------------------------------------------
    print(f"\nPersisting machine-readable datasets to {output_dir}...")
    df_spectrum = pd.DataFrame(all_spectrum_records)
    df_val = pd.DataFrame(all_validation_records)
    df_rand = pd.DataFrame(all_random_records)
    df_curves = pd.DataFrame(all_curves_records)
    df_depth = pd.DataFrame(all_depth_records)
    df_ft = pd.DataFrame(all_feat_token_records)
    df_rep = pd.DataFrame(all_rep_records)
    df_spatial = pd.DataFrame(all_spatial_records) if all_spatial_records else None

    df_spectrum.to_csv(os.path.join(output_dir, "operator_spectrum.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "singular_mode_validation.csv"), index=False)
    df_rand.to_csv(os.path.join(output_dir, "random_pattern_prediction.csv"), index=False)
    df_curves.to_csv(os.path.join(output_dir, "finite_radius_curves.csv"), index=False)
    df_depth.to_csv(os.path.join(output_dir, "depth_spectrum.csv"), index=False)
    df_ft.to_csv(os.path.join(output_dir, "feature_token_matrix.csv"), index=False)
    df_rep.to_csv(os.path.join(output_dir, "replication_summary.csv"), index=False)
    if df_spatial is not None:
        df_spatial.to_csv(os.path.join(output_dir, "spatial_mode_analysis.csv"), index=False)

    # Generate Publication Figures
    print(f"Generating publication figures in {figures_dir}...")
    generate_predictive_operator_figures(
        df_spectrum, df_val, df_rand, df_curves, df_depth, df_rep, df_spatial, figures_dir
    )

    # Compute Summary Statistics
    # 1. Pearson r and Spearman rho between predicted transmission and observed logit L2 on random patterns
    rand_corr = {}
    for mk in ["deit_small", "vit_base"]:
        sub_rand = df_rand[df_rand["model_key"] == mk]
        if not sub_rand.empty:
            p_r = float(np.corrcoef(sub_rand["predicted_transmission"], sub_rand["logit_l2"])[0, 1])
            s_rho = float(sub_rand["predicted_transmission"].corr(sub_rand["logit_l2"], method="spearman"))
            rand_corr[mk] = {"pearson_r": p_r, "spearman_rho": s_rho}

    # 2. Contrast ratio: Top mode vs Null mode damage
    top_null_contrast = {}
    for mk in ["deit_small", "vit_base"]:
        sub_v = df_val[(df_val["model_key"] == mk) & (df_val["feature_dir"] == "jac_top")]
        if not sub_v.empty:
            l2_top = sub_v[sub_v["mode_label"] == "top_mode"]["logit_l2"].values[0]
            l2_null = sub_v[sub_v["mode_label"] == "bottom_mode"]["logit_l2"].values[0]
            top_null_contrast[mk] = {
                "l2_top": float(l2_top),
                "l2_null": float(l2_null),
                "contrast_ratio": float(l2_top / (l2_null + 1e-10))
            }

    elapsed = time.time() - start_time
    manifest = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
        "device": device_str,
        "exploratory_configs": exploratory_configs,
        "replication_configs": rep_configs,
        "num_eval_images": num_eval_images,
        "random_pattern_correlations": rand_corr,
        "top_vs_null_contrast": top_null_contrast
    }

    with open(os.path.join(output_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n====================================================================")
    print(f"TOKEN-TRANSMISSION OPERATOR EXPERIMENT COMPLETED IN {elapsed:.2f}s")
    print(f"Manifest saved to: {os.path.join(output_dir, 'validation_manifest.json')}")
    print(f"====================================================================")


if __name__ == "__main__":
    run_full_predictive_operator_experiment()
