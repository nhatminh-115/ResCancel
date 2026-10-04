"""
Runner script for Multi-Block Operator and Downstream Low-Transmission Geometry Suite.

Executes:
1. Multi-block Jacobian J_{l->L} construction across DeiT-Small, ViT-Base, DeiT-Tiny, DINOv2
2. Mode extraction (multi_top, multi_null, single_top, single_null, random)
3. Null-space rotation measurement across downstream blocks
4. Block-by-block leakage tracing
5. Finite-radius curves (s in [0.0, 4.0])
6. Out-of-sample held-out prediction (M=100) comparing tau_single vs tau_multi
7. Power-law scaling fitting (Damage ~ alpha^p)
8. Historical intervention projections
9. Depth comparison (depths 2, 8, 11)
10. Cross-architecture replication summary table
11. Figure generation
"""

import os
import sys
sys.path.insert(0, ".")
import time
import json
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset

from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.full_fungibility_operator import construct_full_operator
from patch_fungibility.multiblock_operator import (
    construct_downstream_jacobian,
    extract_multiblock_modes,
    evaluate_nullspace_rotation,
    evaluate_downstream_leakage_trace,
    evaluate_finite_radius_curves,
    evaluate_heldout_prediction,
    evaluate_scaling_exponents,
    evaluate_historical_interventions,
    evaluate_depth_comparison,
    generate_multiblock_figures
)


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Multi-Block Operator Suite on: {device}")
    
    out_dir = "outputs/fungibility_multiblock_operator"
    fig_dir = "figures/fungibility_multiblock_operator"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    calib_ds, _, _, _ = get_disjoint_imagenet_splits()
    
    # =============================================================
    # PHASE 1: PRIMARY DECISIVE MODEL (DEIT-SMALL AT DEPTH 8)
    # =============================================================
    print("\n" + "="*70)
    print("PHASE 1: DEI-T-SMALL AT DEPTH 8 (PRIMARY DECISIVE ARCHITECTURE)")
    print("="*70)
    model_deit, transform_deit, _ = load_model_and_transform("deit_small", device)
    calib_ds.transform = transform_deit
    loader_deit = DataLoader(Subset(calib_ds, list(range(20))), batch_size=20, shuffle=False)
    for eval_imgs_deit, eval_tgts_deit in loader_deit:
        break
    eval_imgs_deit = eval_imgs_deit.to(device)
    eval_tgts_deit = eval_tgts_deit.to(device)
    
    clean_logits_deit, clean_collected_deit = forward_block_by_block(
        model_deit, "deit_small", x=eval_imgs_deit, start_depth=0, collect_depths=tuple(range(8, 13))
    )
    h_clean_deit = clean_collected_deit[8]
    
    # 1.1 Single-block operator A_8
    t0 = time.time()
    op_single_deit = construct_full_operator(model_deit, "deit_small", 8, h_clean_deit, device)
    print(f"Constructed single-block A_8 in {time.time() - t0:.4f}s")
    
    # 1.2 Multi-block Jacobian J_{8->12}
    t1 = time.time()
    op_multi_deit = construct_downstream_jacobian(model_deit, "deit_small", 8, h_clean_deit, device)
    print(f"Constructed multi-block J_{{8->12}} of shape {op_multi_deit['J'].shape} in {time.time() - t1:.4f}s")
    print(f"Top 5 singular values of J: {op_multi_deit['S_J'][:5].cpu().numpy()}")
    print(f"Bottom 5 singular values of J: {op_multi_deit['S_J'][-5:].cpu().numpy()}")
    
    # 1.3 Mode extraction
    modes_deit = extract_multiblock_modes(op_multi_deit, op_single_deit, seed=42)
    print(f"Extracted {len(modes_deit)} modes (multi_top, multi_null, single_top, single_null, random)")
    
    # 1.4 Single vs Multi Null Survival Analysis
    print("Measuring survival of single-block nullity in multi-block Jacobian...")
    tau_single_on_single = torch.norm(op_single_deit['A_l'] @ modes_deit['single_null'].reshape(-1)).item()
    tau_single_on_multi = torch.norm(op_multi_deit['J'] @ modes_deit['single_null'].reshape(-1)).item()
    tau_multi_on_multi = torch.norm(op_multi_deit['J'] @ modes_deit['multi_null'].reshape(-1)).item()
    tau_multi_top = torch.norm(op_multi_deit['J'] @ modes_deit['multi_top'].reshape(-1)).item()
    survival_ratio = tau_single_on_multi / (tau_multi_top + 1e-12)
    
    df_survival = pd.DataFrame([{
        "arch": "deit_small",
        "depth": 8,
        "single_null_transmission_through_single_A": tau_single_on_single,
        "single_null_transmission_through_multi_J": tau_single_on_multi,
        "multi_null_transmission_through_multi_J": tau_multi_on_multi,
        "multi_top_transmission_through_multi_J": tau_multi_top,
        "leakage_ratio_single_null": survival_ratio
    }])
    df_survival.to_csv(os.path.join(out_dir, "single_vs_multi_null.csv"), index=False)
    print(f"Single-block null transmission through J: {tau_single_on_multi:.6f}")
    print(f"Multi-block null transmission through J:  {tau_multi_on_multi:.6e}")
    print(f"Leakage ratio of single null: {survival_ratio:.4f}")
    
    # 1.5 Null-Space Rotation across Downstream Blocks
    print("Measuring null-space rotation between adjacent blocks...")
    df_rot_deit = evaluate_nullspace_rotation(model_deit, "deit_small", 8, clean_collected_deit, device)
    df_rot_deit.to_csv(os.path.join(out_dir, "nullspace_rotation.csv"), index=False)
    print(df_rot_deit[["block_b", "block_b_plus_1", "top_principal_cosine", "mean_principal_cosine", "mean_principal_angle_deg"]])
    
    # 1.6 Downstream Leakage Trace
    print("Tracing block-by-block leakage...")
    df_leak_deit = evaluate_downstream_leakage_trace(model_deit, "deit_small", 8, clean_collected_deit, modes_deit, device)
    df_leak_deit.to_csv(os.path.join(out_dir, "leakage_trace.csv"), index=False)
    
    # 1.7 Finite-Radius Response Curves
    print("Evaluating finite-radius curves...")
    df_curves_deit = evaluate_finite_radius_curves(
        model_deit, "deit_small", 8, clean_logits_deit, h_clean_deit, modes_deit, eval_tgts_deit, op_multi_deit["sigma_P"], device
    )
    
    # 1.8 Out-of-Sample Prediction (M=100)
    print("Evaluating M=100 held-out perturbations (Single vs Multi predictor)...")
    df_pred_deit = evaluate_heldout_prediction(
        model_deit, "deit_small", 8, clean_logits_deit, h_clean_deit, op_multi_deit, op_single_deit, eval_tgts_deit, device, M=100, seed=42
    )
    df_pred_deit.to_csv(os.path.join(out_dir, "random_prediction.csv"), index=False)
    r_multi, _ = pearsonr(df_pred_deit["predicted_tau_multi"], df_pred_deit["logit_l2"])
    rho_multi, _ = spearmanr(df_pred_deit["predicted_tau_multi"], df_pred_deit["logit_l2"])
    r_single, _ = pearsonr(df_pred_deit["predicted_tau_single"], df_pred_deit["logit_l2"])
    rho_single, _ = spearmanr(df_pred_deit["predicted_tau_single"], df_pred_deit["logit_l2"])
    print(f"Prediction: Multi-Block r = {r_multi:.4f} (rho = {rho_multi:.4f}) vs Single-Block r = {r_single:.4f} (rho = {rho_single:.4f})")
    
    # 1.9 Power-Law Scaling Exponents
    print("Fitting power-law scaling exponents (Damage ~ alpha^p)...")
    df_scaling_summary, _ = evaluate_scaling_exponents(
        model_deit, "deit_small", 8, clean_logits_deit, h_clean_deit, modes_deit, device
    )
    df_scaling_summary.to_csv(os.path.join(out_dir, "scaling_exponents.csv"), index=False)
    print(df_scaling_summary)
    
    # 1.10 Historical Intervention Projection
    print("Projecting historical interventions onto multi-block spectrum...")
    df_hist = evaluate_historical_interventions(op_multi_deit, op_single_deit, h_clean_deit, device)
    df_hist.to_csv(os.path.join(out_dir, "historical_intervention_projection.csv"), index=False)
    print(df_hist[["intervention", "tau_multi", "tau_single", "multi_high_energy_pct", "multi_null_energy_pct"]])
    
    # 1.11 Depth Comparison (Depths 2, 8, 11)
    print("Evaluating depth comparison across depths 2, 8, 11...")
    df_depth_deit, df_spec_deit = evaluate_depth_comparison(
        model_deit, "deit_small", [2, 8, 11], eval_imgs_deit, device
    )
    
    # =============================================================
    # PHASE 2: SECONDARY EXPLORATORY (VIT-BASE AT DEPTH 8)
    # =============================================================
    print("\n" + "="*70)
    print("PHASE 2: VIT-BASE AT DEPTH 8")
    print("="*70)
    model_vitb, transform_vitb, _ = load_model_and_transform("vit_base", device)
    calib_ds.transform = transform_vitb
    loader_vitb = DataLoader(Subset(calib_ds, list(range(20))), batch_size=20, shuffle=False)
    for eval_imgs_vitb, eval_tgts_vitb in loader_vitb:
        break
    eval_imgs_vitb = eval_imgs_vitb.to(device)
    eval_tgts_vitb = eval_tgts_vitb.to(device)
    
    clean_logits_vitb, clean_collected_vitb = forward_block_by_block(
        model_vitb, "vit_base", x=eval_imgs_vitb, start_depth=0, collect_depths=tuple(range(8, 13))
    )
    h_clean_vitb = clean_collected_vitb[8]
    op_single_vitb = construct_full_operator(model_vitb, "vit_base", 8, h_clean_vitb, device)
    op_multi_vitb = construct_downstream_jacobian(model_vitb, "vit_base", 8, h_clean_vitb, device)
    print(f"Constructed ViT-Base multi-block J of shape {op_multi_vitb['J'].shape}")
    modes_vitb = extract_multiblock_modes(op_multi_vitb, op_single_vitb, seed=42)
    df_curves_vitb = evaluate_finite_radius_curves(
        model_vitb, "vit_base", 8, clean_logits_vitb, h_clean_vitb, modes_vitb, eval_tgts_vitb, op_multi_vitb["sigma_P"], device
    )
    df_depth_vitb, df_spec_vitb = evaluate_depth_comparison(
        model_vitb, "vit_base", [2, 8, 11], eval_imgs_vitb, device
    )
    
    # =============================================================
    # PHASE 3: DECISIVE REPLICATION (DEIT-TINY & DINOV2)
    # =============================================================
    print("\n" + "="*70)
    print("PHASE 3: DECISIVE REPLICATION (DEIT-TINY & DINOV2)")
    print("="*70)
    # 3.1 DeiT-Tiny
    print("Replicating on DeiT-Tiny...")
    model_tiny, transform_tiny, _ = load_model_and_transform("deit_tiny", device)
    calib_ds.transform = transform_tiny
    loader_tiny = DataLoader(Subset(calib_ds, list(range(20))), batch_size=20, shuffle=False)
    for eval_imgs_tiny, eval_tgts_tiny in loader_tiny:
        break
    eval_imgs_tiny = eval_imgs_tiny.to(device)
    eval_tgts_tiny = eval_tgts_tiny.to(device)
    
    clean_logits_tiny, clean_collected_tiny = forward_block_by_block(
        model_tiny, "deit_tiny", x=eval_imgs_tiny, start_depth=0, collect_depths=tuple(range(8, 13))
    )
    h_clean_tiny = clean_collected_tiny[8]
    op_single_tiny = construct_full_operator(model_tiny, "deit_tiny", 8, h_clean_tiny, device)
    op_multi_tiny = construct_downstream_jacobian(model_tiny, "deit_tiny", 8, h_clean_tiny, device)
    modes_tiny = extract_multiblock_modes(op_multi_tiny, op_single_tiny, seed=42)
    df_curves_tiny = evaluate_finite_radius_curves(
        model_tiny, "deit_tiny", 8, clean_logits_tiny, h_clean_tiny, modes_tiny, eval_tgts_tiny, op_multi_tiny["sigma_P"], device
    )
    
    # 3.2 DINOv2
    print("Replicating on DINOv2 (Dual CLS + mean patch pooling readout)...")
    model_dino, transform_dino, _ = load_model_and_transform("dinov2", device)
    calib_ds.transform = transform_dino
    loader_dino = DataLoader(Subset(calib_ds, list(range(20))), batch_size=20, shuffle=False)
    for eval_imgs_dino, eval_tgts_dino in loader_dino:
        break
    eval_imgs_dino = eval_imgs_dino.to(device)
    eval_tgts_dino = eval_tgts_dino.to(device)
    
    clean_logits_dino, clean_collected_dino = forward_block_by_block(
        model_dino, "dinov2", x=eval_imgs_dino, start_depth=0, collect_depths=tuple(range(8, 13))
    )
    h_clean_dino = clean_collected_dino[8]
    op_single_dino = construct_full_operator(model_dino, "dinov2", 8, h_clean_dino, device)
    op_multi_dino = construct_downstream_jacobian(model_dino, "dinov2", 8, h_clean_dino, device)
    modes_dino = extract_multiblock_modes(op_multi_dino, op_single_dino, seed=42)
    df_curves_dino = evaluate_finite_radius_curves(
        model_dino, "dinov2", 8, clean_logits_dino, h_clean_dino, modes_dino, eval_tgts_dino, op_multi_dino["sigma_P"], device
    )
    
    # =============================================================
    # COMBINE DATA AND GENERATE DELIVERABLES
    # =============================================================
    df_all_curves = pd.concat([df_curves_deit, df_curves_vitb, df_curves_tiny, df_curves_dino], ignore_index=True)
    df_all_curves.to_csv(os.path.join(out_dir, "finite_radius_curves.csv"), index=False)
    
    df_all_depth = pd.concat([df_depth_deit, df_depth_vitb], ignore_index=True)
    df_all_depth.to_csv(os.path.join(out_dir, "depth_comparison.csv"), index=False)
    
    df_all_spec = pd.concat([df_spec_deit, df_spec_vitb], ignore_index=True)
    df_all_spec.to_csv(os.path.join(out_dir, "multiblock_spectrum.csv"), index=False)
    
    # Cross-Architecture Replication Summary Table
    rep_rows = []
    for arch, df_c in [("deit_small", df_curves_deit), ("vit_base", df_curves_vitb), 
                       ("deit_tiny", df_curves_tiny), ("dinov2", df_curves_dino)]:
        multi_top_s04 = df_c[(df_c["mode_name"] == "multi_top") & (df_c["scale_factor"] == 0.4)]["logit_l2"].values[0]
        single_null_s04 = df_c[(df_c["mode_name"] == "single_null") & (df_c["scale_factor"] == 0.4)]["logit_l2"].values[0]
        multi_null_s04 = df_c[(df_c["mode_name"] == "multi_null") & (df_c["scale_factor"] == 0.4)]["logit_l2"].values[0]
        rand_s04 = df_c[(df_c["mode_name"] == "random_gaussian") & (df_c["scale_factor"] == 0.4)]["logit_l2"].values[0]
        
        ratio_multi_to_null = multi_top_s04 / (multi_null_s04 + 1e-12)
        ratio_single_to_multi_null = single_null_s04 / (multi_null_s04 + 1e-12)
        
        rep_rows.append({
            "arch": arch,
            "multi_top_damage_s04": multi_top_s04,
            "single_null_damage_s04": single_null_s04,
            "multi_null_damage_s04": multi_null_s04,
            "random_damage_s04": rand_s04,
            "multi_top_to_multi_null_ratio": ratio_multi_to_null,
            "single_null_to_multi_null_ratio": ratio_single_to_multi_null,
            "persistent_null_advantage": "CONFIRMED" if ratio_single_to_multi_null > 1.15 else "MARGINAL"
        })
    df_rep = pd.DataFrame(rep_rows)
    df_rep.to_csv(os.path.join(out_dir, "replication_summary.csv"), index=False)
    print("\nReplication Summary Table:")
    print(df_rep)
    
    # Validation Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": time.time() - start_time,
        "models_evaluated": ["deit_small", "vit_base", "deit_tiny", "dinov2"],
        "depths_evaluated": [2, 8, 11],
        "scale_sweep": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0],
        "num_heldout_perturbations": 100,
        "prediction_correlations": {
            "multi_block_pearson_r": float(r_multi),
            "multi_block_spearman_rho": float(rho_multi),
            "single_block_pearson_r": float(r_single),
            "single_block_spearman_rho": float(rho_single)
        },
        "primary_findings": {
            "single_null_to_multi_null_ratio_deit_small": float(df_rep[df_rep['arch'] == 'deit_small']['single_null_to_multi_null_ratio'].values[0]),
            "multi_top_to_multi_null_ratio_deit_small": float(df_rep[df_rep['arch'] == 'deit_small']['multi_top_to_multi_null_ratio'].values[0]),
            "mean_principal_cosine_b8_to_b9": float(df_rot_deit.iloc[0]['mean_principal_cosine']),
            "mean_principal_angle_deg_b8_to_b9": float(df_rot_deit.iloc[0]['mean_principal_angle_deg']),
            "coherent_pc1_tau_multi": float(df_hist[df_hist['intervention'] == 'Global Coherent (PC1)']['tau_multi'].values[0]),
            "checkerboard_tau_multi": float(df_hist[df_hist['intervention'] == 'Checkerboard (PC1)']['tau_multi'].values[0])
        }
    }
    with open(os.path.join(out_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {os.path.join(out_dir, 'validation_manifest.json')}")
    
    # Generate all 8 figures
    print("\nGenerating figures A through H...")
    generate_multiblock_figures(out_dir, fig_dir)
    print(f"\nMulti-Block Suite complete in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    main()
