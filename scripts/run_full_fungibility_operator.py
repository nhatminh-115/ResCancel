"""
Runner script for Full Fungibility Operator Experiment.

Executes:
1. Operator construction and Gram SVD across DeiT-Small, ViT-B/16, DeiT-Tiny, DINOv2
2. Mode extraction (top, mid, bottom non-zero, exact null, controls)
3. Finite-radius sweeps (s in [0.0, 4.0])
4. Random held-out perturbation prediction (M=100)
5. Mode factorization (rank-1 energy)
6. Depth evolution (fragile: 2, peak: 8, terminal: 11)
7. Canonical intervention projection
8. Residual damage breakdown and power-law scaling
9. Cross-architecture replication summary
10. Figure generation
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
from patch_fungibility.full_fungibility_operator import (
    construct_full_operator,
    extract_operator_modes,
    evaluate_mode_factorization,
    evaluate_finite_radius_curves,
    evaluate_random_held_out,
    evaluate_depth_evolution,
    evaluate_intervention_projections,
    evaluate_residual_and_scaling,
    generate_full_operator_figures
)


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Full Fungibility Operator Suite on: {device}")
    
    out_dir = "outputs/fungibility_full_operator"
    fig_dir = "figures/fungibility_full_operator"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load dataset: 20 validation images for clean state activations and fast evaluation
    calib_ds, _, _, _ = get_disjoint_imagenet_splits()
    
    # -------------------------------------------------------------
    # 1. Primary Model: DeiT-Small at Depth 8
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 1: DEI-T-SMALL AT DEPTH 8 (PRIMARY DECISIVE ARCHITECTURE)")
    print("="*70)
    model_deit, transform_deit, _ = load_model_and_transform("deit_small", device)
    calib_ds.transform = transform_deit
    loader_deit = DataLoader(Subset(calib_ds, list(range(20))), batch_size=20, shuffle=False)
    for eval_imgs, eval_tgts in loader_deit:
        break
    eval_imgs = eval_imgs.to(device)
    eval_tgts = eval_tgts.to(device)
    
    clean_logits_deit, clean_collected_deit = forward_block_by_block(
        model_deit, "deit_small", x=eval_imgs, start_depth=0, collect_depths=(8,)
    )
    h_clean_deit = clean_collected_deit[8]
    
    # Construct full operator
    t0 = time.time()
    op_deit = construct_full_operator(model_deit, "deit_small", 8, h_clean_deit, device)
    print(f"Constructed DeiT-Small A_l of shape {op_deit['A_l'].shape} in {time.time() - t0:.4f}s")
    print(f"Top 5 singular values: {op_deit['S'][:5].cpu().numpy()}")
    print(f"Bottom 5 singular values: {op_deit['S'][-5:].cpu().numpy()}")
    
    # Extract modes
    modes_deit = extract_operator_modes(op_deit, seed=42)
    print(f"Extracted {len(modes_deit)} representative modes.")
    
    # 1.1 Mode Factorization (Rank-1 energy)
    print("Computing mode factorization...")
    df_factor = evaluate_mode_factorization(op_deit, num_modes=100)
    df_factor.to_csv(os.path.join(out_dir, "mode_factorization.csv"), index=False)
    print(f"Mode 0 Rank-1 Energy: {df_factor.iloc[0]['rank1_energy_fraction']:.4f}")
    print(f"Mode 50 Rank-1 Energy: {df_factor.iloc[50]['rank1_energy_fraction']:.4f}")
    
    # 1.2 Finite Radius Curves
    print("Evaluating finite-radius curves...")
    df_curves_deit = evaluate_finite_radius_curves(
        model_deit, "deit_small", 8, clean_logits_deit, h_clean_deit, modes_deit, eval_tgts, op_deit["sigma_P"], device
    )
    
    # 1.3 Random Held-Out Perturbations (M=100)
    print("Evaluating M=100 random held-out perturbations...")
    df_heldout_deit = evaluate_random_held_out(
        model_deit, "deit_small", 8, clean_logits_deit, h_clean_deit, op_deit, eval_tgts, device, M=100, seed=42
    )
    df_heldout_deit.to_csv(os.path.join(out_dir, "random_perturbation_prediction.csv"), index=False)
    
    sub_div = df_heldout_deit[df_heldout_deit["set_type"] == "diverse_geometry"]
    r_val, _ = pearsonr(sub_div["predicted_tau"], sub_div["logit_l2"])
    rho_val, _ = spearmanr(sub_div["predicted_tau"], sub_div["logit_l2"])
    print(f"Diverse Held-Out Prediction: Pearson r = {r_val:.4f}, Spearman rho = {rho_val:.4f}")
    
    # 1.4 Historical Intervention Projections
    print("Evaluating canonical intervention projections...")
    df_proj = evaluate_intervention_projections(op_deit, h_clean_deit, device)
    df_proj.to_csv(os.path.join(out_dir, "intervention_projection.csv"), index=False)
    print(df_proj[["intervention", "high_trans_energy_pct", "low_null_energy_pct", "transmission_tau"]])
    
    # 1.5 Residual Damage and Power-Law Scaling
    print("Evaluating residual damage and scaling exponent...")
    df_resid, df_scaling, df_power = evaluate_residual_and_scaling(
        model_deit, "deit_small", 8, clean_logits_deit, h_clean_deit, modes_deit, device
    )
    df_resid.to_csv(os.path.join(out_dir, "residual_damage.csv"), index=False)
    df_scaling.to_csv(os.path.join(out_dir, "null_scaling.csv"), index=False)
    print("Scaling exponent summary:")
    print(df_power)
    
    # 1.6 Depth Evolution
    print("Evaluating depth evolution across depths 2, 8, 11...")
    df_depth_deit, df_spec_deit = evaluate_depth_evolution(
        model_deit, "deit_small", [2, 8, 11], eval_imgs, device
    )
    
    # -------------------------------------------------------------
    # 2. Secondary Exploratory: ViT-B/16 AugReg
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 2: VIT-B/16 AUGREG AT DEPTH 8")
    print("="*70)
    model_vitb, transform_vitb, _ = load_model_and_transform("vit_base", device)
    calib_ds.transform = transform_vitb
    loader_vitb = DataLoader(Subset(calib_ds, list(range(20))), batch_size=20, shuffle=False)
    for eval_imgs_b, eval_tgts_b in loader_vitb:
        break
    eval_imgs_b = eval_imgs_b.to(device)
    eval_tgts_b = eval_tgts_b.to(device)
    
    clean_logits_vitb, clean_collected_vitb = forward_block_by_block(
        model_vitb, "vit_base", x=eval_imgs_b, start_depth=0, collect_depths=(8,)
    )
    h_clean_vitb = clean_collected_vitb[8]
    op_vitb = construct_full_operator(model_vitb, "vit_base", 8, h_clean_vitb, device)
    print(f"Constructed ViT-B/16 A_l of shape {op_vitb['A_l'].shape}")
    modes_vitb = extract_operator_modes(op_vitb, seed=42)
    df_curves_vitb = evaluate_finite_radius_curves(
        model_vitb, "vit_base", 8, clean_logits_vitb, h_clean_vitb, modes_vitb, eval_tgts_b, op_vitb["sigma_P"], device
    )
    df_depth_vitb, df_spec_vitb = evaluate_depth_evolution(
        model_vitb, "vit_base", [2, 8, 11], eval_imgs_b, device
    )
    
    # -------------------------------------------------------------
    # 3. Decisive Replication: DeiT-Tiny and DINOv2 ViT-S/14
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 3: DECISIVE REPLICATION (DEIT-TINY & DINOV2)")
    print("="*70)
    # 3.1 DeiT-Tiny
    print("Replicating on DeiT-Tiny...")
    model_tiny, transform_tiny, _ = load_model_and_transform("deit_tiny", device)
    calib_ds.transform = transform_tiny
    loader_tiny = DataLoader(Subset(calib_ds, list(range(20))), batch_size=20, shuffle=False)
    for eval_imgs_t, eval_tgts_t in loader_tiny:
        break
    eval_imgs_t = eval_imgs_t.to(device)
    eval_tgts_t = eval_tgts_t.to(device)
    
    clean_logits_tiny, clean_collected_tiny = forward_block_by_block(
        model_tiny, "deit_tiny", x=eval_imgs_t, start_depth=0, collect_depths=(8,)
    )
    h_clean_tiny = clean_collected_tiny[8]
    op_tiny = construct_full_operator(model_tiny, "deit_tiny", 8, h_clean_tiny, device)
    modes_tiny = extract_operator_modes(op_tiny, seed=42)
    df_curves_tiny = evaluate_finite_radius_curves(
        model_tiny, "deit_tiny", 8, clean_logits_tiny, h_clean_tiny, modes_tiny, eval_tgts_t, op_tiny["sigma_P"], device
    )
    
    # 3.2 DINOv2
    print("Replicating on DINOv2 ViT-S/14 (Dual-channel readout)...")
    model_dino, transform_dino, _ = load_model_and_transform("dinov2", device)
    calib_ds.transform = transform_dino
    loader_dino = DataLoader(Subset(calib_ds, list(range(20))), batch_size=20, shuffle=False)
    for eval_imgs_d, eval_tgts_d in loader_dino:
        break
    eval_imgs_d = eval_imgs_d.to(device)
    eval_tgts_d = eval_tgts_d.to(device)
    
    clean_logits_dino, clean_collected_dino = forward_block_by_block(
        model_dino, "dinov2", x=eval_imgs_d, start_depth=0, collect_depths=(8,)
    )
    h_clean_dino = clean_collected_dino[8]
    op_dino = construct_full_operator(model_dino, "dinov2", 8, h_clean_dino, device)
    modes_dino = extract_operator_modes(op_dino, seed=42)
    df_curves_dino = evaluate_finite_radius_curves(
        model_dino, "dinov2", 8, clean_logits_dino, h_clean_dino, modes_dino, eval_tgts_d, op_dino["sigma_P"], device
    )
    
    # Combine finite radius curves across all models
    df_all_curves = pd.concat([df_curves_deit, df_curves_vitb, df_curves_tiny, df_curves_dino], ignore_index=True)
    df_all_curves.to_csv(os.path.join(out_dir, "finite_radius_curves.csv"), index=False)
    
    # Mode validation summary at s = 0.4
    df_s04 = df_all_curves[df_all_curves["scale_factor"] == 0.4].copy()
    df_s04.to_csv(os.path.join(out_dir, "singular_mode_validation.csv"), index=False)
    
    # Depth evolution & spectrum across models
    df_all_depth = pd.concat([df_depth_deit, df_depth_vitb], ignore_index=True)
    df_all_depth.to_csv(os.path.join(out_dir, "depth_evolution.csv"), index=False)
    
    df_all_spec = pd.concat([df_spec_deit, df_spec_vitb], ignore_index=True)
    df_all_spec.to_csv(os.path.join(out_dir, "operator_spectrum.csv"), index=False)
    
    # Cross-Architecture Replication Summary Table
    rep_rows = []
    for arch, df_c in [("deit_small", df_curves_deit), ("vit_base", df_curves_vitb), 
                       ("deit_tiny", df_curves_tiny), ("dinov2", df_curves_dino)]:
        top_s04 = df_c[(df_c["mode_name"] == "top_0") & (df_c["scale_factor"] == 0.4)]["logit_l2"].values[0]
        null_s04 = df_c[(df_c["mode_name"] == "exact_null") & (df_c["scale_factor"] == 0.4)]["logit_l2"].values[0]
        rand_s04 = df_c[(df_c["mode_name"] == "random_gaussian") & (df_c["scale_factor"] == 0.4)]["logit_l2"].values[0]
        ratio = top_s04 / (null_s04 + 1e-12)
        rep_rows.append({
            "arch": arch,
            "top_damage_s04": top_s04,
            "null_damage_s04": null_s04,
            "random_damage_s04": rand_s04,
            "top_to_null_damage_ratio": ratio,
            "null_advantage": "TOLERANT" if ratio > 2.0 else "FRAGILE"
        })
    df_rep = pd.DataFrame(rep_rows)
    df_rep.to_csv(os.path.join(out_dir, "replication_summary.csv"), index=False)
    print("\nReplication Summary Table:")
    print(df_rep)
    
    # -------------------------------------------------------------
    # 4. Generate Validation Manifest
    # -------------------------------------------------------------
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": time.time() - start_time,
        "models_evaluated": ["deit_small", "vit_base", "deit_tiny", "dinov2"],
        "depths_evaluated": [2, 8, 11],
        "scale_sweep": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0],
        "num_heldout_perturbations": 100,
        "diverse_heldout_r": float(r_val),
        "diverse_heldout_rho": float(rho_val),
        "primary_findings": {
            "top_to_null_ratio_deit_small": float(df_rep[df_rep['arch'] == 'deit_small']['top_to_null_damage_ratio'].values[0]),
            "top_mode_rank1_energy_fraction": float(df_factor.iloc[0]['rank1_energy_fraction']),
            "mid_mode_rank1_energy_fraction": float(df_factor.iloc[50]['rank1_energy_fraction']),
            "exact_null_space_dimension_pct": float(df_depth_deit[df_depth_deit['depth'] == 8]['null_dim_pct'].values[0]),
            "coherent_pc1_tau": float(df_proj[df_proj['intervention'] == 'Global Coherent (PC1)']['transmission_tau'].values[0]),
            "checkerboard_tau": float(df_proj[df_proj['intervention'] == 'Checkerboard (PC1)']['transmission_tau'].values[0])
        }
    }
    with open(os.path.join(out_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {os.path.join(out_dir, 'validation_manifest.json')}")
    
    # -------------------------------------------------------------
    # 5. Generate Figures A through H
    # -------------------------------------------------------------
    print("\nGenerating figures A through H...")
    generate_full_operator_figures(out_dir, fig_dir)
    print(f"\nSuite complete in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    main()
