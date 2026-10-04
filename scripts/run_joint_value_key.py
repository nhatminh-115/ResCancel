"""
Runner Script for Joint Value-Key Downstream Low-Transmission Geometry

Executes complete experimental suite:
1. Operator construction (Value V_l, Key R_l, Joint C_l, Cumulative C_{l->L})
2. Spectrum analysis, effective rank, and principal angle subspace overlap
3. Finite-radius full-model evaluations (damage and attention shift metrics)
4. Power-law scaling fitting (p and intercept c)
5. Blockwise downstream rerouting trace
6. Out-of-sample prediction comparison (M=100 held-out perturbations)
7. Depth analysis (depths 2, 8, 11)
8. Cross-architecture replication (DeiT-Small, ViT-Base, DeiT-Tiny, DINOv2)
9. Generates all 12 CSV/JSON deliverables and all 8 publication figures
"""

import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.multiblock_operator import construct_downstream_jacobian
from patch_fungibility.joint_value_key import (
    construct_value_and_key_operators,
    construct_cumulative_multiblock_joint_operator,
    extract_all_modes,
    evaluate_finite_radius_and_attention,
    fit_power_law_scaling,
    trace_blockwise_rerouting,
    evaluate_held_out_predictions
)
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from torch.utils.data import DataLoader, Subset


def run_joint_value_key_pipeline():
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"RUNNING JOINT VALUE-KEY DOWNSTREAM PIPELINE ON DEVICE: {device}")
    print("=" * 80)
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "fungibility_joint_value_key"))
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "fungibility_joint_value_key"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Dataset & Image Setup
    # -------------------------------------------------------------
    print("\n[Step 1/8] Loading dataset and test images...")
    calib_ds, _, _, _ = get_disjoint_imagenet_splits()
    
    # Primary Model: DeiT-Small
    primary_arch = "deit_small"
    primary_depth = 8
    model_s, transform_s, meta_s = load_model_and_transform(primary_arch, device)
    model_s.eval()
    
    calib_ds.transform = transform_s
    sample_indices = list(range(10))
    subset = Subset(calib_ds, sample_indices)
    loader = DataLoader(subset, batch_size=len(sample_indices), shuffle=False)
    batch_imgs, batch_targets = next(iter(loader))
    batch_imgs = batch_imgs.to(device)
    batch_targets = batch_targets.to(device)
    
    # Capture clean activations for DeiT-Small
    clean_logits_s, clean_acts_s = forward_block_by_block(
        model_s, primary_arch, x=batch_imgs, collect_depths=tuple(range(13))
    )
    clean_acts_8 = clean_acts_s[primary_depth]
    
    # -------------------------------------------------------------
    # 2. Construct Value, Key, Joint, and Downstream Operators
    # -------------------------------------------------------------
    print("\n[Step 2/8] Constructing Value (V_8), Key (R_8), Joint (C_8), and Downstream Jacobian (J_{8->12})...")
    t0 = time.time()
    op_vk_8 = construct_value_and_key_operators(model_s, primary_arch, primary_depth, clean_acts_8, device)
    op_multi_8 = construct_downstream_jacobian(model_s, primary_arch, primary_depth, clean_acts_8, device)
    op_cum_8 = construct_cumulative_multiblock_joint_operator(model_s, primary_arch, primary_depth, clean_acts_s, op_multi_8, device)
    print(f"Operators constructed in {time.time() - t0:.2f}s.")
    print(f"  ||V_8||_F: {op_vk_8['norm_V']:.4f} | Top sigma_1(V): {op_vk_8['S_V'][0].item():.4f} | EffRank: {op_vk_8['eff_rank_V']:.1f}")
    print(f"  ||R_8||_F: {op_vk_8['norm_R']:.4f} | Top sigma_1(R): {op_vk_8['S_R'][0].item():.4f} | EffRank: {op_vk_8['eff_rank_R']:.1f}")
    print(f"  ||J_{{8->12}}||_F: {torch.norm(op_multi_8['J']).item():.4f} | Top sigma_1(J): {op_multi_8['S_J'][0].item():.4f}")
    print(f"  RowSpace Overlap: Mean Cosine = {op_vk_8['mean_cos']:.4f} ({op_vk_8['mean_angle']:.1f} deg) | Top Cosine = {op_vk_8['top_cos']:.4f} ({op_vk_8['top_angle']:.1f} deg)")
    
    # Save Spectra CSVs
    df_spec_v = pd.DataFrame({
        "rank_k": np.arange(1, len(op_vk_8["S_V"]) + 1),
        "singular_value": op_vk_8["S_V"].cpu().numpy(),
        "normalized_energy": (op_vk_8["S_V"]**2 / torch.sum(op_vk_8["S_V"]**2)).cpu().numpy(),
        "cumulative_energy": (torch.cumsum(op_vk_8["S_V"]**2, dim=0) / torch.sum(op_vk_8["S_V"]**2)).cpu().numpy()
    })
    df_spec_v.to_csv(os.path.join(out_dir, "value_operator_spectrum.csv"), index=False)
    
    df_spec_r = pd.DataFrame({
        "rank_k": np.arange(1, len(op_vk_8["S_R"]) + 1),
        "singular_value": op_vk_8["S_R"].cpu().numpy(),
        "normalized_energy": (op_vk_8["S_R"]**2 / torch.sum(op_vk_8["S_R"]**2)).cpu().numpy(),
        "cumulative_energy": (torch.cumsum(op_vk_8["S_R"]**2, dim=0) / torch.sum(op_vk_8["S_R"]**2)).cpu().numpy()
    })
    df_spec_r.to_csv(os.path.join(out_dir, "key_operator_spectrum.csv"), index=False)
    
    df_spec_c = pd.DataFrame({
        "rank_k": np.arange(1, len(op_vk_8["S_C"]) + 1),
        "singular_value": op_vk_8["S_C"].cpu().numpy(),
        "normalized_energy": (op_vk_8["S_C"]**2 / torch.sum(op_vk_8["S_C"]**2)).cpu().numpy(),
        "cumulative_energy": (torch.cumsum(op_vk_8["S_C"]**2, dim=0) / torch.sum(op_vk_8["S_C"]**2)).cpu().numpy()
    })
    df_spec_c.to_csv(os.path.join(out_dir, "joint_operator_spectrum.csv"), index=False)
    
    df_overlap = pd.DataFrame({
        "principal_index": np.arange(1, len(op_vk_8["s_overlap"]) + 1),
        "principal_cosine": op_vk_8["s_overlap"],
        "principal_angle_deg": np.arccos(np.clip(op_vk_8["s_overlap"], 0.0, 1.0)) * 180.0 / np.pi
    })
    df_overlap.to_csv(os.path.join(out_dir, "vk_subspace_overlap.csv"), index=False)
    
    # -------------------------------------------------------------
    # 3. Extract Modes & Finite-Radius Evaluation
    # -------------------------------------------------------------
    print("\n[Step 3/8] Extracting modes and evaluating finite-radius curves & attention shifts...")
    modes = extract_all_modes(op_vk_8, op_multi_8, op_cum_8, seed=42)
    
    df_eval, df_attn = evaluate_finite_radius_and_attention(
        model_s, primary_arch, primary_depth, clean_acts_s, clean_logits_s, modes, batch_targets, device
    )
    df_eval.to_csv(os.path.join(out_dir, "finite_radius_curves.csv"), index=False)
    df_attn.to_csv(os.path.join(out_dir, "attention_shift.csv"), index=False)
    
    # -------------------------------------------------------------
    # 4. Power-Law Scaling Fitting
    # -------------------------------------------------------------
    print("\n[Step 4/8] Fitting power-law scaling exponents p...")
    df_scaling = fit_power_law_scaling(df_eval)
    df_scaling.to_csv(os.path.join(out_dir, "scaling_exponents.csv"), index=False)
    print("Scaling exponent results:")
    for _, row in df_scaling.iterrows():
        print(f"  {row['mode_name']:25s} | p = {row['scaling_exponent_p']:.4f} | c = {row['intercept_c']:.4f} | R^2 = {row['r_squared']:.4f}")
        
    # -------------------------------------------------------------
    # 5. Blockwise Downstream Rerouting Trace
    # -------------------------------------------------------------
    print("\n[Step 5/8] Tracing blockwise downstream rerouting and query disturbance...")
    df_block_trace = trace_blockwise_rerouting(model_s, primary_arch, primary_depth, clean_acts_s, modes, device)
    df_block_trace.to_csv(os.path.join(out_dir, "blockwise_rerouting.csv"), index=False)
    
    # -------------------------------------------------------------
    # 6. Held-Out Out-of-Sample Predictions (M=100)
    # -------------------------------------------------------------
    print("\n[Step 6/8] Evaluating held-out random perturbations (M=100)...")
    df_pred, corr_summary = evaluate_held_out_predictions(
        model_s, primary_arch, primary_depth, clean_acts_8, clean_logits_s, op_vk_8, op_multi_8, device, M=100, seed=123
    )
    df_pred.to_csv(os.path.join(out_dir, "random_prediction.csv"), index=False)
    print("Prediction Correlations:")
    print(f"  Value Operator:   r = {corr_summary['r_value']:.4f}, rho = {corr_summary['rho_value']:.4f}, R^2 = {corr_summary['r2_value']:.4f}")
    print(f"  Key Operator:     r = {corr_summary['r_key']:.4f}, rho = {corr_summary['rho_key']:.4f}, R^2 = {corr_summary['r2_key']:.4f}")
    print(f"  Joint VK Operator: r = {corr_summary['r_joint']:.4f}, rho = {corr_summary['rho_joint']:.4f}, R^2 = {corr_summary['r2_joint']:.4f}")
    print(f"  Multi-Block J:    r = {corr_summary['r_multi']:.4f}, rho = {corr_summary['rho_multi']:.4f}, R^2 = {corr_summary['r2_multi']:.4f}")

    # -------------------------------------------------------------
    # 7. Depth Comparison (Depths 2, 8, 11)
    # -------------------------------------------------------------
    print("\n[Step 7/8] Comparing depths (Early: 2, Peak: 8, Late: 11)...")
    depth_rows = []
    for d in [2, 8, 11]:
        clean_d = clean_acts_s[d]
        op_vk_d = construct_value_and_key_operators(model_s, primary_arch, d, clean_d, device)
        op_j_d = construct_downstream_jacobian(model_s, primary_arch, d, clean_d, device)
        
        # Damage of top joint vs joint null at s=0.4
        modes_d = extract_all_modes(op_vk_d, op_j_d, seed=42)
        alpha_d = 0.4 * op_vk_d["sigma_P"]
        
        hp_top = clean_d.clone(); hp_top[:, 1:, :] += (alpha_d * modes_d["top_high_transmission"]).unsqueeze(0)
        pert_l_top, _ = forward_block_by_block(model_s, primary_arch, h_start=hp_top, start_depth=d)
        dam_top = torch.norm(pert_l_top - clean_logits_s).item()
        
        hp_null = clean_d.clone(); hp_null[:, 1:, :] += (alpha_d * modes_d["joint_vk_null"]).unsqueeze(0)
        pert_l_null, _ = forward_block_by_block(model_s, primary_arch, h_start=hp_null, start_depth=d)
        dam_null = torch.norm(pert_l_null - clean_logits_s).item()
        
        depth_rows.append({
            "arch": primary_arch,
            "depth": d,
            "norm_V": op_vk_d["norm_V"],
            "norm_R": op_vk_d["norm_R"],
            "ratio_R_to_V": op_vk_d["norm_R"] / op_vk_d["norm_V"],
            "top_sigma_V": op_vk_d["S_V"][0].item(),
            "top_sigma_R": op_vk_d["S_R"][0].item(),
            "top_sigma_J": op_j_d["S_J"][0].item(),
            "mean_principal_angle_deg": op_vk_d["mean_angle"],
            "damage_top_mode": dam_top,
            "damage_joint_null": dam_null,
            "damage_ratio": dam_top / (dam_null + 1e-12)
        })
    df_depth = pd.DataFrame(depth_rows)
    df_depth.to_csv(os.path.join(out_dir, "depth_comparison.csv"), index=False)
    
    # -------------------------------------------------------------
    # 8. Cross-Architecture Replication
    # -------------------------------------------------------------
    print("\n[Step 8/8] Replicating across architectures (ViT-Base, DeiT-Tiny, DINOv2)...")
    rep_rows = []
    
    # Add primary model DeiT-Small
    rep_rows.append({
        "arch": "deit_small",
        "primary_depth": 8,
        "input_dim_ND": op_vk_8["total_dim"],
        "readout_dim": op_vk_8["D_readout"],
        "norm_V": op_vk_8["norm_V"],
        "norm_R": op_vk_8["norm_R"],
        "mean_principal_angle_deg": op_vk_8["mean_angle"],
        "damage_top": depth_rows[1]["damage_top_mode"],
        "damage_joint_null": depth_rows[1]["damage_joint_null"],
        "damage_ratio": depth_rows[1]["damage_ratio"]
    })
    
    other_archs = [
        ("vit_base", 7),
        ("deit_tiny", 8),
        ("dinov2", 8)
    ]
    
    for arch_name, d_target in other_archs:
        print(f"  Testing {arch_name} at depth {d_target}...")
        m_other, t_other, meta_other = load_model_and_transform(arch_name, device)
        m_other.eval()
        
        calib_ds.transform = t_other
        loader_other = DataLoader(Subset(calib_ds, sample_indices), batch_size=len(sample_indices), shuffle=False)
        batch_other, _ = next(iter(loader_other))
        batch_other = batch_other.to(device)
        c_logits_other, c_acts_other = forward_block_by_block(
            m_other, arch_name, x=batch_other, collect_depths=tuple(range(13))
        )
        c_act_d = c_acts_other[d_target]
        
        op_vk_other = construct_value_and_key_operators(m_other, arch_name, d_target, c_act_d, device)
        op_j_other = construct_downstream_jacobian(m_other, arch_name, d_target, c_act_d, device)
        
        modes_other = extract_all_modes(op_vk_other, op_j_other, seed=42)
        alpha_other = 0.4 * op_vk_other["sigma_P"]
        
        hp_top = c_act_d.clone(); hp_top[:, 1:, :] += (alpha_other * modes_other["top_high_transmission"]).unsqueeze(0)
        p_logits_top, _ = forward_block_by_block(m_other, arch_name, h_start=hp_top, start_depth=d_target)
        d_top = torch.norm(p_logits_top - c_logits_other).item()
        
        hp_null = c_act_d.clone(); hp_null[:, 1:, :] += (alpha_other * modes_other["joint_vk_null"]).unsqueeze(0)
        p_logits_null, _ = forward_block_by_block(m_other, arch_name, h_start=hp_null, start_depth=d_target)
        d_null = torch.norm(p_logits_null - c_logits_other).item()
        
        rep_rows.append({
            "arch": arch_name,
            "primary_depth": d_target,
            "input_dim_ND": op_vk_other["total_dim"],
            "readout_dim": op_vk_other["D_readout"],
            "norm_V": op_vk_other["norm_V"],
            "norm_R": op_vk_other["norm_R"],
            "mean_principal_angle_deg": op_vk_other["mean_angle"],
            "damage_top": d_top,
            "damage_joint_null": d_null,
            "damage_ratio": d_top / (d_null + 1e-12)
        })
        
    df_rep = pd.DataFrame(rep_rows)
    df_rep.to_csv(os.path.join(out_dir, "replication_summary.csv"), index=False)
    
    # -------------------------------------------------------------
    # 9. Plotting All 8 Publication Figures
    # -------------------------------------------------------------
    print("\nGenerating all 8 publication figures...")
    
    # Figure A: Value vs Key Spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    ax1.plot(df_spec_v["rank_k"], df_spec_v["singular_value"], color="#1f77b4", lw=2, label="Value Operator V_l")
    ax1.plot(df_spec_r["rank_k"], df_spec_r["singular_value"], color="#d62728", lw=2, linestyle="--", label="Key Operator R_l")
    ax1.set_xlabel("Singular Mode Index k", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Singular Value sigma_k", fontsize=11, fontweight="bold")
    ax1.set_title("Singular Value Spectra: Value vs Key Pathways", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.legend(frameon=True)
    
    ax2.plot(df_spec_v["rank_k"], df_spec_v["cumulative_energy"], color="#1f77b4", lw=2, label="Value Operator V_l")
    ax2.plot(df_spec_r["rank_k"], df_spec_r["cumulative_energy"], color="#d62728", lw=2, linestyle="--", label="Key Operator R_l")
    ax2.set_xlabel("Singular Mode Index k", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Spectral Energy", fontsize=11, fontweight="bold")
    ax2.set_title("Cumulative Energy Concentration", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_a_value_vs_key_spectrum.png"))
    plt.close(fig)
    
    # Figure B: Subspace Overlap & Principal Angles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    ax1.plot(df_overlap["principal_index"], df_overlap["principal_cosine"], color="#2ca02c", lw=2)
    ax1.set_xlabel("Principal Dimension Index k", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Principal Cosine cos(theta_k)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Principal Cosines: RowSpace(V_l) vs RowSpace(R_l)\n(Mean Cosine = {op_vk_8['mean_cos']:.3f})", fontsize=11, fontweight="bold")
    ax1.set_ylim([0, 1.05])
    
    ax2.plot(df_overlap["principal_index"], df_overlap["principal_angle_deg"], color="#9467bd", lw=2)
    ax2.set_xlabel("Principal Dimension Index k", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Principal Angle theta_k (degrees)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Principal Angles (Mean Angle = {op_vk_8['mean_angle']:.1f} deg)", fontsize=11, fontweight="bold")
    ax2.set_ylim([0, 95])
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_b_subspace_overlap.png"))
    plt.close(fig)
    
    # Figure C: Value-Null vs Joint VK vs Multi-Block Null Modes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    colors = {
        "top_high_transmission": "#d62728",
        "value_only_null": "#ff7f0e",
        "key_only_null": "#bcbd22",
        "joint_vk_null": "#2ca02c",
        "multiblock_J_null": "#1f77b4",
        "joint_cum_null": "#9467bd",
        "random_matched_norm": "#7f7f7f"
    }
    for mode in ["top_high_transmission", "value_only_null", "joint_vk_null", "multiblock_J_null", "joint_cum_null"]:
        sub = df_eval[df_eval["mode_name"] == mode]
        ax1.plot(sub["radius_s"], sub["logit_l2"], marker='o', label=mode, color=colors[mode], lw=2)
        ax2.plot(sub["radius_s"], sub["margin_damage"], marker='s', label=mode, color=colors[mode], lw=2)
    ax1.set_xlabel("Perturbation Radius s = alpha / sigma_P", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Logit L2 Damage", fontsize=11, fontweight="bold")
    ax1.set_title("Damage vs Radius: Mode Comparison", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    
    ax2.set_xlabel("Perturbation Radius s = alpha / sigma_P", fontsize=11, fontweight="bold")
    ax2.set_ylabel("True-Class Margin Damage", fontsize=11, fontweight="bold")
    ax2.set_title("Margin Damage vs Radius", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_c_value_null_vs_joint_vk.png"))
    plt.close(fig)
    
    # Figure D: Finite Radius Full Response Curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    for mode, col in colors.items():
        if mode not in df_eval["mode_name"].values: continue
        sub = df_eval[df_eval["mode_name"] == mode]
        axes[0, 0].plot(sub["radius_s"], sub["logit_l2"], marker='o', label=mode, color=col, lw=2)
        axes[0, 1].plot(sub["radius_s"], sub["kl_div"], marker='^', label=mode, color=col, lw=2)
        axes[1, 0].plot(sub["radius_s"], sub["top1_accuracy"] * 100, marker='s', label=mode, color=col, lw=2)
        axes[1, 1].plot(sub["radius_s"], sub["top1_flip_rate"] * 100, marker='d', label=mode, color=col, lw=2)
        
    axes[0, 0].set_title("Logit L2 Damage", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("KL Divergence", fontsize=11, fontweight="bold")
    axes[1, 0].set_title("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    axes[1, 1].set_title("Top-1 Flip Rate (%)", fontsize=11, fontweight="bold")
    for ax in axes.flat:
        ax.set_xlabel("Perturbation Radius s", fontsize=10, fontweight="bold")
        ax.legend(frameon=True, fontsize=8)
    plt.suptitle("Figure D: Finite-Radius Multi-Metric Model Response", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_d_finite_radius_curves.png"))
    plt.close(fig)
    
    # Figure E: Attention Shift Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for mode in ["top_high_transmission", "value_only_null", "key_only_null", "joint_vk_null", "multiblock_J_null", "joint_cum_null"]:
        if mode not in df_attn["mode_name"].values: continue
        sub = df_attn[df_attn["mode_name"] == mode]
        ax1.plot(sub["radius_s"], sub["attention_frob_shift"], marker='o', label=mode, color=colors[mode], lw=2)
        ax2.plot(sub["radius_s"], sub["readout_to_patch_shift"], marker='^', label=mode, color=colors[mode], lw=2)
    ax1.set_xlabel("Radius s", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Attention Frobenius Shift ||A_p - A_c||_F", fontsize=11, fontweight="bold")
    ax1.set_title("Total Attention Matrix Shift", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8)
    
    ax2.set_xlabel("Radius s", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Readout-to-Patch Attention Shift", fontsize=11, fontweight="bold")
    ax2.set_title("Readout Attention Weight Perturbation", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_e_attention_shift.png"))
    plt.close(fig)
    
    # Figure F: Scaling Exponents
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    sub_small = df_eval[(df_eval["radius_s"] >= 0.05) & (df_eval["radius_s"] <= 0.4)]
    for mode in ["value_only_null", "joint_vk_null", "multiblock_J_null", "joint_cum_null"]:
        if mode not in sub_small["mode_name"].values: continue
        sub = sub_small[sub_small["mode_name"] == mode]
        ax1.plot(np.log(sub["alpha"]), np.log(sub["logit_l2"]), marker='o', label=mode, color=colors[mode], lw=2)
    ax1.set_xlabel("ln(alpha)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("ln(Logit L2 Damage)", fontsize=11, fontweight="bold")
    ax1.set_title("Log-Log Response at Small Radii", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)
    
    ax2.bar(df_scaling["mode_name"], df_scaling["scaling_exponent_p"], color=[colors.get(m, "#333333") for m in df_scaling["mode_name"]])
    ax2.axhline(1.0, color="gray", linestyle="--", label="p = 1.0 (Linear)")
    ax2.axhline(2.0, color="black", linestyle=":", label="p = 2.0 (Quadratic)")
    ax2.set_ylabel("Scaling Exponent p", fontsize=11, fontweight="bold")
    ax2.set_title("Fitted Power-Law Exponents", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(df_scaling["mode_name"], rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax2.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_f_scaling_exponents.png"))
    plt.close(fig)
    
    # Figure G: Blockwise Rerouting Trace
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for mode in ["value_only_null", "joint_vk_null", "multiblock_J_null", "joint_cum_null"]:
        if mode not in df_block_trace["mode_name"].values: continue
        sub = df_block_trace[df_block_trace["mode_name"] == mode]
        ax1.plot(sub["block"], sub["cls_attention_shift"], marker='o', label=mode, color=colors[mode], lw=2)
        ax2.plot(sub["block"], sub["query_relative_shift"], marker='s', label=mode, color=colors[mode], lw=2)
    ax1.set_xlabel("Downstream Block b", fontsize=11, fontweight="bold")
    ax1.set_ylabel("CLS Attention Shift ||A_p - A_c||", fontsize=11, fontweight="bold")
    ax1.set_title("Attention Shift Persistence Across Depth", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8)
    
    ax2.set_xlabel("Downstream Block b", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Relative Query Shift ||delta Q_b|| / ||Q_b||", fontsize=11, fontweight="bold")
    ax2.set_title("Query Disturbance Across Depth", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_g_blockwise_rerouting.png"))
    plt.close(fig)
    
    # Figure H: Cross-Architecture Replication
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    x = np.arange(len(df_rep))
    width = 0.35
    ax1.bar(x - width/2, df_rep["norm_V"], width, label="||V_l||_F", color="#1f77b4")
    ax1.bar(x + width/2, df_rep["norm_R"], width, label="||R_l||_F", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_rep["arch"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Operator Frobenius Norm", fontsize=11, fontweight="bold")
    ax1.set_title("Value vs Key Pathway Norms Across Architectures", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    
    ax2.bar(df_rep["arch"], df_rep["damage_ratio"], color="#2ca02c")
    ax2.set_ylabel("Top Mode vs Joint-Null Damage Ratio", fontsize=11, fontweight="bold")
    ax2.set_title("Joint Null Tolerance Across Architectures", fontsize=12, fontweight="bold")
    for i, v in enumerate(df_rep["damage_ratio"]):
        ax2.text(i, v + 0.3, f"{v:.1f}x", ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_h_cross_architecture_replication.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # 10. Manifest Generation
    # -------------------------------------------------------------
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": time.time() - total_start_time,
        "device": str(device),
        "primary_arch": primary_arch,
        "primary_depth": primary_depth,
        "replicated_archs": ["deit_small", "vit_base", "deit_tiny", "dinov2"],
        "num_held_out_samples": 100,
        "summary_metrics": {
            "norm_V": op_vk_8["norm_V"],
            "norm_R": op_vk_8["norm_R"],
            "mean_principal_angle_deg": op_vk_8["mean_angle"],
            "corr_r_val": corr_summary["r_value"],
            "corr_r_key": corr_summary["r_key"],
            "corr_r_joint": corr_summary["r_joint"],
            "corr_r_multi": corr_summary["r_multi"],
            "scaling_exponent_val_null": float(df_scaling[df_scaling["mode_name"] == "value_only_null"]["scaling_exponent_p"].values[0]),
            "scaling_exponent_joint_vk_null": float(df_scaling[df_scaling["mode_name"] == "joint_vk_null"]["scaling_exponent_p"].values[0]),
            "scaling_exponent_J_null": float(df_scaling[df_scaling["mode_name"] == "multiblock_J_null"]["scaling_exponent_p"].values[0]),
            "scaling_exponent_joint_cum_null": float(df_scaling[df_scaling["mode_name"] == "joint_cum_null"]["scaling_exponent_p"].values[0]) if "joint_cum_null" in df_scaling["mode_name"].values else None
        }
    }
    with open(os.path.join(out_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - total_start_time:.2f}s.")
    print(f"Artifacts saved in: {out_dir}")
    print(f"Figures saved in: {fig_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_joint_value_key_pipeline()
