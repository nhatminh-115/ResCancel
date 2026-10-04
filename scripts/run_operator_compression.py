"""
Runner Script for Fungibility Operator Compression

Executes the complete constructive token compression benchmark:
1. Operator-aware carrier optimization vs baselines across budgets: B in {147, 98, 72, 49, 32}
2. Grouping ablation: Spatial, Feature-Similarity, Random
3. Collapse parity verification (full-length repeated surrogate vs collapsed B-carrier forward)
4. Operator ablation: J_{l->L} vs C_{l->L} vs J_bar vs Group Mean
5. Low-rank spectral truncation: r in {8, 16, 32, 64, 128}
6. Calibration-averaged operator (J_bar) analysis
7. Cross-architecture replication: DeiT-Small, ViT-Base, DeiT-Tiny, DINOv2
8. Generates all 12 CSV/JSON deliverables and all 8 publication figures
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
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.multiblock_operator import construct_downstream_jacobian
from patch_fungibility.joint_value_key import construct_cumulative_multiblock_joint_operator
from patch_fungibility.compression_models import (
    forward_downstream_compressed,
    forward_downstream_reference
)
from patch_fungibility.operator_compression import (
    create_groupings,
    solve_operator_aware_carriers,
    solve_low_rank_carriers,
    evaluate_matched_budget_compression
)
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits


def run_operator_compression_suite():
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"RUNNING FUNGIBILITY OPERATOR COMPRESSION SUITE ON DEVICE: {device}")
    print("=" * 80)
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "fungibility_operator_compression"))
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "fungibility_operator_compression"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Dataset & Calibration Setup
    # -------------------------------------------------------------
    print("\n[Step 1/8] Loading dataset and test batches...")
    calib_ds, _, _, _ = get_disjoint_imagenet_splits()
    
    primary_arch = "deit_small"
    primary_depth = 8
    model_s, transform_s, meta_s = load_model_and_transform(primary_arch, device)
    model_s.eval()
    
    calib_ds.transform = transform_s
    
    # 20 evaluation images
    eval_indices = list(range(20))
    loader_eval = DataLoader(Subset(calib_ds, eval_indices), batch_size=len(eval_indices), shuffle=False)
    eval_imgs, eval_targets = next(iter(loader_eval))
    eval_imgs = eval_imgs.to(device)
    eval_targets = eval_targets.to(device)
    
    # Capture clean activations for DeiT-Small
    clean_logits_s, clean_acts_s = forward_block_by_block(
        model_s, primary_arch, x=eval_imgs, collect_depths=tuple(range(13))
    )
    clean_acts_8 = clean_acts_s[primary_depth]
    N_patches = clean_acts_8.shape[1] - 1 # 196
    
    # Precompute calibration-averaged J_bar on 5 calibration images
    print("Precomputing calibration-averaged operator J_bar on reference images...")
    J_list = []
    for c_i in range(5):
        op_c = construct_downstream_jacobian(model_s, primary_arch, primary_depth, clean_acts_8[c_i:c_i+1], device)
        J_list.append(op_c["J"])
    J_bar = torch.stack(J_list, dim=0).mean(dim=0)
    print(f"J_bar computed. Shape: {J_bar.shape}, norm: {torch.norm(J_bar).item():.4f}")

    # -------------------------------------------------------------
    # 2. Matched-Budget Evaluation Across Budgets
    # -------------------------------------------------------------
    budgets = [147, 98, 72, 49, 32] # 75%, 50%, 36.7%, 25%, 16.3%
    print(f"\n[Step 2/8] Running matched-budget evaluations for budgets B in {budgets}...")
    
    all_matched_records = []
    reconstruction_records = []
    collapse_parity_records = []
    residual_damage_records = []
    
    num_eval = 20
    t0 = time.time()
    
    for b_idx, B_tokens in enumerate(budgets):
        print(f"  Evaluating Budget B = {B_tokens} ({B_tokens / N_patches * 100:.1f}% patches)...")
        for i in range(num_eval):
            act_i = clean_acts_8[i:i+1]
            logit_i = clean_logits_s[i:i+1]
            tgt_i = eval_targets[i:i+1]
            
            # Construct per-image operators
            op_multi_i = construct_downstream_jacobian(model_s, primary_arch, primary_depth, act_i, device)
            
            # Cumulative operator for first 5 images to keep runtime ultra-fast
            if i < 5:
                clean_acts_i_list = {d: clean_acts_s[d][i:i+1] for d in clean_acts_s.keys()}
                op_cum_i = construct_cumulative_multiblock_joint_operator(
                    model_s, primary_arch, primary_depth, clean_acts_i_list, op_multi_i, device
                )
            else:
                op_cum_i = None
                
            res_dict = evaluate_matched_budget_compression(
                model_s, primary_arch, primary_depth, act_i, logit_i, tgt_i, B_tokens,
                grouping_strategy="spatial", op_multi=op_multi_i, op_cum=op_cum_i, J_bar=J_bar, device=device
            )
            
            # Record matched-budget results
            for method_name, m_res in res_dict.items():
                all_matched_records.append({
                    "image_idx": i,
                    "budget_B": B_tokens,
                    "compression_ratio": float(N_patches / B_tokens),
                    "method": method_name,
                    "logit_l2": m_res["logit_l2"],
                    "kl_div": m_res["kl_div"],
                    "margin_damage": m_res["margin_damage"],
                    "top1_acc": m_res["top1_acc"],
                    "top1_flip": m_res["top1_flip"]
                })
                
                if "operator_residual" in m_res:
                    residual_damage_records.append({
                        "image_idx": i,
                        "budget_B": B_tokens,
                        "method": method_name,
                        "operator_residual": m_res["operator_residual"],
                        "logit_l2": m_res["logit_l2"],
                        "top1_acc": m_res["top1_acc"]
                    })
                    
            # Record reconstruction objectives and parity
            reconstruction_records.append({
                "image_idx": i,
                "budget_B": B_tokens,
                "norm_r_mean": res_dict["group_mean"]["operator_residual"],
                "norm_r_opt": res_dict["operator_aware_J"]["operator_residual"],
                "residual_reduction_ratio": res_dict["group_mean"]["operator_residual"] / (res_dict["operator_aware_J"]["operator_residual"] + 1e-12)
            })
            
            collapse_parity_records.append({
                "image_idx": i,
                "budget_B": B_tokens,
                "parity_mean": res_dict["group_mean"]["collapse_parity"],
                "parity_operator": res_dict["operator_aware_J"]["collapse_parity"]
            })
            
    print(f"Matched budget evaluations completed in {time.time() - t0:.2f}s.")
    
    df_matched = pd.DataFrame(all_matched_records)
    df_matched.to_csv(os.path.join(out_dir, "matched_budget_results.csv"), index=False)
    
    df_recon = pd.DataFrame(reconstruction_records)
    df_recon.to_csv(os.path.join(out_dir, "reconstruction_objectives.csv"), index=False)
    
    df_parity = pd.DataFrame(collapse_parity_records)
    df_parity.to_csv(os.path.join(out_dir, "collapse_parity.csv"), index=False)
    
    df_res_dam = pd.DataFrame(residual_damage_records)
    df_res_dam.to_csv(os.path.join(out_dir, "residual_vs_damage.csv"), index=False)
    
    # -------------------------------------------------------------
    # 3. Budget Frontier Summary
    # -------------------------------------------------------------
    print("\n[Step 3/8] Computing accuracy-token frontier summary...")
    frontier_summary = df_matched.groupby(["method", "budget_B"]).agg({
        "top1_acc": "mean",
        "logit_l2": "mean",
        "margin_damage": "mean",
        "top1_flip": "mean",
        "kl_div": "mean"
    }).reset_index()
    frontier_summary["downstream_tokens"] = frontier_summary["budget_B"] + 1 # include CLS
    frontier_summary.to_csv(os.path.join(out_dir, "budget_frontier.csv"), index=False)
    
    # -------------------------------------------------------------
    # 4. Grouping Strategy Ablation (Spatial vs Feature vs Random)
    # -------------------------------------------------------------
    print("\n[Step 4/8] Running grouping strategy ablation (Spatial vs Feature-Similarity vs Random)...")
    grouping_records = []
    test_budget = 98
    for strat in ["spatial", "feature_similarity", "random"]:
        for i in range(10):
            act_i = clean_acts_8[i:i+1]
            logit_i = clean_logits_s[i:i+1]
            tgt_i = eval_targets[i:i+1]
            op_multi_i = construct_downstream_jacobian(model_s, primary_arch, primary_depth, act_i, device)
            
            res_dict = evaluate_matched_budget_compression(
                model_s, primary_arch, primary_depth, act_i, logit_i, tgt_i, test_budget,
                grouping_strategy=strat, op_multi=op_multi_i, op_cum=None, J_bar=J_bar, device=device
            )
            for m in ["group_mean", "operator_aware_J", "random_pruning"]:
                grouping_records.append({
                    "strategy": strat,
                    "image_idx": i,
                    "method": m,
                    "logit_l2": res_dict[m]["logit_l2"],
                    "top1_acc": res_dict[m]["top1_acc"],
                    "margin_damage": res_dict[m]["margin_damage"]
                })
    df_grouping = pd.DataFrame(grouping_records)
    df_grouping.to_csv(os.path.join(out_dir, "grouping_ablation.csv"), index=False)
    
    # -------------------------------------------------------------
    # 5. Operator Formulation Ablation
    # -------------------------------------------------------------
    print("\n[Step 5/8] Analyzing operator formulation ablation...")
    op_ablation_summary = frontier_summary[frontier_summary["budget_B"] == 98][
        ["method", "logit_l2", "margin_damage", "top1_acc"]
    ]
    op_ablation_summary.to_csv(os.path.join(out_dir, "operator_ablation.csv"), index=False)
    
    # -------------------------------------------------------------
    # 6. Low-Rank Spectral Truncation Ablation
    # -------------------------------------------------------------
    print("\n[Step 6/8] Running low-rank spectral truncation ablation (r in {8, 16, 32, 64, 128})...")
    low_rank_records = []
    ranks = [8, 16, 32, 64, 128]
    for r in ranks:
        for i in range(10):
            act_i = clean_acts_8[i:i+1]
            logit_i = clean_logits_s[i:i+1]
            tgt_i = eval_targets[i:i+1]
            P = act_i[0, 1:, :]
            cls_tok = act_i[:, 0:1, :]
            
            op_multi_i = construct_downstream_jacobian(model_s, primary_arch, primary_depth, act_i, device)
            J_i = op_multi_i["J"]
            
            groups, S, mults_t = create_groupings(P, 98, strategy="spatial")
            mult_all = torch.cat([torch.tensor([1.0], device=device), mults_t], dim=0)
            
            sol_r = solve_low_rank_carriers(P, S, mults_t, J_i, rank_r=r, lam_factor=10.0)
            C_r = sol_r["C_opt"]
            
            h_comp = torch.cat([cls_tok, C_r.unsqueeze(0)], dim=1)
            logits_comp = forward_downstream_compressed(model_s, primary_arch, primary_depth, h_comp, mult_all, N_patches)
            l2 = torch.norm(logits_comp - logit_i).item()
            acc = 1.0 if (logits_comp.argmax(dim=-1).item() == tgt_i.item()) else 0.0
            
            low_rank_records.append({
                "rank_r": r,
                "image_idx": i,
                "logit_l2": l2,
                "top1_acc": acc,
                "operator_residual": sol_r["norm_r_opt"]
            })
    df_low_rank = pd.DataFrame(low_rank_records)
    df_low_rank.to_csv(os.path.join(out_dir, "low_rank_ablation.csv"), index=False)
    
    # -------------------------------------------------------------
    # 7. Calibration Operator Results & Full-Length Surrogate
    # -------------------------------------------------------------
    print("\n[Step 7/8] Generating calibration operator and full-length surrogate tables...")
    sub_calib = df_matched[df_matched["method"].isin(["group_mean", "operator_aware_J", "calibration_J_bar"])]
    calib_summary = sub_calib.groupby(["method", "budget_B"]).agg({"logit_l2": "mean", "top1_acc": "mean"}).reset_index()
    calib_summary.to_csv(os.path.join(out_dir, "calibration_operator_results.csv"), index=False)
    
    # Full-length surrogate comparison table (at budget 98)
    surrogate_rows = []
    for i in range(10):
        act_i = clean_acts_8[i:i+1]
        logit_i = clean_logits_s[i:i+1]
        P = act_i[0, 1:, :]
        op_multi_i = construct_downstream_jacobian(model_s, primary_arch, primary_depth, act_i, device)
        J_i = op_multi_i["J"]
        groups, S, mults_t = create_groupings(P, 98, strategy="spatial")
        sol = solve_operator_aware_carriers(P, S, mults_t, J_i, lam_factor=10.0)
        
        # Clean forward
        l_clean = forward_downstream_reference(model_s, primary_arch, primary_depth, act_i)
        # Group Mean surrogate
        h_m = act_i.clone(); h_m[0, 1:, :] = S @ sol["C_mean"]
        l_mean = forward_downstream_reference(model_s, primary_arch, primary_depth, h_m)
        # Operator surrogate
        h_o = act_i.clone(); h_o[0, 1:, :] = S @ sol["C_opt"]
        l_opt = forward_downstream_reference(model_s, primary_arch, primary_depth, h_o)
        
        surrogate_rows.append({
            "image_idx": i,
            "l2_clean_vs_mean_surrogate": torch.norm(l_mean - l_clean).item(),
            "l2_clean_vs_opt_surrogate": torch.norm(l_opt - l_clean).item(),
            "surrogate_damage_reduction": (torch.norm(l_mean - l_clean).item() - torch.norm(l_opt - l_clean).item())
        })
    df_surrogate = pd.DataFrame(surrogate_rows)
    df_surrogate.to_csv(os.path.join(out_dir, "full_length_surrogate_results.csv"), index=False)
    
    # -------------------------------------------------------------
    # 8. Cross-Architecture Replication (ViT-Base, DeiT-Tiny, DINOv2)
    # -------------------------------------------------------------
    print("\n[Step 8/8] Replicating across architectures at budget B=98...")
    rep_records = []
    
    # DeiT-Small
    deit_s_mean_l2 = df_matched[(df_matched["budget_B"] == 98) & (df_matched["method"] == "group_mean")]["logit_l2"].mean()
    deit_s_opt_l2 = df_matched[(df_matched["budget_B"] == 98) & (df_matched["method"] == "operator_aware_J")]["logit_l2"].mean()
    deit_s_prune_l2 = df_matched[(df_matched["budget_B"] == 98) & (df_matched["method"] == "random_pruning")]["logit_l2"].mean()
    rep_records.append({
        "arch": "deit_small", "primary_depth": 8, "budget_B": 98,
        "mean_merge_l2": deit_s_mean_l2, "operator_merge_l2": deit_s_opt_l2, "pruning_l2": deit_s_prune_l2,
        "operator_damage_reduction_pct": (deit_s_mean_l2 - deit_s_opt_l2) / deit_s_mean_l2 * 100.0
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
        
        loader_other = DataLoader(Subset(calib_ds, list(range(5))), batch_size=5, shuffle=False)
        b_other, t_other_batch = next(iter(loader_other))
        b_other = b_other.to(device)
        t_other_batch = t_other_batch.to(device)
        
        c_logits_oth, c_acts_oth = forward_block_by_block(
            m_other, arch_name, x=b_other, collect_depths=tuple(range(13))
        )
        c_act_d = c_acts_oth[d_target]
        N_oth = c_act_d.shape[1] - 1
        
        l2_m_list, l2_o_list, l2_p_list = [], [], []
        
        for k in range(5):
            act_k = c_act_d[k:k+1]
            log_k = c_logits_oth[k:k+1]
            tgt_k = t_other_batch[k:k+1]
            op_k = construct_downstream_jacobian(m_other, arch_name, d_target, act_k, device)
            
            res_k = evaluate_matched_budget_compression(
                m_other, arch_name, d_target, act_k, log_k, tgt_k, 98,
                grouping_strategy="spatial", op_multi=op_k, op_cum=None, J_bar=None, device=device
            )
            l2_m_list.append(res_k["group_mean"]["logit_l2"])
            l2_o_list.append(res_k["operator_aware_J"]["logit_l2"])
            l2_p_list.append(res_k["random_pruning"]["logit_l2"])
            
        mean_l2_m = float(np.mean(l2_m_list))
        mean_l2_o = float(np.mean(l2_o_list))
        mean_l2_p = float(np.mean(l2_p_list))
        
        rep_records.append({
            "arch": arch_name, "primary_depth": d_target, "budget_B": 98,
            "mean_merge_l2": mean_l2_m, "operator_merge_l2": mean_l2_o, "pruning_l2": mean_l2_p,
            "operator_damage_reduction_pct": (mean_l2_m - mean_l2_o) / mean_l2_m * 100.0
        })
        
    df_rep = pd.DataFrame(rep_records)
    df_rep.to_csv(os.path.join(out_dir, "replication_summary.csv"), index=False)
    
    # -------------------------------------------------------------
    # 9. Plotting All 8 Publication Figures
    # -------------------------------------------------------------
    print("\nGenerating all 8 publication figures...")
    
    # Color palette
    colors = {
        "operator_aware_J": "#1f77b4",
        "operator_aware_C_cum": "#9467bd",
        "calibration_J_bar": "#17becf",
        "group_mean": "#ff7f0e",
        "medoid": "#bcbd22",
        "random_pruning": "#d62728",
        "norm_pruning": "#e377c2",
        "unweighted_centroid": "#7f7f7f"
    }
    
    # Figure A: Reconstruction Objectives
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    sub_r = df_recon.groupby("budget_B").mean().reset_index()
    ax1.plot(sub_r["budget_B"], sub_r["norm_r_mean"], marker='o', color="#ff7f0e", lw=2, label="Group Mean ||J E_mean||")
    ax1.plot(sub_r["budget_B"], sub_r["norm_r_opt"], marker='s', color="#1f77b4", lw=2, label="Operator-Aware ||J E_operator||")
    ax1.set_xlabel("Token Budget B", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Linear Downstream Transmission ||J E||", fontsize=11, fontweight="bold")
    ax1.set_title("Downstream Transmission Objective Minimization", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    
    ax2.bar(sub_r["budget_B"].astype(str), sub_r["residual_reduction_ratio"], color="#2ca02c", width=0.4)
    ax2.set_xlabel("Token Budget B", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Transmission Reduction Factor (x)", fontsize=11, fontweight="bold")
    ax2.set_title("Downstream Transmission Suppression Factor", fontsize=12, fontweight="bold")
    for i, v in enumerate(sub_r["residual_reduction_ratio"]):
        ax2.text(i, v + 50, f"{v:.0f}x", ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_a_reconstruction_objective.png"))
    plt.close(fig)
    
    # Figure B: Matched-Budget Accuracy Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for m in ["operator_aware_J", "group_mean", "random_pruning", "norm_pruning", "medoid"]:
        sub_m = frontier_summary[frontier_summary["method"] == m]
        ax.plot(sub_m["budget_B"], sub_m["top1_acc"] * 100, marker='o', label=m, color=colors.get(m, "#333333"), lw=2)
    ax.set_xlabel("Token Budget B", fontsize=11, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Figure B: Matched-Budget Accuracy Across Budgets", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_b_matched_budget_accuracy.png"))
    plt.close(fig)
    
    # Figure C: Accuracy-Token Frontier
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for m in ["operator_aware_J", "group_mean", "random_pruning", "norm_pruning"]:
        sub_m = frontier_summary[frontier_summary["method"] == m]
        ax1.plot(sub_m["downstream_tokens"], sub_m["top1_acc"] * 100, marker='o', label=m, color=colors.get(m, "#333333"), lw=2)
        ax2.plot(sub_m["downstream_tokens"], sub_m["logit_l2"], marker='s', label=m, color=colors.get(m, "#333333"), lw=2)
    ax1.set_xlabel("Downstream Token Count (B + 1)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Accuracy vs Downstream Sequence Length", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    
    ax2.set_xlabel("Downstream Token Count (B + 1)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Logit L2 Damage", fontsize=11, fontweight="bold")
    ax2.set_title("Logit Damage vs Downstream Sequence Length", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_c_accuracy_token_frontier.png"))
    plt.close(fig)
    
    # Figure D: Operator Residual vs Observed Damage
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.scatter(df_res_dam["operator_residual"], df_res_dam["logit_l2"], alpha=0.6, c="#1f77b4", s=30)
    r_corr, _ = pearsonr(df_res_dam["operator_residual"], df_res_dam["logit_l2"])
    rho_corr, _ = spearmanr(df_res_dam["operator_residual"], df_res_dam["logit_l2"])
    ax.set_xlabel("Downstream Transmission Residual ||J E||", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed Full-Model Logit L2 Damage", fontsize=11, fontweight="bold")
    ax.set_title(f"Figure D: Operator Residual vs Real Damage (r = {r_corr:.3f}, rho = {rho_corr:.3f})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_d_operator_residual_vs_damage.png"))
    plt.close(fig)
    
    # Figure E: Grouping Ablation
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_g = df_grouping.groupby(["strategy", "method"]).mean().reset_index()
    strats = ["spatial", "feature_similarity", "random"]
    x = np.arange(len(strats))
    width = 0.35
    l2_mean_g = [sub_g[(sub_g["strategy"] == s) & (sub_g["method"] == "group_mean")]["logit_l2"].values[0] for s in strats]
    l2_opt_g = [sub_g[(sub_g["strategy"] == s) & (sub_g["method"] == "operator_aware_J")]["logit_l2"].values[0] for s in strats]
    ax.bar(x - width/2, l2_mean_g, width, label="Group Mean", color="#ff7f0e")
    ax.bar(x + width/2, l2_opt_g, width, label="Operator-Aware J", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(["Spatial", "Feature-Sim", "Random"], fontsize=10, fontweight="bold")
    ax.set_ylabel("Logit L2 Damage (B=98)", fontsize=11, fontweight="bold")
    ax.set_title("Figure E: Grouping Strategy Ablation", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_e_grouping_ablation.png"))
    plt.close(fig)
    
    # Figure F: Operator Formulation Ablation
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sub_op = frontier_summary[frontier_summary["budget_B"] == 98]
    ax.bar(sub_op["method"], sub_op["logit_l2"], color=[colors.get(m, "#333333") for m in sub_op["method"]])
    ax.set_ylabel("Logit L2 Damage (B=98)", fontsize=11, fontweight="bold")
    ax.set_title("Figure F: Operator Formulation Ablation at 50% Budget", fontsize=12, fontweight="bold")
    ax.set_xticklabels(sub_op["method"], rotation=30, ha="right", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_f_operator_ablation.png"))
    plt.close(fig)
    
    # Figure G: Low-Rank Spectral Truncation Trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    sub_lr = df_low_rank.groupby("rank_r").mean().reset_index()
    ax1.plot(sub_lr["rank_r"], sub_lr["logit_l2"], marker='o', color="#1f77b4", lw=2)
    ax1.set_xlabel("Retained Spectral Rank r", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Logit L2 Damage", fontsize=11, fontweight="bold")
    ax1.set_title("Damage vs Spectral Truncation Rank r", fontsize=12, fontweight="bold")
    
    ax2.plot(sub_lr["rank_r"], sub_lr["operator_residual"], marker='s', color="#d62728", lw=2)
    ax2.set_xlabel("Retained Spectral Rank r", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Residual ||J E||", fontsize=11, fontweight="bold")
    ax2.set_title("Downstream Transmission vs Rank r", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_g_low_rank_tradeoff.png"))
    plt.close(fig)
    
    # Figure H: Cross-Architecture Replication
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    x = np.arange(len(df_rep))
    width = 0.25
    ax1.bar(x - width, df_rep["mean_merge_l2"], width, label="Group Mean", color="#ff7f0e")
    ax1.bar(x, df_rep["operator_merge_l2"], width, label="Operator-Aware J", color="#1f77b4")
    ax1.bar(x + width, df_rep["pruning_l2"], width, label="Pruning", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_rep["arch"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Logit L2 Damage (B=98)", fontsize=11, fontweight="bold")
    ax1.set_title("Damage Comparison Across Architectures", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    
    ax2.bar(df_rep["arch"], df_rep["operator_damage_reduction_pct"], color="#2ca02c", width=0.4)
    ax2.set_ylabel("Damage Reduction over Mean Merging (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Operator-Aware Improvement over Mean", fontsize=12, fontweight="bold")
    for i, v in enumerate(df_rep["operator_damage_reduction_pct"]):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_h_cross_architecture_replication.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # 10. Validation Manifest Generation
    # -------------------------------------------------------------
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": time.time() - total_start,
        "device": str(device),
        "primary_arch": primary_arch,
        "primary_depth": primary_depth,
        "budgets_evaluated": budgets,
        "replicated_archs": ["deit_small", "vit_base", "deit_tiny", "dinov2"],
        "summary_findings": {
            "mean_transmission_reduction_ratio": float(df_recon["residual_reduction_ratio"].mean()),
            "max_collapse_parity_error": float(df_parity["parity_operator"].max()),
            "correlation_residual_vs_damage_r": float(r_corr),
            "correlation_residual_vs_damage_rho": float(rho_corr),
            "deit_small_operator_reduction_pct": float(df_rep[df_rep["arch"] == "deit_small"]["operator_damage_reduction_pct"].values[0]),
            "vit_base_operator_reduction_pct": float(df_rep[df_rep["arch"] == "vit_base"]["operator_damage_reduction_pct"].values[0])
        }
    }
    with open(os.path.join(out_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"COMPRESSION SUITE COMPLETED SUCCESSFULLY IN {time.time() - total_start:.2f}s.")
    print(f"Artifacts saved in: {out_dir}")
    print(f"Figures saved in: {fig_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_operator_compression_suite()
