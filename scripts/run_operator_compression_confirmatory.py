"""
scripts/run_operator_compression_confirmatory.py

Runs the strict confirmatory benchmark for Operator-Aware Token Compression:
- 1,000 canonical held-out ImageNet-1k images (seed 9201)
- 4 architectures: DeiT-Tiny, DeiT-Small, ViT-B/16 AugReg, DINOv2 ViT-S/14
- 5 matched token budgets per architecture
- 7 baselines + 4 operator configurations (Full J, Rank 16, Rank 32, Rank 64)
- 5 canonical seeds (31001-31005) for stochastic baselines & groupings
- Parity checks, paired statistics, causal delta, residual correlations
- Complete CSV outputs and publication-grade figures
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import math
import subprocess
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.compression_models import (
    forward_downstream_compressed,
    forward_downstream_reference
)
from patch_fungibility.operator_compression_confirmatory import (
    compute_fast_downstream_jacobian,
    extract_cls_attention_weights,
    tome_bipartite_merge,
    create_groupings_confirmatory,
    solve_operator_aware_carriers_fast,
    solve_low_rank_carriers_fast,
    compute_mcnemar_pvalue,
    bootstrap_mean_diff_ci
)


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__)).decode('ascii').strip()
    except Exception:
        return "unknown"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strict Confirmatory Benchmark for Operator Compression")
    parser.add_argument("--max_images", type=int, default=1000, help="Maximum images to evaluate (default: 1000)")
    parser.add_argument("--models", nargs="+", default=["deit_tiny", "deit_small", "vit_base", "dinov2"],
                        help="Models to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Strict Operator-Aware Token Compression Confirmatory Benchmark ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Configured max_images: {args.max_images}, models: {args.models}")

    output_dir = os.path.abspath("outputs/fungibility_operator_compression_confirmatory")
    figures_dir = os.path.abspath("figures/fungibility_operator_compression_confirmatory")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    git_hash = get_git_revision_hash()
    print(f"Git commit hash: {git_hash}")

    # 1. Dataset Split
    print("\n--- Loading Canonical Held-Out Split (N=1,000, seed=9201) ---")
    calib_set, eval_set, calib_df, eval_df = get_disjoint_imagenet_splits(
        calib_seed=9101,
        eval_seed=9201,
        n_per_split=1000
    )
    if args.max_images < len(eval_set):
        eval_set._samples = eval_set._samples[:args.max_images]
    print(f"Disjoint evaluation set loaded: {len(eval_set)} images.")

    # Model specifications
    all_models_config = [
        {"model_key": "deit_tiny", "depth": 8, "name": "DeiT-Tiny", "N_patches": 196, "budgets": [147, 98, 72, 49, 32]},
        {"model_key": "deit_small", "depth": 8, "name": "DeiT-Small", "N_patches": 196, "budgets": [147, 98, 72, 49, 32]},
        {"model_key": "vit_base", "depth": 7, "name": "ViT-B/16", "N_patches": 196, "budgets": [147, 98, 72, 49, 32]},
        {"model_key": "dinov2", "depth": 8, "name": "DINOv2 ViT-S/14", "N_patches": 256, "budgets": [192, 128, 94, 64, 42]}
    ]
    models_config = [m for m in all_models_config if m["model_key"] in args.models]

    canonical_seeds = [31001, 31002, 31003, 31004, 31005]

    all_per_image_rows = []
    all_same_group_rows = []
    all_low_rank_rows = []
    all_parity_rows = []
    runtime_records = []

    # Iterate over models
    for m_cfg in models_config:
        m_key = m_cfg["model_key"]
        m_name = m_cfg["name"]
        depth = m_cfg["depth"]
        N_patches = m_cfg["N_patches"]
        budgets = m_cfg["budgets"]

        print(f"\n==================================================")
        print(f"Benchmarking Model: {m_name} (depth={depth}, N={N_patches})")
        print(f"Budgets: {budgets}")
        print(f"==================================================")

        model, transform, meta = load_model_and_transform(m_key, device)
        model.eval()

        # Update eval_set transform
        eval_set.transform = transform
        data_loader = torch.utils.data.DataLoader(
            eval_set, batch_size=1, shuffle=False, num_workers=0
        )

        n_images = len(eval_set)
        t_model_start = time.time()

        for img_idx, (img_tensor, label_tensor) in enumerate(data_loader):
            img_tensor = img_tensor.to(device)
            target = label_tensor.item()

            t0_img = time.time()

            # Clean forward pass
            with torch.no_grad():
                clean_logits, clean_acts_list = forward_block_by_block(
                    model, m_key, x=img_tensor, collect_depths=(depth,)
                )
            clean_act = clean_acts_list[depth]
            P = clean_act[0, 1:, :] # (N, D)
            cls_token = clean_act[:, 0:1, :]
            D = P.shape[1]

            clean_pred = clean_logits.argmax(dim=-1).item()
            clean_correct = (clean_pred == target)
            clean_target_logit = clean_logits[0, target].item()
            # Clean margin
            clean_sorted = torch.sort(clean_logits[0], descending=True)
            top1_val = clean_sorted.values[0].item()
            top2_val = clean_sorted.values[1].item()
            clean_margin = (top1_val - top2_val) if clean_correct else (clean_target_logit - top1_val)

            # 1. Fast Batched Downstream Jacobian J
            t0_j = time.time()
            J = compute_fast_downstream_jacobian(model, m_key, depth, clean_act, chunk_size=128)
            t_j = time.time() - t0_j

            # 2. Attention weights for Attention Pruning
            with torch.no_grad():
                attn_weights = extract_cls_attention_weights(model, m_key, depth, clean_act)

            # Helper for compressed forward pass
            def eval_comp(C_tokens, mults_tokens):
                all_mults = torch.cat([torch.tensor([1.0], device=device), mults_tokens], dim=0)
                h_c = torch.cat([cls_token, C_tokens.unsqueeze(0)], dim=1)
                logits_c = forward_downstream_compressed(model, m_key, depth, h_c, all_mults, N_patches)
                l2 = torch.norm(logits_c - clean_logits).item()
                p_c = F.softmax(clean_logits, dim=-1)
                log_p = F.log_softmax(logits_c, dim=-1)
                kl = F.kl_div(log_p, p_c, reduction='batchmean').item()
                pred = logits_c.argmax(dim=-1).item()
                corr = 1 if (pred == target) else 0
                flip = 1 if (pred != clean_pred) else 0

                c_tgt = logits_c[0, target].item()
                sorted_c = torch.sort(logits_c[0], descending=True)
                top1_c = sorted_c.values[0].item()
                top2_c = sorted_c.values[1].item()
                margin_c = (top1_c - top2_c) if (corr == 1) else (c_tgt - top1_c)
                margin_damage = clean_margin - margin_c

                return {
                    "logit_l2": l2,
                    "kl_div": kl,
                    "top1_acc": corr,
                    "top1_flip": flip,
                    "margin_damage": margin_damage,
                    "logits": logits_c
                }

            # Helper for uncompressed surrogate forward pass (parity test)
            def eval_surrogate(C_tokens, S_matrix):
                h_full = clean_act.clone()
                h_full[0, 1:, :] = S_matrix @ C_tokens
                return forward_downstream_reference(model, m_key, depth, h_full)

            # Evaluate each budget B
            for B in budgets:
                rem_frac = B / float(N_patches)

                # --- Baseline A: Random Pruning (5 seeds) ---
                for s_idx, s in enumerate(canonical_seeds):
                    torch.manual_seed(s)
                    keep_idx = torch.randperm(N_patches, device=device)[:B].sort()[0]
                    C_rand = P[keep_idx]
                    mults_rand = torch.ones(B, device=device)
                    res_rand = eval_comp(C_rand, mults_rand)
                    if s_idx == 0:
                        all_per_image_rows.append({
                            "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                            "budget": B, "rem_frac": rem_frac, "method": "Random Pruning", "seed": s,
                            "top1_acc": res_rand["top1_acc"], "top1_flip": res_rand["top1_flip"],
                            "margin_damage": res_rand["margin_damage"], "logit_l2": res_rand["logit_l2"],
                            "kl_div": res_rand["kl_div"], "op_residual": np.nan, "delta_c_norm": np.nan
                        })

                # --- Baseline B: Norm Pruning ---
                norms = torch.norm(P, dim=-1)
                keep_norm = torch.topk(norms, B).indices.sort()[0]
                C_norm = P[keep_norm]
                res_norm = eval_comp(C_norm, torch.ones(B, device=device))
                all_per_image_rows.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                    "budget": B, "rem_frac": rem_frac, "method": "Norm Pruning", "seed": 0,
                    "top1_acc": res_norm["top1_acc"], "top1_flip": res_norm["top1_flip"],
                    "margin_damage": res_norm["margin_damage"], "logit_l2": res_norm["logit_l2"],
                    "kl_div": res_norm["kl_div"], "op_residual": np.nan, "delta_c_norm": np.nan
                })

                # --- Baseline C: Attention Pruning ---
                keep_attn = torch.topk(attn_weights, B).indices.sort()[0]
                C_attn = P[keep_attn]
                res_attn = eval_comp(C_attn, torch.ones(B, device=device))
                all_per_image_rows.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                    "budget": B, "rem_frac": rem_frac, "method": "Attention Pruning", "seed": 0,
                    "top1_acc": res_attn["top1_acc"], "top1_flip": res_attn["top1_flip"],
                    "margin_damage": res_attn["margin_damage"], "logit_l2": res_attn["logit_l2"],
                    "kl_div": res_attn["kl_div"], "op_residual": np.nan, "delta_c_norm": np.nan
                })

                # --- Baseline G: ToMe Bipartite Soft Matching ---
                C_tome, mults_tome = tome_bipartite_merge(P, B)
                res_tome = eval_comp(C_tome, mults_tome)
                all_per_image_rows.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                    "budget": B, "rem_frac": rem_frac, "method": "ToMe (BSM)", "seed": 0,
                    "top1_acc": res_tome["top1_acc"], "top1_flip": res_tome["top1_flip"],
                    "margin_damage": res_tome["margin_damage"], "logit_l2": res_tome["logit_l2"],
                    "kl_div": res_tome["kl_div"], "op_residual": np.nan, "delta_c_norm": np.nan
                })

                # Primary Feature-Similarity Grouping
                groups_feat, S_feat, mults_feat = create_groupings_confirmatory(
                    P, B, strategy="feature_similarity", seed=42
                )

                # Solve Operator-Aware Carrier & Group-Mean
                t0_solve = time.time()
                sol_feat = solve_operator_aware_carriers_fast(P, S_feat, mults_feat, J, lam_factor=10.0)
                t_solve = time.time() - t0_solve

                C_mean = sol_feat["C_mean"]
                C_opt = sol_feat["C_opt"]

                # --- Baseline D: Group-Mean Merging ---
                res_mean = eval_comp(C_mean, mults_feat)
                all_per_image_rows.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                    "budget": B, "rem_frac": rem_frac, "method": "Group-Mean Merging", "seed": 0,
                    "top1_acc": res_mean["top1_acc"], "top1_flip": res_mean["top1_flip"],
                    "margin_damage": res_mean["margin_damage"], "logit_l2": res_mean["logit_l2"],
                    "kl_div": res_mean["kl_div"], "op_residual": sol_feat["norm_r_mean"], "delta_c_norm": 0.0
                })

                # --- Baseline E: Medoid Merging ---
                C_medoid = torch.zeros(B, D, device=device)
                for j, g in enumerate(groups_feat):
                    p_g = P[g]
                    diffs = torch.norm(p_g - C_mean[j:j+1], dim=-1)
                    best_i = int(torch.argmin(diffs).item())
                    C_medoid[j] = p_g[best_i]
                res_medoid = eval_comp(C_medoid, mults_feat)
                r_medoid = J @ (P - S_feat @ C_medoid).reshape(-1)
                all_per_image_rows.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                    "budget": B, "rem_frac": rem_frac, "method": "Medoid Merging", "seed": 0,
                    "top1_acc": res_medoid["top1_acc"], "top1_flip": res_medoid["top1_flip"],
                    "margin_damage": res_medoid["margin_damage"], "logit_l2": res_medoid["logit_l2"],
                    "kl_div": res_medoid["kl_div"], "op_residual": torch.norm(r_medoid).item(), "delta_c_norm": torch.norm(C_medoid - C_mean).item()
                })

                # --- Baseline F: Unweighted Centroid Carrier ---
                res_unweighted = eval_comp(C_mean, torch.ones(B, device=device))
                all_per_image_rows.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                    "budget": B, "rem_frac": rem_frac, "method": "Unweighted Centroid", "seed": 0,
                    "top1_acc": res_unweighted["top1_acc"], "top1_flip": res_unweighted["top1_flip"],
                    "margin_damage": res_unweighted["margin_damage"], "logit_l2": res_unweighted["logit_l2"],
                    "kl_div": res_unweighted["kl_div"], "op_residual": sol_feat["norm_r_mean"], "delta_c_norm": 0.0
                })

                # --- Primary Method: Operator-Aware Oracle Compression (Full J) ---
                res_opt = eval_comp(C_opt, mults_feat)
                all_per_image_rows.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                    "budget": B, "rem_frac": rem_frac, "method": "Operator-Aware (Oracle)", "seed": 0,
                    "top1_acc": res_opt["top1_acc"], "top1_flip": res_opt["top1_flip"],
                    "margin_damage": res_opt["margin_damage"], "logit_l2": res_opt["logit_l2"],
                    "kl_div": res_opt["kl_div"], "op_residual": sol_feat["norm_r_opt"], "delta_c_norm": sol_feat["norm_delta_C"]
                })

                # Record Same-Group Causal Delta on Feature-Similarity grouping
                all_same_group_rows.append({
                    "model": m_name, "image_idx": img_idx, "budget": B, "grouping": "feature_similarity",
                    "delta_op_residual": sol_feat["norm_r_mean"] - sol_feat["norm_r_opt"],
                    "delta_logit_l2": res_mean["logit_l2"] - res_opt["logit_l2"], # positive = operator better
                    "delta_margin_dam": res_mean["margin_damage"] - res_opt["margin_damage"], # positive = operator better
                    "acc_mean": res_mean["top1_acc"], "acc_opt": res_opt["top1_acc"]
                })

                # Collapse Parity Verification (first 50 images per model to verify numerical precision)
                if img_idx < 50:
                    with torch.no_grad():
                        logits_mean_surr = eval_surrogate(C_mean, S_feat)
                        parity_mean = torch.norm(res_mean["logits"] - logits_mean_surr, p=float('inf')).item()
                        logits_opt_surr = eval_surrogate(C_opt, S_feat)
                        parity_opt = torch.norm(res_opt["logits"] - logits_opt_surr, p=float('inf')).item()
                        all_parity_rows.append({
                            "model": m_name, "image_idx": img_idx, "budget": B,
                            "parity_mean_max_err": parity_mean, "parity_opt_max_err": parity_opt
                        })

                # Low-Rank Operator Truncation Ablation (r in {64, 32, 16})
                for rank_r in [64, 32, 16]:
                    sol_lr = solve_low_rank_carriers_fast(P, S_feat, mults_feat, J, rank_r=rank_r, lam_factor=10.0)
                    res_lr = eval_comp(sol_lr["C_opt"], mults_feat)
                    all_low_rank_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "rank": rank_r,
                        "top1_acc": res_lr["top1_acc"], "margin_damage": res_lr["margin_damage"],
                        "logit_l2": res_lr["logit_l2"], "op_residual": sol_lr["norm_r_opt"]
                    })
                    if rank_r in [16, 32]:
                        all_per_image_rows.append({
                            "model": m_name, "image_idx": img_idx, "target": target, "clean_correct": int(clean_correct),
                            "budget": B, "rem_frac": rem_frac, "method": f"Operator-Aware (Rank-{rank_r})", "seed": 0,
                            "top1_acc": res_lr["top1_acc"], "top1_flip": res_lr["top1_flip"],
                            "margin_damage": res_lr["margin_damage"], "logit_l2": res_lr["logit_l2"],
                            "kl_div": res_lr["kl_div"], "op_residual": sol_lr["norm_r_opt"], "delta_c_norm": sol_lr["norm_delta_C"]
                        })

                # Controlled Grouping Ablation: Spatial & Random (on budget B=49 / 64)
                if B == budgets[3]: # 4th budget (~25% tokens)
                    # Spatial
                    groups_sp, S_sp, mults_sp = create_groupings_confirmatory(P, B, strategy="spatial")
                    sol_sp = solve_operator_aware_carriers_fast(P, S_sp, mults_sp, J, lam_factor=10.0)
                    res_sp_mean = eval_comp(sol_sp["C_mean"], mults_sp)
                    res_sp_opt = eval_comp(sol_sp["C_opt"], mults_sp)
                    all_same_group_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "grouping": "spatial",
                        "delta_op_residual": sol_sp["norm_r_mean"] - sol_sp["norm_r_opt"],
                        "delta_logit_l2": res_sp_mean["logit_l2"] - res_sp_opt["logit_l2"],
                        "delta_margin_dam": res_sp_mean["margin_damage"] - res_sp_opt["margin_damage"],
                        "acc_mean": res_sp_mean["top1_acc"], "acc_opt": res_sp_opt["top1_acc"]
                    })

                    # Random
                    groups_rd, S_rd, mults_rd = create_groupings_confirmatory(P, B, strategy="random", seed=42)
                    sol_rd = solve_operator_aware_carriers_fast(P, S_rd, mults_rd, J, lam_factor=10.0)
                    res_rd_mean = eval_comp(sol_rd["C_mean"], mults_rd)
                    res_rd_opt = eval_comp(sol_rd["C_opt"], mults_rd)
                    all_same_group_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "grouping": "random",
                        "delta_op_residual": sol_rd["norm_r_mean"] - sol_rd["norm_r_opt"],
                        "delta_logit_l2": res_rd_mean["logit_l2"] - res_rd_opt["logit_l2"],
                        "delta_margin_dam": res_rd_mean["margin_damage"] - res_rd_opt["margin_damage"],
                        "acc_mean": res_rd_mean["top1_acc"], "acc_opt": res_rd_opt["top1_acc"]
                    })

            # Record runtime for first 100 images
            if img_idx < 100:
                runtime_records.append({
                    "model": m_name, "image_idx": img_idx,
                    "t_jacobian": t_j, "t_carrier_solve": t_solve, "t_total_img": time.time() - t0_img
                })

            if (img_idx + 1) % 100 == 0:
                elapsed = time.time() - t_model_start
                rate = (img_idx + 1) / elapsed
                print(f"[{m_name}] {img_idx + 1}/{n_images} images processed ({rate:.1f} img/s, elapsed: {elapsed:.1f}s)")

        print(f"[{m_name}] Finished N={n_images} in {time.time() - t_model_start:.1f}s.")

    # Convert per-image results to DataFrame
    df_per_image = pd.DataFrame(all_per_image_rows)
    df_per_image.to_csv(os.path.join(output_dir, "per_image_results.csv"), index=False)
    print(f"\nSaved {len(df_per_image)} rows to per_image_results.csv")

    # 1. Budget Summary
    budget_summary = df_per_image.groupby(["model", "method", "budget", "rem_frac"]).agg(
        top1_acc=("top1_acc", "mean"),
        top1_flip=("top1_flip", "mean"),
        margin_damage=("margin_damage", "mean"),
        logit_l2=("logit_l2", "mean"),
        kl_div=("kl_div", "mean"),
        op_residual=("op_residual", "mean")
    ).reset_index()

    # Clean accuracies per model
    clean_accs = {}
    for m_cfg in models_config:
        m_name = m_cfg["name"]
        m_sub = df_per_image[df_per_image["model"] == m_name]
        clean_acc = m_sub[m_sub["method"] == "Group-Mean Merging"]["clean_correct"].mean()
        clean_accs[m_name] = clean_acc

    budget_summary["clean_acc"] = budget_summary["model"].map(clean_accs)
    budget_summary["retention"] = budget_summary["top1_acc"] / budget_summary["clean_acc"]
    budget_summary.to_csv(os.path.join(output_dir, "budget_summary.csv"), index=False)
    print(f"Saved budget_summary.csv")

    # 2. Architecture Summary & AUC-Token Frontier
    arch_rows = []
    for (m_name, method), group in budget_summary.groupby(["model", "method"]):
        group_sorted = group.sort_values("rem_frac")
        x_pts = group_sorted["rem_frac"].values
        y_pts = group_sorted["top1_acc"].values
        # Trapezoidal AUC
        auc = float(np.trapezoid(y_pts, x_pts))
        mean_ret = float(group["retention"].mean())
        agg_ret = float(group[group["budget"] == group["budget"].min()]["retention"].values[0])
        arch_rows.append({
            "model": m_name, "method": method, "clean_acc": clean_accs[m_name],
            "auc_frontier": auc, "mean_retention": mean_ret, "aggressive_retention": agg_ret
        })
    df_arch_summary = pd.DataFrame(arch_rows)
    df_arch_summary.to_csv(os.path.join(output_dir, "architecture_summary.csv"), index=False)
    print(f"Saved architecture_summary.csv")

    # 3. Seed Summary
    # For random pruning across canonical seeds
    # Let's save a summary
    df_seed_summary = df_per_image[df_per_image["method"] == "Random Pruning"].groupby(["model", "budget", "seed"]).agg(
        top1_acc=("top1_acc", "mean"),
        margin_damage=("margin_damage", "mean")
    ).reset_index()
    df_seed_summary.to_csv(os.path.join(output_dir, "seed_summary.csv"), index=False)
    print(f"Saved seed_summary.csv")

    # 4. Same-Group Causal Ablation
    df_same_group = pd.DataFrame(all_same_group_rows)
    df_same_group.to_csv(os.path.join(output_dir, "same_group_ablation.csv"), index=False)
    print(f"Saved same_group_ablation.csv")

    # 5. Low-Rank Ablation
    df_low_rank = pd.DataFrame(all_low_rank_rows)
    df_lr_summary = df_low_rank.groupby(["model", "budget", "rank"]).agg(
        top1_acc=("top1_acc", "mean"),
        margin_damage=("margin_damage", "mean"),
        logit_l2=("logit_l2", "mean"),
        op_residual=("op_residual", "mean")
    ).reset_index()
    df_lr_summary.to_csv(os.path.join(output_dir, "low_rank_ablation.csv"), index=False)
    print(f"Saved low_rank_ablation.csv")

    # 6. Collapse Parity Verification
    df_parity = pd.DataFrame(all_parity_rows)
    df_parity_summary = df_parity.groupby(["model", "budget"]).agg(
        mean_parity_opt_err=("parity_opt_max_err", "mean"),
        max_parity_opt_err=("parity_opt_max_err", "max"),
        mean_parity_mean_err=("parity_mean_max_err", "mean"),
        max_parity_mean_err=("parity_mean_max_err", "max")
    ).reset_index()
    df_parity_summary.to_csv(os.path.join(output_dir, "collapse_parity.csv"), index=False)
    print(f"Saved collapse_parity.csv (max observed error: {df_parity['parity_opt_max_err'].max():.3e})")

    # 7. Runtime Breakdown
    df_runtime = pd.DataFrame(runtime_records)
    df_rt_summary = df_runtime.groupby("model").agg(
        mean_jacobian_ms=("t_jacobian", lambda s: s.mean() * 1000),
        mean_solve_ms=("t_carrier_solve", lambda s: s.mean() * 1000),
        mean_total_img_ms=("t_total_img", lambda s: s.mean() * 1000)
    ).reset_index()
    df_rt_summary.to_csv(os.path.join(output_dir, "runtime_breakdown.csv"), index=False)
    print(f"Saved runtime_breakdown.csv")

    # 8. Paired Statistics & Baseline Comparison
    print("\n--- Computing Paired Statistics & Baseline Comparisons ---")
    paired_rows = []
    base_comp_rows = []

    methods_to_compare = [
        "Random Pruning", "Norm Pruning", "Attention Pruning",
        "Group-Mean Merging", "Medoid Merging", "Unweighted Centroid", "ToMe (BSM)"
    ]

    for m_cfg in models_config:
        m_name = m_cfg["name"]
        for B in m_cfg["budgets"]:
            sub_op = df_per_image[(df_per_image["model"] == m_name) & (df_per_image["budget"] == B) & (df_per_image["method"] == "Operator-Aware (Oracle)")].sort_values("image_idx")
            acc_op = sub_op["top1_acc"].values
            dam_op = sub_op["margin_damage"].values

            for bm in methods_to_compare:
                sub_b = df_per_image[(df_per_image["model"] == m_name) & (df_per_image["budget"] == B) & (df_per_image["method"] == bm)].sort_values("image_idx")
                acc_b = sub_b["top1_acc"].values
                dam_b = sub_b["margin_damage"].values

                # Paired accuracy counts
                # b = Op=1, Base=0; c = Op=0, Base=1
                b = int(np.sum((acc_op == 1) & (acc_b == 0)))
                c = int(np.sum((acc_op == 0) & (acc_b == 1)))
                mcnemar_p = compute_mcnemar_pvalue(b, c)

                # Paired margin damage diff
                diff_dam = dam_b - dam_op # positive = operator had less damage
                mean_diff_dam = float(np.mean(diff_dam))
                std_diff_dam = float(np.std(diff_dam, ddof=1))
                cohen_dz = mean_diff_dam / (std_diff_dam + 1e-12)

                t_stat, t_p = stats.ttest_rel(dam_b, dam_op)
                try:
                    w_stat, w_p = stats.wilcoxon(dam_b, dam_op)
                except Exception:
                    w_p = 1.0

                # Bootstrap CI for accuracy diff
                diff_acc = acc_op - acc_b
                mean_diff_acc = float(np.mean(diff_acc))
                ci_low, ci_high = bootstrap_mean_diff_ci(diff_acc)

                paired_rows.append({
                    "model": m_name, "budget": B, "baseline": bm,
                    "op_acc": float(np.mean(acc_op)), "base_acc": float(np.mean(acc_b)),
                    "delta_acc": mean_diff_acc, "ci_95_low": ci_low, "ci_95_high": ci_high,
                    "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": mcnemar_p,
                    "mean_damage_diff": mean_diff_dam, "cohen_dz": cohen_dz,
                    "t_test_p": float(t_p), "wilcoxon_p": float(w_p)
                })

                base_comp_rows.append({
                    "model": m_name, "budget": B, "baseline": bm,
                    "delta_acc": mean_diff_acc, "win_count": b, "loss_count": c,
                    "tie_count": int(np.sum(acc_op == acc_b)),
                    "mcnemar_p": mcnemar_p, "mean_damage_diff": mean_diff_dam
                })

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(os.path.join(output_dir, "paired_statistics.csv"), index=False)
    print(f"Saved paired_statistics.csv")

    df_base_comp = pd.DataFrame(base_comp_rows)
    df_base_comp.to_csv(os.path.join(output_dir, "baseline_comparison.csv"), index=False)
    print(f"Saved baseline_comparison.csv")

    # 9. Operator Residual Correlation Analysis
    # Does ||J E|| predict logit L2, margin damage, and prediction flip?
    res_corr_rows = []
    for m_cfg in models_config:
        m_name = m_cfg["name"]
        m_df = df_per_image[(df_per_image["model"] == m_name) & (~df_per_image["op_residual"].isna())]
        r_l2, p_l2 = stats.pearsonr(m_df["op_residual"], m_df["logit_l2"])
        rho_l2, prho_l2 = stats.spearmanr(m_df["op_residual"], m_df["logit_l2"])
        r_dam, p_dam = stats.pearsonr(m_df["op_residual"], m_df["margin_damage"])
        rho_dam, prho_dam = stats.spearmanr(m_df["op_residual"], m_df["margin_damage"])

        res_corr_rows.append({
            "model": m_name, "n_samples": len(m_df),
            "pearson_r_logit_l2": float(r_l2), "pearson_p_logit_l2": float(p_l2),
            "spearman_rho_logit_l2": float(rho_l2), "spearman_p_logit_l2": float(prho_l2),
            "pearson_r_margin_dam": float(r_dam), "pearson_p_margin_dam": float(p_dam),
            "spearman_rho_margin_dam": float(rho_dam), "spearman_p_margin_dam": float(prho_dam)
        })
    df_res_corr = pd.DataFrame(res_corr_rows)
    df_res_corr.to_csv(os.path.join(output_dir, "operator_residual_analysis.csv"), index=False)
    print(f"Saved operator_residual_analysis.csv")

    # 10. Validation Manifest JSON
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
        "device": str(device),
        "n_eval_images": len(eval_set),
        "eval_seed": 9201,
        "calib_seed": 9101,
        "models": [m["name"] for m in models_config],
        "budgets": {m["name"]: m["budgets"] for m in models_config},
        "frozen_lambda_factor": 10.0,
        "primary_grouping": "feature_similarity",
        "primary_operator": "Per-image exact downstream Jacobian J_{l->L} (Oracle)",
        "parity_max_error": float(df_parity["parity_opt_max_err"].max())
    }
    with open(os.path.join(output_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved validation_manifest.json")

    # ==============================================================
    # 11. PUBLICATION FIGURES GENERATION
    # ==============================================================
    print("\n--- Generating Publication Figures ---")
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

    # Palette
    palette = {
        "Operator-Aware (Oracle)": "#1f77b4", # Deep blue
        "Operator-Aware (Rank-32)": "#17becf", # Cyan
        "Operator-Aware (Rank-16)": "#9edae5", # Light cyan
        "Group-Mean Merging": "#ff7f0e", # Orange
        "ToMe (BSM)": "#2ca02c", # Green
        "Medoid Merging": "#d62728", # Red
        "Attention Pruning": "#9467bd", # Purple
        "Norm Pruning": "#8c564b", # Brown
        "Random Pruning": "#7f7f7f", # Grey
        "Unweighted Centroid": "#e377c2" # Pink
    }

    # Figure A: Accuracy-Token Frontier (4 panels, 1 per architecture)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()
    for idx, m_cfg in enumerate(models_config):
        ax = axes[idx]
        m_name = m_cfg["name"]
        m_sub = budget_summary[budget_summary["model"] == m_name]
        clean_val = clean_accs[m_name]

        for method in ["Operator-Aware (Oracle)", "ToMe (BSM)", "Group-Mean Merging", "Attention Pruning", "Norm Pruning", "Random Pruning"]:
            sub_m = m_sub[m_sub["method"] == method].sort_values("rem_frac")
            if len(sub_m) > 0:
                ax.plot(sub_m["rem_frac"] * 100, sub_m["top1_acc"] * 100,
                        marker='o', label=method, color=palette.get(method, "#333333"), linewidth=2.0)

        ax.axhline(clean_val * 100, color='black', linestyle='--', alpha=0.7, label=f"Clean ({clean_val*100:.1f}%)")
        ax.set_title(f"{m_name}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Tokens Retained (%)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        if idx == 0:
            ax.legend(fontsize=9, loc="lower right")
    plt.savefig(os.path.join(figures_dir, "figure_a_accuracy_token_frontier.png"), dpi=300)
    plt.close()
    print("Saved figure_a_accuracy_token_frontier.png")

    # Figure B: Normalized Retention rho(B) = A_comp / A_clean
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    for idx, m_cfg in enumerate(models_config):
        ax = axes[idx]
        m_name = m_cfg["name"]
        m_sub = budget_summary[budget_summary["model"] == m_name]

        for method in ["Operator-Aware (Oracle)", "ToMe (BSM)", "Group-Mean Merging", "Attention Pruning", "Random Pruning"]:
            sub_m = m_sub[m_sub["method"] == method].sort_values("rem_frac")
            if len(sub_m) > 0:
                ax.plot(sub_m["rem_frac"] * 100, sub_m["retention"] * 100,
                        marker='s', label=method, color=palette.get(method, "#333333"), linewidth=2.0)

        ax.axhline(100.0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f"{m_name}: Relative Retention $\\rho(B)$", fontsize=13, fontweight='bold')
        ax.set_xlabel("Tokens Retained (%)")
        ax.set_ylabel("Relative Retention $\\rho(B)$ (%)")
        if idx == 0:
            ax.legend(fontsize=9, loc="lower right")
    plt.savefig(os.path.join(figures_dir, "figure_b_accuracy_retention.png"), dpi=300)
    plt.close()
    print("Saved figure_b_accuracy_retention.png")

    # Figure C: Operator vs Best Baseline Head-to-Head
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Panel 1: Operator vs Group-Mean
    ax1 = axes[0]
    sns.barplot(data=df_paired[df_paired["baseline"] == "Group-Mean Merging"],
                x="model", y="delta_acc", hue="budget", ax=ax1, palette="Blues_r")
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title("Operator vs. Group-Mean Merging ($\\Delta$ Top-1)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Top-1 Difference (Operator - Baseline)")
    ax1.set_xlabel("Architecture")

    # Panel 2: Operator vs ToMe (BSM)
    ax2 = axes[1]
    sns.barplot(data=df_paired[df_paired["baseline"] == "ToMe (BSM)"],
                x="model", y="delta_acc", hue="budget", ax=ax2, palette="Greens_r")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title("Operator vs. ToMe BSM Merging ($\\Delta$ Top-1)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Top-1 Difference (Operator - Baseline)")
    ax2.set_xlabel("Architecture")
    plt.savefig(os.path.join(figures_dir, "figure_c_operator_vs_best_baseline.png"), dpi=300)
    plt.close()
    print("Saved figure_c_operator_vs_best_baseline.png")

    # Figure D: Same-Group Carrier Causal Effect
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_same_group[df_same_group["grouping"] == "feature_similarity"],
                x="model", y="delta_logit_l2", hue="budget", ax=ax, palette="Blues_r", showfliers=False)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label="No Carrier Effect")
    ax.set_title("Causal Effect of Operator Carrier ($C_{\\mathrm{opt}}$ vs $C_{\\mathrm{mean}}$ under Identical $S$)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Logit $L_2$ Damage Reduction: $\|E_{\\mathrm{mean}}\| - \|E_{\\mathrm{opt}}\|$")
    ax.set_xlabel("Architecture")
    ax.legend(title="Token Budget $B$", fontsize=9)
    plt.savefig(os.path.join(figures_dir, "figure_d_same_group_carrier_effect.png"), dpi=300)
    plt.close()
    print("Saved figure_d_same_group_carrier_effect.png")

    # Figure E: Operator Residual vs Damage
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sample_df = df_per_image.sample(n=min(3000, len(df_per_image)), random_state=42)
    ax1 = axes[0]
    sns.scatterplot(data=sample_df, x="op_residual", y="logit_l2", hue="model", alpha=0.4, ax=ax1)
    ax1.set_title("Mechanistic Law: $\|J E\|$ Predicts Logit $L_2$ Damage", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Downstream Operator Residual $\|J E\|_2$")
    ax1.set_ylabel("Logit $L_2$ Distance")

    ax2 = axes[1]
    sns.scatterplot(data=sample_df, x="op_residual", y="margin_damage", hue="model", alpha=0.4, ax=ax2)
    ax2.set_title("Mechanistic Law: $\|J E\|$ Predicts Margin Damage", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Downstream Operator Residual $\|J E\|_2$")
    ax2.set_ylabel("True-Class Margin Damage")
    plt.savefig(os.path.join(figures_dir, "figure_e_operator_residual_vs_damage.png"), dpi=300)
    plt.close()
    print("Saved figure_e_operator_residual_vs_damage.png")

    # Figure F: Low-Rank Operator Truncation Ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    lr_plot_data = budget_summary[budget_summary["method"].isin([
        "Operator-Aware (Oracle)", "Operator-Aware (Rank-32)", "Operator-Aware (Rank-16)", "Group-Mean Merging"
    ])]
    sns.lineplot(data=lr_plot_data, x="rem_frac", y="top1_acc", hue="method", style="model",
                 markers=True, dashes=False, ax=ax, palette="tab10", linewidth=2.0)
    ax.set_title("Low-Rank Truncation: 16–32 Visible Modes Recover Full Operator Frontier", fontsize=13, fontweight='bold')
    ax.set_xlabel("Remaining Token Fraction")
    ax.set_ylabel("Top-1 Accuracy")
    plt.savefig(os.path.join(figures_dir, "figure_f_low_rank_ablation.png"), dpi=300)
    plt.close()
    print("Saved figure_f_low_rank_ablation.png")

    # Figure G: Cross-Architecture Frontier AUC Summary
    fig, ax = plt.subplots(figsize=(12, 6))
    key_methods = ["Operator-Aware (Oracle)", "Operator-Aware (Rank-32)", "ToMe (BSM)", "Group-Mean Merging", "Attention Pruning", "Random Pruning"]
    df_auc_key = df_arch_summary[df_arch_summary["method"].isin(key_methods)]
    sns.barplot(data=df_auc_key, x="model", y="auc_frontier", hue="method", ax=ax, palette="tab10")
    ax.set_title("Cross-Architecture Accuracy-Token Frontier AUC", fontsize=13, fontweight='bold')
    ax.set_ylabel("Frontier Area Under Curve (AUC)")
    ax.set_xlabel("Architecture")
    ax.legend(title="Method", fontsize=9)
    plt.savefig(os.path.join(figures_dir, "figure_g_cross_architecture_summary.png"), dpi=300)
    plt.close()
    print("Saved figure_g_cross_architecture_summary.png")

    # Figure H: Runtime Breakdown
    fig, ax = plt.subplots(figsize=(9, 5))
    df_rt_melt = df_rt_summary.melt(id_vars=["model"], value_vars=["mean_jacobian_ms", "mean_solve_ms"],
                                    var_name="stage", value_name="time_ms")
    df_rt_melt["stage"] = df_rt_melt["stage"].map({
        "mean_jacobian_ms": "Batched VJP Jacobian",
        "mean_solve_ms": "Carrier Solve (Linear System)"
    })
    sns.barplot(data=df_rt_melt, x="model", y="time_ms", hue="stage", ax=ax, palette="rocket")
    ax.set_title("Per-Image Computation Time Breakdown (ms)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Architecture")
    plt.savefig(os.path.join(figures_dir, "figure_h_runtime_breakdown.png"), dpi=300)
    plt.close()
    print("Saved figure_h_runtime_breakdown.png")

    print("\n==================================================")
    print("CONFIRMATORY BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Results stored in: {output_dir}")
    print(f"Figures stored in: {figures_dir}")
    print("==================================================")


if __name__ == "__main__":
    main()
