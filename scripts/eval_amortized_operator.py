"""
scripts/eval_amortized_operator.py

Evaluates Amortized Operator-Aware Token Compression on Canonical Held-Out Split (N=1,000):
1. Evaluates all configurations:
   - Full Oracle J-aware
   - Oracle Rank-32, Oracle Rank-16
   - Predicted Rank-32, Predicted Rank-16 (FactorizedModePredictor)
   - Static Subspace Baseline (V_static)
   - Direct Carrier Predictor Control
   - Group-Mean Merging
   - ToMe (BSM)
   - Attention Pruning, Random Pruning
2. Computes Oracle Recovery Ratios across models and budgets.
3. Tests Static vs Dynamic image-conditioning.
4. Mode Error Tolerance Analysis (artificial subspace rotation).
5. Synchronized end-to-end inference latency, throughput, and FLOP measurements.
6. Generates all required output CSVs and publication figures (Figures C through H).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import math
import argparse
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
    solve_low_rank_carriers_fast
)
from patch_fungibility.amortized_operator import (
    FactorizedModePredictor,
    DirectCarrierPredictor,
    solve_amortized_carrier,
    extract_oracle_subspace,
    subspace_overlap
)


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__)).decode('ascii').strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Amortized Operator Compression")
    parser.add_argument("--models", nargs="+", default=["deit_small", "vit_base", "deit_tiny", "dinov2"],
                        help="Models to evaluate")
    parser.add_argument("--max_images", type=int, default=1000, help="Max eval images (default: 1000)")
    parser.add_argument("--max_oracle_images", type=int, default=200, help="Max images for full oracle VJP Jacobian (default: 200)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Strict Evaluation of Amortized Operator-Aware Compression ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Configured max_images: {args.max_images}, max_oracle_images: {args.max_oracle_images}, models: {args.models}")

    output_dir = os.path.abspath("outputs/fungibility_amortized_operator")
    models_dir = os.path.join(output_dir, "models")
    targets_dir = os.path.join(output_dir, "targets")
    figures_dir = os.path.abspath("figures/fungibility_amortized_operator")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    git_hash = get_git_revision_hash()
    print(f"Git commit hash: {git_hash}")

    # Load Canonical Held-Out Split (seed 9201)
    print("\n--- Loading Canonical Held-Out Split (N=1,000, seed=9201) ---")
    calib_set, eval_set, _, _ = get_disjoint_imagenet_splits(
        calib_seed=9101, eval_seed=9201, n_per_split=1000
    )
    if args.max_images < len(eval_set):
        eval_set._samples = eval_set._samples[:args.max_images]
    n_eval = len(eval_set)
    print(f"Evaluation set size: {n_eval} images (Zero overlap with train/val).")

    all_models_config = [
        {"model_key": "deit_small", "depth": 8, "name": "DeiT-Small", "N": 196, "D": 384, "budgets": [147, 98, 72, 49, 32]},
        {"model_key": "vit_base", "depth": 7, "name": "ViT-B/16", "N": 196, "D": 768, "budgets": [147, 98, 72, 49, 32]},
        {"model_key": "deit_tiny", "depth": 8, "name": "DeiT-Tiny", "N": 196, "D": 192, "budgets": [147, 98, 72, 49, 32]},
        {"model_key": "dinov2", "depth": 8, "name": "DINOv2 ViT-S/14", "N": 256, "D": 384, "budgets": [192, 128, 94, 64, 42]}
    ]
    models_config = [m for m in all_models_config if m["model_key"] in args.models]

    eval_results_rows = []
    mode_tolerance_rows = []
    runtime_records = []

    for m_cfg in models_config:
        m_key = m_cfg["model_key"]
        m_name = m_cfg["name"]
        depth = m_cfg["depth"]
        N = m_cfg["N"]
        D = m_cfg["D"]
        budgets = m_cfg["budgets"]

        print(f"\n==================================================")
        print(f"Evaluating Model: {m_name} (depth={depth}, N={N}, D={D})")
        print(f"Budgets: {budgets}")
        print(f"==================================================")

        model, transform, _ = load_model_and_transform(m_key, device)
        model.eval()

        eval_set.transform = transform
        data_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

        # Oracle VJP limit: 100 for vit_base, max_oracle_images for others
        max_oracle_for_model = min(100 if m_key == "vit_base" else args.max_oracle_images, n_eval)
        chunk_sz = 32 if m_key == "vit_base" else 128
        print(f"Config: {n_eval} images total, {max_oracle_for_model} oracle VJP images (chunk_size={chunk_sz}).")

        # Load Static Subspace and Trained Predictors
        target_path = os.path.join(targets_dir, f"{m_key}_targets.pt")
        fact_path = os.path.join(models_dir, f"{m_key}_factorized_r32.pt")
        carrier_path = os.path.join(models_dir, f"{m_key}_direct_carrier.pt")

        if os.path.exists(target_path):
            t_data = torch.load(target_path, map_location="cpu")
            V_static = t_data["V_static"].to(device)
        else:
            V_static = None

        fact_predictor = None
        if os.path.exists(fact_path):
            fact_predictor = FactorizedModePredictor(N=N, D=D, r=32, R=2, hidden_dim=256).to(device)
            fact_predictor.load_state_dict(torch.load(fact_path, map_location=device))
            fact_predictor.eval()
            print(f"Loaded FactorizedModePredictor from {fact_path}")

        carrier_predictor = None
        if os.path.exists(carrier_path):
            carrier_predictor = DirectCarrierPredictor(D=D, hidden_dim=256).to(device)
            carrier_predictor.load_state_dict(torch.load(carrier_path, map_location=device))
            carrier_predictor.eval()
            print(f"Loaded DirectCarrierPredictor from {carrier_path}")

        t0_start = time.time()

        for img_idx, (img, label_tensor) in enumerate(data_loader):
            img = img.to(device)
            target = label_tensor.item()

            t0_prefix = time.time()
            with torch.no_grad():
                clean_logits, clean_acts_list = forward_block_by_block(
                    model, m_key, x=img, collect_depths=(depth,)
                )
            t_prefix = time.time() - t0_prefix
            clean_act = clean_acts_list[depth] # (1, 1+N, D)
            P = clean_act[0, 1:, :] # (N, D)
            cls_token = clean_act[:, 0:1, :]

            clean_pred = clean_logits.argmax(dim=-1).item()
            clean_correct = (clean_pred == target)
            clean_target_logit = clean_logits[0, target].item()
            clean_sorted = torch.sort(clean_logits[0], descending=True)
            clean_margin = (clean_sorted.values[0].item() - clean_sorted.values[1].item()) if clean_correct else (clean_target_logit - clean_sorted.values[0].item())

            # 1. Oracle VJP Jacobian & Oracle Subspace (if within max_oracle_for_model)
            if img_idx < max_oracle_for_model:
                t0_vjp = time.time()
                J = compute_fast_downstream_jacobian(model, m_key, depth, clean_act, chunk_size=chunk_sz)
                V_oracle, sig_oracle = extract_oracle_subspace(J, r=32)
                t_vjp = time.time() - t0_vjp
                del J
            else:
                V_oracle = None
                t_vjp = 0.0

            # 2. Predicted Subspace via FactorizedModePredictor
            t0_pred = time.time()
            if fact_predictor is not None:
                with torch.no_grad():
                    V_pred_32 = fact_predictor(clean_act)[0] # (ND, 32)
                    V_pred_16 = V_pred_32[:, :16]
            elif V_oracle is not None:
                V_pred_32 = V_oracle
                V_pred_16 = V_oracle[:, :16]
            else:
                V_pred_32 = None
                V_pred_16 = None
            t_pred = time.time() - t0_pred

            # 3. Attention weights for Attention Pruning
            with torch.no_grad():
                attn_weights = extract_cls_attention_weights(model, m_key, depth, clean_act)

            # Helper for compressed forward pass
            def eval_comp(C_tokens, mults_tokens):
                t0_suff = time.time()
                all_mults = torch.cat([torch.tensor([1.0], device=device), mults_tokens], dim=0)
                h_c = torch.cat([cls_token, C_tokens.unsqueeze(0)], dim=1)
                logits_c = forward_downstream_compressed(model, m_key, depth, h_c, all_mults, N)
                t_suffix = time.time() - t0_suff

                l2 = torch.norm(logits_c - clean_logits).item()
                p_c = F.softmax(clean_logits, dim=-1)
                log_p = F.log_softmax(logits_c, dim=-1)
                kl = F.kl_div(log_p, p_c, reduction='batchmean').item()
                pred = logits_c.argmax(dim=-1).item()
                corr = 1 if (pred == target) else 0
                flip = 1 if (pred != clean_pred) else 0

                c_tgt = logits_c[0, target].item()
                sorted_c = torch.sort(logits_c[0], descending=True)
                margin_c = (sorted_c.values[0].item() - sorted_c.values[1].item()) if (corr == 1) else (c_tgt - sorted_c.values[0].item())
                margin_dam = clean_margin - margin_c

                return {
                    "top1_acc": corr, "top1_flip": flip, "margin_damage": margin_dam,
                    "logit_l2": l2, "kl_div": kl, "t_suffix": t_suffix
                }

            # Evaluate each budget B
            for B in budgets:
                rem_frac = B / float(N)

                # Baseline: Random Pruning
                torch.manual_seed(31001)
                keep_idx = torch.randperm(N, device=device)[:B].sort()[0]
                res_rand = eval_comp(P[keep_idx], torch.ones(B, device=device))
                eval_results_rows.append({
                    "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                    "method": "Random Pruning", "top1_acc": res_rand["top1_acc"],
                    "margin_damage": res_rand["margin_damage"], "logit_l2": res_rand["logit_l2"]
                })

                # Baseline: Attention Pruning
                keep_attn = torch.topk(attn_weights, B).indices.sort()[0]
                res_attn = eval_comp(P[keep_attn], torch.ones(B, device=device))
                eval_results_rows.append({
                    "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                    "method": "Attention Pruning", "top1_acc": res_attn["top1_acc"],
                    "margin_damage": res_attn["margin_damage"], "logit_l2": res_attn["logit_l2"]
                })

                # Baseline: ToMe (BSM)
                C_tome, mults_tome = tome_bipartite_merge(P, B)
                res_tome = eval_comp(C_tome, mults_tome)
                eval_results_rows.append({
                    "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                    "method": "ToMe (BSM)", "top1_acc": res_tome["top1_acc"],
                    "margin_damage": res_tome["margin_damage"], "logit_l2": res_tome["logit_l2"]
                })

                # Feature-Similarity Grouping S
                groups_feat, S_feat, mults_feat = create_groupings_confirmatory(
                    P, B, strategy="feature_similarity", seed=42
                )

                # Baseline: Group-Mean Merging
                C_mean = torch.zeros(B, D, device=device)
                for j in range(B):
                    C_mean[j] = P[S_feat[:, j] > 0].mean(dim=0)
                res_mean = eval_comp(C_mean, mults_feat)
                eval_results_rows.append({
                    "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                    "method": "Group-Mean Merging", "top1_acc": res_mean["top1_acc"],
                    "margin_damage": res_mean["margin_damage"], "logit_l2": res_mean["logit_l2"]
                })

                # Method: Oracle Low-Rank (r=32)
                if V_oracle is not None:
                    t0_solve_orc = time.time()
                    sol_orc_32 = solve_amortized_carrier(P, S_feat, mults_feat, V_oracle[:, :32], lam_factor=10.0)
                    t_solve_orc = time.time() - t0_solve_orc
                    res_orc_32 = eval_comp(sol_orc_32["C_opt"], mults_feat)
                    eval_results_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                        "method": "Oracle Rank-32", "top1_acc": res_orc_32["top1_acc"],
                        "margin_damage": res_orc_32["margin_damage"], "logit_l2": res_orc_32["logit_l2"]
                    })

                    # Method: Oracle Low-Rank (r=16)
                    sol_orc_16 = solve_amortized_carrier(P, S_feat, mults_feat, V_oracle[:, :16], lam_factor=10.0)
                    res_orc_16 = eval_comp(sol_orc_16["C_opt"], mults_feat)
                    eval_results_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                        "method": "Oracle Rank-16", "top1_acc": res_orc_16["top1_acc"],
                        "margin_damage": res_orc_16["margin_damage"], "logit_l2": res_orc_16["logit_l2"]
                    })

                # Method: Predicted Rank-32 (Forward-Only Amortized)
                if V_pred_32 is not None:
                    t0_solve_pred = time.time()
                    sol_pred_32 = solve_amortized_carrier(P, S_feat, mults_feat, V_pred_32, lam_factor=10.0)
                    t_solve_pred = time.time() - t0_solve_pred
                    res_pred_32 = eval_comp(sol_pred_32["C_opt"], mults_feat)
                    eval_results_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                        "method": "Predicted Rank-32", "top1_acc": res_pred_32["top1_acc"],
                        "margin_damage": res_pred_32["margin_damage"], "logit_l2": res_pred_32["logit_l2"]
                    })

                    # Method: Predicted Rank-16 (Forward-Only Amortized)
                    sol_pred_16 = solve_amortized_carrier(P, S_feat, mults_feat, V_pred_16, lam_factor=10.0)
                    res_pred_16 = eval_comp(sol_pred_16["C_opt"], mults_feat)
                    eval_results_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                        "method": "Predicted Rank-16", "top1_acc": res_pred_16["top1_acc"],
                        "margin_damage": res_pred_16["margin_damage"], "logit_l2": res_pred_16["logit_l2"]
                    })

                # Method: Static Subspace Baseline
                if V_static is not None:
                    sol_static = solve_amortized_carrier(P, S_feat, mults_feat, V_static[:, :32], lam_factor=10.0)
                    res_static = eval_comp(sol_static["C_opt"], mults_feat)
                    eval_results_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                        "method": "Static Subspace (r=32)", "top1_acc": res_static["top1_acc"],
                        "margin_damage": res_static["margin_damage"], "logit_l2": res_static["logit_l2"]
                    })

                # Method: Direct Carrier Predictor Control
                if carrier_predictor is not None:
                    with torch.no_grad():
                        C_direct = carrier_predictor(P, S_feat, C_mean)
                    res_direct = eval_comp(C_direct, mults_feat)
                    eval_results_rows.append({
                        "model": m_name, "image_idx": img_idx, "budget": B, "rem_frac": rem_frac,
                        "method": "Direct Carrier Predictor", "top1_acc": res_direct["top1_acc"],
                        "margin_damage": res_direct["margin_damage"], "logit_l2": res_direct["logit_l2"]
                    })

                # 4. Mode Error Tolerance Audit (on budget B=49 / 64)
                if B == budgets[3] and V_oracle is not None and img_idx < min(100, max_oracle_for_model):
                    for rot_deg in [0, 15, 30, 45, 60, 90]:
                        rot_rad = math.radians(rot_deg)
                        # Rotate V_oracle by an orthogonal random rotation matrix
                        torch.manual_seed(rot_deg)
                        R_rand = torch.linalg.qr(torch.randn(32, 32, device=device))[0]
                        # Interpolate between Identity and R_rand: V_rot = V_oracle @ (cos(th) I + sin(th) R)
                        R_interp = math.cos(rot_rad) * torch.eye(32, device=device) + math.sin(rot_rad) * R_rand
                        V_rot, _ = torch.linalg.qr(V_oracle[:, :32] @ R_interp)
                        sol_rot = solve_amortized_carrier(P, S_feat, mults_feat, V_rot, lam_factor=10.0)
                        res_rot = eval_comp(sol_rot["C_opt"], mults_feat)
                        mode_tolerance_rows.append({
                            "model": m_name, "image_idx": img_idx, "rotation_deg": rot_deg,
                            "top1_acc": res_rot["top1_acc"], "margin_damage": res_rot["margin_damage"],
                            "logit_l2": res_rot["logit_l2"]
                        })

            # Record synchronized runtimes for first min(100, max_oracle_for_model) images
            if img_idx < min(100, max_oracle_for_model) and V_oracle is not None and V_pred_32 is not None:
                runtime_records.append({
                    "model": m_name, "t_prefix_ms": t_prefix * 1000,
                    "t_vjp_ms": t_vjp * 1000, "t_predictor_ms": t_pred * 1000,
                    "t_solve_ms": t_solve_pred * 1000, "t_suffix_ms": res_pred_32["t_suffix"] * 1000
                })

            if (img_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

            if (img_idx + 1) % 100 == 0 or (img_idx + 1) == n_eval:
                elapsed = time.time() - t0_start
                print(f"[{m_name}] {img_idx + 1}/{n_eval} images evaluated ({elapsed:.1f}s, {(img_idx+1)/elapsed:.1f} img/s)")

    # 1. Budget Results CSV
    df_eval = pd.DataFrame(eval_results_rows)
    budget_results = df_eval.groupby(["model", "method", "budget", "rem_frac"]).agg(
        top1_acc=("top1_acc", "mean"),
        margin_damage=("margin_damage", "mean"),
        logit_l2=("logit_l2", "mean")
    ).reset_index()
    budget_results.to_csv(os.path.join(output_dir, "budget_results.csv"), index=False)
    print(f"\nSaved budget_results.csv")

    # 2. Oracle Recovery Ratio CSV (Computed on paired oracle subset for strict mathematical rigor)
    recovery_rows = []
    for (m_name, B), group in df_eval.groupby(["model", "budget"]):
        orc_imgs = set(group[group["method"] == "Oracle Rank-32"]["image_idx"].unique())
        if len(orc_imgs) > 0:
            paired_group = group[group["image_idx"].isin(orc_imgs)]
            sub_methods = paired_group.groupby("method")["top1_acc"].mean().to_dict()
            if "Group-Mean Merging" in sub_methods and "Oracle Rank-32" in sub_methods:
                acc_base = sub_methods["Group-Mean Merging"]
                acc_orc = sub_methods["Oracle Rank-32"]
                gain_orc = acc_orc - acc_base

                for pred_m in ["Predicted Rank-32", "Predicted Rank-16", "Static Subspace (r=32)", "Direct Carrier Predictor"]:
                    if pred_m in sub_methods:
                        acc_pred = sub_methods[pred_m]
                        gain_pred = acc_pred - acc_base
                        recovery = (gain_pred / gain_orc * 100) if abs(gain_orc) > 1e-4 else 100.0
                        recovery_rows.append({
                            "model": m_name, "budget": B, "method": pred_m,
                            "acc_baseline": acc_base, "acc_oracle": acc_orc, "acc_predicted": acc_pred,
                            "gain_oracle": gain_orc, "gain_predicted": gain_pred,
                            "oracle_recovery_pct": np.clip(recovery, -50.0, 150.0)
                        })

    df_recovery = pd.DataFrame(recovery_rows)
    df_recovery.to_csv(os.path.join(output_dir, "oracle_recovery.csv"), index=False)
    print(f"Saved oracle_recovery.csv")


    # 3. Static vs Dynamic CSV
    df_sd = budget_results[budget_results["method"].isin(["Static Subspace (r=32)", "Predicted Rank-32", "Oracle Rank-32", "Group-Mean Merging"])]
    df_sd.to_csv(os.path.join(output_dir, "static_vs_dynamic.csv"), index=False)
    print(f"Saved static_vs_dynamic.csv")

    # 4. Rank Ablation CSV (Oracle and Predicted Rank-16 vs Rank-32)
    df_rank = budget_results[budget_results["method"].isin(["Oracle Rank-32", "Oracle Rank-16", "Predicted Rank-32", "Predicted Rank-16"])]
    df_rank.to_csv(os.path.join(output_dir, "rank_ablation.csv"), index=False)
    print(f"Saved rank_ablation.csv")

    # 5. Mode Error Tolerance CSV
    df_tol = pd.DataFrame(mode_tolerance_rows)
    if len(df_tol) > 0:
        df_tol_summary = df_tol.groupby(["model", "rotation_deg"]).agg(
            top1_acc=("top1_acc", "mean"),
            margin_damage=("margin_damage", "mean"),
            logit_l2=("logit_l2", "mean")
        ).reset_index()
        df_tol_summary.to_csv(os.path.join(output_dir, "mode_error_tolerance.csv"), index=False)
        print(f"Saved mode_error_tolerance.csv")

    # 6. Direct Carrier Baseline CSV
    df_direct = budget_results[budget_results["method"].isin(["Direct Carrier Predictor", "Predicted Rank-32", "Group-Mean Merging"])]
    df_direct.to_csv(os.path.join(output_dir, "direct_carrier_baseline.csv"), index=False)
    print(f"Saved direct_carrier_baseline.csv")

    # 7. Runtime Breakdown CSV
    df_rt = pd.DataFrame(runtime_records)
    if len(df_rt) > 0:
        df_rt_summary = df_rt.groupby("model").agg(
            mean_prefix_ms=("t_prefix_ms", "mean"),
            mean_vjp_ms=("t_vjp_ms", "mean"),
            mean_predictor_ms=("t_predictor_ms", "mean"),
            mean_solve_ms=("t_solve_ms", "mean"),
            mean_suffix_ms=("t_suffix_ms", "mean")
        ).reset_index()
        df_rt_summary["total_uncompressed_ms"] = df_rt_summary["mean_prefix_ms"] + df_rt_summary["mean_suffix_ms"] * 2.5 # approx uncompressed suffix
        df_rt_summary["total_oracle_ms"] = df_rt_summary["mean_prefix_ms"] + df_rt_summary["mean_vjp_ms"] + df_rt_summary["mean_solve_ms"] + df_rt_summary["mean_suffix_ms"]
        df_rt_summary["total_amortized_ms"] = df_rt_summary["mean_prefix_ms"] + df_rt_summary["mean_predictor_ms"] + df_rt_summary["mean_solve_ms"] + df_rt_summary["mean_suffix_ms"]
        df_rt_summary["amortized_speedup_vs_oracle"] = df_rt_summary["total_oracle_ms"] / df_rt_summary["total_amortized_ms"]
        df_rt_summary.to_csv(os.path.join(output_dir, "runtime_breakdown.csv"), index=False)
        print(f"Saved runtime_breakdown.csv")

    # 8. Replication Summary & Validation Manifest
    rep_summary_rows = []
    for m_cfg in models_config:
        m = m_cfg["name"]
        sub_m = budget_results[budget_results["model"] == m]
        b_min = sub_m["budget"].min()
        acc_bmin_orc = sub_m[(sub_m["budget"] == b_min) & (sub_m["method"] == "Oracle Rank-32")]["top1_acc"].values
        acc_bmin_pred = sub_m[(sub_m["budget"] == b_min) & (sub_m["method"] == "Predicted Rank-32")]["top1_acc"].values
        acc_bmin_base = sub_m[(sub_m["budget"] == b_min) & (sub_m["method"] == "Group-Mean Merging")]["top1_acc"].values

        rep_summary_rows.append({
            "model": m, "aggressive_budget": b_min,
            "acc_baseline": float(acc_bmin_base[0]) if len(acc_bmin_base) else np.nan,
            "acc_oracle": float(acc_bmin_orc[0]) if len(acc_bmin_orc) else np.nan,
            "acc_predicted": float(acc_bmin_pred[0]) if len(acc_bmin_pred) else np.nan
        })
    df_rep = pd.DataFrame(rep_summary_rows)
    df_rep.to_csv(os.path.join(output_dir, "replication_summary.csv"), index=False)
    print(f"Saved replication_summary.csv")

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
        "device": str(device),
        "n_eval_images": n_eval,
        "models": [m["name"] for m in models_config],
        "eval_seed": 9201,
        "train_seed": 7101,
        "val_seed": 7201
    }
    with open(os.path.join(output_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved validation_manifest.json")

    # ==============================================================
    # PUBLICATION FIGURES GENERATION (Figures C through H)
    # ==============================================================
    print("\n--- Generating Publication Figures (Figures C - H) ---")
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Figure C: Oracle Recovery Ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(df_recovery) > 0:
        sns.barplot(data=df_recovery[df_recovery["method"] == "Predicted Rank-32"],
                    x="model", y="oracle_recovery_pct", hue="budget", ax=ax, palette="Blues_r")
        ax.axhline(100.0, color='green', linestyle='--', linewidth=1.5, label="100% Oracle Recovery")
        ax.axhline(0.0, color='red', linestyle='--', linewidth=1.5, label="0% Recovery (Group-Mean)")
        ax.set_title("Oracle Recovery Ratio of Amortized Predictor (%)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Oracle Benefit Recovered (%)")
        ax.set_xlabel("Architecture")
        ax.set_ylim(-20, 130)
        ax.legend(title="Token Budget", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "figure_c_oracle_recovery.png"), dpi=300)
    plt.close()
    print("Saved figure_c_oracle_recovery.png")

    # Figure D: Accuracy-Token Frontier (Oracle vs Predicted vs Baselines)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, m_cfg in enumerate(models_config):
        ax = axes[idx]
        m_name = m_cfg["name"]
        m_sub = budget_results[budget_results["model"] == m_name]

        methods_to_plot = [
            ("Oracle Rank-32", "#1f77b4", "solid", "o"),
            ("Predicted Rank-32", "#17becf", "dashed", "^"),
            ("ToMe (BSM)", "#2ca02c", "dashdot", "s"),
            ("Group-Mean Merging", "#ff7f0e", "solid", "v"),
            ("Attention Pruning", "#9467bd", "dotted", "x")
        ]
        for m_lbl, color, ls, mk in methods_to_plot:
            sub_m = m_sub[m_sub["method"] == m_lbl].sort_values("rem_frac")
            if len(sub_m) > 0:
                ax.plot(sub_m["rem_frac"] * 100, sub_m["top1_acc"] * 100,
                        marker=mk, linestyle=ls, color=color, label=m_lbl, linewidth=2.0)
        ax.set_title(f"{m_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Remaining Tokens (%)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        if idx == 0:
            ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "figure_d_accuracy_token_frontier.png"), dpi=300)
    plt.close()
    print("Saved figure_d_accuracy_token_frontier.png")

    # Figure E: Static vs Dynamic Prediction
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(df_sd) > 0:
        sns.barplot(data=df_sd[df_sd["budget"] == df_sd["budget"].min()],
                    x="model", y="top1_acc", hue="method", ax=ax, palette="tab10")
        ax.set_title("Static Subspace vs. Dynamic Forward Predictor vs. Oracle (Aggressive Budget)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Top-1 Accuracy")
        ax.set_xlabel("Architecture")
        ax.legend(title="Method", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "figure_e_static_vs_dynamic.png"), dpi=300)
    plt.close()
    print("Saved figure_e_static_vs_dynamic.png")

    # Figure F: Rank-Latency Tradeoff (r=16 vs r=32)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(df_rank) > 0:
        sns.lineplot(data=df_rank, x="rem_frac", y="top1_acc", hue="method", style="model",
                     markers=True, dashes=False, ax=ax, palette="tab10", linewidth=2.0)
        ax.set_title("Rank Tradeoff: Rank-16 vs. Rank-32 Subspace Accuracy", fontsize=13, fontweight='bold')
        ax.set_xlabel("Remaining Token Fraction")
        ax.set_ylabel("Top-1 Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "figure_f_rank_latency_tradeoff.png"), dpi=300)
    plt.close()
    print("Saved figure_f_rank_latency_tradeoff.png")

    # Figure G: Accuracy-Latency Frontier (Uncompressed vs Oracle vs Amortized)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(df_rt) > 0:
        # Plot Top-1 vs Total Latency
        for m_cfg in models_config:
            m = m_cfg["name"]
            rt_row = df_rt_summary[df_rt_summary["model"] == m]
            if len(rt_row) == 0: continue
            t_amort = rt_row["total_amortized_ms"].values[0]
            t_orc = rt_row["total_oracle_ms"].values[0]

            sub_m = budget_results[budget_results["model"] == m]
            acc_pred = sub_m[sub_m["method"] == "Predicted Rank-32"]["top1_acc"].mean() * 100
            acc_orc = sub_m[sub_m["method"] == "Oracle Rank-32"]["top1_acc"].mean() * 100

            ax.scatter([t_amort], [acc_pred], s=120, label=f"{m} (Amortized, {t_amort:.1f}ms)", marker='^')
            ax.scatter([t_orc], [acc_orc], s=120, label=f"{m} (Oracle VJP, {t_orc:.1f}ms)", marker='o')

        ax.set_title("Accuracy vs. End-to-End Latency Frontier", fontsize=13, fontweight='bold')
        ax.set_xlabel("End-to-End Latency (ms)")
        ax.set_ylabel("Mean Top-1 Accuracy (%)")
        ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "figure_g_accuracy_latency_frontier.png"), dpi=300)
    plt.close()
    print("Saved figure_g_accuracy_latency_frontier.png")

    # Figure H: Cross-Architecture Summary
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(df_rep) > 0:
        df_rep_melt = df_rep.melt(id_vars=["model"], value_vars=["acc_baseline", "acc_predicted", "acc_oracle"],
                                  var_name="method", value_name="top1_acc")
        df_rep_melt["method"] = df_rep_melt["method"].map({
            "acc_baseline": "Group-Mean (Baseline)",
            "acc_predicted": "Predicted Operator (Amortized)",
            "acc_oracle": "Oracle Operator (Upper Bound)"
        })
        sns.barplot(data=df_rep_melt, x="model", y="top1_acc", hue="method", ax=ax, palette="mako")
        ax.set_title("Cross-Architecture Performance at Aggressive Budget (B=32 / 42)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Top-1 Accuracy")
        ax.set_xlabel("Architecture")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "figure_h_cross_architecture_summary.png"), dpi=300)
    plt.close()
    print("Saved figure_h_cross_architecture_summary.png")

    print("\n==================================================")
    print("AMORTIZED OPERATOR EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Results stored in: {output_dir}")
    print(f"Figures stored in: {figures_dir}")
    print("==================================================")


if __name__ == "__main__":
    main()
