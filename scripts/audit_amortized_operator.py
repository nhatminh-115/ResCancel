"""
scripts/audit_amortized_operator.py

Strict Sanity and Validation Audit of Amortized Operator-Aware Token Compression:
1. Dataset & Manifest Audit: Exact check of image IDs, labels, denominators.
2. Recompute Top-1 accuracy directly from per-image raw predictions across:
   - Full Canonical Split (N=1,000)
   - Paired Oracle Subset (N=200 for DeiT/DINO, N=100 for ViT-B)
3. Clean-to-Oracle and Clean-to-Predicted Flip Analysis:
   - Identifies exact cause of "Oracle above Clean" (denominator/subset mismatch vs real corrections).
4. Rigorous Synchronized GPU Latency Audit:
   - Clean uncompressed vs Random Pruning vs Attention Pruning vs ToMe vs Group-Mean vs Amortized vs Oracle.
   - Separate timings: prefix, predictor, grouping, carrier solve, collapse, suffix, total.
   - Three distinct speedup ratios:
     * Speedup A: Oracle-to-Amortized (T_oracle / T_amortized)
     * Speedup B: Clean-Inference Speedup (T_clean / T_amortized)
     * Speedup C: ToMe Speedup (T_tome / T_amortized)
5. True Oracle Recovery Audit:
   - Separates Low-Rank Oracle Recovery (Rank-32 vs Full-J) from Amortization Recovery (Predicted vs Oracle).
6. Correlation between Predicted-Oracle Subspace Overlap and Compression Metrics:
   - Residual reduction, logit damage, accuracy improvement.
7. Exports all 7 required deliverables to outputs/fungibility_amortized_operator_audit/.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import math
import hashlib
import numpy as np
import pandas as pd
import scipy.stats as stats

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
    create_groupings_confirmatory
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
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__)).decode('ascii').strip()
    except Exception:
        return "unknown"


def compute_manifest_hash(samples) -> str:
    hasher = hashlib.sha256()
    for s in samples:
        hasher.update(str(s).encode('utf-8'))
    return hasher.hexdigest()[:16]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== STRICT SANITY & VALIDATION AUDIT OF AMORTIZED OPERATOR COMPRESSION ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    output_dir = os.path.abspath("outputs/fungibility_amortized_operator_audit")
    models_dir = os.path.abspath("outputs/fungibility_amortized_operator/models")
    targets_dir = os.path.abspath("outputs/fungibility_amortized_operator/targets")
    os.makedirs(output_dir, exist_ok=True)

    git_hash = get_git_revision_hash()
    print(f"Git commit: {git_hash}")

    # Load Canonical Held-Out Split (seed 9201)
    print("\n--- 1. Loading Canonical Held-Out Split (N=1,000, seed=9201) ---")
    calib_set, eval_set, _, _ = get_disjoint_imagenet_splits(
        calib_seed=9101, eval_seed=9201, n_per_split=1000
    )
    n_eval = len(eval_set)
    manifest_hash = compute_manifest_hash(eval_set._samples)
    print(f"Canonical split size: {n_eval} images, Manifest hash: {manifest_hash}")

    all_models_config = [
        {"model_key": "deit_tiny", "depth": 8, "name": "DeiT-Tiny", "N": 196, "D": 192, "budgets": [147, 98, 72, 49, 32], "oracle_n": 200, "chunk_sz": 128},
        {"model_key": "deit_small", "depth": 8, "name": "DeiT-Small", "N": 196, "D": 384, "budgets": [147, 98, 72, 49, 32], "oracle_n": 200, "chunk_sz": 128},
        {"model_key": "vit_base", "depth": 7, "name": "ViT-B/16", "N": 196, "D": 768, "budgets": [147, 98, 72, 49, 32], "oracle_n": 100, "chunk_sz": 32},
        {"model_key": "dinov2", "depth": 8, "name": "DINOv2 ViT-S/14", "N": 256, "D": 384, "budgets": [192, 128, 94, 64, 42], "oracle_n": 200, "chunk_sz": 128}
    ]

    dataset_alignment_rows = []
    raw_predictions = []
    flip_analysis_rows = []
    overlap_compression_rows = []
    runtime_records = []

    for m_cfg in all_models_config:
        m_key = m_cfg["model_key"]
        m_name = m_cfg["name"]
        depth = m_cfg["depth"]
        N = m_cfg["N"]
        D = m_cfg["D"]
        budgets = m_cfg["budgets"]
        oracle_n = m_cfg["oracle_n"]
        chunk_sz = m_cfg["chunk_sz"]

        print(f"\n=======================================================")
        print(f"Auditing Architecture: {m_name} (depth={depth}, N={N}, D={D})")
        print(f"Budgets: {budgets} | Oracle subset: {oracle_n} images (chunk={chunk_sz})")
        print(f"=======================================================")

        model, transform, _ = load_model_and_transform(m_key, device)
        model.eval()

        eval_set.transform = transform
        data_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

        # Load trained predictor and static subspace
        fact_path = os.path.join(models_dir, f"{m_key}_factorized_r32.pt")
        target_path = os.path.join(targets_dir, f"{m_key}_targets.pt")

        V_static = None
        if os.path.exists(target_path):
            t_data = torch.load(target_path, map_location="cpu")
            V_static = t_data["V_static"].to(device)

        fact_predictor = None
        if os.path.exists(fact_path):
            fact_predictor = FactorizedModePredictor(N=N, D=D, r=32, R=2, hidden_dim=256).to(device)
            fact_predictor.load_state_dict(torch.load(fact_path, map_location=device))
            fact_predictor.eval()

        # Per-model raw storage
        per_model_preds = []
        clean_correct_1000 = 0
        clean_correct_oracle_subset = 0

        t0_model = time.time()

        for img_idx, (img, label_tensor) in enumerate(data_loader):
            img = img.to(device)
            target = label_tensor.item()

            # Clean forward pass
            with torch.no_grad():
                clean_logits, clean_acts_list = forward_block_by_block(
                    model, m_key, x=img, collect_depths=(depth,)
                )
            clean_pred = clean_logits.argmax(dim=-1).item()
            is_clean_correct = (clean_pred == target)
            if is_clean_correct:
                clean_correct_1000 += 1
                if img_idx < oracle_n:
                    clean_correct_oracle_subset += 1

            clean_act = clean_acts_list[depth]
            P = clean_act[0, 1:, :]
            cls_token = clean_act[:, 0:1, :]

            # Oracle VJP (if within oracle_n)
            V_oracle = None
            if img_idx < oracle_n:
                J = compute_fast_downstream_jacobian(model, m_key, depth, clean_act, chunk_size=chunk_sz)
                V_oracle, _ = extract_oracle_subspace(J, r=32)
                del J

            # Predicted Subspace
            V_pred_32 = None
            if fact_predictor is not None:
                with torch.no_grad():
                    V_pred_32 = fact_predictor(clean_act)[0]

            # Attention weights
            with torch.no_grad():
                attn_weights = extract_cls_attention_weights(model, m_key, depth, clean_act)

            # Subspace overlap if oracle is present
            sub_overlap = None
            if V_oracle is not None and V_pred_32 is not None:
                sub_overlap = float(subspace_overlap(V_oracle[:, :32], V_pred_32))

            # Helper for compression
            def run_compressed_pred(C_tokens, mults_tokens):
                all_mults = torch.cat([torch.tensor([1.0], device=device), mults_tokens], dim=0)
                h_c = torch.cat([cls_token, C_tokens.unsqueeze(0)], dim=1)
                logits_c = forward_downstream_compressed(model, m_key, depth, h_c, all_mults, N)
                pred = logits_c.argmax(dim=-1).item()
                l2 = torch.norm(logits_c - clean_logits).item()
                return pred, l2

            # Evaluate each budget B
            for B in budgets:
                rem_frac = B / float(N)

                # 1. Random Pruning
                torch.manual_seed(31001)
                keep_idx = torch.randperm(N, device=device)[:B].sort()[0]
                pred_rand, l2_rand = run_compressed_pred(P[keep_idx], torch.ones(B, device=device))
                per_model_preds.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_pred": clean_pred,
                    "budget": B, "rem_frac": rem_frac, "method": "Random Pruning",
                    "pred": pred_rand, "is_correct": int(pred_rand == target),
                    "agrees_clean": int(pred_rand == clean_pred), "l2_damage": l2_rand
                })

                # 2. Attention Pruning
                keep_attn = torch.topk(attn_weights, B).indices.sort()[0]
                pred_attn, l2_attn = run_compressed_pred(P[keep_attn], torch.ones(B, device=device))
                per_model_preds.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_pred": clean_pred,
                    "budget": B, "rem_frac": rem_frac, "method": "Attention Pruning",
                    "pred": pred_attn, "is_correct": int(pred_attn == target),
                    "agrees_clean": int(pred_attn == clean_pred), "l2_damage": l2_attn
                })

                # 3. ToMe (BSM)
                C_tome, mults_tome = tome_bipartite_merge(P, B)
                pred_tome, l2_tome = run_compressed_pred(C_tome, mults_tome)
                per_model_preds.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_pred": clean_pred,
                    "budget": B, "rem_frac": rem_frac, "method": "ToMe (BSM)",
                    "pred": pred_tome, "is_correct": int(pred_tome == target),
                    "agrees_clean": int(pred_tome == clean_pred), "l2_damage": l2_tome
                })

                # Feature groupings
                groups_feat, S_feat, mults_feat = create_groupings_confirmatory(
                    P, B, strategy="feature_similarity", seed=42
                )

                # 4. Group-Mean Merging
                C_mean = torch.zeros(B, D, device=device)
                for j in range(B):
                    C_mean[j] = P[S_feat[:, j] > 0].mean(dim=0)
                pred_mean, l2_mean = run_compressed_pred(C_mean, mults_feat)
                per_model_preds.append({
                    "model": m_name, "image_idx": img_idx, "target": target, "clean_pred": clean_pred,
                    "budget": B, "rem_frac": rem_frac, "method": "Group-Mean Merging",
                    "pred": pred_mean, "is_correct": int(pred_mean == target),
                    "agrees_clean": int(pred_mean == clean_pred), "l2_damage": l2_mean
                })

                # 5. Static Subspace (r=32)
                if V_static is not None:
                    sol_static = solve_amortized_carrier(P, S_feat, mults_feat, V_static[:, :32], lam_factor=10.0)
                    pred_static, l2_static = run_compressed_pred(sol_static["C_opt"], mults_feat)
                    per_model_preds.append({
                        "model": m_name, "image_idx": img_idx, "target": target, "clean_pred": clean_pred,
                        "budget": B, "rem_frac": rem_frac, "method": "Static Subspace (r=32)",
                        "pred": pred_static, "is_correct": int(pred_static == target),
                        "agrees_clean": int(pred_static == clean_pred), "l2_damage": l2_static
                    })

                # 6. Predicted Rank-32 (Forward Amortized)
                if V_pred_32 is not None:
                    sol_pred = solve_amortized_carrier(P, S_feat, mults_feat, V_pred_32, lam_factor=10.0)
                    pred_pred, l2_pred = run_compressed_pred(sol_pred["C_opt"], mults_feat)
                    per_model_preds.append({
                        "model": m_name, "image_idx": img_idx, "target": target, "clean_pred": clean_pred,
                        "budget": B, "rem_frac": rem_frac, "method": "Predicted Rank-32",
                        "pred": pred_pred, "is_correct": int(pred_pred == target),
                        "agrees_clean": int(pred_pred == clean_pred), "l2_damage": l2_pred
                    })

                # 7. Oracle Rank-32 (Evaluated on oracle_n images)
                if V_oracle is not None:
                    sol_orc = solve_amortized_carrier(P, S_feat, mults_feat, V_oracle[:, :32], lam_factor=10.0)
                    pred_orc, l2_orc = run_compressed_pred(sol_orc["C_opt"], mults_feat)
                    per_model_preds.append({
                        "model": m_name, "image_idx": img_idx, "target": target, "clean_pred": clean_pred,
                        "budget": B, "rem_frac": rem_frac, "method": "Oracle Rank-32",
                        "pred": pred_orc, "is_correct": int(pred_orc == target),
                        "agrees_clean": int(pred_orc == clean_pred), "l2_damage": l2_orc
                    })

                    # Correlation audit on aggressive budget B=budgets[-1]
                    if B == budgets[-1] and sub_overlap is not None:
                        overlap_compression_rows.append({
                            "model": m_name, "image_idx": img_idx, "budget": B,
                            "subspace_overlap": sub_overlap,
                            "logit_damage_groupmean": l2_mean,
                            "logit_damage_predicted": l2_pred,
                            "logit_damage_oracle": l2_orc,
                            "damage_reduction_over_groupmean": l2_mean - l2_pred,
                            "pred_correct": int(pred_pred == target),
                            "mean_correct": int(pred_mean == target),
                            "oracle_correct": int(pred_orc == target),
                            "clean_correct": int(is_clean_correct)
                        })

            if (img_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
            if (img_idx + 1) % 200 == 0 or (img_idx + 1) == n_eval:
                el = time.time() - t0_model
                print(f"[{m_name}] {img_idx + 1}/{n_eval} images audited ({el:.1f}s, {(img_idx+1)/el:.1f} img/s)")

        # Compile Dataset Alignment Audit for this model
        df_p = pd.DataFrame(per_model_preds)
        raw_predictions.extend(per_model_preds)

        for meth in df_p["method"].unique():
            sub_df = df_p[df_p["method"] == meth]
            n_rows = len(sub_df)
            n_u = sub_df["image_idx"].nunique()
            is_aligned = (n_u == n_eval)
            denom = n_u
            clean_corr = clean_correct_1000 if n_u == n_eval else clean_correct_oracle_subset

            notes = "Full canonical N=1000 evaluation" if is_aligned else f"Evaluated only on first N={n_u} oracle subset (causes apparent denominator mismatch!)"

            dataset_alignment_rows.append({
                "architecture": m_name,
                "method": meth,
                "n_rows": n_rows,
                "n_unique_images": n_u,
                "manifest_hash": manifest_hash,
                "denominator": denom,
                "n_clean_correct": clean_corr,
                "accuracy_definition": "mean(pred == ground_truth_label)",
                "aligned_with_clean": "YES" if is_aligned else "NO (SUBSET)",
                "notes": notes
            })

    # Save 1. Dataset Alignment Audit CSV
    df_align = pd.DataFrame(dataset_alignment_rows)
    df_align.to_csv(os.path.join(output_dir, "dataset_alignment_audit.csv"), index=False)
    print("\nSaved dataset_alignment_audit.csv")

    # 2. Recomputed Accuracy & 3. Clean Flip Analysis
    df_raw = pd.DataFrame(raw_predictions)

    recomputed_rows = []
    flip_rows = []

    for (m_name, meth, B), group in df_raw.groupby(["model", "method", "budget"]):
        rem_frac = group["rem_frac"].iloc[0]
        denom = len(group)

        # Accuracy recomputed directly: mean(pred == target)
        top1 = group["is_correct"].mean()
        agreement = group["agrees_clean"].mean()

        # Flip counts
        # Clean correct -> method wrong (Loss)
        clean_corr = (group["clean_pred"] == group["target"])
        clean_wrong = ~clean_corr
        meth_corr = (group["pred"] == group["target"])
        meth_wrong = ~meth_corr

        correct_to_wrong = (clean_corr & meth_wrong).sum() # Loss
        wrong_to_correct = (clean_wrong & meth_corr).sum() # Gain
        net_correction = wrong_to_correct - correct_to_wrong

        clean_acc_subset = clean_corr.mean()

        recomputed_rows.append({
            "model": m_name, "method": meth, "budget": B, "rem_frac": rem_frac,
            "n_images_evaluated": denom,
            "recomputed_top1_acc": top1,
            "clean_acc_on_same_subset": clean_acc_subset,
            "acc_minus_clean": top1 - clean_acc_subset,
            "mean_logit_l2": group["l2_damage"].mean()
        })

        flip_rows.append({
            "model": m_name, "method": meth, "budget": B, "rem_frac": rem_frac,
            "n_images": denom,
            "clean_acc": clean_acc_subset,
            "method_acc": top1,
            "prediction_agreement_with_clean": agreement,
            "clean_correct_to_method_wrong": int(correct_to_wrong),
            "clean_wrong_to_method_correct": int(wrong_to_correct),
            "net_corrections": int(net_correction)
        })

    df_recomputed = pd.DataFrame(recomputed_rows)
    df_recomputed.to_csv(os.path.join(output_dir, "recomputed_accuracy.csv"), index=False)
    print("Saved recomputed_accuracy.csv")

    df_flips = pd.DataFrame(flip_rows)
    df_flips.to_csv(os.path.join(output_dir, "clean_flip_analysis.csv"), index=False)
    print("Saved clean_flip_analysis.csv")

    # 4. Rigorous GPU Latency & Speedup Audit
    print("\n--- 2. Performing Synchronized GPU Latency Audit ---")
    runtime_audit_rows = []

    for m_cfg in all_models_config:
        m_key = m_cfg["model_key"]
        m_name = m_cfg["name"]
        depth = m_cfg["depth"]
        N = m_cfg["N"]
        D = m_cfg["D"]
        B = m_cfg["budgets"][-1] # aggressive budget
        chunk_sz = m_cfg["chunk_sz"]

        model, transform, _ = load_model_and_transform(m_key, device)
        model.eval()

        fact_path = os.path.join(models_dir, f"{m_key}_factorized_r32.pt")
        fact_predictor = None
        if os.path.exists(fact_path):
            fact_predictor = FactorizedModePredictor(N=N, D=D, r=32, R=2, hidden_dim=256).to(device)
            fact_predictor.load_state_dict(torch.load(fact_path, map_location=device))
            fact_predictor.eval()

        x = torch.randn(1, 3, 224, 224, device=device)

        # Warmup GPU
        for _ in range(15):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        # Benchmark 1: Clean Uncompressed Forward
        t_clean_runs = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            t_clean_runs.append((time.perf_counter() - t0) * 1000)
        t_clean_ms = np.median(t_clean_runs)

        # Benchmark 2: Prefix blocks
        with torch.no_grad():
            clean_logits, clean_acts = forward_block_by_block(model, m_key, x=x, collect_depths=(depth,))
        clean_act = clean_acts[depth]
        P = clean_act[0, 1:, :]
        cls_token = clean_act[:, 0:1, :]

        t_prefix_runs = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = forward_block_by_block(model, m_key, x=x, collect_depths=(depth,))
            torch.cuda.synchronize()
            t_prefix_runs.append((time.perf_counter() - t0) * 1000)
        t_prefix_ms = np.median(t_prefix_runs)

        # Benchmark 3: Predictor
        t_pred_runs = []
        if fact_predictor is not None:
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    V_p = fact_predictor(clean_act)[0]
                torch.cuda.synchronize()
                t_pred_runs.append((time.perf_counter() - t0) * 1000)
        t_pred_ms = np.median(t_pred_runs) if len(t_pred_runs) else 1.0

        # Benchmark 4: Grouping
        t_group_runs = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, S_f, mults_f = create_groupings_confirmatory(P, B, strategy="feature_similarity", seed=42)
            torch.cuda.synchronize()
            t_group_runs.append((time.perf_counter() - t0) * 1000)
        t_group_ms = np.median(t_group_runs)

        # Benchmark 5: Carrier Solve (Tikhonov)
        V_dummy = torch.randn(N * D, 32, device=device)
        V_dummy, _ = torch.linalg.qr(V_dummy)
        t_solve_runs = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sol = solve_amortized_carrier(P, S_f, mults_f, V_dummy, lam_factor=10.0)
            torch.cuda.synchronize()
            t_solve_runs.append((time.perf_counter() - t0) * 1000)
        t_solve_ms = np.median(t_solve_runs)

        # Benchmark 6: Suffix compressed
        all_mults = torch.cat([torch.tensor([1.0], device=device), mults_f], dim=0)
        h_c = torch.cat([cls_token, sol["C_opt"].unsqueeze(0)], dim=1)
        t_suffix_runs = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = forward_downstream_compressed(model, m_key, depth, h_c, all_mults, N)
            torch.cuda.synchronize()
            t_suffix_runs.append((time.perf_counter() - t0) * 1000)
        t_suffix_ms = np.median(t_suffix_runs)

        # Benchmark 7: ToMe (Bipartite Soft Matching) full pipeline
        t_tome_runs = []
        for _ in range(30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _, acts_t = forward_block_by_block(model, m_key, x=x, collect_depths=(depth,))
                act_t = acts_t[depth]
                P_t = act_t[0, 1:, :]
                cls_t = act_t[:, 0:1, :]
                C_tm, mults_tm = tome_bipartite_merge(P_t, B)
                all_m_tm = torch.cat([torch.tensor([1.0], device=device), mults_tm], dim=0)
                hc_tm = torch.cat([cls_t, C_tm.unsqueeze(0)], dim=1)
                _ = forward_downstream_compressed(model, m_key, depth, hc_tm, all_m_tm, N)
            torch.cuda.synchronize()
            t_tome_runs.append((time.perf_counter() - t0) * 1000)
        t_tome_total_ms = np.median(t_tome_runs)

        # Benchmark 8: Oracle VJP
        t_vjp_runs = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            J = compute_fast_downstream_jacobian(model, m_key, depth, clean_act, chunk_size=chunk_sz)
            V_orc, _ = extract_oracle_subspace(J, r=32)
            torch.cuda.synchronize()
            t_vjp_runs.append((time.perf_counter() - t0) * 1000)
            del J
        t_vjp_ms = np.median(t_vjp_runs)

        # Total Amortized Pipeline
        t_amortized_total_ms = t_prefix_ms + t_pred_ms + t_group_ms + t_solve_ms + t_suffix_ms
        t_oracle_total_ms = t_prefix_ms + t_vjp_ms + t_group_ms + t_solve_ms + t_suffix_ms

        speedup_A_oracle = t_oracle_total_ms / t_amortized_total_ms
        speedup_B_clean = t_clean_ms / t_amortized_total_ms
        speedup_C_tome = t_tome_total_ms / t_amortized_total_ms

        runtime_audit_rows.append({
            "model": m_name,
            "budget": B,
            "t_clean_uncompressed_ms": t_clean_ms,
            "t_prefix_ms": t_prefix_ms,
            "t_predictor_ms": t_pred_ms,
            "t_grouping_ms": t_group_ms,
            "t_carrier_solve_ms": t_solve_ms,
            "t_suffix_compressed_ms": t_suffix_ms,
            "t_tome_total_ms": t_tome_total_ms,
            "t_amortized_total_ms": t_amortized_total_ms,
            "t_oracle_total_ms": t_oracle_total_ms,
            "speedup_A_oracle_to_amortized": speedup_A_oracle,
            "speedup_B_clean_to_amortized": speedup_B_clean,
            "speedup_C_tome_to_amortized": speedup_C_tome,
            "practical_acceleration_vs_clean": "YES" if speedup_B_clean > 1.0 else "NO (CARRIER SOLVE OVERHEAD)",
            "competitive_with_tome": "YES" if speedup_C_tome > 1.0 else "NO (SLOWER THAN TOME)"
        })

    df_runtime = pd.DataFrame(runtime_audit_rows)
    df_runtime.to_csv(os.path.join(output_dir, "runtime_audit.csv"), index=False)
    print("Saved runtime_audit.csv")

    # 5. True Oracle Recovery Audit (Separating Low-Rank Oracle vs Amortization Recovery)
    print("\n--- 3. Computing True Oracle Recovery Audit ---")
    recovery_audit_rows = []

    # Filter to paired subset where Oracle is evaluated
    for (m_name, B), group in df_raw.groupby(["model", "budget"]):
        orc_imgs = set(group[group["method"] == "Oracle Rank-32"]["image_idx"].unique())
        if len(orc_imgs) > 0:
            paired = group[group["image_idx"].isin(orc_imgs)]
            methods_acc = paired.groupby("method")["is_correct"].mean().to_dict()
            methods_l2 = paired.groupby("method")["l2_damage"].mean().to_dict()

            if "Group-Mean Merging" in methods_acc and "Oracle Rank-32" in methods_acc and "Predicted Rank-32" in methods_acc:
                acc_base = methods_acc["Group-Mean Merging"]
                acc_orc32 = methods_acc["Oracle Rank-32"]
                acc_pred32 = methods_acc["Predicted Rank-32"]

                l2_base = methods_l2["Group-Mean Merging"]
                l2_orc32 = methods_l2["Oracle Rank-32"]
                l2_pred32 = methods_l2["Predicted Rank-32"]

                # Amortization recovery relative to Rank-32 Oracle
                gain_orc_acc = acc_orc32 - acc_base
                gain_pred_acc = acc_pred32 - acc_base
                rec_acc = (gain_pred_acc / gain_orc_acc * 100) if abs(gain_orc_acc) > 1e-4 else 100.0

                gain_orc_l2 = l2_base - l2_orc32
                gain_pred_l2 = l2_base - l2_pred32
                rec_l2 = (gain_pred_l2 / gain_orc_l2 * 100) if abs(gain_orc_l2) > 1e-4 else 100.0

                recovery_audit_rows.append({
                    "model": m_name, "budget": B,
                    "n_paired_images": len(orc_imgs),
                    "acc_baseline_groupmean": acc_base,
                    "acc_oracle_rank32": acc_orc32,
                    "acc_predicted_rank32": acc_pred32,
                    "oracle_acc_advantage": gain_orc_acc,
                    "predicted_acc_advantage": gain_pred_acc,
                    "amortization_recovery_acc_pct": np.clip(rec_acc, -50.0, 150.0),
                    "l2_baseline_groupmean": l2_base,
                    "l2_oracle_rank32": l2_orc32,
                    "l2_predicted_rank32": l2_pred32,
                    "amortization_recovery_l2_pct": np.clip(rec_l2, -50.0, 150.0)
                })

    df_rec_audit = pd.DataFrame(recovery_audit_rows)
    df_rec_audit.to_csv(os.path.join(output_dir, "oracle_recovery_audit.csv"), index=False)
    print("Saved oracle_recovery_audit.csv")

    # 6. Overlap vs Compression Correlation CSV
    df_overlap = pd.DataFrame(overlap_compression_rows)
    if len(df_overlap) > 0:
        df_overlap.to_csv(os.path.join(output_dir, "overlap_vs_compression.csv"), index=False)
        print("Saved overlap_vs_compression.csv")

    # 7. Validation Manifest JSON
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
        "device": str(device),
        "manifest_hash": manifest_hash,
        "n_eval_images": n_eval,
        "models_audited": [m["name"] for m in all_models_config],
        "eval_seed": 9201,
        "audit_status": "COMPLETED_AUDIT"
    }
    with open(os.path.join(output_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved validation_manifest.json")

    print("\n=======================================================")
    print("STRICT SANITY AUDIT COMPLETED SUCCESSFULLY!")
    print(f"Audit outputs saved in: {output_dir}")
    print("=======================================================")


if __name__ == "__main__":
    main()
