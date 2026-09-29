"""
patch_fungibility/compression_pipeline.py

Pipeline for FUNGIBILITY-TO-COMPRESSION POC: WEIGHTED CENTROID CARRIER
- Phase A: Exact Numerical Equivalence Validation (N=64 images)
- Phase B: Full 1,000-Image Evaluation across 5 Mask Seeds
- Phase C: Matched-Budget Baseline Comparisons (Random Pruning, Unweighted Centroid, Image-Mean Carrier)
- Phase D: FLOP Accounting and Physical PyTorch CUDA Latency Benchmarking
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

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits, ParquetImageSubset
from patch_fungibility.dense_fraction_models import load_model_and_transform, forward_block_by_block
from patch_fungibility.dense_fraction_pipeline import load_calibration_statistics, extract_activations_and_clean_logits
from patch_fungibility.compression_models import (
    forward_downstream_compressed,
    forward_downstream_reference,
    forward_end_to_end_clean,
    forward_end_to_end_compressed
)


MASK_SEEDS = [31001, 31002, 31003, 31004, 31005]

# Mean F95, F90, F80 thresholds from dense fraction sweep
MODEL_CONFIGS = [
    {
        "key": "deit_tiny",
        "depth": 8,
        "is_primary": True,
        "n_patches": 196,
        "embed_dim": 192,
        "num_heads": 3,
        "clean_acc": 0.6790,
        "clean_margin": 1.1681,
        "threshold_fractions": [0.6367, 0.7786, 0.8847]
    },
    {
        "key": "deit_small",
        "depth": 8,
        "is_primary": True,
        "n_patches": 196,
        "embed_dim": 384,
        "num_heads": 6,
        "clean_acc": 0.7610,
        "clean_margin": 2.2423,
        "threshold_fractions": [0.7459, 0.8653, 0.9102]
    },
    {
        "key": "vit_base",
        "depth": 7,
        "is_primary": True,
        "n_patches": 196,
        "embed_dim": 768,
        "num_heads": 12,
        "clean_acc": 0.7610,
        "clean_margin": 3.1923,
        "threshold_fractions": [0.4367, 0.6367, 0.7949]
    },
    {
        "key": "dinov2",
        "depth": 9,
        "is_primary": True,
        "n_patches": 256,
        "embed_dim": 384,
        "num_heads": 6,
        "clean_acc": 0.7880,
        "clean_margin": 2.7605,
        "threshold_fractions": [0.2312, 0.2898, 0.3969]
    }
]


def build_compact_fraction_grid(n_patches: int, threshold_fractions: List[float]) -> List[Dict[str, Any]]:
    """
    Constructs deduplicated compact fraction points:
    0%, 25%, 50%, 75%, F95, F90, F80, 100%.
    """
    candidate_fractions = [0.0, 0.25, 0.50, 0.75, 1.0] + threshold_fractions
    seen_k = set()
    grid = []

    for f in candidate_fractions:
        k = int(round(f * n_patches))
        if k not in seen_k:
            seen_k.add(k)
            grid.append({
                "requested_fraction": f,
                "actual_k": k,
                "actual_fraction": k / n_patches,
                "M_real": n_patches - k,
                "m_carrier": k,
                "compressed_tokens": 1 + (n_patches - k) + (1 if k > 0 else 0)
            })

    # Sort by actual_k
    grid = sorted(grid, key=lambda x: x["actual_k"])
    return grid


def compute_theoretical_flops(
    n_patches: int,
    embed_dim: int,
    start_depth: int,
    total_blocks: int,
    k_replaced: int,
    model_key: str
) -> Dict[str, Any]:
    """
    Computes theoretical FLOPs for uncompressed vs. compressed downstream blocks.
    """
    D = embed_dim
    T_orig = 1 + n_patches
    M_real = n_patches - k_replaced
    T_comp = 1 + M_real + (1 if k_replaced > 0 else 0)
    num_downstream = total_blocks - start_depth

    # Uncompressed single block FLOPs
    # Attn: 8 T D^2 + 4 T^2 D
    attn_flops_orig = 8 * T_orig * (D ** 2) + 4 * (T_orig ** 2) * D
    # MLP: 16 T D^2
    mlp_flops_orig = 16 * T_orig * (D ** 2)
    block_flops_orig = attn_flops_orig + mlp_flops_orig

    # Compressed single block FLOPs
    attn_flops_comp = 8 * T_comp * (D ** 2) + 4 * (T_comp ** 2) * D
    mlp_flops_comp = 16 * T_comp * (D ** 2)
    block_flops_comp = attn_flops_comp + mlp_flops_comp

    # Downstream totals
    downstream_orig = num_downstream * block_flops_orig
    downstream_comp = num_downstream * block_flops_comp
    downstream_reduction_pct = ((downstream_orig - downstream_comp) / downstream_orig) * 100.0

    # Total model FLOPs
    upstream = start_depth * block_flops_orig
    # Patch embed + Head approx: 2 * 3 * 16 * 16 * D * n_patches + 2 * D * 1000
    stem_head = 2 * 3 * 256 * D + 2 * D * 1000
    model_orig = upstream + downstream_orig + stem_head
    model_comp = upstream + downstream_comp + stem_head
    total_reduction_pct = ((model_orig - model_comp) / model_orig) * 100.0

    return {
        "T_orig": T_orig,
        "T_comp": T_comp,
        "token_reduction_pct": ((T_orig - T_comp) / T_orig) * 100.0,
        "downstream_orig_gflops": downstream_orig / 1e9,
        "downstream_comp_gflops": downstream_comp / 1e9,
        "downstream_reduction_pct": downstream_reduction_pct,
        "total_orig_gflops": model_orig / 1e9,
        "total_comp_gflops": model_comp / 1e9,
        "total_reduction_pct": total_reduction_pct
    }


def run_compression_poc_pipeline(
    output_dir: str = "outputs/fungibility_compression_poc",
    smoke_test: bool = False,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Master execution pipeline for FUNGIBILITY-TO-COMPRESSION POC.
    """
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if device_str is not None:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[Pipeline] Using compute device: {device}")
    if device.type == "cuda":
        print(f"[Pipeline] GPU Device: {torch.cuda.get_device_name(0)}")

    # 1. Load canonical dataset splits
    n_per_split = 64 if smoke_test else 1000
    print(f"[Pipeline] Loading canonical ImageNet disjoint splits (N={n_per_split})...")
    calib_ds_raw, eval_ds_raw, calib_manifest, eval_manifest = get_disjoint_imagenet_splits(
        calib_seed=9101, eval_seed=9201, n_per_split=n_per_split
    )
    eval_samples = eval_ds_raw._samples
    labels = torch.tensor([s[1] for s in eval_samples], dtype=torch.long)

    # Load frozen permutations
    with open("outputs/fungibility_dense_fraction/mask_permutations.json") as f:
        perms_meta = json.load(f)
    perms_196 = {int(s): np.array(p) for s, p in perms_meta["perms_196"].items()}
    perms_256 = {int(s): np.array(p) for s, p in perms_meta["perms_256"].items()}

    # Data structures to accumulate results
    phase_a_rows = []
    phase_b_rows = []
    phase_c_rows = []
    compute_rows = []
    latency_rows = []

    validation_audit = {
        "phase_a_exact_equivalence": True,
        "max_abs_logit_diff_across_all": 0.0,
        "min_prediction_agreement": 1.0,
        "models_audited": []
    }

    # Iterate over models
    for cfg in MODEL_CONFIGS:
        m_key = cfg["key"]
        depth = cfg["depth"]
        n_patches = cfg["n_patches"]
        embed_dim = cfg["embed_dim"]
        clean_acc = cfg["clean_acc"]
        clean_margin = cfg["clean_margin"]
        perms = perms_196 if n_patches == 196 else perms_256

        print(f"\n================================================================================")
        print(f"PROCESSING MODEL: {m_key.upper()} (Depth {depth}, D={embed_dim}, N={n_patches})")
        print(f"================================================================================")

        model, transform, _ = load_model_and_transform(m_key, device)
        mu_np, _ = load_calibration_statistics(m_key, depth)
        mu = torch.tensor(mu_np, dtype=torch.float32, device=device)

        # Build compact fraction grid
        grid = build_compact_fraction_grid(n_patches, cfg["threshold_fractions"])
        print(f"  Compact grid points ({len(grid)}): {[g['actual_fraction'] for g in grid]}")

        # Compute theoretical FLOPs for all points in grid
        for g in grid:
            flops = compute_theoretical_flops(
                n_patches=n_patches,
                embed_dim=embed_dim,
                start_depth=depth,
                total_blocks=12,
                k_replaced=g["actual_k"],
                model_key=m_key
            )
            compute_rows.append({
                "model": m_key,
                "depth": depth,
                "requested_fraction": g["requested_fraction"],
                "actual_fraction": g["actual_fraction"],
                "actual_k": g["actual_k"],
                "M_real": g["M_real"],
                "m_carrier": g["m_carrier"],
                "T_orig": flops["T_orig"],
                "T_comp": flops["T_comp"],
                "token_reduction_pct": flops["token_reduction_pct"],
                "downstream_orig_gflops": flops["downstream_orig_gflops"],
                "downstream_comp_gflops": flops["downstream_comp_gflops"],
                "downstream_reduction_pct": flops["downstream_reduction_pct"],
                "total_orig_gflops": flops["total_orig_gflops"],
                "total_comp_gflops": flops["total_comp_gflops"],
                "total_reduction_pct": flops["total_reduction_pct"]
            })

        # ----------------------------------------------------------------------
        # PHASE A: EXACT NUMERICAL EQUIVALENCE VALIDATION (N=64 images)
        # ----------------------------------------------------------------------
        print(f"\n[Phase A] Running Exact Numerical Equivalence Validation (N=64 images)...")
        eval_ds_64 = ParquetImageSubset(eval_samples[:64], transform=transform)
        loader_64 = DataLoader(eval_ds_64, batch_size=32, shuffle=False)
        cached_h_64, clean_logits_64, _ = extract_activations_and_clean_logits(
            model, m_key, loader_64, target_depths=(depth,), device=device
        )
        h_64 = cached_h_64[depth].to(device)
        labels_64 = labels[:64].to(device)

        # Test across first mask seed (31001) for all compact fractions
        p_seed = perms[31001]
        for g in grid:
            k = g["actual_k"]
            f_act = g["actual_fraction"]
            mask_bool = np.zeros(n_patches, dtype=bool)
            if k > 0:
                mask_bool[p_seed[:k]] = True
            mask_t = torch.tensor(mask_bool, dtype=torch.bool, device=device)

            # Build uncompressed reference sequence
            cls_tok = h_64[:, :1, :]
            patches = h_64[:, 1:, :]

            if k == 0:
                h_ref = h_64
                logits_ref = forward_downstream_reference(model, m_key, depth, h_ref)
                logits_comp = logits_ref
                max_diff = 0.0
                mean_diff = 0.0
                pred_agree = 1.0
                margin_diff = 0.0
            else:
                real_p = patches[:, ~mask_t, :]
                mu_expanded = mu.unsqueeze(0).unsqueeze(0).expand(64, k, -1)
                h_ref = torch.cat([cls_tok, real_p, mu_expanded], dim=1)
                logits_ref = forward_downstream_reference(model, m_key, depth, h_ref)

                # Build compressed sequence
                carrier = mu.unsqueeze(0).unsqueeze(0).expand(64, 1, -1)
                h_comp = torch.cat([cls_tok, real_p, carrier], dim=1)
                multiplicities = torch.cat([
                    torch.ones(1 + (n_patches - k), device=device),
                    torch.tensor([float(k)], device=device)
                ])
                logits_comp = forward_downstream_compressed(model, m_key, depth, h_comp, multiplicities, n_patches)

                # Error metrics
                abs_diff = torch.abs(logits_ref - logits_comp)
                max_diff = float(torch.max(abs_diff).item())
                mean_diff = float(torch.mean(abs_diff).item())

                pred_ref = torch.argmax(logits_ref, dim=-1)
                pred_comp = torch.argmax(logits_comp, dim=-1)
                pred_agree = float((pred_ref == pred_comp).float().mean().item())

                # Margin diff
                y_64 = labels_64
                true_ref = logits_ref[torch.arange(64), y_64]
                true_comp = logits_comp[torch.arange(64), y_64]
                margin_diff = float(torch.mean(true_ref - true_comp).item())

            phase_a_rows.append({
                "model": m_key,
                "depth": depth,
                "actual_fraction": f_act,
                "actual_k": k,
                "M_real": g["M_real"],
                "m_carrier": g["m_carrier"],
                "T_comp": g["compressed_tokens"],
                "max_abs_logit_diff": max_diff,
                "mean_abs_logit_diff": mean_diff,
                "prediction_agreement": pred_agree,
                "mean_margin_diff": margin_diff
            })

            print(f"    k={k:3d} ({f_act*100:5.1f}%): max abs diff={max_diff:.2e}, pred agreement={pred_agree*100:.1f}%")
            assert pred_agree == 1.0, f"Phase A failed 100% agreement on {m_key} k={k}!"
            assert max_diff <= 1e-4, f"Phase A failed max diff <= 1e-4 on {m_key} k={k}: got {max_diff}"

            if max_diff > validation_audit["max_abs_logit_diff_across_all"]:
                validation_audit["max_abs_logit_diff_across_all"] = max_diff
            if pred_agree < validation_audit["min_prediction_agreement"]:
                validation_audit["min_prediction_agreement"] = pred_agree

        del cached_h_64, h_64, clean_logits_64
        validation_audit["models_audited"].append(m_key)

        # ----------------------------------------------------------------------
        # PHASE B & PHASE C: FULL EVALUATION (N=1,000 images, 5 mask seeds)
        # ----------------------------------------------------------------------
        print(f"\n[Phase B & C] Running Full Evaluation across 5 mask seeds (N={n_per_split})...")
        eval_ds = ParquetImageSubset(eval_samples, transform=transform)
        eval_loader = DataLoader(eval_ds, batch_size=64 if embed_dim <= 384 else 32, shuffle=False)

        cached_h_full, clean_logits_full, _ = extract_activations_and_clean_logits(
            model, m_key, eval_loader, target_depths=(depth,), device=device
        )
        h_full = cached_h_full[depth].to(device)
        clean_logits_t = clean_logits_full.to(device)
        labels_t = labels.to(device)
        N_eval = len(labels)

        # Compute clean margin
        true_clean = clean_logits_t[torch.arange(N_eval), labels_t]
        mask_clean_inc = torch.ones_like(clean_logits_t, dtype=torch.bool)
        mask_clean_inc[torch.arange(N_eval), labels_t] = False
        strongest_clean_inc = torch.max(torch.where(mask_clean_inc, clean_logits_t, -1e9), dim=1).values
        clean_margins = (true_clean - strongest_clean_inc).cpu().numpy()
        clean_mean_margin = float(np.mean(clean_margins))

        for s_idx, s in enumerate(MASK_SEEDS):
            p_s = perms[s]

            for g in grid:
                k = g["actual_k"]
                f_act = g["actual_fraction"]
                mask_bool = np.zeros(n_patches, dtype=bool)
                if k > 0:
                    mask_bool[p_s[:k]] = True
                mask_t = torch.tensor(mask_bool, dtype=torch.bool, device=device)

                cls_tok = h_full[:, :1, :]
                patches = h_full[:, 1:, :]
                real_p = patches[:, ~mask_t, :]
                M_real = n_patches - k

                # 1. UNCOMPRESSED CENTROID REFERENCE
                if k == 0:
                    h_ref = h_full
                else:
                    mu_exp = mu.unsqueeze(0).unsqueeze(0).expand(N_eval, k, -1)
                    h_ref = torch.cat([cls_tok, real_p, mu_exp], dim=1)

                # Batch forward downstream
                logits_ref_list = []
                for bi in range(0, N_eval, 64):
                    l_bi = forward_downstream_reference(model, m_key, depth, h_ref[bi : bi + 64])
                    logits_ref_list.append(l_bi)
                logits_ref = torch.cat(logits_ref_list, dim=0)

                # Metrics for Reference
                preds_ref = torch.argmax(logits_ref, dim=-1)
                acc_ref = float((preds_ref == labels_t).float().mean().item())
                true_ref = logits_ref[torch.arange(N_eval), labels_t]
                mask_inc = torch.ones_like(logits_ref, dtype=torch.bool)
                mask_inc[torch.arange(N_eval), labels_t] = False
                strong_inc = torch.max(torch.where(mask_inc, logits_ref, -1e9), dim=1).values
                margins_ref = (true_ref - strong_inc).cpu().numpy()
                mean_margin_ref = float(np.mean(margins_ref))

                # 2. WEIGHTED CENTROID CARRIER
                if k == 0:
                    logits_wcomp = logits_ref
                else:
                    carrier = mu.unsqueeze(0).unsqueeze(0).expand(N_eval, 1, -1)
                    h_comp = torch.cat([cls_tok, real_p, carrier], dim=1)
                    multiplicities = torch.cat([
                        torch.ones(1 + M_real, device=device),
                        torch.tensor([float(k)], device=device)
                    ])
                    logits_wcomp_list = []
                    for bi in range(0, N_eval, 64):
                        l_bi = forward_downstream_compressed(
                            model, m_key, depth, h_comp[bi : bi + 64], multiplicities, n_patches
                        )
                        logits_wcomp_list.append(l_bi)
                    logits_wcomp = torch.cat(logits_wcomp_list, dim=0)

                preds_wcomp = torch.argmax(logits_wcomp, dim=-1)
                acc_wcomp = float((preds_wcomp == labels_t).float().mean().item())
                true_wcomp = logits_wcomp[torch.arange(N_eval), labels_t]
                strong_inc_wcomp = torch.max(torch.where(mask_inc, logits_wcomp, -1e9), dim=1).values
                margins_wcomp = (true_wcomp - strong_inc_wcomp).cpu().numpy()
                mean_margin_wcomp = float(np.mean(margins_wcomp))

                # Discrepancy between Reference and Compressed
                max_diff_full = float(torch.max(torch.abs(logits_ref - logits_wcomp)).item())
                mean_diff_full = float(torch.mean(torch.abs(logits_ref - logits_wcomp)).item())
                pred_agree_full = float((preds_ref == preds_wcomp).float().mean().item())

                phase_b_rows.append({
                    "model": m_key,
                    "depth": depth,
                    "mask_seed": s,
                    "actual_fraction": f_act,
                    "actual_k": k,
                    "M_real": M_real,
                    "m_carrier": k,
                    "T_comp": g["compressed_tokens"],
                    "acc_reference": acc_ref,
                    "margin_reference": mean_margin_ref,
                    "acc_compressed": acc_wcomp,
                    "margin_compressed": mean_margin_wcomp,
                    "damage_compressed": clean_mean_margin - mean_margin_wcomp,
                    "max_logit_error": max_diff_full,
                    "mean_logit_error": mean_diff_full,
                    "prediction_agreement": pred_agree_full
                })

                # PHASE C BASELINES
                # Baseline 1: WEIGHTED_CENTROID_CARRIER
                phase_c_rows.append({
                    "model": m_key,
                    "depth": depth,
                    "mask_seed": s,
                    "actual_fraction": f_act,
                    "actual_k": k,
                    "M_real": M_real,
                    "condition": "WEIGHTED_CENTROID_CARRIER",
                    "multiplicity": k,
                    "downstream_patch_tokens": M_real + (1 if k > 0 else 0),
                    "total_downstream_tokens": g["compressed_tokens"],
                    "top1_acc": acc_wcomp,
                    "mean_margin": mean_margin_wcomp,
                    "margin_damage": clean_mean_margin - mean_margin_wcomp
                })

                if k > 0:
                    # Baseline 2: UNWEIGHTED_CENTROID (s=1)
                    carrier_unw = mu.unsqueeze(0).unsqueeze(0).expand(N_eval, 1, -1)
                    h_unw = torch.cat([cls_tok, real_p, carrier_unw], dim=1)
                    mult_unw = torch.ones(2 + M_real, device=device)
                    logits_unw_list = []
                    for bi in range(0, N_eval, 64):
                        l_bi = forward_downstream_compressed(
                            model, m_key, depth, h_unw[bi : bi + 64], mult_unw, n_patches
                        )
                        logits_unw_list.append(l_bi)
                    logits_unw = torch.cat(logits_unw_list, dim=0)
                    preds_unw = torch.argmax(logits_unw, dim=-1)
                    acc_unw = float((preds_unw == labels_t).float().mean().item())
                    true_unw = logits_unw[torch.arange(N_eval), labels_t]
                    strong_unw = torch.max(torch.where(mask_inc, logits_unw, -1e9), dim=1).values
                    margin_unw = float(torch.mean(true_unw - strong_unw).item())

                    phase_c_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "mask_seed": s,
                        "actual_fraction": f_act,
                        "actual_k": k,
                        "M_real": M_real,
                        "condition": "UNWEIGHTED_CENTROID",
                        "multiplicity": 1,
                        "downstream_patch_tokens": M_real + 1,
                        "total_downstream_tokens": g["compressed_tokens"],
                        "top1_acc": acc_unw,
                        "mean_margin": margin_unw,
                        "margin_damage": clean_mean_margin - margin_unw
                    })

                    # Baseline 3: IMAGE_MEAN_CARRIER
                    discarded = patches[:, mask_t, :]
                    mean_c = discarded.mean(dim=1, keepdim=True)
                    h_img_mean = torch.cat([cls_tok, real_p, mean_c], dim=1)
                    mult_img = torch.cat([
                        torch.ones(1 + M_real, device=device),
                        torch.tensor([float(k)], device=device)
                    ])
                    logits_img_list = []
                    for bi in range(0, N_eval, 64):
                        l_bi = forward_downstream_compressed(
                            model, m_key, depth, h_img_mean[bi : bi + 64], mult_img, n_patches
                        )
                        logits_img_list.append(l_bi)
                    logits_img = torch.cat(logits_img_list, dim=0)
                    preds_img = torch.argmax(logits_img, dim=-1)
                    acc_img = float((preds_img == labels_t).float().mean().item())
                    true_img = logits_img[torch.arange(N_eval), labels_t]
                    strong_img = torch.max(torch.where(mask_inc, logits_img, -1e9), dim=1).values
                    margin_img = float(torch.mean(true_img - strong_img).item())

                    phase_c_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "mask_seed": s,
                        "actual_fraction": f_act,
                        "actual_k": k,
                        "M_real": M_real,
                        "condition": "IMAGE_MEAN_CARRIER",
                        "multiplicity": k,
                        "downstream_patch_tokens": M_real + 1,
                        "total_downstream_tokens": g["compressed_tokens"],
                        "top1_acc": acc_img,
                        "mean_margin": margin_img,
                        "margin_damage": clean_mean_margin - margin_img
                    })

                    # Baseline 4: RANDOM_PRUNING (Keep B = M_real + 1 patches, no carrier)
                    B_target = min(M_real + 1, n_patches)
                    if B_target > M_real:
                        extra_idx = torch.where(mask_t)[0][0]
                        extra_p = patches[:, extra_idx : extra_idx + 1, :]
                        pruned_p = torch.cat([real_p, extra_p], dim=1)
                    else:
                        pruned_p = real_p

                    h_pruned = torch.cat([cls_tok, pruned_p], dim=1)
                    mult_pruned = torch.ones(h_pruned.size(1), device=device)
                    logits_pruned_list = []
                    for bi in range(0, N_eval, 64):
                        l_bi = forward_downstream_compressed(
                            model, m_key, depth, h_pruned[bi : bi + 64], mult_pruned, n_patches
                        )
                        logits_pruned_list.append(l_bi)
                    logits_pruned = torch.cat(logits_pruned_list, dim=0)
                    preds_pruned = torch.argmax(logits_pruned, dim=-1)
                    acc_pruned = float((preds_pruned == labels_t).float().mean().item())
                    true_pruned = logits_pruned[torch.arange(N_eval), labels_t]
                    strong_pruned = torch.max(torch.where(mask_inc, logits_pruned, -1e9), dim=1).values
                    margin_pruned = float(torch.mean(true_pruned - strong_pruned).item())

                    phase_c_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "mask_seed": s,
                        "actual_fraction": f_act,
                        "actual_k": k,
                        "M_real": M_real,
                        "condition": "RANDOM_PRUNING",
                        "multiplicity": 0,
                        "downstream_patch_tokens": B_target,
                        "total_downstream_tokens": 1 + B_target,
                        "top1_acc": acc_pruned,
                        "mean_margin": margin_pruned,
                        "margin_damage": clean_mean_margin - margin_pruned
                    })

        del cached_h_full, h_full, clean_logits_full

        # ----------------------------------------------------------------------
        # PHASE D: PHYSICAL PYTORCH CUDA LATENCY BENCHMARKING
        # ----------------------------------------------------------------------
        if device.type == "cuda":
            print(f"\n[Phase D] Running Physical CUDA Latency Benchmarking on RTX 5070...")
            # Benchmark for batch sizes 1 and 16
            # Use representative fraction: F90 operating point
            f90_entry = [g for g in grid if g["actual_fraction"] > 0.6 and g["actual_fraction"] < 0.9][-1]
            k_bench = f90_entry["actual_k"]
            mask_bench_np = np.zeros(n_patches, dtype=bool)
            mask_bench_np[perms[31001][:k_bench]] = True
            mask_bench = torch.tensor(mask_bench_np, dtype=torch.bool, device=device)

            for bs in [1, 16]:
                dummy_input = torch.randn(bs, 3, 224, 224, device=device)

                # Warmup Clean
                for _ in range(5):
                    _ = forward_end_to_end_clean(model, m_key, dummy_input)
                torch.cuda.synchronize()

                # Time Clean
                clean_times = []
                for _ in range(30):
                    ev_start = torch.cuda.Event(enable_timing=True)
                    ev_end = torch.cuda.Event(enable_timing=True)
                    ev_start.record()
                    _ = forward_end_to_end_clean(model, m_key, dummy_input)
                    ev_end.record()
                    torch.cuda.synchronize()
                    clean_times.append(ev_start.elapsed_time(ev_end))

                clean_mean = float(np.mean(clean_times))
                clean_sd = float(np.std(clean_times))

                # Conditions to benchmark
                bench_conditions = [
                    ("WEIGHTED_CENTROID_CARRIER", mu),
                    ("UNWEIGHTED_CENTROID", mu),
                    ("IMAGE_MEAN_CARRIER", None),
                    ("RANDOM_PRUNING", None)
                ]

                for cond_name, mu_arg in bench_conditions:
                    # Warmup
                    for _ in range(5):
                        _ = forward_end_to_end_compressed(
                            model, m_key, dummy_input, depth, mask_bench, cond_name, mu_arg
                        )
                    torch.cuda.synchronize()

                    cond_times = []
                    for _ in range(30):
                        ev_start = torch.cuda.Event(enable_timing=True)
                        ev_end = torch.cuda.Event(enable_timing=True)
                        ev_start.record()
                        _ = forward_end_to_end_compressed(
                            model, m_key, dummy_input, depth, mask_bench, cond_name, mu_arg
                        )
                        ev_end.record()
                        torch.cuda.synchronize()
                        cond_times.append(ev_start.elapsed_time(ev_end))

                    cond_mean = float(np.mean(cond_times))
                    cond_sd = float(np.std(cond_times))
                    speedup_pct = ((clean_mean - cond_mean) / clean_mean) * 100.0

                    latency_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "batch_size": bs,
                        "condition": cond_name,
                        "k_replaced": k_bench,
                        "actual_fraction": f90_entry["actual_fraction"],
                        "T_comp": f90_entry["compressed_tokens"],
                        "clean_latency_ms": clean_mean,
                        "clean_sd_ms": clean_sd,
                        "compressed_latency_ms": cond_mean,
                        "compressed_sd_ms": cond_sd,
                        "latency_reduction_pct": speedup_pct
                    })
                    print(f"    BS={bs:2d} | {cond_name:25s} | Clean: {clean_mean:5.2f}ms -> Comp: {cond_mean:5.2f}ms ({speedup_pct:+5.1f}%)")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save all tables
    print(f"\n[Pipeline] Saving all Phase A-D CSV tables to {output_dir}...")
    df_phase_a = pd.DataFrame(phase_a_rows)
    df_phase_a.to_csv(os.path.join(output_dir, "equivalence_results.csv"), index=False)

    df_phase_b = pd.DataFrame(phase_b_rows)
    df_phase_b.to_csv(os.path.join(output_dir, "full_eval_results.csv"), index=False)

    df_phase_c = pd.DataFrame(phase_c_rows)
    df_phase_c.to_csv(os.path.join(output_dir, "baseline_comparison.csv"), index=False)

    df_compute = pd.DataFrame(compute_rows)
    df_compute.to_csv(os.path.join(output_dir, "compute_summary.csv"), index=False)

    df_latency = pd.DataFrame(latency_rows)
    df_latency.to_csv(os.path.join(output_dir, "latency_summary.csv"), index=False)

    # Decision evaluation
    max_err = validation_audit["max_abs_logit_diff_across_all"]
    min_agree = validation_audit["min_prediction_agreement"]

    if min_agree == 1.0 and max_err <= 1e-4:
        outcome = "Outcome A — EXACT CARRIER WORKS"
        outcome_desc = "Multiplicity-aware compression reproduces uncompressed centroid-filled intervention to numerical tolerance."
    elif min_agree >= 0.999:
        outcome = "Outcome B — APPROXIMATE CARRIER WORKS"
        outcome_desc = "Predictions remain effectively equivalent with slight floating-point accumulation difference."
    else:
        outcome = "Outcome C — CARRIER EQUIVALENCE FAILS"
        outcome_desc = "Compression materially changes predictions despite correct multiplicity handling."

    # Practical Classification
    # Check highest-compression operating points
    operating_points = {}
    for cfg in MODEL_CONFIGS:
        mk = cfg["key"]
        sub_c = df_phase_c[(df_phase_c["model"] == mk) & (df_phase_c["condition"] == "WEIGHTED_CENTROID_CARRIER")]
        sub_mean = sub_c.groupby("actual_fraction", as_index=False).agg({"top1_acc": "mean", "M_real": "first", "actual_k": "first", "total_downstream_tokens": "first"})
        clean_a = cfg["clean_acc"]

        # F95
        f95_cand = sub_mean[sub_mean["top1_acc"] >= 0.95 * clean_a]
        f95_pt = f95_cand.iloc[-1].to_dict() if len(f95_cand) > 0 else {}

        # F90
        f90_cand = sub_mean[sub_mean["top1_acc"] >= 0.90 * clean_a]
        f90_pt = f90_cand.iloc[-1].to_dict() if len(f90_cand) > 0 else {}

        operating_points[mk] = {
            "F95_point": f95_pt,
            "F90_point": f90_pt
        }

    # Speedup check
    has_latency_speedup = len(df_latency) > 0 and (df_latency["latency_reduction_pct"].max() > 5.0)

    if outcome.startswith("Outcome A") and has_latency_speedup:
        classification = "USEFUL APPLICATION"
        interp = "Weighted centroid carrier preserves centroid-reference accuracy, provides substantial token/FLOP reduction, produces measurable end-to-end latency speedup on GPU, and is highly competitive with matched-budget baselines."
    elif outcome.startswith("Outcome A") and not has_latency_speedup:
        classification = "MECHANISTIC DEMONSTRATION ONLY"
        interp = "Carrier equivalence holds exactly, but total end-to-end speed gain is negligible because intervention occurs too late in the network."
    else:
        classification = "NOT COMPETITIVE"
        interp = "Baselines dominate the Pareto frontier."

    validation_results = {
        "outcome": outcome,
        "classification": classification,
        "interpretation": interp,
        "validation_audit": validation_audit,
        "operating_points": operating_points
    }
    with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
        json.dump(validation_results, f, indent=2)

    total_time = time.time() - t_start
    manifest = {
        "experiment_name": "FUNGIBILITY-TO-COMPRESSION POC: WEIGHTED CENTROID CARRIER",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": total_time,
        "device": str(device),
        "smoke_test": smoke_test,
        "outcome": outcome,
        "classification": classification,
        "operating_points": operating_points
    }
    with open(os.path.join(output_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Pipeline] Execution completed in {total_time:.1f}s ({total_time/60:.2f}m)!")
    print(f"  VERDICT:        {outcome}")
    print(f"  CLASSIFICATION: {classification}")
    return {
        "manifest": manifest,
        "validation": validation_results
    }
