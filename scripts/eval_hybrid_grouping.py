"""
scripts/eval_hybrid_grouping.py

Controlled Evaluation of Hybrid Dynamic Grouping (IP-SM):
Isolates grouping benefit from operator carrier benefit by strictly comparing:
1. Spatial Group Mean (Centroid)
2. Spatial + Operator Carrier
3. Hybrid Group Mean (Centroid)
4. Hybrid + Operator Carrier
5. Attention Pruning (EViT)
6. Token Merging (ToMe)

Evaluates on identical images, depths, and budgets across 4 architectures:
- DeiT-Tiny
- DeiT-Small
- ViT-Base
- DINOv2-S/14

Produces:
- outputs/fungibility_batched_operator/hybrid_grouping_results.csv
- figures/fungibility_batched_operator/figure_e_hybrid_grouping.png
"""

import os
import sys
import time
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.dense_fraction_models import load_model_and_transform
from patch_fungibility.amortized_operator import FactorizedModePredictor
from patch_fungibility.practical_operator_compression import (
    create_fixed_spatial_grouping,
    forward_prefix_only
)
from patch_fungibility.batched_operator_compression import (
    forward_prefix_batched,
    forward_suffix_batched,
    batched_solve_amortized_carrier,
    create_hybrid_dynamic_grouping,
    run_evit_pruning_batched,
    run_tome_bipartite_batched
)


def evaluate_hybrid_grouping(
    models_dict: Dict[str, nn.Module],
    predictors_dict: Dict[Tuple[str, int], nn.Module],
    eval_loader: DataLoader,
    device: torch.device = torch.device("cuda"),
    num_eval_batches: int = 15
) -> List[Dict[str, Any]]:
    """
    Controlled evaluation across identical batches, depths, and budgets.
    """
    eval_configs = [
        {"model_key": "deit_tiny", "l": 6, "B": 49, "K_keep": 16, "N": 196, "D": 192},
        {"model_key": "deit_small", "l": 6, "B": 49, "K_keep": 16, "N": 196, "D": 384},
        {"model_key": "vit_base", "l": 5, "B": 49, "K_keep": 16, "N": 196, "D": 768},
        {"model_key": "dinov2", "l": 6, "B": 64, "K_keep": 16, "N": 256, "D": 384},
    ]

    all_results = []

    for cfg in eval_configs:
        model_key = cfg["model_key"]
        l = cfg["l"]
        B = cfg["B"]
        K_keep = cfg["K_keep"]
        N = cfg["N"]
        D = cfg["D"]

        model = models_dict[model_key].to(device).eval()
        predictor = predictors_dict.get((model_key, l))
        if predictor is not None:
            predictor = predictor.to(device).eval()

        print(f"\n=======================================================")
        print(f"Evaluating Hybrid Grouping on {model_key.upper()} at block {l}, budget {B}")
        print(f"=======================================================")

        # Precompute static spatial grouping
        grid_sz = int(math.isqrt(N))
        S_spatial, mults_spatial, _ = create_fixed_spatial_grouping(N, B, grid_h=grid_sz, grid_w=grid_sz, device=device)
        mults_spatial_cls = torch.cat([torch.tensor([1.0], device=device), mults_spatial])

        # Methods to evaluate
        methods = [
            "Clean",
            "Spatial_Group_Mean",
            "Spatial_Operator_Carrier",
            "Hybrid_Group_Mean",
            "Hybrid_Operator_Carrier",
            "Attention_Pruning_EViT",
            "Token_Merging_ToMe"
        ]

        method_correct = {m: 0 for m in methods}
        method_logits_diff = {m: [] for m in methods}
        method_latencies = {m: [] for m in methods}
        grouping_latencies = {"Spatial": [], "Hybrid": [], "EViT": [], "ToMe": []}
        total_samples = 0

        batch_idx = 0
        for images, targets in eval_loader:
            images = images.to(device)
            targets = targets.to(device)
            BS = images.shape[0]
            total_samples += BS

            with torch.no_grad():
                # 0. Clean forward reference
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                clean_logits = model(images)
                torch.cuda.synchronize()
                clean_time = (time.perf_counter() - t0) * 1000.0
                method_latencies["Clean"].append(clean_time)
                method_correct["Clean"] += (clean_logits.argmax(dim=-1) == targets).sum().item()

                # 1. Prefix pass
                h_l = forward_prefix_batched(model, model_key, l, images)
                cls_tok = h_l[:, :1, :]
                patches = h_l[:, 1:, :]

                # 2. Predictor subspace
                if predictor is not None:
                    V_pred = predictor(h_l)
                else:
                    V_pred = torch.randn(BS, N * D, 32, device=device)
                    V_pred, _ = torch.linalg.qr(V_pred)

                # --- Method 1: Spatial Group Mean ---
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                # Grouping is instant
                t_grp_spatial = (time.perf_counter() - t0) * 1000.0
                grouping_latencies["Spatial"].append(t_grp_spatial)

                C_mean_sp = torch.bmm(S_spatial.unsqueeze(0).expand(BS, -1, -1).transpose(1, 2), patches) / mults_spatial.view(1, B, 1)
                h_sp_mean = torch.cat([cls_tok, C_mean_sp], dim=1)
                logits_sp_mean = forward_suffix_batched(model, model_key, l, h_sp_mean, mults_spatial_cls, N, route="reference")
                torch.cuda.synchronize()
                method_latencies["Spatial_Group_Mean"].append((time.perf_counter() - t0) * 1000.0)
                method_correct["Spatial_Group_Mean"] += (logits_sp_mean.argmax(dim=-1) == targets).sum().item()
                method_logits_diff["Spatial_Group_Mean"].append(torch.norm(logits_sp_mean - clean_logits, dim=-1).mean().item())

                # --- Method 2: Spatial + Operator Carrier ---
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                res_sp_op = batched_solve_amortized_carrier(patches, S_spatial, mults_spatial, V_pred)
                h_sp_op = torch.cat([cls_tok, res_sp_op["C_opt"]], dim=1)
                logits_sp_op = forward_suffix_batched(model, model_key, l, h_sp_op, mults_spatial_cls, N, route="reference")
                torch.cuda.synchronize()
                method_latencies["Spatial_Operator_Carrier"].append((time.perf_counter() - t0) * 1000.0)
                method_correct["Spatial_Operator_Carrier"] += (logits_sp_op.argmax(dim=-1) == targets).sum().item()
                method_logits_diff["Spatial_Operator_Carrier"].append(torch.norm(logits_sp_op - clean_logits, dim=-1).mean().item())

                # --- Method 3: Hybrid Group Mean ---
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                S_hyb, mults_hyb = create_hybrid_dynamic_grouping(patches, B=B, K_keep=K_keep, grid_h=grid_sz, grid_w=grid_sz)
                torch.cuda.synchronize()
                t_grp_hyb = (time.perf_counter() - t0) * 1000.0
                grouping_latencies["Hybrid"].append(t_grp_hyb)

                C_mean_hyb = torch.bmm(S_hyb.transpose(1, 2), patches) / mults_hyb.unsqueeze(-1).clamp(min=1.0)
                h_hyb_mean = torch.cat([cls_tok, C_mean_hyb], dim=1)
                # Mults with CLS: (BS, 1+B)
                mults_hyb_cls = torch.cat([torch.ones(BS, 1, device=device), mults_hyb], dim=1)
                logits_hyb_mean = forward_suffix_batched(model, model_key, l, h_hyb_mean, mults_hyb_cls[0], N, route="reference")
                torch.cuda.synchronize()
                method_latencies["Hybrid_Group_Mean"].append((time.perf_counter() - t0) * 1000.0)
                method_correct["Hybrid_Group_Mean"] += (logits_hyb_mean.argmax(dim=-1) == targets).sum().item()
                method_logits_diff["Hybrid_Group_Mean"].append(torch.norm(logits_hyb_mean - clean_logits, dim=-1).mean().item())

                # --- Method 4: Hybrid + Operator Carrier ---
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                res_hyb_op = batched_solve_amortized_carrier(patches, S_hyb, mults_hyb, V_pred)
                h_hyb_op = torch.cat([cls_tok, res_hyb_op["C_opt"]], dim=1)
                logits_hyb_op = forward_suffix_batched(model, model_key, l, h_hyb_op, mults_hyb_cls[0], N, route="reference")
                torch.cuda.synchronize()
                method_latencies["Hybrid_Operator_Carrier"].append((time.perf_counter() - t0) * 1000.0)
                method_correct["Hybrid_Operator_Carrier"] += (logits_hyb_op.argmax(dim=-1) == targets).sum().item()
                method_logits_diff["Hybrid_Operator_Carrier"].append(torch.norm(logits_hyb_op - clean_logits, dim=-1).mean().item())

                # --- Method 5: Attention Pruning (EViT) ---
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                h_evit, mults_evit = run_evit_pruning_batched(h_l, B=B)
                torch.cuda.synchronize()
                grouping_latencies["EViT"].append((time.perf_counter() - t0) * 1000.0)
                logits_evit = forward_suffix_batched(model, model_key, l, h_evit, mults_evit, N, route="unweighted")
                torch.cuda.synchronize()
                method_latencies["Attention_Pruning_EViT"].append((time.perf_counter() - t0) * 1000.0)
                method_correct["Attention_Pruning_EViT"] += (logits_evit.argmax(dim=-1) == targets).sum().item()
                method_logits_diff["Attention_Pruning_EViT"].append(torch.norm(logits_evit - clean_logits, dim=-1).mean().item())

                # --- Method 6: Token Merging (ToMe) ---
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                h_tome, mults_tome = run_tome_bipartite_batched(h_l, target_tokens=B)
                torch.cuda.synchronize()
                grouping_latencies["ToMe"].append((time.perf_counter() - t0) * 1000.0)
                logits_tome = forward_suffix_batched(model, model_key, l, h_tome, mults_tome, N, route="unweighted")
                torch.cuda.synchronize()
                method_latencies["Token_Merging_ToMe"].append((time.perf_counter() - t0) * 1000.0)
                method_correct["Token_Merging_ToMe"] += (logits_tome.argmax(dim=-1) == targets).sum().item()
                method_logits_diff["Token_Merging_ToMe"].append(torch.norm(logits_tome - clean_logits, dim=-1).mean().item())

            batch_idx += 1
            if batch_idx >= num_eval_batches:
                break

        # Record metrics for this model
        for m in methods:
            acc = (method_correct[m] / total_samples) * 100.0
            lat = float(np.median(method_latencies[m]))
            l2 = float(np.mean(method_logits_diff[m])) if method_logits_diff[m] else 0.0

            # Determine grouping latency
            if "Spatial" in m:
                grp_lat = float(np.median(grouping_latencies["Spatial"]))
            elif "Hybrid" in m:
                grp_lat = float(np.median(grouping_latencies["Hybrid"]))
            elif "EViT" in m:
                grp_lat = float(np.median(grouping_latencies["EViT"]))
            elif "ToMe" in m:
                grp_lat = float(np.median(grouping_latencies["ToMe"]))
            else:
                grp_lat = 0.0

            throughput = (1000.0 * eval_loader.batch_size) / max(1e-4, lat)

            all_results.append({
                "model_key": model_key,
                "depth": l,
                "budget": B,
                "method": m,
                "top1_acc": acc,
                "clean_top1": (method_correct["Clean"] / total_samples) * 100.0,
                "retention_pp": acc - (method_correct["Clean"] / total_samples) * 100.0,
                "logit_l2": l2,
                "grouping_latency_ms": grp_lat,
                "total_batch_latency_ms": lat,
                "throughput_img_per_sec": throughput
            })
            print(f"[{model_key.upper()}][{m:25s}] Top-1: {acc:.2f}% (d={acc - (method_correct['Clean'] / total_samples) * 100.0:+.2f} pp) | Grp Lat: {grp_lat:.3f} ms | Batch Lat: {lat:.2f} ms")

    return all_results


def plot_hybrid_grouping_results(df: pd.DataFrame, out_path: str):
    """
    Renders Figure E: Hybrid Grouping Control (Separating Grouping from Carrier Benefit).
    """
    models = df["model_key"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes = [axes]

    method_colors = {
        "Clean": "#7f7f7f",
        "Spatial_Group_Mean": "#aec7e8",
        "Spatial_Operator_Carrier": "#1f77b4",
        "Hybrid_Group_Mean": "#98df8a",
        "Hybrid_Operator_Carrier": "#2ca02c",
        "Attention_Pruning_EViT": "#ffbb78",
        "Token_Merging_ToMe": "#ff7f0e"
    }

    for idx, model_key in enumerate(models):
        sub_df = df[df["model_key"] == model_key]
        ax = axes[idx]

        methods = sub_df["method"].values
        accs = sub_df["top1_acc"].values
        colors = [method_colors.get(m, "#333333") for m in methods]

        bars = ax.bar(range(len(methods)), accs, color=colors, edgecolor="black")
        clean_acc = sub_df[sub_df["method"] == "Clean"]["top1_acc"].values[0]
        ax.axhline(clean_acc, color="gray", linestyle="--", alpha=0.7, label=f"Clean ({clean_acc:.1f}%)")

        ax.set_xticks(range(len(methods)))
        clean_labels = [m.replace("_", "\n") for m in methods]
        ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{model_key.upper()} (l={sub_df['depth'].iloc[0]}, B={sub_df['budget'].iloc[0]})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Top-1 Accuracy (%)" if idx == 0 else "")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Highlight value on top
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.suptitle("Figure E: Fair Hybrid Control — Separating Grouping from Carrier Correction", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure E to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Hybrid Dynamic Grouping Evaluation on {device}...")

    # Load disjoint ImageNet split
    _, eval_set, _, _ = get_disjoint_imagenet_splits(calib_seed=9101, eval_seed=9201, n_per_split=1000)
    eval_loader = DataLoader(eval_set, batch_size=32, shuffle=False, num_workers=2)

    # Load models
    models = {
        "deit_tiny": load_model_and_transform("deit_tiny", device)[0],
        "deit_small": load_model_and_transform("deit_small", device)[0],
        "vit_base": load_model_and_transform("vit_base", device)[0],
        "dinov2": load_model_and_transform("dinov2", device)[0],
    }

    # Load pre-trained amortized predictors from checkpoints if available
    predictors = {}
    ckp_dir = os.path.join(REPO_ROOT, "outputs", "fungibility_amortized_operator", "checkpoints")
    for (m_key, l, d) in [("deit_tiny", 6, 192), ("deit_small", 6, 384), ("vit_base", 5, 768), ("dinov2", 6, 384)]:
        ckp_path = os.path.join(ckp_dir, f"predictor_{m_key}_l{l}_rank32.pt")
        pred = FactorizedModePredictor(N=256 if m_key == "dinov2" else 196, D=d, r=32)
        if os.path.exists(ckp_path):
            pred.load_state_dict(torch.load(ckp_path, map_location=device))
            print(f"Loaded trained predictor for {m_key} block {l}")
        else:
            print(f"No checkpoint found at {ckp_path}, using initialized predictor")
        predictors[(m_key, l)] = pred

    results = evaluate_hybrid_grouping(models, predictors, eval_loader, device=device, num_eval_batches=15)
    df = pd.DataFrame(results)

    # Save CSV
    out_csv = os.path.join(REPO_ROOT, "outputs", "fungibility_batched_operator", "hybrid_grouping_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved hybrid grouping results to {out_csv}")

    # Generate Figure E
    out_fig = os.path.join(REPO_ROOT, "figures", "fungibility_batched_operator", "figure_e_hybrid_grouping.png")
    plot_hybrid_grouping_results(df, out_fig)


if __name__ == "__main__":
    main()
