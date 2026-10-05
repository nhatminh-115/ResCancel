"""
scripts/generate_operator_targets.py

Offline Target Generation and Structural Analysis:
1. Generates oracle downstream Jacobian right singular vectors (V_r, sigma_r)
   for training (N=1,000) and validation (N=200) splits.
2. Computes Target Structure Statistics:
   - Pairwise subspace overlap across images (Grassmannian distance)
   - Singular value distribution and variability
3. Performs Mode Separability Audit:
   - Reshapes mode vectors q_k -> Q_k in R^{N x D}
   - Measures energy captured by rank 1, 2, 4, 8 approximations
4. Computes Static Calibration Subspace Baseline V_static.
5. Saves target_statistics.csv, mode_separability.csv, and figure_a_target_subspace_variability.png.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.operator_compression_confirmatory import compute_fast_downstream_jacobian
from patch_fungibility.amortized_operator import (
    get_amortized_imagenet_splits,
    extract_oracle_subspace,
    subspace_overlap
)


def main():
    parser = argparse.ArgumentParser(description="Generate Oracle Targets and Audit Mode Separability")
    parser.add_argument("--models", nargs="+", default=["deit_small", "vit_base", "deit_tiny", "dinov2"],
                        help="Models to process")
    parser.add_argument("--n_train", type=int, default=1000, help="Number of training targets (default: 1000)")
    parser.add_argument("--n_val", type=int, default=200, help="Number of validation targets (default: 200)")
    parser.add_argument("--r", type=int, default=32, help="Rank r of subspace (default: 32)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Oracle Target Generation & Mode Separability Audit ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Models: {args.models}, n_train: {args.n_train}, n_val: {args.n_val}, r: {args.r}")

    output_dir = os.path.abspath("outputs/fungibility_amortized_operator")
    targets_dir = os.path.join(output_dir, "targets")
    figures_dir = os.path.abspath("figures/fungibility_amortized_operator")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load 4-way disjoint splits
    print("\n--- Loading 4-way Disjoint ImageNet Splits ---")
    calib_set, eval_set, train_set, val_set = get_amortized_imagenet_splits(
        calib_seed=9101, eval_seed=9201, train_seed=7101, val_seed=7201,
        n_calib=1000, n_eval=1000, n_train=args.n_train, n_val=args.n_val
    )
    if args.n_train < len(train_set):
        train_set._samples = train_set._samples[:args.n_train]
    if args.n_val < len(val_set):
        val_set._samples = val_set._samples[:args.n_val]

    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    models_config = [
        {"model_key": "deit_small", "depth": 8, "name": "DeiT-Small", "N": 196, "D": 384},
        {"model_key": "vit_base", "depth": 7, "name": "ViT-B/16", "N": 196, "D": 768},
        {"model_key": "deit_tiny", "depth": 8, "name": "DeiT-Tiny", "N": 196, "D": 192},
        {"model_key": "dinov2", "depth": 8, "name": "DINOv2 ViT-S/14", "N": 256, "D": 384}
    ]
    models_config = [m for m in models_config if m["model_key"] in args.models]

    target_stats_rows = []
    mode_sep_rows = []
    overlap_distributions = {}

    for m_cfg in models_config:
        m_key = m_cfg["model_key"]
        m_name = m_cfg["name"]
        depth = m_cfg["depth"]
        N = m_cfg["N"]
        D = m_cfg["D"]

        print(f"\n==================================================")
        print(f"Processing Model: {m_name} (depth={depth}, N={N}, D={D})")
        print(f"==================================================")

        target_path = os.path.join(targets_dir, f"{m_key}_targets.pt")
        if os.path.exists(target_path):
            print(f"[{m_name}] Target checkpoint exists: {target_path}. Loading precomputed targets...")
            chk = torch.load(target_path, map_location="cpu")
            val_acts_tensor = chk["val_acts"]
            val_V_tensor = chk["val_V"]
            val_sigmas_tensor = chk["val_sigmas"]
            V_static = chk["V_static"]
        else:
            model, transform, meta = load_model_and_transform(m_key, device)
            model.eval()

            train_set.transform = transform
            val_set.transform = transform

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

            # Storage for targets
            train_acts = []
            train_V_r = []
            train_sigmas = []

            val_acts = []
            val_V_r = []
            val_sigmas = []

            t0_model = time.time()

            # 1. Process Training Set
            print(f"[{m_name}] Generating {len(train_set)} training targets...")
            for idx, (img, label) in enumerate(train_loader):
                img = img.to(device)
                with torch.no_grad():
                    _, acts = forward_block_by_block(model, m_key, x=img, collect_depths=(depth,))
                act = acts[depth] # (1, 1+N, D)

                J = compute_fast_downstream_jacobian(model, m_key, depth, act, chunk_size=128)
                V_r, sig_r = extract_oracle_subspace(J, r=args.r)

                # Store on CPU in float16 to preserve memory
                train_acts.append(act.squeeze(0).cpu().to(torch.float32))
                train_V_r.append(V_r.cpu().to(torch.float16))
                train_sigmas.append(sig_r.cpu().to(torch.float32))

                if (idx + 1) % 50 == 0:
                    elapsed = time.time() - t0_model
                    print(f"[{m_name}] Train: {idx + 1}/{len(train_set)} ({elapsed:.1f}s, {(idx+1)/elapsed:.1f} img/s)", flush=True)

            # 2. Process Validation Set
            print(f"[{m_name}] Generating {len(val_set)} validation targets...")
            for idx, (img, label) in enumerate(val_loader):
                img = img.to(device)
                with torch.no_grad():
                    _, acts = forward_block_by_block(model, m_key, x=img, collect_depths=(depth,))
                act = acts[depth]

                J = compute_fast_downstream_jacobian(model, m_key, depth, act, chunk_size=128)
                V_r, sig_r = extract_oracle_subspace(J, r=args.r)

                val_acts.append(act.squeeze(0).cpu().to(torch.float32))
                val_V_r.append(V_r.cpu().to(torch.float16))
                val_sigmas.append(sig_r.cpu().to(torch.float32))

            # Stack into tensors
            train_acts_tensor = torch.stack(train_acts) # (N_train, 1+N, D)
            train_V_tensor = torch.stack(train_V_r) # (N_train, ND, r) float16
            train_sigmas_tensor = torch.stack(train_sigmas) # (N_train, r)

            val_acts_tensor = torch.stack(val_acts)
            val_V_tensor = torch.stack(val_V_r)
            val_sigmas_tensor = torch.stack(val_sigmas)

            # 3. Compute Static Global Subspace (average projection matrix over train set)
            print(f"[{m_name}] Computing Global Static Subspace Baseline...")
            sample_size = min(200, len(train_V_r))
            cols = train_V_tensor[:sample_size].float().permute(1, 0, 2).reshape(N * D, sample_size * args.r).to(device)
            Gram_cols = cols.T @ cols # (Mr, Mr)
            evals_g, U_g = torch.linalg.eigh(Gram_cols)
            evals_g = evals_g.flip(0)
            U_g = U_g.flip(1)
            sig_g = torch.sqrt(torch.clamp(evals_g[:args.r], min=1e-12))
            V_static = (cols @ U_g[:, :args.r]) / sig_g.unsqueeze(0) # (ND, r)
            V_static, _ = torch.linalg.qr(V_static)
            V_static = V_static.cpu().to(torch.float32)

            del cols, Gram_cols, evals_g, U_g, model
            torch.cuda.empty_cache()

            # Save target checkpoint
            torch.save({
                "model_key": m_key,
                "depth": depth,
                "N": N, "D": D, "r": args.r,
                "train_acts": train_acts_tensor,
                "train_V": train_V_tensor,
                "train_sigmas": train_sigmas_tensor,
                "val_acts": val_acts_tensor,
                "val_V": val_V_tensor,
                "val_sigmas": val_sigmas_tensor,
                "V_static": V_static
            }, target_path)
            print(f"[{m_name}] Targets saved to: {target_path} (Size: {os.path.getsize(target_path)/1024/1024:.1f} MB)", flush=True)

        # 4. Measure Subspace Overlap Across Distinct Images
        print(f"[{m_name}] Computing pairwise subspace overlap across images...", flush=True)
        n_pairs = min(50, len(val_V_tensor))
        overlaps = []
        static_overlaps = []
        V_static_dev = V_static.to(device)

        for i in range(n_pairs):
            val_i = val_V_tensor[i].float().to(device)
            ol_static = subspace_overlap(val_i, V_static_dev)
            static_overlaps.append(ol_static)
            for j in range(i + 1, min(i + 6, n_pairs)):
                val_j = val_V_tensor[j].float().to(device)
                ol = subspace_overlap(val_i, val_j)
                overlaps.append(ol)
                del val_j
            del val_i

        del V_static_dev
        torch.cuda.empty_cache()

        overlap_distributions[m_name] = overlaps
        mean_ol = float(np.mean(overlaps))
        std_ol = float(np.std(overlaps))
        mean_static_ol = float(np.mean(static_overlaps))

        target_stats_rows.append({
            "model": m_name, "r": args.r,
            "pairwise_subspace_overlap_mean": mean_ol,
            "pairwise_subspace_overlap_std": std_ol,
            "static_subspace_overlap_mean": mean_static_ol,
            "singular_value_top1_mean": float(val_sigmas_tensor[:, 0].mean().item()),
            "singular_value_top32_mean": float(val_sigmas_tensor[:, -1].mean().item()),
            "singular_value_ratio_32_to_1": float((val_sigmas_tensor[:, -1] / val_sigmas_tensor[:, 0]).mean().item())
        })
        print(f"[{m_name}] Pairwise Overlap: {mean_ol*100:.2f}% +/- {std_ol*100:.2f}%, Static Overlap: {mean_static_ol*100:.2f}%", flush=True)

        # 5. Mode Separability Audit
        print(f"[{m_name}] Auditing mode separability (token x feature SVD)...", flush=True)
        r1_e, r2_e, r4_e, r8_e = [], [], [], []
        n_sep = min(20, len(val_V_tensor))

        for img_idx in range(n_sep):
            V_img = val_V_tensor[img_idx].float().to(device) # (ND, r)
            for k in range(args.r):
                Q_k = V_img[:, k].view(N, D)
                s = torch.linalg.svdvals(Q_k)
                e = s ** 2
                tot = e.sum().item()
                r1_e.append(e[0].item() / tot)
                r2_e.append(e[:2].sum().item() / tot)
                r4_e.append(e[:4].sum().item() / tot)
                r8_e.append(e[:8].sum().item() / tot)
            del V_img

        torch.cuda.empty_cache()

        mode_sep_rows.append({
            "model": m_name,
            "rank1_energy_mean": float(np.mean(r1_e)), "rank1_energy_std": float(np.std(r1_e)),
            "rank2_energy_mean": float(np.mean(r2_e)), "rank2_energy_std": float(np.std(r2_e)),
            "rank4_energy_mean": float(np.mean(r4_e)), "rank4_energy_std": float(np.std(r4_e)),
            "rank8_energy_mean": float(np.mean(r8_e)), "rank8_energy_std": float(np.std(r8_e))
        })
        print(f"[{m_name}] Separability: Rank-1={np.mean(r1_e)*100:.1f}%, Rank-2={np.mean(r2_e)*100:.1f}%, Rank-4={np.mean(r4_e)*100:.1f}%, Rank-8={np.mean(r8_e)*100:.1f}%", flush=True)

    # Save CSV summaries
    df_stats = pd.DataFrame(target_stats_rows)
    df_stats.to_csv(os.path.join(output_dir, "target_statistics.csv"), index=False)
    print(f"\nSaved target_statistics.csv")

    df_sep = pd.DataFrame(mode_sep_rows)
    df_sep.to_csv(os.path.join(output_dir, "mode_separability.csv"), index=False)
    print(f"Saved mode_separability.csv")

    # Figure A: Target Subspace Variability & Separability
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Pairwise Subspace Overlap Distribution
    ax1 = axes[0]
    overlap_data = []
    for m_name, ols in overlap_distributions.items():
        for ol in ols:
            overlap_data.append({"model": m_name, "overlap": ol * 100})
    df_ol_plot = pd.DataFrame(overlap_data)
    sns.barplot(data=df_ol_plot, x="model", y="overlap", ax=ax1, palette="Blues_d", errorbar="sd")
    ax1.set_title("Pairwise Subspace Overlap Across Images (%)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Grassmannian Overlap (%)")
    ax1.set_xlabel("Architecture")
    ax1.set_ylim(0, 5)

    # Panel 2: Mode Separability
    ax2 = axes[1]
    sep_plot_data = []
    for _, row in df_sep.iterrows():
        m = row["model"]
        sep_plot_data.append({"model": m, "rank": "Rank 1", "energy": row["rank1_energy_mean"] * 100})
        sep_plot_data.append({"model": m, "rank": "Rank 2", "energy": row["rank2_energy_mean"] * 100})
        sep_plot_data.append({"model": m, "rank": "Rank 4", "energy": row["rank4_energy_mean"] * 100})
        sep_plot_data.append({"model": m, "rank": "Rank 8", "energy": row["rank8_energy_mean"] * 100})
    df_sep_plot = pd.DataFrame(sep_plot_data)
    sns.barplot(data=df_sep_plot, x="model", y="energy", hue="rank", ax=ax2, palette="crest")
    ax2.set_title("Mode Separability: Low-Rank Structure in Token x Feature Space", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Energy Captured (%)")
    ax2.set_xlabel("Architecture")
    ax2.set_ylim(0, 100)
    ax2.legend(title="Approximation", loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "figure_a_target_subspace_variability.png"), dpi=300)
    plt.close()
    print("Saved figure_a_target_subspace_variability.png")

    print("\nTarget Generation Completed Successfully!")


if __name__ == "__main__":
    main()
