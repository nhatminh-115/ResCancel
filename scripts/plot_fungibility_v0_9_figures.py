import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "figure.dpi": 300
})


def plot_all_v0_9_figures(
    output_dir: str = "outputs/fungibility_v0_9",
    figures_dir: str = "figures/fungibility_v0_9"
):
    os.makedirs(figures_dir, exist_ok=True)

    spec_df = pd.read_csv(os.path.join(output_dir, "eigenvalue_spectrum.csv"))
    comp_df = pd.read_csv(os.path.join(output_dir, "natural_vs_energy_matched_rank.csv"))
    pc_id_df = pd.read_csv(os.path.join(output_dir, "pc_identity_results.csv"))
    scale_df = pd.read_csv(os.path.join(output_dir, "pc1_scale_sweep.csv"))
    rand_df = pd.read_csv(os.path.join(output_dir, "random_direction_results.csv"))
    prop_df = pd.read_csv(os.path.join(output_dir, "rank_propagation.csv"))
    attn_df = pd.read_csv(os.path.join(output_dir, "attention_diagnostics.csv"))

    colors = {
        "Tiny_Nat": "#1f77b4",
        "Tiny_EM": "#aec7e8",
        "Small_Nat": "#d62728",
        "Small_EM": "#ff9896",
        "Centroid": "#7f7f7f",
        "Clean": "#17becf",
        "Gaussian": "#2ca02c"
    }

    # -----------------------------------------------------------------
    # Figure 1: eigenvalue_spectrum.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1A: Scree plot (log eigenvalues)
    ax = axes[0]
    for m in ["deit_tiny_patch16_224", "deit_small_patch16_224"]:
        m_sub = spec_df[spec_df["model"] == m].sort_values("component")
        lbl = "DeiT-Tiny (D=192)" if "tiny" in m else "DeiT-Small (D=384)"
        col = colors["Tiny_Nat"] if "tiny" in m else colors["Small_Nat"]
        ax.plot(m_sub["component"], m_sub["eigenvalue"], label=lbl, color=col, linewidth=2.2)

    ax.set_yscale("log")
    ax.set_xlabel("Principal Component Index (r)")
    ax.set_ylabel("Eigenvalue (Variance along PC_r)")
    ax.set_title("Calibration Patch Eigenvalue Spectrum")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 1B: Cumulative Explained Variance Ratio
    ax = axes[1]
    for m in ["deit_tiny_patch16_224", "deit_small_patch16_224"]:
        m_sub = spec_df[spec_df["model"] == m].sort_values("component")
        lbl = "DeiT-Tiny" if "tiny" in m else "DeiT-Small"
        col = colors["Tiny_Nat"] if "tiny" in m else colors["Small_Nat"]
        ax.plot(m_sub["component"], m_sub["cumulative_evr"] * 100, label=lbl, color=col, linewidth=2.2)

    ax.set_xlim(1, 64)
    ax.set_xlabel("Number of Top Principal Components (R)")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Cumulative Variance Captured by Top-R PCs")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "eigenvalue_spectrum.png"))
    plt.close(fig)
    print("Saved eigenvalue_spectrum.png")

    # -----------------------------------------------------------------
    # Figure 2: natural_vs_energy_matched_rank.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = comp_df[comp_df["model"] == model]

        nat_sub = m_df[m_df["formulation"] == "natural"].groupby("rank")["accuracy"].agg(["mean", "std"]).reset_index()
        em_sub = m_df[m_df["formulation"] == "energy_matched"].groupby("rank")["accuracy"].agg(["mean", "std"]).reset_index()

        col_nat = colors["Tiny_Nat"] if "tiny" in model else colors["Small_Nat"]
        col_em = colors["Tiny_EM"] if "tiny" in model else colors["Small_EM"]

        ax.errorbar(
            nat_sub["rank"], nat_sub["mean"] * 100, yerr=nat_sub["std"] * 100,
            marker="o", linewidth=2.4, capsize=4, label="PCA Natural Energy (Unscaled)",
            color=col_nat
        )
        ax.errorbar(
            em_sub["rank"], em_sub["mean"] * 100, yerr=em_sub["std"] * 100,
            marker="s", linestyle="--", linewidth=2.0, capsize=4, label="PCA Energy-Matched (V0.8)",
            color=col_em
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Subspace Rank R")
        if idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name}: Natural vs Energy-Matched Rank")
        ax.legend(frameon=True, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "natural_vs_energy_matched_rank.png"))
    plt.close(fig)
    print("Saved natural_vs_energy_matched_rank.png")

    # -----------------------------------------------------------------
    # Figure 3: accuracy_vs_natural_rank.png
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in ["deit_tiny_patch16_224", "deit_small_patch16_224"]:
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = comp_df[(comp_df["model"] == model) & (comp_df["formulation"] == "natural")]
        sub = m_df.groupby("rank")["accuracy"].agg(["mean", "std"]).reset_index()

        col = colors["Tiny_Nat"] if "tiny" in model else colors["Small_Nat"]
        ax.errorbar(
            sub["rank"], sub["mean"] * 100, yerr=sub["std"] * 100,
            marker="o", linewidth=2.4, capsize=4, label=f"{m_name} (Natural Variance)",
            color=col
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Natural PCA Subspace Rank R")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Classification Recovery vs Natural Low-Rank Diversity")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "accuracy_vs_natural_rank.png"))
    plt.close(fig)
    print("Saved accuracy_vs_natural_rank.png")

    # -----------------------------------------------------------------
    # Figure 4: pc_identity_comparison.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = pc_id_df[pc_id_df["model"] == model]

        nat_sub = m_df[m_df["condition_type"] == "natural"].groupby("pc_index")["accuracy"].mean() * 100
        mat_sub = m_df[m_df["condition_type"] == "matched"].groupby("pc_index")["accuracy"].mean() * 100

        # Random 1D benchmark
        rand_acc = rand_df[rand_df["model"] == model]["accuracy"].mean() * 100

        pc_labels = [f"PC{k}" for k in sorted(nat_sub.index)]
        x = np.arange(len(pc_labels))
        width = 0.35

        ax.bar(x - width/2, [nat_sub[k] for k in sorted(nat_sub.index)], width, label="Natural Eigenvalue (Var=lambda_k)", color="#1f77b4" if "tiny" in model else "#d62728", alpha=0.85)
        ax.bar(x + width/2, [mat_sub[k] for k in sorted(mat_sub.index)], width, label="PC1-Energy-Matched (Var=lambda_1)", color="#aec7e8" if "tiny" in model else "#ff9896", alpha=0.85)

        ax.axhline(rand_acc, color="#ff7f0e", linestyle="--", linewidth=2.0, label=f"Random 1D (Var=lambda_1): {rand_acc:.1f}%")

        ax.set_xticks(x)
        ax.set_xticklabels(pc_labels)
        ax.set_xlabel("Principal Component Direction")
        if idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name}: PC Identity vs Random 1D Direction")
        ax.legend(frameon=True, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "pc_identity_comparison.png"))
    plt.close(fig)
    print("Saved pc_identity_comparison.png")

    # -----------------------------------------------------------------
    # Figure 5: pc1_amplitude_sweep.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = scale_df[scale_df["model"] == model]

        sub = m_df.groupby("scale_label")["accuracy"].agg(["mean", "std"]).reindex(["0.25", "0.50", "1.00", "2.00", "4.00", "E_MATCH"]).reset_index()

        col = colors["Tiny_Nat"] if "tiny" in model else colors["Small_Nat"]
        ax.errorbar(
            sub["scale_label"], sub["mean"] * 100, yerr=sub["std"] * 100,
            marker="o", linewidth=2.4, capsize=4, color=col, label=f"{m_name} PC1 Scale"
        )
        ax.axvline("1.00", color="gray", linestyle=":", label="Natural Scale (s=1.0)")
        ax.axvline("E_MATCH", color="red", linestyle="--", alpha=0.7, label="V0.8 Energy-Match (s=E_MATCH)")

        ax.set_xlabel("PC1 Amplitude Multiplier (s)")
        if idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name}: PC1 Amplitude Sensitivity Sweep")
        ax.legend(frameon=True, loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "pc1_amplitude_sweep.png"))
    plt.close(fig)
    print("Saved pc1_amplitude_sweep.png")

    # -----------------------------------------------------------------
    # Figure 6: rank_expansion_through_blocks.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = prop_df[prop_df["model"] == model]

        key_conds = ["clean", "static_centroid", "natural_pca_rank_1_seed_16001", "energy_matched_pca_rank_1_seed_16001", "diagonal_gaussian_seed_16001"]
        labels = ["Clean Baseline", "Static Centroid", "Natural PCA Rank 1", "Energy-Matched PCA Rank 1", "Full Gaussian"]
        line_styles = ["-", "--", "-", "-.", ":"]
        line_colors = ["#17becf", "#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"]

        for c, lbl, ls, col in zip(key_conds, labels, line_styles, line_colors):
            c_df = m_df[m_df["condition"] == c].sort_values("block_depth")
            if not c_df.empty:
                ax.plot(
                    c_df["block_depth"], c_df["effective_rank"],
                    marker="o", label=lbl, linestyle=ls, color=col, linewidth=2.2
                )

        ax.set_xticks([8, 9, 10, 11])
        ax.set_xticklabels(["Depth 8\n(Injection)", "Block 9", "Block 10", "Block 11"])
        ax.set_xlabel("Transformer Depth")
        if idx == 0:
            ax.set_ylabel("Representation Effective Rank (r_eff)")
        ax.set_title(f"{m_name}: Rank Expansion Through Blocks")
        ax.legend(frameon=True, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "rank_expansion_through_blocks.png"))
    plt.close(fig)
    print("Saved rank_expansion_through_blocks.png")

    # -----------------------------------------------------------------
    # Figure 7: attention_recovery_through_blocks.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = attn_df[attn_df["model"] == model]

        key_conds = ["clean", "static_centroid", "natural_pca_rank_1_seed_16001", "diagonal_gaussian_seed_16001"]
        labels = ["Clean", "Static Centroid", "Natural PCA Rank 1", "Full Gaussian"]
        line_colors = ["#17becf", "#7f7f7f", "#ff7f0e", "#2ca02c"]

        for c, lbl, col in zip(key_conds, labels, line_colors):
            c_df = m_df[m_df["condition"] == c].sort_values("block_depth")
            if not c_df.empty:
                ax.plot(
                    c_df["block_depth"], c_df["cls_attention_entropy"],
                    marker="s", label=lbl, color=col, linewidth=2.2
                )

        ax.set_xticks([9, 10, 11])
        ax.set_xticklabels(["Block 9", "Block 10", "Block 11"])
        ax.set_xlabel("Downstream Block")
        if idx == 0:
            ax.set_ylabel("CLS-to-Patch Attention Entropy (nats)")
        ax.set_title(f"{m_name}: Attention Entropy Recovery")
        ax.legend(frameon=True, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "attention_recovery_through_blocks.png"))
    plt.close(fig)
    print("Saved attention_recovery_through_blocks.png")
