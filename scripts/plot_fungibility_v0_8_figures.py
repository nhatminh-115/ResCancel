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


def plot_all_v0_8_figures(
    output_dir: str = "outputs/fungibility_v0_8",
    figures_dir: str = "figures/fungibility_v0_8"
):
    os.makedirs(figures_dir, exist_ok=True)

    rank_df = pd.read_csv(os.path.join(output_dir, "rank_condition_results.csv"))
    grouped_df = pd.read_csv(os.path.join(output_dir, "grouped_diversity_results.csv"))
    shared_df = pd.read_csv(os.path.join(output_dir, "shared_vs_independent_results.csv"))
    rep_df = pd.read_csv(os.path.join(output_dir, "representation_rank_diagnostics.csv"))
    attn_df = pd.read_csv(os.path.join(output_dir, "attention_rank_diagnostics.csv"))

    colors = {
        "Tiny_PCA": "#1f77b4",
        "Tiny_Random": "#aec7e8",
        "Small_PCA": "#d62728",
        "Small_Random": "#ff9896",
        "Centroid": "#7f7f7f",
        "Shared": "#ff7f0e",
        "Independent": "#2ca02c",
        "Clean": "#17becf"
    }

    # -----------------------------------------------------------------
    # Figure 1: accuracy_vs_designed_rank.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = rank_df[rank_df["model"] == model]

        pca_sub = m_df[m_df["basis_type"] == "pca"].groupby("rank")["accuracy"].agg(["mean", "std"]).reset_index()
        rand_sub = m_df[m_df["basis_type"] == "random"].groupby("rank")["accuracy"].agg(["mean", "std"]).reset_index()

        ax.errorbar(
            pca_sub["rank"], pca_sub["mean"] * 100, yerr=pca_sub["std"] * 100,
            marker="o", linewidth=2.2, capsize=4, label="PCA Subspace",
            color=colors["Tiny_PCA"] if "Tiny" in m_name else colors["Small_PCA"]
        )
        ax.errorbar(
            rand_sub["rank"], rand_sub["mean"] * 100, yerr=rand_sub["std"] * 100,
            marker="s", linestyle="--", linewidth=2.0, capsize=4, label="Random Subspace",
            color=colors["Tiny_Random"] if "Tiny" in m_name else colors["Small_Random"]
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Designed Subspace Rank R")
        if idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name}: Accuracy vs Subspace Rank")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "accuracy_vs_designed_rank.png"))
    plt.close(fig)
    print("Saved accuracy_vs_designed_rank.png")

    # -----------------------------------------------------------------
    # Figure 2: margin_vs_effective_rank.png
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in ["deit_tiny_patch16_224", "deit_small_patch16_224"]:
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_rep = rep_df[(rep_df["model"] == model) & (rep_df["block_depth"] == 8)]
        # Merge with rank_df on condition
        merged = pd.merge(m_rep, rank_df[rank_df["model"] == model], on=["model", "condition"])

        color = colors["Tiny_PCA"] if "Tiny" in m_name else colors["Small_PCA"]
        ax.scatter(
            merged["effective_rank"], merged["mean_margin"],
            label=f"{m_name} (Depth 8 r_eff)", s=60, alpha=0.85, color=color
        )
        # Trendline
        z = np.polyfit(merged["effective_rank"], merged["mean_margin"], 1)
        p = np.poly1d(z)
        x_vals = np.linspace(merged["effective_rank"].min(), merged["effective_rank"].max(), 50)
        ax.plot(x_vals, p(x_vals), linestyle=":", color=color, alpha=0.7)

    ax.set_xlabel("Measured Participation-Ratio Effective Rank (r_eff)")
    ax.set_ylabel("Mean True-Class Margin")
    ax.set_title("True-Class Margin vs Measured Effective Rank at Injection")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "margin_vs_effective_rank.png"))
    plt.close(fig)
    print("Saved margin_vs_effective_rank.png")

    # -----------------------------------------------------------------
    # Figure 3: pca_vs_random_subspace.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = rank_df[rank_df["model"] == model]

        pca_sub = m_df[m_df["basis_type"] == "pca"].groupby("rank")["accuracy"].mean()
        rand_sub = m_df[m_df["basis_type"] == "random"].groupby("rank")["accuracy"].mean()
        diff = (pca_sub - rand_sub) * 100

        ranks = sorted(diff.index.tolist())
        diff_vals = [diff[r] for r in ranks]

        bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in diff_vals]
        ax.bar([str(r) for r in ranks], diff_vals, color=bar_colors, alpha=0.85, edgecolor="black")
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xlabel("Subspace Rank R")
        if idx == 0:
            ax.set_ylabel("Accuracy Advantage: PCA - Random (%)")
        ax.set_title(f"{m_name}: PCA Subspace Geometry Advantage")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "pca_vs_random_subspace.png"))
    plt.close(fig)
    print("Saved pca_vs_random_subspace.png")

    # -----------------------------------------------------------------
    # Figure 4: shared_vs_independent_noise.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = shared_df[shared_df["model"] == model]

        cond_order = ["static_centroid", "shared_noise", "independent_noise"]
        labels = ["Static Centroid\n(R=0, K=1)", "Shared Noise\n(R=0, K=1, Stoch)", "Independent Noise\n(Full Diversity)"]
        means = []
        stds = []
        all_seed_points = []

        for c in cond_order:
            sub = m_df[m_df["condition_type"] == c]
            means.append(sub["accuracy"].mean() * 100)
            stds.append(sub["accuracy"].std() * 100 if len(sub) > 1 else 0.0)
            all_seed_points.append((sub["accuracy"].values * 100).tolist())

        bar_colors = ["#7f7f7f", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(labels, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="black")

        # Overlay individual seed points
        for i, pts in enumerate(all_seed_points):
            x_pts = np.random.normal(i, 0.04, size=len(pts))
            ax.scatter(x_pts, pts, color="black", s=35, zorder=5, alpha=0.8)

        if idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name}: Shared vs Independent Noise")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "shared_vs_independent_noise.png"))
    plt.close(fig)
    print("Saved shared_vs_independent_noise.png")

    # -----------------------------------------------------------------
    # Figure 5: accuracy_vs_unique_token_count.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_df = grouped_df[grouped_df["model"] == model]

        k_sub = m_df.groupby("k")["accuracy"].agg(["mean", "std"]).reset_index()

        color = colors["Tiny_PCA"] if "Tiny" in m_name else colors["Small_PCA"]
        ax.errorbar(
            k_sub["k"], k_sub["mean"] * 100, yerr=k_sub["std"] * 100,
            marker="D", linewidth=2.2, capsize=4, color=color, label=f"{m_name} Grouped K"
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 196])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Number of Unique Replacement Tokens (K)")
        if idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name}: Accuracy vs Unique Token Count (K)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "accuracy_vs_unique_token_count.png"))
    plt.close(fig)
    print("Saved accuracy_vs_unique_token_count.png")

    # -----------------------------------------------------------------
    # Figure 6: effective_rank_through_blocks.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_rep = rep_df[rep_df["model"] == model]

        key_conds = ["clean", "static_centroid", "pca_rank_4_seed_11001", "pca_rank_16_seed_11001", "diagonal_gaussian_seed_11001"]
        labels = ["Clean Baseline", "Static Centroid", "PCA Rank 4", "PCA Rank 16", "Full Gaussian"]
        line_styles = ["-", "--", "-.", ":", "-"]
        line_colors = ["#17becf", "#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"]

        for c, lbl, ls, col in zip(key_conds, labels, line_styles, line_colors):
            c_df = m_rep[m_rep["condition"] == c].sort_values("block_depth")
            if not c_df.empty:
                ax.plot(
                    c_df["block_depth"], c_df["effective_rank"],
                    marker="o", label=lbl, linestyle=ls, color=col, linewidth=2.0
                )

        ax.set_xticks([8, 9, 10, 11])
        ax.set_xticklabels(["Depth 8\n(Injection)", "Block 9", "Block 10", "Block 11"])
        ax.set_xlabel("Transformer Depth")
        if idx == 0:
            ax.set_ylabel("Representation Effective Rank (r_eff)")
        ax.set_title(f"{m_name}: Effective Rank Through Blocks")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "effective_rank_through_blocks.png"))
    plt.close(fig)
    print("Saved effective_rank_through_blocks.png")

    # -----------------------------------------------------------------
    # Figure 7: attention_rank_through_blocks.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, model in enumerate(["deit_tiny_patch16_224", "deit_small_patch16_224"]):
        ax = axes[idx]
        m_name = "DeiT-Tiny" if "tiny" in model else "DeiT-Small"
        m_attn = attn_df[attn_df["model"] == model]

        key_conds = ["clean", "static_centroid", "pca_rank_4_seed_11001", "diagonal_gaussian_seed_11001"]
        labels = ["Clean", "Static Centroid", "PCA Rank 4", "Full Gaussian"]
        line_colors = ["#17becf", "#7f7f7f", "#1f77b4", "#2ca02c"]

        for c, lbl, col in zip(key_conds, labels, line_colors):
            c_df = m_attn[m_attn["condition"] == c].sort_values("block_depth")
            if not c_df.empty:
                ax.plot(
                    c_df["block_depth"], c_df["cls_attention_entropy"],
                    marker="s", label=lbl, color=col, linewidth=2.0
                )

        ax.set_xticks([9, 10, 11])
        ax.set_xticklabels(["Block 9", "Block 10", "Block 11"])
        ax.set_xlabel("Downstream Block")
        if idx == 0:
            ax.set_ylabel("CLS-to-Patch Attention Entropy (nats)")
        ax.set_title(f"{m_name}: CLS Attention Entropy")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "attention_rank_through_blocks.png"))
    plt.close(fig)
    print("Saved attention_rank_through_blocks.png")
