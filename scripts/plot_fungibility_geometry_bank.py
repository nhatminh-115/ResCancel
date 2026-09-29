"""
scripts/plot_fungibility_geometry_bank.py

Generates publication-quality figures for GEOMETRY-DIVERSITY BANK POC:
1. figures/fungibility_geometry_bank/accuracy_vs_token_budget.png
2. figures/fungibility_geometry_bank/delta_vs_pruning.png
3. figures/fungibility_geometry_bank/pca_vs_kmeans.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_geometry_bank_plots(
    data_dir: str = "outputs/fungibility_geometry_bank",
    fig_dir: str = "figures/fungibility_geometry_bank"
):
    os.makedirs(fig_dir, exist_ok=True)
    summary_path = os.path.join(data_dir, "matched_budget_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)

    # Style configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300
    })

    models = ["deit_tiny", "deit_small", "vit_base", "dinov2"]
    titles = {
        "deit_tiny": "DeiT-Tiny (Depth 8)",
        "deit_small": "DeiT-Small (Depth 8)",
        "vit_base": "ViT-B/16 AugReg (Depth 7)",
        "dinov2": "DINOv2 ViT-S/14 (Depth 9)"
    }

    # --------------------------------------------------------------------------
    # Figure 1: Accuracy vs Token Budget B
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)

    conditions_to_plot = [
        ("RANDOM_PRUNING", 0, "Random Pruning (No Carrier)", "#1f77b4", "o", "-"),
        ("REAL_PLUS_CENTROID", 1, "Single Centroid (K=1)", "#2ca02c", "s", "--"),
        ("REAL_PLUS_PCA_BANK", 4, "PCA Bank (K=4)", "#ff7f0e", "^", "-."),
        ("REAL_PLUS_PCA_BANK", 8, "PCA Bank (K=8)", "#d62728", "v", "-."),
        ("REAL_PLUS_KMEANS_BANK", 4, "K-Means Bank (K=4)", "#9467bd", "D", ":"),
        ("REAL_PLUS_KMEANS_BANK", 8, "K-Means Bank (K=8)", "#8c564b", "X", ":")
    ]

    for ax, m in zip(axes, models):
        m_df = df[df["model"] == m]
        budgets = sorted(m_df["budget_B"].unique())

        for cond, k, label, color, marker, ls in conditions_to_plot:
            sub = m_df[(m_df["condition"] == cond) & (m_df["K_synth"] == k)]
            if len(sub) > 0:
                sub = sub.sort_values("budget_B")
                ax.plot(
                    sub["budget_B"],
                    sub["mean_top1_acc"] * 100.0,
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    linewidth=1.8,
                    markersize=6
                )

        ax.set_title(titles[m], fontweight="bold")
        ax.set_xlabel("Downstream Patch Tokens (B)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(budgets)

    axes[0].legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "accuracy_vs_token_budget.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {p1}")

    # --------------------------------------------------------------------------
    # Figure 2: Delta Accuracy vs Random Pruning
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    delta_conds = [
        ("REAL_PLUS_CENTROID", 1, "Single Centroid (K=1)", "#2ca02c"),
        ("REAL_PLUS_PCA_BANK", 4, "PCA (K=4)", "#ff7f0e"),
        ("REAL_PLUS_PCA_BANK", 8, "PCA (K=8)", "#d62728"),
        ("REAL_PLUS_PCA_BANK", 16, "PCA (K=16)", "#e377c2"),
        ("REAL_PLUS_KMEANS_BANK", 4, "K-Means (K=4)", "#9467bd"),
        ("REAL_PLUS_KMEANS_BANK", 8, "K-Means (K=8)", "#8c564b"),
        ("REAL_PLUS_KMEANS_BANK", 16, "K-Means (K=16)", "#17becf")
    ]

    for ax, m in zip(axes, models):
        m_df = df[df["model"] == m]
        budgets = sorted(m_df["budget_B"].unique())

        x = np.arange(len(budgets))
        total_bars = len(delta_conds)
        width = 0.8 / total_bars

        for idx, (cond, k, label, color) in enumerate(delta_conds):
            vals = []
            for B in budgets:
                row = m_df[(m_df["condition"] == cond) & (m_df["K_synth"] == k) & (m_df["budget_B"] == B)]
                if len(row) > 0:
                    vals.append(row["mean_delta_acc"].values[0] * 100.0)
                else:
                    vals.append(np.nan)
            offset = (idx - total_bars / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=label if m == models[0] else "", color=color, alpha=0.85)

        ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
        ax.axhline(1.0, color="green", linestyle="--", linewidth=1.0, alpha=0.7, label="+1% Promising Threshold" if m == models[0] else "")
        ax.set_title(titles[m], fontweight="bold")
        ax.set_xlabel("Downstream Patch Tokens (B)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"B={b}" for b in budgets])
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    axes[0].set_ylabel("Δ Top-1 Acc vs. Random Pruning (%)")
    axes[0].legend(loc="lower left", framealpha=0.85, fontsize=8)
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "delta_vs_pruning.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {p2}")

    # --------------------------------------------------------------------------
    # Figure 3: PCA vs K-Means Direct Comparison
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, m in zip(axes, models):
        m_df = df[df["model"] == m]
        pca_rows = m_df[m_df["condition"] == "REAL_PLUS_PCA_BANK"].sort_values(["budget_B", "K_synth"])
        km_rows = m_df[m_df["condition"] == "REAL_PLUS_KMEANS_BANK"].sort_values(["budget_B", "K_synth"])

        merged = pd.merge(pca_rows, km_rows, on=["budget_B", "K_synth"], suffixes=("_pca", "_km"))

        ax.scatter(
            merged["mean_top1_acc_pca"] * 100.0,
            merged["mean_top1_acc_km"] * 100.0,
            c=merged["K_synth"],
            cmap="viridis",
            s=70,
            edgecolors="black",
            linewidth=1.2,
            zorder=3
        )

        # Plot y=x line
        all_vals = list(merged["mean_top1_acc_pca"] * 100.0) + list(merged["mean_top1_acc_km"] * 100.0)
        min_v = min(all_vals) - 1.0
        max_v = max(all_vals) + 1.0
        ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.6, label="Parity (y = x)")

        ax.set_title(titles[m], fontweight="bold")
        ax.set_xlabel("PCA Bank Top-1 Acc (%)")
        ax.set_ylabel("K-Means Bank Top-1 Acc (%)")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p3 = os.path.join(fig_dir, "pca_vs_kmeans.png")
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {p3}")


if __name__ == "__main__":
    generate_geometry_bank_plots()
