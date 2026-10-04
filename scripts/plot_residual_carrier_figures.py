"""
scripts/plot_residual_carrier_figures.py

Generates publication-quality figures for IMAGE-CONDITIONED RESIDUAL CARRIER POC:
1. figures/fungibility_residual_carrier/accuracy_vs_method.png
2. figures/fungibility_residual_carrier/delta_accuracy_heatmap.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_residual_carrier_plots(
    data_dir: str = "outputs/fungibility_residual_carrier",
    fig_dir: str = "figures/fungibility_residual_carrier"
):
    os.makedirs(fig_dir, exist_ok=True)
    summary_path = os.path.join(data_dir, "summary_by_condition.csv")
    heatmap_path = os.path.join(data_dir, "heatmap_grid.csv")

    if not os.path.exists(summary_path) or not os.path.exists(heatmap_path):
        print(f"Error: {summary_path} or {heatmap_path} not found.")
        return

    df_summary = pd.read_csv(summary_path)
    df_heatmap = pd.read_csv(heatmap_path)

    # Styling
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

    models = ["deit_small", "vit_base"]
    titles = {
        "deit_small": "DeiT-Small (Depth 8, Budget B=50)",
        "vit_base": "ViT-B/16 AugReg (Depth 7, Budget B=72)"
    }

    # --------------------------------------------------------------------------
    # Figure 1: Accuracy vs Method / Subspace Rank r across gammas
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, m in zip(axes, models):
        m_df = df_summary[df_summary["model"] == m]

        # Extract baseline pruning
        pruning_row = m_df[m_df["condition"] == "PURE_RANDOM_PRUNING"]
        pruning_acc = pruning_row["mean_top1_acc"].values[0] * 100.0
        pruning_std = pruning_row["std_top1_acc"].values[0] * 100.0

        # Plot horizontal line for Pure Pruning
        ax.axhline(
            pruning_acc, color="#1f77b4", linestyle="-", linewidth=2.0,
            label=f"Pure Random Pruning ({pruning_acc:.2f}%)"
        )
        ax.axhspan(
            pruning_acc - pruning_std, pruning_acc + pruning_std,
            color="#1f77b4", alpha=0.15
        )

        # Plot Centroid Carrier
        cent_row = m_df[m_df["condition"] == "UNWEIGHTED_CENTROID"]
        if len(cent_row) > 0:
            cent_acc = cent_row["mean_top1_acc"].values[0] * 100.0
            ax.axhline(
                cent_acc, color="#2ca02c", linestyle="--", linewidth=1.5,
                label=f"Centroid Carrier γ=0 ({cent_acc:.2f}%)"
            )

        # Plot curves for each gamma > 0 across r in [1, 4, 16, 64, D]
        r_order = ["1", "4", "16", "64", "D"]
        colors = {"0.25": "#ff7f0e", "0.5": "#d62728", "0.75": "#9467bd", "1.0": "#8c564b"}

        for g in [0.25, 0.5, 0.75, 1.0]:
            sub = m_df[(m_df["gamma"] == g) & (m_df["is_carrier"])].copy()
            sub["r_order"] = sub["r_rank"].map(lambda x: r_order.index(str(x)))
            sub = sub.sort_values("r_order")

            ax.plot(
                sub["r_rank"],
                sub["mean_top1_acc"] * 100.0,
                marker="o",
                label=f"Residual Carrier γ={g}",
                color=colors[str(g)],
                linewidth=1.8,
                markersize=6
            )

        ax.set_title(titles[m], fontweight="bold")
        ax.set_xlabel("PCA Subspace Rank r (D = Full Residual)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="lower left", framealpha=0.9, fontsize=8)
    axes[1].legend(loc="lower left", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "accuracy_vs_method.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {p1}")

    # --------------------------------------------------------------------------
    # Figure 2: Delta Accuracy Heatmap vs (r, gamma)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    r_order = ["1", "4", "16", "64", "D"]
    gamma_order = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ax, m in zip(axes, models):
        m_heat = df_heatmap[df_heatmap["model"] == m]
        # Pivot table: rows = gamma, cols = r
        pivot = m_heat.pivot(index="gamma", columns="r_rank", values="mean_delta_acc_pct")
        pivot = pivot.reindex(index=reversed(gamma_order), columns=r_order)

        # Plot heatmap with diverging colormap centered at 0
        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1.0)
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt="+.2f",
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={"label": "Δ Top-1 Acc vs. Pure Pruning (%)"},
            linewidths=0.5
        )
        ax.set_title(f"{titles[m]}\nHeatmap of Δ Accuracy (%)", fontweight="bold")
        ax.set_xlabel("PCA Subspace Rank r")
        ax.set_ylabel("Residual Shrinkage Factor γ")

    plt.tight_layout()
    p2 = os.path.join(fig_dir, "delta_accuracy_heatmap.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {p2}")


if __name__ == "__main__":
    generate_residual_carrier_plots()
