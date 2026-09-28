import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_all_fungibility_v0_6_figures(
    summary_df: pd.DataFrame,
    output_dir: str = "figures/fungibility_v0_6"
):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    models = ["deit_tiny_patch16_224", "deit_small_patch16_224"]
    fractions = ["10%", "25%", "50%"]
    depths = sorted(summary_df["depth"].unique())

    model_display_names = {
        "deit_tiny_patch16_224": "DeiT-Tiny",
        "deit_small_patch16_224": "DeiT-Small"
    }

    # -----------------------------------------------------------------
    # Figure 1: gaussian_recovery_by_depth.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=True)
    frac_colors = {"10%": "#1f77b4", "25%": "#2ca02c", "50%": "#ff7f0e"}

    for ax_idx, m in enumerate(models):
        ax = axes[ax_idx]
        m_df = summary_df[summary_df["model"] == m]
        for f in fractions:
            f_df = m_df[m_df["fraction"] == f].sort_values("depth")
            ax.plot(
                f_df["depth"], f_df["recovery_fraction"] * 100.0,
                marker="o", linewidth=2.5, markersize=7,
                label=f"Budget {f}", color=frac_colors[f]
            )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="No Recovery (Zero)")
        ax.axhline(100.0, color="green", linestyle=":", linewidth=1.2, alpha=0.7, label="Full Recovery (Baseline)")
        ax.set_title(f"{model_display_names[m]}: Gaussian Recovery Fraction", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Block Depth $l$", fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Recovery Fraction (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"Block {d}" for d in depths])
        ax.legend(frameon=True, loc="lower right")

    plt.suptitle("Held-Out Gaussian Recovery Fraction Across Depth and Budget", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gaussian_recovery_by_depth.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 2: damage_by_depth_fraction.png (Focus on 25% primary budget)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=False)
    cond_styles = [
        ("zero_damage", "Zero Ablation", "#d62728", "s--"),
        ("sphere_damage", "Random Sphere", "#8c564b", "x-"),
        ("gaussian_damage", "Held-Out Gaussian", "#2ca02c", "o-"),
        ("same_mean_damage", "Same-Image Mean", "#9467bd", "^-.")
    ]

    for ax_idx, m in enumerate(models):
        ax = axes[ax_idx]
        f25_df = summary_df[(summary_df["model"] == m) & (summary_df["fraction"] == "25%")].sort_values("depth")
        for col_name, lbl, color, style in cond_styles:
            ax.plot(f25_df["depth"], f25_df[col_name], style, label=lbl, color=color, linewidth=2.2, markersize=7)
        ax.axhline(0.0, color="black", linestyle=":", alpha=0.6)
        ax.set_title(f"{model_display_names[m]}: Margin Damage (25% Budget)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Block Depth $l$", fontsize=11, fontweight="bold")
        ax.set_ylabel("True-Class Margin Damage", fontsize=11, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"Block {d}" for d in depths])
        ax.legend(frameon=True, loc="upper right")

    plt.suptitle("Causal Margin Damage Across Depths Under Held-Out Gaussian vs Nulls", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "damage_by_depth_fraction.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 3: accuracy_by_depth_fraction.png (25% primary budget)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=False)
    for ax_idx, m in enumerate(models):
        ax = axes[ax_idx]
        f25_df = summary_df[(summary_df["model"] == m) & (summary_df["fraction"] == "25%")].sort_values("depth")
        clean_acc = f25_df["clean_acc"].iloc[0] * 100.0

        ax.axhline(clean_acc, color="blue", linestyle="--", linewidth=1.5, label=f"Clean Baseline ({clean_acc:.1f}%)")
        ax.plot(f25_df["depth"], f25_df["zero_acc"] * 100.0, "s--", label="Zero Ablation", color="#d62728", linewidth=2.0)
        ax.plot(f25_df["depth"], f25_df["sphere_acc"] * 100.0, "x-", label="Random Sphere", color="#8c564b", linewidth=2.0)
        ax.plot(f25_df["depth"], f25_df["gaussian_acc"] * 100.0, "o-", label="Held-Out Gaussian", color="#2ca02c", linewidth=2.5, markersize=8)
        ax.plot(f25_df["depth"], f25_df["same_mean_acc"] * 100.0, "^-.", label="Same-Image Mean", color="#9467bd", linewidth=2.0)

        ax.set_title(f"{model_display_names[m]}: Top-1 Accuracy (25% Budget)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Block Depth $l$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"Block {d}" for d in depths])
        ax.legend(frameon=True, loc="lower left")

    plt.suptitle("Top-1 Accuracy Trajectory Across Depth Under Held-Out Interventions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_depth_fraction.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 4: gaussian_vs_random_sphere.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=True)
    for ax_idx, m in enumerate(models):
        ax = axes[ax_idx]
        m_df = summary_df[summary_df["model"] == m]
        for f in fractions:
            f_df = m_df[m_df["fraction"] == f].sort_values("depth")
            ax.plot(
                f_df["depth"], f_df["gauss_vs_sphere_diff"],
                marker="o", linewidth=2.5, markersize=7,
                label=f"Budget {f}", color=frac_colors[f]
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"{model_display_names[m]}: Gaussian vs Sphere Margin Advantage", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Block Depth $l$", fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Margin Advantage ($\Delta m$)", fontsize=11, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"Block {d}" for d in depths])
        ax.legend(frameon=True, loc="upper right")

    plt.suptitle("Testing Distribution Dependence: Gaussian Significantly Outperforms Norm-Matched Sphere", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gaussian_vs_random_sphere.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 5: depth_transition_heatmap.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    for ax_idx, m in enumerate(models):
        ax = axes[ax_idx]
        m_df = summary_df[summary_df["model"] == m]
        pivot = m_df.pivot(index="fraction", columns="depth", values="recovery_fraction") * 100.0
        pivot = pivot.loc[fractions] # preserve 10, 25, 50 order
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True, ax=ax, vmin=0.0, vmax=100.0)
        ax.set_title(f"{model_display_names[m]}: Recovery % (Depth x Budget)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Block Depth $l$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Intervention Fraction", fontsize=11, fontweight="bold")
        ax.set_xticklabels([f"Block {d}" for d in depths])

    plt.suptitle("Depth-Dependent Transition of Gaussian Recovery Across Intervention Fractions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "depth_transition_heatmap.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 6: correct_vs_wrong_depth_gaussian.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=False)
    for ax_idx, m in enumerate(models):
        ax = axes[ax_idx]
        f25_df = summary_df[(summary_df["model"] == m) & (summary_df["fraction"] == "25%")].sort_values("depth")
        ax.plot(f25_df["depth"], f25_df["gaussian_damage"], "o-", label="Correct-Depth Gaussian", color="#2ca02c", linewidth=2.5, markersize=8)
        ax.plot(f25_df["depth"], f25_df["wrong_depth_damage"], "x--", label="Wrong-Depth Gaussian", color="#d62728", linewidth=2.2, markersize=7)
        ax.axhline(0.0, color="black", linestyle=":", alpha=0.6)
        ax.set_title(f"{model_display_names[m]}: Depth Specificity of Gaussian Stats", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Block Depth $l$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Margin Damage", fontsize=11, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"Block {d}" for d in depths])
        ax.legend(frameon=True, loc="upper right")

    plt.suptitle("Testing Depth-Specificity: Mismatched Depth Gaussian Fails Across Depths", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correct_vs_wrong_depth_gaussian.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 7: global_mean_vs_gaussian.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=False)
    for ax_idx, m in enumerate(models):
        ax = axes[ax_idx]
        f25_df = summary_df[(summary_df["model"] == m) & (summary_df["fraction"] == "25%")].sort_values("depth")
        ax.plot(f25_df["depth"], f25_df["gaussian_damage"], "o-", label="Held-Out Gaussian", color="#2ca02c", linewidth=2.5, markersize=8)
        ax.plot(f25_df["depth"], f25_df["global_mean_damage"], "d-.", label="Global Calibration Mean", color="#17becf", linewidth=2.2, markersize=7)
        ax.axhline(0.0, color="black", linestyle=":", alpha=0.6)
        ax.set_title(f"{model_display_names[m]}: Stochastic Gaussian vs Global Mean", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Block Depth $l$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Margin Damage", fontsize=11, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"Block {d}" for d in depths])
        ax.legend(frameon=True, loc="upper right")

    plt.suptitle("Role of Stochastic Variability: Held-Out Gaussian vs Global Prototype Vector", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_mean_vs_gaussian.png"))
    plt.close()

    print(f"Generated all 7 figures successfully in {output_dir}/")
