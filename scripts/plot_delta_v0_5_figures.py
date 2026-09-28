import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_clean_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 300


def plot_historical_vs_random_oracle(
    image_df: pd.DataFrame,
    hist_vs_rand: Dict[str, Any],
    seed_df: pd.DataFrame,
    model_name: str,
    output_path: str
):
    """
    Plots the primary falsification test: Historical Oracle vs. Matched Random-Direction Oracle.
    """
    set_clean_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Histogram & Bootstrap CI of paired differences
    diffs = image_df["historical_vs_random_diff"].values
    mean_d = hist_vs_rand["mean_diff"]
    ci = hist_vs_rand["bootstrap_ci_95"]
    p_val = hist_vs_rand["p_value"]
    dz = hist_vs_rand["cohens_dz"]

    sns.histplot(diffs, kde=True, ax=axes[0], color="#1f77b4", bins=30, alpha=0.6)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero Advantage")
    axes[0].axvline(mean_d, color="#d62728", linestyle="-", linewidth=2.0, label=f"Mean Diff: {mean_d:+.4f}")
    axes[0].axvspan(ci[0], ci[1], color="#d62728", alpha=0.15, label=f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    axes[0].set_xlabel("Paired Margin Advantage (Historical - Random Oracle)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Image Count", fontsize=11, fontweight="bold")
    axes[0].set_title(f"[{model_name}] Primary Falsification: Historical vs Random\np = {p_val:.3e} | dz = {dz:.3f}", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=8.5)

    # Right: Per-seed stability breakdown
    rand_seeds = seed_df[seed_df["control_type"] == "random_direction"]
    seed_names = [f"Seed {int(r['seed'])}" for _, r in rand_seeds.iterrows()]
    seed_means = rand_seeds["margin_diff"].values
    seed_cis_lower = rand_seeds["ci_lower"].values
    seed_cis_upper = rand_seeds["ci_upper"].values
    yerr = [seed_means - seed_cis_lower, seed_cis_upper - seed_means]

    axes[1].errorbar(range(len(seed_names)), seed_means, yerr=yerr, fmt="o", color="#2ca02c",
                     ecolor="#2ca02c", elinewidth=2, capsize=5, markersize=7)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(mean_d, color="#d62728", linestyle=":", linewidth=1.5, label="5-Seed Mean")
    axes[1].set_xticks(range(len(seed_names)))
    axes[1].set_xticklabels(seed_names, fontsize=9.5)
    axes[1].set_ylabel("Margin Difference (Historical - Random)", fontsize=11, fontweight="bold")
    axes[1].set_title(f"[{model_name}] Stability Across 5 Frozen Random Seeds", fontsize=11)
    axes[1].legend(loc="lower right", fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_historical_vs_cross_image(
    image_df: pd.DataFrame,
    hist_vs_cross: Dict[str, Any],
    model_name: str,
    output_path: str
):
    """
    Plots Historical vs. Cross-Image Donor Oracle comparison.
    """
    set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    diffs = image_df["historical_vs_cross_diff"].values
    mean_d = hist_vs_cross["mean_diff"]
    ci = hist_vs_cross["bootstrap_ci_95"]
    p_val = hist_vs_cross["p_value"]
    dz = hist_vs_cross["cohens_dz"]

    sns.histplot(diffs, kde=True, ax=ax, color="#ff7f0e", bins=30, alpha=0.6)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero Advantage")
    ax.axvline(mean_d, color="#d62728", linestyle="-", linewidth=2.0, label=f"Mean Diff: {mean_d:+.4f}")
    ax.axvspan(ci[0], ci[1], color="#d62728", alpha=0.15, label=f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    ax.set_xlabel("Paired Margin Advantage (Historical - Cross-Image)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Image Count", fontsize=11, fontweight="bold")
    ax.set_title(f"[{model_name}] Sample Specificity: Historical vs Cross-Image\np = {p_val:.3e} | dz = {dz:.3f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_historical_vs_spatial_shuffle(
    image_df: pd.DataFrame,
    hist_vs_shuff: Dict[str, Any],
    seed_df: pd.DataFrame,
    model_name: str,
    output_path: str
):
    """
    Plots Historical vs. Spatially Shuffled Historical Oracle comparison.
    """
    set_clean_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    diffs = image_df["historical_vs_shuffled_diff"].values
    mean_d = hist_vs_shuff["mean_diff"]
    ci = hist_vs_shuff["bootstrap_ci_95"]
    p_val = hist_vs_shuff["p_value"]
    dz = hist_vs_shuff["cohens_dz"]

    sns.histplot(diffs, kde=True, ax=ax, color="#9467bd", bins=30, alpha=0.6)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero Advantage")
    ax.axvline(mean_d, color="#d62728", linestyle="-", linewidth=2.0, label=f"Mean Diff: {mean_d:+.4f}")
    ax.axvspan(ci[0], ci[1], color="#d62728", alpha=0.15, label=f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    ax.set_xlabel("Paired Margin Advantage (Historical - Spatially Shuffled)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Image Count", fontsize=11, fontweight="bold")
    ax.set_title(f"[{model_name}] Spatial Specificity: Historical vs Spatial Shuffle\np = {p_val:.3e} | dz = {dz:.3f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_oracle_capacity_comparison(
    image_df: pd.DataFrame,
    model_name: str,
    output_path: str
):
    """
    Multi-condition boxplot comparing all matched-capacity conditions against Baseline, Global, and Historical.
    """
    set_clean_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    cols = [
        "base_margin",
        "global_oracle_margin",
        "random_oracle_mean_margin",
        "cross_image_margin",
        "spatial_shuffled_mean_margin",
        "pooled_delta_margin",
        "historical_margin"
    ]
    labels = [
        "Baseline\n(No Oracle)",
        "Global\nOracle (V0)",
        "Random\nOracle (5s)",
        "Cross-Image\nOracle",
        "Spatial-Shuff\nOracle (3s)",
        "Pooled-Delta\nOracle",
        "True Historical\nOracle (V0)"
    ]
    colors = ["#7f7f7f", "#8c564b", "#d62728", "#ff7f0e", "#9467bd", "#e377c2", "#2ca02c"]

    data_to_plot = [image_df[col].values for col in cols]
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Plot means as diamonds
    means = [np.mean(vals) for vals in data_to_plot]
    ax.plot(range(1, len(cols) + 1), means, "d", color="black", markersize=6, label="Mean Margin")

    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("True-Class Logit Margin", fontsize=11, fontweight="bold")
    ax.set_title(f"[{model_name}] Matched-Oracle Capacity Spectrum (N = 1,000 Images)", fontsize=12, pad=12)
    ax.legend(loc="upper left", fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
