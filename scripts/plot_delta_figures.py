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


def plot_source_layer_histogram(
    source_df: pd.DataFrame,
    model_name: str,
    output_path: str
):
    """
    Plots the aggregate distribution of historical source layers chosen by the Tokenwise Oracle.
    """
    set_clean_style()
    layer_cols = [f"count_layer_{l}" for l in range(8)]
    total_counts = source_df[layer_cols].sum().values
    total_tokens = np.sum(total_counts)
    proportions = total_counts / total_tokens

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(8), proportions, color="#2b5c8f", width=0.65, edgecolor="black", alpha=0.85)

    for bar, prop in zip(bars, proportions):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f"{prop:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(8))
    ax.set_xticklabels([f"Block {l}" for l in range(8)], fontsize=10)
    ax.set_xlabel("Historical Source Block", fontsize=11, fontweight="bold")
    ax.set_ylabel("Selection Fraction Across All Patches", fontsize=11, fontweight="bold")
    ax.set_title(f"[{model_name}] Tokenwise Oracle Historical Source Distribution\n(Total Evaluated Tokens = {int(total_tokens):,})", fontsize=11, pad=12)
    ax.set_ylim(0, max(proportions) * 1.18)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_tokenwise_source_maps(
    diversity_records: List[Dict[str, Any]],
    model_name: str,
    output_path: str,
    num_samples: int = 4
):
    """
    Visualizes 14x14 spatial patch maps of historical source layer choices for representative images.
    """
    set_clean_style()
    fig, axes = plt.subplots(1, num_samples, figsize=(3.8 * num_samples, 3.8))
    if num_samples == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("tab10", 8)

    for idx in range(num_samples):
        rec = diversity_records[idx]
        grid = np.array(rec["spatial_map_14x14"])
        global_l = rec["global_selected_layer"]
        entropy = rec["entropy"]

        im = axes[idx].imshow(grid, cmap=cmap, vmin=-0.5, vmax=7.5, interpolation="nearest")
        axes[idx].set_title(f"Image {idx+1}\nGlobal: Block {global_l} | H: {entropy:.2f}b", fontsize=10)
        axes[idx].axis("off")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.046, pad=0.15, ticks=range(8))
    cbar.ax.set_xticklabels([f"Blk {l}" for l in range(8)], fontsize=9)
    cbar.set_label("Historical Source Block Index", fontsize=10, fontweight="bold")

    fig.suptitle(f"[{model_name}] Spatial Source Selection Maps (14x14 Patches)", fontsize=12, fontweight="bold", y=0.98)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_oracle_margin_comparison(
    image_df: pd.DataFrame,
    token_vs_global: Dict[str, Any],
    model_name: str,
    output_path: str
):
    """
    Plots policy margin comparisons and the paired difference distribution (Tokenwise - Global).
    """
    set_clean_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: Policy Boxplot / Violinplot
    policies = ["base_margin", "most_recent_margin", "rand_mean_margin", "global_margin", "tokenwise_margin"]
    labels = ["Baseline", "Most-Recent\n(Block 7)", "Random\n(Mean 3 Seeds)", "Global\nOracle", "Tokenwise\nOracle"]
    colors = ["#7f7f7f", "#8c564b", "#9467bd", "#1f77b4", "#2ca02c"]

    data_to_plot = [image_df[col].values for col in policies]
    bp = axes[0].boxplot(data_to_plot, patch_artist=True, notch=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("True-Class Logit Margin", fontsize=11, fontweight="bold")
    axes[0].set_title(f"[{model_name}] Policy Margin Comparison (N=1,000)", fontsize=11)

    # Right: Paired Difference Histogram (Tokenwise - Global)
    diffs = image_df["tokenwise_vs_global_margin_diff"].values
    mean_d = token_vs_global["mean_diff"]
    ci = token_vs_global["bootstrap_ci_95"]
    p_val = token_vs_global["p_value"]
    dz = token_vs_global["cohens_dz"]

    sns.histplot(diffs, kde=True, ax=axes[1], color="#2ca02c", bins=30, alpha=0.6)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero Advantage")
    axes[1].axvline(mean_d, color="#d62728", linestyle="-", linewidth=2.0, label=f"Mean Diff: {mean_d:+.4f}")
    axes[1].axvspan(ci[0], ci[1], color="#d62728", alpha=0.15, label=f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    axes[1].set_xlabel("Paired Margin Advantage (Tokenwise - Global)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Image Count", fontsize=11, fontweight="bold")
    axes[1].set_title(f"[{model_name}] Tokenwise vs Global Effect\np = {p_val:.3e} | dz = {dz:.3f}", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=8.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
