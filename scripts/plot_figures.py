import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_scientific_style():
    """Sets publication-quality matplotlib / seaborn style."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })


def plot_cancellation_landscape(event_df: pd.DataFrame, model_name: str, output_path: str):
    """
    Plots layer-wise cancellation metrics (cos_sim, contraction q, and C_L1)
    comparing Attention vs MLP across layers.
    """
    set_scientific_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    cls_events = event_df[event_df["token_type"] == "cls"]

    # 1. Cosine similarity vs Layer
    sns.lineplot(
        data=cls_events,
        x="layer",
        y="cos_sim",
        hue="site",
        marker="o",
        ax=axes[0],
        palette={"attn": "#1f77b4", "mlp": "#ff7f0e"}
    )
    axes[0].set_title(r"Cosine Similarity $\cos(x, \delta)$")
    axes[0].set_xlabel("Transformer Layer")
    axes[0].set_ylabel(r"$\cos(x, \delta)$")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.6)

    # 2. Residual Contraction q vs Layer
    sns.lineplot(
        data=cls_events,
        x="layer",
        y="q_contract",
        hue="site",
        marker="s",
        ax=axes[1],
        palette={"attn": "#1f77b4", "mlp": "#ff7f0e"}
    )
    axes[1].set_title(r"Residual Contraction $q = ||x+\delta|| / ||x||$")
    axes[1].set_xlabel("Transformer Layer")
    axes[1].set_ylabel(r"Norm Contraction Ratio $q$")
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.6)

    # 3. Coordinate cancellation C_L1 vs Layer
    sns.lineplot(
        data=cls_events,
        x="layer",
        y="c_l1",
        hue="site",
        marker="^",
        ax=axes[2],
        palette={"attn": "#1f77b4", "mlp": "#ff7f0e"}
    )
    axes[2].set_title(r"Coordinate Cancellation $C_{L1}$")
    axes[2].set_xlabel("Transformer Layer")
    axes[2].set_ylabel(r"$C_{L1}$ Ratio")
    axes[2].axhline(1.0, color="gray", linestyle="--", alpha=0.6)

    plt.suptitle(f"Residual Cancellation Landscape across Layers & Sites ({model_name})", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_geometry_joint_distribution(event_df: pd.DataFrame, model_name: str, output_path: str):
    """
    Plots joint distribution of cos(x, delta) vs contraction q, highlighting
    the frozen extreme cancellation criteria.
    """
    set_scientific_style()
    cls_events = event_df[event_df["token_type"] == "cls"].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(
        cls_events["cos_sim"],
        cls_events["q_contract"],
        c=cls_events["c_l1"],
        cmap="viridis",
        alpha=0.4,
        s=15,
        edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r"Coordinate Cancellation $C_{L1}$")

    # Mark frozen extreme cancellation threshold zone: cos <= -0.60, q <= 0.85
    ax.axvline(-0.60, color="red", linestyle="--", label="Extreme Thresholds")
    ax.axhline(0.85, color="red", linestyle="--")
    ax.fill_between([-1.0, -0.60], 0.0, 0.85, color="red", alpha=0.12, label="Extreme Regime")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, max(2.5, cls_events["q_contract"].quantile(0.99)))
    ax.set_xlabel(r"Cosine Similarity $\cos(x, \delta)$")
    ax.set_ylabel(r"Contraction Ratio $q = ||x+\delta|| / ||x||$")
    ax.set_title(f"Residual Vector Geometry & Candidate Extreme Cancellation ({model_name})")
    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_matched_comparison(matched_df: pd.DataFrame, model_name: str, output_path: str):
    """
    Plots matched control comparison: clean margin and flip rate between
    extreme events and rigorously matched controls.
    """
    set_scientific_style()
    if len(matched_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 1. Clean Logit Margin
    sns.boxplot(
        data=matched_df,
        x="group",
        y="margin",
        hue="group",
        palette={"extreme": "#d62728", "control": "#2ca02c"},
        ax=axes[0],
        width=0.4,
        legend=False
    )
    axes[0].set_title("Classification Margin (Top-1 vs Top-2)")
    axes[0].set_xlabel("Residual Observation Group")
    axes[0].set_ylabel("Clean Logit Margin")

    # 2. Perturbation Flip Rate
    flip_rates = matched_df.groupby("group")["flipped"].mean().reset_index()
    sns.barplot(
        data=flip_rates,
        x="group",
        y="flipped",
        hue="group",
        palette={"extreme": "#d62728", "control": "#2ca02c"},
        ax=axes[1],
        width=0.4,
        legend=False
    )
    axes[1].set_title("Perturbation Flip Rate (Fragility)")
    axes[1].set_xlabel("Residual Observation Group")
    axes[1].set_ylabel("Fraction Flipped under Perturbation")

    plt.suptitle(f"Confound-Controlled Comparison: Extreme Cancellation vs Matched Controls ({model_name})", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_causal_intervention(intervention_df: pd.DataFrame, model_name: str, output_path: str):
    """
    Plots causal intervention effects across alpha in {1.0, 0.75, 0.50, 0.25}
    for weaken_opposing vs random_direction vs uniform_scaling.
    """
    set_scientific_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    modes_palette = {
        "baseline": "black",
        "weaken_opposing": "#d62728",
        "random_direction": "#7f7f7f",
        "uniform_scaling": "#1f77b4"
    }

    # 1. Clean Accuracy
    sns.lineplot(
        data=intervention_df,
        x="alpha",
        y="clean_acc",
        hue="mode",
        style="mode",
        markers=True,
        dashes=False,
        ax=axes[0],
        palette=modes_palette
    )
    axes[0].set_title("Clean Top-1 Accuracy")
    axes[0].set_xlabel(r"Weakening Parameter $\alpha$")
    axes[0].set_ylabel("Top-1 Accuracy")
    axes[0].invert_xaxis()  # 1.0 (untouched) to 0.25 (strong suppression)

    # 2. Margin Shift
    sns.lineplot(
        data=intervention_df,
        x="alpha",
        y="margin_delta",
        hue="mode",
        style="mode",
        markers=True,
        dashes=False,
        ax=axes[1],
        palette=modes_palette
    )
    axes[1].set_title(r"Logit Margin Change ($\Delta$ Margin)")
    axes[1].set_xlabel(r"Weakening Parameter $\alpha$")
    axes[1].set_ylabel(r"Margin Change vs Baseline ($\Delta z$)")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axes[1].invert_xaxis()

    # 3. Flip Rate under Perturbation
    sns.lineplot(
        data=intervention_df,
        x="alpha",
        y="flip_rate",
        hue="mode",
        style="mode",
        markers=True,
        dashes=False,
        ax=axes[2],
        palette=modes_palette
    )
    axes[2].set_title("Perturbation Flip Rate (Fragility)")
    axes[2].set_xlabel(r"Weakening Parameter $\alpha$")
    axes[2].set_ylabel("Flip Rate")
    axes[2].invert_xaxis()

    plt.suptitle(f"Causal Intervention & Matched Controls Sweep ({model_name})", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
