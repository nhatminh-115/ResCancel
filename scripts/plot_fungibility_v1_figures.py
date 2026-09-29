"""
scripts/plot_fungibility_v1_figures.py

Generates the 6 required publication-quality figures for Patch Fungibility V1:
1. depth_generalization.png
2. content_fungibility_across_models.png
3. geometry_controls_across_models.png
4. shared_vs_independent_across_models.png
5. replacement_fraction_generalization.png
6. learned_vs_random_1d.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def set_publication_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "grid.alpha": 0.5,
        "grid.linestyle": "--"
    })


def plot_depth_generalization(outputs_dir: str, figures_dir: str):
    """
    Figure 1: depth_generalization.png
    Compares Clean, Zero, Centroid, and Diagonal Gaussian across depths {5, 7, 8, 9, 10}
    for both ViT-B AugReg and DINOv2 ViT-S/14.
    """
    df_vb = pd.read_csv(os.path.join(outputs_dir, "vitb_depth_results.csv"))
    df_dino = pd.read_csv(os.path.join(outputs_dir, "dinov2_depth_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, df, title in zip(axes, [df_vb, df_dino], ["Model A: ViT-B/16 AugReg (Supervised)", "Model B: DINOv2 ViT-S/14 (Self-Supervised)"]):
        depths = [5, 7, 8, 9, 10]
        clean_row = df[df["condition"] == "CLEAN"].iloc[0]
        clean_acc = clean_row["top1_accuracy"] * 100.0

        # Plot Clean horizontal reference
        ax.axhline(clean_acc, color="#2b2d42", linestyle="--", linewidth=1.8, label=f"Clean ({clean_acc:.1f}%)", zorder=2)

        # Zero
        zero_df = df[df["condition"] == "ZERO"].sort_values("depth")
        ax.plot(zero_df["depth"], zero_df["top1_accuracy"] * 100.0, marker="o", color="#d90429", label="Zero (25% Patches)", linewidth=2.2, zorder=3)

        # Centroid
        cent_df = df[df["condition"] == "CENTROID"].sort_values("depth")
        ax.plot(cent_df["depth"], cent_df["top1_accuracy"] * 100.0, marker="s", color="#3a86ff", label=r"Calibration Centroid $\mu_l$", linewidth=2.2, zorder=4)

        # Diagonal Gaussian (group by depth, compute mean and std across seeds)
        gauss_df = df[df["condition"] == "DIAGONAL_GAUSSIAN"].groupby("depth")["top1_accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            gauss_df["depth"], gauss_df["mean"] * 100.0, yerr=gauss_df["std"] * 100.0,
            marker="^", color="#8338ec", label=r"Diagonal Gaussian $\mathcal{N}(\mu_l, \sigma_l^2)$",
            capsize=4, linewidth=2.0, zorder=5
        )

        ax.set_title(title, pad=12)
        ax.set_xlabel("Transformer Block Depth (1-12)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_xticks(depths)
        ax.legend(frameon=True, loc="lower left" if "DINOv2" in title else "best")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Late-Layer Patch Content Fungibility Across Depth", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "depth_generalization.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_content_fungibility_across_models(outputs_dir: str, figures_dir: str):
    """
    Figure 2: content_fungibility_across_models.png
    Bar chart at standardized Depth 8 comparing Clean, Zero, Centroid, and Gaussian
    for both models, highlighting recovery percentage.
    """
    df_vb = pd.read_csv(os.path.join(outputs_dir, "vitb_depth_results.csv"))
    df_dino = pd.read_csv(os.path.join(outputs_dir, "dinov2_depth_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    models = ["ViT-B AugReg", "DINOv2 ViT-S/14"]
    dfs = [df_vb, df_dino]

    for ax, df, m_name in zip(axes, dfs, models):
        clean_acc = df[df["condition"] == "CLEAN"].iloc[0]["top1_accuracy"] * 100.0
        z_acc = df[(df["depth"] == 8) & (df["condition"] == "ZERO")].iloc[0]["top1_accuracy"] * 100.0
        c_acc = df[(df["depth"] == 8) & (df["condition"] == "CENTROID")].iloc[0]["top1_accuracy"] * 100.0

        g_sub = df[(df["depth"] == 8) & (df["condition"] == "DIAGONAL_GAUSSIAN")]
        g_acc_mean = g_sub["top1_accuracy"].mean() * 100.0
        g_acc_std = g_sub["top1_accuracy"].std() * 100.0

        conditions = ["Clean", "Zero\n(25%)", "Centroid\n(25%)", "Gaussian\n(25%)"]
        accs = [clean_acc, z_acc, c_acc, g_acc_mean]
        errors = [0.0, 0.0, 0.0, g_acc_std]
        colors = ["#2b2d42", "#d90429", "#3a86ff", "#8338ec"]

        bars = ax.bar(conditions, accs, yerr=errors, capsize=5, color=colors, alpha=0.9, edgecolor="black", width=0.55)

        # Annotate values on top of bars
        for bar, val in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10
            )

        # Compute recovery
        dmg_zero = clean_acc - z_acc
        dmg_cent = clean_acc - c_acc
        recovery = (dmg_zero - dmg_cent) / dmg_zero * 100.0 if dmg_zero > 0 else 0.0

        ax.set_title(f"{m_name} (Depth 8)\nCentroid Recovery: {recovery:.1f}% of Zero Damage", pad=10)
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.suptitle("Patch Content Fungibility: Zero vs. Held-Out Replacements (Depth 8, 25% Mask)", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "content_fungibility_across_models.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_geometry_controls_across_models(outputs_dir: str, figures_dir: str):
    """
    Figure 3: geometry_controls_across_models.png
    Geometry controls at Depth 8 (Centroid vs Coordinate-Permuted vs Sign-Flipped)
    across fractions (25%, 50%, 75%) for both models.
    """
    df_vb = pd.read_csv(os.path.join(outputs_dir, "vitb_fraction_results.csv"))
    df_dino = pd.read_csv(os.path.join(outputs_dir, "dinov2_fraction_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    fractions = [25, 50, 75]

    for ax, df, title in zip(axes, [df_vb, df_dino], ["Model A: ViT-B/16 AugReg", "Model B: DINOv2 ViT-S/14"]):
        # Centroid
        c_accs = [df[(df["fraction"] == f / 100.0) & (df["condition"] == "CENTROID")].iloc[0]["top1_accuracy"] * 100.0 for f in fractions]
        ax.plot(fractions, c_accs, marker="s", color="#3a86ff", label=r"True Centroid $\mu_8$", linewidth=2.2)

        # Coordinate permuted (mean +/- std across seeds)
        p_means, p_stds = [], []
        for f in fractions:
            sub = df[(df["fraction"] == f / 100.0) & (df["condition"] == "COORDINATE_PERMUTED_CENTROID")]
            p_means.append(sub["top1_accuracy"].mean() * 100.0)
            p_stds.append(sub["top1_accuracy"].std() * 100.0)
        ax.errorbar(fractions, p_means, yerr=p_stds, marker="d", color="#fb5607", label=r"Coord-Permuted $\pi(\mu_8)$", capsize=4, linewidth=2.0)

        # Sign-flipped
        sf_accs = [df[(df["fraction"] == f / 100.0) & (df["condition"] == "SIGN_FLIPPED_CENTROID")].iloc[0]["top1_accuracy"] * 100.0 for f in fractions]
        ax.plot(fractions, sf_accs, marker="x", color="#ff006e", label=r"Sign-Flipped $-\mu_8$", linewidth=2.0, linestyle=":")

        # Zero for reference
        z_accs = [df[(df["fraction"] == f / 100.0) & (df["condition"] == "ZERO")].iloc[0]["top1_accuracy"] * 100.0 for f in fractions]
        ax.plot(fractions, z_accs, marker="o", color="#d90429", label="Zero Patches", linewidth=1.8, linestyle="--", alpha=0.7)

        ax.set_title(title, pad=12)
        ax.set_xlabel("Replaced Spatial Patch Fraction (%)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_xticks(fractions)
        ax.legend(frameon=True, loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Geometric Constraints: Feature-Space Alignment vs. Negative Controls (Depth 8)", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "geometry_controls_across_models.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_shared_vs_independent_across_models(outputs_dir: str, figures_dir: str):
    """
    Figure 4: shared_vs_independent_across_models.png
    At 100% spatial patch replacement:
    Static Centroid vs Shared Gaussian Noise vs Independent Gaussian Noise vs Isotropic Independent Noise.
    """
    df_vb = pd.read_csv(os.path.join(outputs_dir, "vitb_diversity_results.csv"))
    df_dino = pd.read_csv(os.path.join(outputs_dir, "dinov2_diversity_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, df, title in zip(axes, [df_vb, df_dino], ["Model A: ViT-B/16 AugReg (100% Replacement)", "Model B: DINOv2 ViT-S/14 (100% Replacement)"]):
        sc_acc = df[df["condition"] == "STATIC_CENTROID"].iloc[0]["top1_accuracy"] * 100.0

        sh_sub = df[df["condition"] == "SHARED_GAUSSIAN"]
        sh_mean, sh_std = sh_sub["top1_accuracy"].mean() * 100.0, sh_sub["top1_accuracy"].std() * 100.0

        ind_sub = df[df["condition"] == "INDEPENDENT_GAUSSIAN"]
        ind_mean, ind_std = ind_sub["top1_accuracy"].mean() * 100.0, ind_sub["top1_accuracy"].std() * 100.0

        iso_sub = df[df["condition"] == "ISOTROPIC_INDEPENDENT"]
        iso_mean, iso_std = iso_sub["top1_accuracy"].mean() * 100.0, iso_sub["top1_accuracy"].std() * 100.0

        conds = ["Static\nCentroid", "Shared\nGaussian", "Independent\nGaussian", "Isotropic\nIndependent"]
        means = [sc_acc, sh_mean, ind_mean, iso_mean]
        stds = [0.0, sh_std, ind_std, iso_std]
        colors = ["#6c757d", "#e63946", "#2a9d8f", "#457b9d"]

        bars = ax.bar(conds, means, yerr=stds, capsize=6, color=colors, alpha=0.9, edgecolor="black", width=0.55)

        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10
            )

        div_gain = ind_mean - sh_mean
        ax.set_title(f"{title}\nDiversity Gain (Indep - Shared): +{div_gain:.1f}%", pad=12)
        ax.set_ylabel("Top-1 Accuracy (%)")
        upper_limit = max(max(means) * 1.25, 20.0)
        ax.set_ylim(0, min(upper_limit, 100.0))
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.suptitle("Causal Role of Spatial Diversity under 100% Patch Replacement (Depth 8)", y=0.98)
    plt.subplots_adjust(top=0.88)
    out_path = os.path.join(figures_dir, "shared_vs_independent_across_models.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_replacement_fraction_generalization(outputs_dir: str, figures_dir: str):
    """
    Figure 5: replacement_fraction_generalization.png
    Top-1 Accuracy and Mean Margin vs Replacement Fraction (25%, 50%, 75%, 100%).
    """
    df_vb_frac = pd.read_csv(os.path.join(outputs_dir, "vitb_fraction_results.csv"))
    df_dino_frac = pd.read_csv(os.path.join(outputs_dir, "dinov2_fraction_results.csv"))
    df_vb_div = pd.read_csv(os.path.join(outputs_dir, "vitb_diversity_results.csv"))
    df_dino_div = pd.read_csv(os.path.join(outputs_dir, "dinov2_diversity_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    fractions = [25, 50, 75, 100]

    for ax, df_f, df_d, title in zip(
        axes, [df_vb_frac, df_dino_frac], [df_vb_div, df_dino_div],
        ["Model A: ViT-B/16 AugReg", "Model B: DINOv2 ViT-S/14"]
    ):
        # Centroid
        c_accs = [df_f[(df_f["fraction"] == f / 100.0) & (df_f["condition"] == "CENTROID")].iloc[0]["top1_accuracy"] * 100.0 for f in [25, 50, 75]]
        # At 100%, Static Centroid
        c_100 = df_d[df_d["condition"] == "STATIC_CENTROID"].iloc[0]["top1_accuracy"] * 100.0
        c_accs.append(c_100)
        ax.plot(fractions, c_accs, marker="s", color="#3a86ff", label="Centroid (Static)", linewidth=2.2)

        # Independent Gaussian
        g_accs = []
        for f in [25, 50, 75]:
            sub = df_f[(df_f["fraction"] == f / 100.0) & (df_f["condition"] == "DIAGONAL_GAUSSIAN")]
            g_accs.append(sub["top1_accuracy"].mean() * 100.0)
        ind_100 = df_d[df_d["condition"] == "INDEPENDENT_GAUSSIAN"]["top1_accuracy"].mean() * 100.0
        g_accs.append(ind_100)
        ax.plot(fractions, g_accs, marker="^", color="#2a9d8f", label="Independent Gaussian", linewidth=2.2)

        # Zero (only up to 75% in fraction table; 100% can be plotted if present)
        z_accs = [df_f[(df_f["fraction"] == f / 100.0) & (df_f["condition"] == "ZERO")].iloc[0]["top1_accuracy"] * 100.0 for f in [25, 50, 75]]
        ax.plot([25, 50, 75], z_accs, marker="o", color="#d90429", label="Zero Patches", linewidth=2.0, linestyle="--")

        ax.set_title(title, pad=12)
        ax.set_xlabel("Replaced Spatial Patch Fraction (%)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_xticks(fractions)
        ax.legend(frameon=True, loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Downstream ViT Classification vs. Spatial Patch Replacement Fraction (Depth 8)", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "replacement_fraction_generalization.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_learned_vs_random_1d(outputs_dir: str, figures_dir: str):
    """
    Figure 6: learned_vs_random_1d.png
    Low-dimensional 1D variation at 100% replacement:
    Static Centroid vs Natural PC1 vs Natural PC2 vs Random 1D direction.
    """
    df_vb = pd.read_csv(os.path.join(outputs_dir, "vitb_1d_results.csv"))
    df_dino = pd.read_csv(os.path.join(outputs_dir, "dinov2_1d_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, df, title in zip(axes, [df_vb, df_dino], ["Model A: ViT-B/16 AugReg (100% Replaced)", "Model B: DINOv2 ViT-S/14 (100% Replaced)"]):
        sc_acc = df[df["condition"] == "STATIC_CENTROID"].iloc[0]["top1_accuracy"] * 100.0

        pc1_sub = df[df["condition"] == "NATURAL_PC1"]
        pc1_mean, pc1_std = pc1_sub["top1_accuracy"].mean() * 100.0, pc1_sub["top1_accuracy"].std() * 100.0

        pc2_sub = df[df["condition"] == "NATURAL_PC2"]
        pc2_mean, pc2_std = pc2_sub["top1_accuracy"].mean() * 100.0, pc2_sub["top1_accuracy"].std() * 100.0

        r1d_sub = df[df["condition"] == "RANDOM_1D"]
        r1d_mean, r1d_std = r1d_sub["top1_accuracy"].mean() * 100.0, r1d_sub["top1_accuracy"].std() * 100.0

        conds = ["Static\nCentroid", "Natural\nPC1", "Natural\nPC2", "Random\n1D Direction"]
        means = [sc_acc, pc1_mean, pc2_mean, r1d_mean]
        stds = [0.0, pc1_std, pc2_std, r1d_std]
        colors = ["#6c757d", "#1d3557", "#457b9d", "#e76f51"]

        bars = ax.bar(conds, means, yerr=stds, capsize=6, color=colors, alpha=0.9, edgecolor="black", width=0.55)

        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10
            )

        ax.set_title(title, pad=12)
        ax.set_ylabel("Top-1 Accuracy (%)")
        upper_limit = max(max(means) * 1.3, 20.0)
        ax.set_ylim(0, min(upper_limit, 100.0))
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.suptitle("Learned Low-Dimensional Directions vs. Random 1D Direction (Depth 8, 100% Replacement)", y=0.98)
    plt.subplots_adjust(top=0.88)
    out_path = os.path.join(figures_dir, "learned_vs_random_1d.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_all_v1_figures(outputs_dir: str = "outputs/fungibility_v1", figures_dir: str = "figures/fungibility_v1"):
    os.makedirs(figures_dir, exist_ok=True)
    set_publication_style()
    print(f"[Plotting] Generating publication figures in {figures_dir}...")
    plot_depth_generalization(outputs_dir, figures_dir)
    plot_content_fungibility_across_models(outputs_dir, figures_dir)
    plot_geometry_controls_across_models(outputs_dir, figures_dir)
    plot_shared_vs_independent_across_models(outputs_dir, figures_dir)
    plot_replacement_fraction_generalization(outputs_dir, figures_dir)
    plot_learned_vs_random_1d(outputs_dir, figures_dir)
    print("[Plotting] All 6 figures generated successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=str, default="outputs/fungibility_v1")
    parser.add_argument("--figures-dir", type=str, default="figures/fungibility_v1")
    args = parser.parse_args()
    generate_all_v1_figures(args.outputs_dir, args.figures_dir)
