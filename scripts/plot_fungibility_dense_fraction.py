"""
scripts/plot_fungibility_dense_fraction.py

Publication-quality figures for Patch Fungibility Dense Fraction & Spatial-Mask Robustness Sweep:
1. dense_fraction_accuracy.png: 4-panel accuracy vs actual fraction with mask SD bands
2. dense_fraction_margin.png: 4-panel true-class margin vs actual fraction with mask SD bands
3. dense_fraction_recovery.png: Centroid & Gaussian recovery vs fraction (masked where Zero damage < 0.10)
4. mask_seed_robustness.png: 5 individual mask seeds + bold mean for Centroid across 4 models
5. threshold_summary.png: F95, F90, F80 retention thresholds across models and conditions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


COLOR_CLEAN = "#7f8c8d"
COLOR_ZERO = "#c0392b"       # Crimson / Red
COLOR_CENTROID = "#2980b9"   # Blue
COLOR_GAUSSIAN = "#27ae60"   # Emerald Green
COLOR_SEEDS = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]

PRIMARY_MODELS = [
    ("deit_tiny", 8, "DeiT-Tiny (Depth 8)"),
    ("deit_small", 8, "DeiT-Small (Depth 8)"),
    ("vit_base", 7, "ViT-B AugReg (Depth 7 - Primary)"),
    ("dinov2", 9, "DINOv2 ViT-S/14 (Depth 9 - Primary)")
]


def plot_dense_fraction_accuracy(df_across: pd.DataFrame, figures_dir: str):
    """
    Figure 1: 4-panel Top-1 Accuracy (%) vs Actual Fraction Replaced (%).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, (m_key, depth, title) in zip(axes, PRIMARY_MODELS):
        sub_m = df_across[(df_across["model"] == m_key) & (df_across["depth"] == depth)]

        # Clean baseline
        clean_row = sub_m[sub_m["actual_fraction"] == 0.0]
        clean_acc = clean_row["mean_acc"].values[0] * 100.0 if len(clean_row) > 0 else 70.0
        ax.axhline(clean_acc, color=COLOR_CLEAN, linestyle="--", linewidth=1.5, label=f"Clean ({clean_acc:.1f}%)")

        for cond, color, label in [
            ("ZERO", COLOR_ZERO, "Zero Replacement"),
            ("CENTROID", COLOR_CENTROID, r"Centroid $\mu_l$"),
            ("DIAGONAL_GAUSSIAN", COLOR_GAUSSIAN, r"Diagonal Gaussian $\mathcal{N}(\mu_l, \sigma_l^2)$")
        ]:
            sub_c = sub_m[sub_m["condition"] == cond].sort_values(by="actual_fraction")
            if len(sub_c) == 0:
                continue
            x = sub_c["actual_percent"].values
            y = sub_c["mean_acc"].values * 100.0
            sd = sub_c["sd_acc"].values * 100.0

            ax.plot(x, y, color=color, linewidth=2.0, label=label)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.18)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(-1, 101)
        ax.set_ylim(-2, 102)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    axes[2].set_xlabel("Actual Spatial Patches Replaced (%)", fontsize=11)
    axes[3].set_xlabel("Actual Spatial Patches Replaced (%)", fontsize=11)

    # Common legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=True, fontsize=11)

    plt.suptitle("Dense Fraction Dose-Response: Patch Content Fungibility Across Spatial Masks",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "dense_fraction_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def plot_dense_fraction_margin(df_across: pd.DataFrame, figures_dir: str):
    """
    Figure 2: 4-panel True-Class Margin vs Actual Fraction Replaced (%).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, (m_key, depth, title) in zip(axes, PRIMARY_MODELS):
        sub_m = df_across[(df_across["model"] == m_key) & (df_across["depth"] == depth)]

        # Clean baseline margin
        clean_row = sub_m[sub_m["actual_fraction"] == 0.0]
        clean_margin = clean_row["mean_margin"].values[0] if len(clean_row) > 0 else 0.0
        ax.axhline(clean_margin, color=COLOR_CLEAN, linestyle="--", linewidth=1.5, label=f"Clean ({clean_margin:.2f})")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)

        for cond, color, label in [
            ("ZERO", COLOR_ZERO, "Zero Replacement"),
            ("CENTROID", COLOR_CENTROID, r"Centroid $\mu_l$"),
            ("DIAGONAL_GAUSSIAN", COLOR_GAUSSIAN, r"Diagonal Gaussian $\mathcal{N}(\mu_l, \sigma_l^2)$")
        ]:
            sub_c = sub_m[sub_m["condition"] == cond].sort_values(by="actual_fraction")
            if len(sub_c) == 0:
                continue
            x = sub_c["actual_percent"].values
            y = sub_c["mean_margin"].values
            sd = sub_c["sd_margin"].values

            ax.plot(x, y, color=color, linewidth=2.0, label=label)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.18)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(-1, 101)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylabel("Mean True-Class Margin", fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    axes[2].set_xlabel("Actual Spatial Patches Replaced (%)", fontsize=11)
    axes[3].set_xlabel("Actual Spatial Patches Replaced (%)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=True, fontsize=11)

    plt.suptitle("Dense Fraction Dose-Response: Margin Dynamics vs Replacement Fraction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "dense_fraction_margin.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def plot_dense_fraction_recovery(df_across: pd.DataFrame, figures_dir: str):
    """
    Figure 3: 4-panel Recovery(f) vs Actual Fraction Replaced (%).
    Shows Centroid and Gaussian recovery, with N/A where Zero damage < 0.10.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, (m_key, depth, title) in zip(axes, PRIMARY_MODELS):
        sub_m = df_across[(df_across["model"] == m_key) & (df_across["depth"] == depth)]

        ax.axhline(1.0, color=COLOR_CLEAN, linestyle="--", linewidth=1.2, label="100% Recovery (Clean)")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="50% Recovery Threshold")
        ax.axhline(0.0, color=COLOR_ZERO, linestyle="-.", linewidth=1.0, alpha=0.7, label="0% Recovery (Zero Baseline)")

        for cond, color, label in [
            ("CENTROID", COLOR_CENTROID, r"Centroid $\mu_l$ Recovery"),
            ("DIAGONAL_GAUSSIAN", COLOR_GAUSSIAN, r"Gaussian $\mathcal{N}(\mu_l, \sigma_l^2)$ Recovery")
        ]:
            sub_c = sub_m[sub_m["condition"] == cond].sort_values(by="actual_fraction")
            if len(sub_c) == 0:
                continue
            x = sub_c["actual_percent"].values
            y = sub_c["recovery"].values

            # Mask out NaNs cleanly
            valid = ~np.isnan(y)
            ax.plot(x[valid], y[valid], color=color, linewidth=2.0, marker="o", markersize=3, label=label)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(-1, 101)
        ax.set_ylim(-0.2, 1.2)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylabel("Damage Recovery Fraction", fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    axes[2].set_xlabel("Actual Spatial Patches Replaced (%)", fontsize=11)
    axes[3].set_xlabel("Actual Spatial Patches Replaced (%)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, frameon=True, fontsize=10)

    plt.suptitle("Fungibility Recovery Across Replacement Fraction (Undefined where Zero Damage < 0.10)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "dense_fraction_recovery.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def plot_mask_seed_robustness(df_seed: pd.DataFrame, df_across: pd.DataFrame, figures_dir: str):
    """
    Figure 4: 4-panel Centroid Accuracy vs Fraction showing 5 individual mask seeds + bold mean.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    mask_seeds = sorted(df_seed["mask_seed"].unique())

    for ax, (m_key, depth, title) in zip(axes, PRIMARY_MODELS):
        # Individual seeds for Centroid
        for s_idx, s in enumerate(mask_seeds):
            sub_s = df_seed[
                (df_seed["model"] == m_key) &
                (df_seed["depth"] == depth) &
                (df_seed["condition"] == "CENTROID") &
                (df_seed["mask_seed"] == s)
            ].sort_values(by="actual_fraction")

            if len(sub_s) > 0:
                ax.plot(
                    sub_s["actual_percent"].values,
                    sub_s["top1_acc"].values * 100.0,
                    color=COLOR_SEEDS[s_idx % len(COLOR_SEEDS)],
                    linewidth=1.2,
                    alpha=0.6,
                    label=f"Mask Seed {s}"
                )

        # Bold mean curve across mask seeds
        sub_mean = df_across[
            (df_across["model"] == m_key) &
            (df_across["depth"] == depth) &
            (df_across["condition"] == "CENTROID")
        ].sort_values(by="actual_fraction")

        if len(sub_mean) > 0:
            ax.plot(
                sub_mean["actual_percent"].values,
                sub_mean["mean_acc"].values * 100.0,
                color="black",
                linewidth=2.8,
                label=r"Mean Centroid $\mu_l$"
            )

        # Zero mean curve for comparison
        sub_zero = df_across[
            (df_across["model"] == m_key) &
            (df_across["depth"] == depth) &
            (df_across["condition"] == "ZERO")
        ].sort_values(by="actual_fraction")

        if len(sub_zero) > 0:
            ax.plot(
                sub_zero["actual_percent"].values,
                sub_zero["mean_acc"].values * 100.0,
                color=COLOR_ZERO,
                linestyle="--",
                linewidth=1.8,
                label="Mean Zero Baseline"
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(-1, 101)
        ax.set_ylim(-2, 102)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    axes[2].set_xlabel("Actual Spatial Patches Replaced (%)", fontsize=11)
    axes[3].set_xlabel("Actual Spatial Patches Replaced (%)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=7, frameon=True, fontsize=9.5)

    plt.suptitle("Mask-Seed Robustness: 5 Independent Spatial Permutations vs Mean Trajectory",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "mask_seed_robustness.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def plot_threshold_summary(df_thresh: pd.DataFrame, figures_dir: str):
    """
    Figure 5: F95, F90, F80 Thresholds across models and conditions with SD error bars.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    metrics = [("F95", "95% Retention Threshold (F95)"),
               ("F90", "90% Retention Threshold (F90)"),
               ("F80", "80% Retention Threshold (F80)")]

    cond_order = ["ZERO", "CENTROID", "DIAGONAL_GAUSSIAN"]
    cond_labels = ["Zero", r"Centroid $\mu_l$", r"Gaussian $\mathcal{N}$"]
    colors = [COLOR_ZERO, COLOR_CENTROID, COLOR_GAUSSIAN]

    model_keys = ["deit_tiny", "deit_small", "vit_base", "dinov2"]
    model_labels = ["DeiT-Tiny\n(D8)", "DeiT-Small\n(D8)", "ViT-B\n(D7)", "DINOv2\n(D9)"]

    df_mean = df_thresh[df_thresh["mask_seed"] == "MEAN"]

    for ax_idx, (col, title) in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(len(model_keys))
        width = 0.25

        for c_idx, (cond, clabel, color) in enumerate(zip(cond_order, cond_labels, colors)):
            sub_c = df_mean[
                (df_mean["condition"] == cond) &
                (df_mean["is_primary"] == True)
            ]

            vals = []
            errs = []
            for mk in model_keys:
                row = sub_c[sub_c["model"] == mk]
                if len(row) > 0:
                    vals.append(row[col].values[0] * 100.0)
                    errs.append(row[f"{col}_sd"].values[0] * 100.0)
                else:
                    vals.append(0.0)
                    errs.append(0.0)

            ax.bar(
                x + (c_idx - 1) * width,
                vals,
                width=width,
                yerr=errs,
                capsize=4,
                color=color,
                label=clabel if ax_idx == 0 else "",
                alpha=0.88,
                edgecolor="black",
                linewidth=0.8
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        if ax_idx == 0:
            ax.set_ylabel("Max Fraction Retaining Accuracy (%)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True, fontsize=11)

    plt.suptitle("Accuracy Retention Thresholds (F95 / F90 / F80) Across Models",
                 fontsize=14, fontweight="bold", y=1.09)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "threshold_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def generate_all_dense_fraction_figures(output_dir: str, figures_dir: str):
    """
    Master plotting dispatcher.
    """
    print(f"[Plotting] Loading data from {output_dir}...")
    df_seed = pd.read_csv(os.path.join(output_dir, "summary_by_mask_seed.csv"))
    df_across = pd.read_csv(os.path.join(output_dir, "summary_across_masks.csv"))
    df_thresh = pd.read_csv(os.path.join(output_dir, "threshold_crossings.csv"))

    print("[Plotting] Generating Figure 1: dense_fraction_accuracy.png...")
    plot_dense_fraction_accuracy(df_across, figures_dir)

    print("[Plotting] Generating Figure 2: dense_fraction_margin.png...")
    plot_dense_fraction_margin(df_across, figures_dir)

    print("[Plotting] Generating Figure 3: dense_fraction_recovery.png...")
    plot_dense_fraction_recovery(df_across, figures_dir)

    print("[Plotting] Generating Figure 4: mask_seed_robustness.png...")
    plot_mask_seed_robustness(df_seed, df_across, figures_dir)

    print("[Plotting] Generating Figure 5: threshold_summary.png...")
    plot_threshold_summary(df_thresh, figures_dir)
    print("[Plotting] All 5 figures generated successfully.")


if __name__ == "__main__":
    generate_all_dense_fraction_figures(
        output_dir="outputs/fungibility_dense_fraction",
        figures_dir="figures/fungibility_dense_fraction"
    )
