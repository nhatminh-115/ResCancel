import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

OUTPUT_DIR = "outputs/fungibility_v0_7"
FIG_DIR = "figures/fungibility_v0_7"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_prototype_performance_by_fraction():
    """Figure 1: Accuracy across fractions (25%, 50%, 75%, 100%) for key conditions."""
    tiny_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_fraction_summary.csv"))
    small_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "small_fraction_summary.csv"))
    tiny_comp = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_prototype_comparison.csv"))
    small_comp = pd.read_csv(os.path.join(OUTPUT_DIR, "small_prototype_comparison.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    models = [("DeiT-Tiny", tiny_frac, tiny_comp, axes[0]), ("DeiT-Small", small_frac, small_comp, axes[1])]

    fractions = ["25%", "50%", "75%", "100%"]
    x = np.arange(len(fractions))

    for m_name, frac_df, comp_df, ax in models:
        clean_acc = frac_df["clean_acc"].values * 100
        zero_acc = frac_df["zero_acc"].values * 100
        mu8_acc = frac_df["mu8_acc"].values * 100
        gauss_acc = frac_df["gaussian_acc"].values * 100

        # Extract coord perm avg acc
        cp_acc = comp_df[comp_df["control"] == "coord_perm_avg"]["acc_control"].values * 100
        # Extract sign flip 100% acc
        sf_acc = comp_df[comp_df["control"] == "sign_flip_100%"]["acc_control"].values * 100

        ax.axhline(clean_acc[0], color="black", linestyle="--", linewidth=1.5, label="Clean Baseline")
        ax.plot(x, mu8_acc, marker="o", linewidth=2.5, color="#1b9e77", label=r"Block-8 Mean $\mu_8$")
        ax.plot(x, gauss_acc, marker="s", linewidth=2.0, color="#7570b3", label="Held-Out Gaussian")
        ax.plot(x, cp_acc, marker="^", linewidth=1.8, color="#e7298a", label="Coordinate Permuted")
        ax.plot(x, sf_acc, marker="v", linewidth=1.8, color="#d95f02", label=r"Sign Inverted $-\mu_8$")
        ax.plot(x, zero_acc, marker="x", linewidth=2.0, color="#e41a1c", linestyle=":", label="Zero Ablation")

        ax.set_xticks(x)
        ax.set_xticklabels(["25%\n(49 tokens)", "50%\n(98 tokens)", "75%\n(147 tokens)", "100%\n(196 tokens)"])
        ax.set_xlabel("Replaced Spatial Patch Tokens")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name} (Depth 8 Intervention)")
        ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower left")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prototype_performance_by_fraction.png"))
    plt.close(fig)
    print("Saved: prototype_performance_by_fraction.png")


def plot_wrong_depth_mean_comparison():
    """Figure 2: Wrong-depth means (raw vs norm-matched) vs mu_8."""
    tiny_comp = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_prototype_comparison.csv"))
    small_comp = pd.read_csv(os.path.join(OUTPUT_DIR, "small_prototype_comparison.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    models = [("DeiT-Tiny", tiny_comp, axes[0]), ("DeiT-Small", small_comp, axes[1])]

    depths = [5, 6, 7, 8, 9, 10]
    x = np.arange(len(depths))

    for m_name, comp_df, ax in models:
        # At 50% fraction
        df_50 = comp_df[comp_df["fraction"] == "50%"]

        raw_adv = []
        norm_adv = []
        for d in depths:
            if d == 8:
                raw_adv.append(0.0)
                norm_adv.append(0.0)
            else:
                row_raw = df_50[df_50["control"] == f"wrong_depth_{d}"]
                row_norm = df_50[df_50["control"] == f"wrong_depth_norm_{d}"]
                raw_adv.append(row_raw["prototype_advantage"].values[0] if len(row_raw) > 0 else 0.0)
                norm_adv.append(row_norm["prototype_advantage"].values[0] if len(row_norm) > 0 else 0.0)

        width = 0.35
        ax.bar(x - width/2, raw_adv, width=width, label=r"Raw Mean $\mu_k$", color="#386cb0", alpha=0.85)
        ax.bar(x + width/2, norm_adv, width=width, label=r"Norm-Matched Mean $\mu'_k$", color="#f0027f", alpha=0.85)

        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Depth {d}" if d != 8 else "Depth 8\n(Reference)" for d in depths])
        ax.set_xlabel("Source Depth of Prototype Vector")
        ax.set_ylabel(r"Prototype Advantage: $\Delta$ Margin vs $\mu_8$")
        ax.set_title(f"{m_name} (50% Patch Replacement)")
        ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "wrong_depth_mean_comparison.png"))
    plt.close(fig)
    print("Saved: wrong_depth_mean_comparison.png")


def plot_prototype_scale_sweep():
    """Figure 3: Accuracy vs Scale factor of mu_8."""
    tiny_scale = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_scale_sweep.csv"))
    small_scale = pd.read_csv(os.path.join(OUTPUT_DIR, "small_scale_sweep.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    models = [("DeiT-Tiny", tiny_scale, axes[0]), ("DeiT-Small", small_scale, axes[1])]

    scales = [0.25, 0.50, 1.00, 2.00, 4.00]
    colors = {"25%": "#4daf4a", "50%": "#377eb8", "75%": "#ff7f00", "100%": "#e41a1c"}

    for m_name, scale_df, ax in models:
        for frac, col in colors.items():
            sub = scale_df[scale_df["fraction"] == frac].sort_values("scale")
            ax.plot(sub["scale"], sub["accuracy"] * 100, marker="o", linewidth=2.0, color=col, label=f"{frac} Replaced")

        ax.set_xscale("log", base=2)
        ax.set_xticks(scales)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel(r"Scale Factor $s$ applied to $\mu_8$ ($s \cdot \mu_8$)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name} Scale Sweep")
        ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prototype_scale_sweep.png"))
    plt.close(fig)
    print("Saved: prototype_scale_sweep.png")


def plot_prototype_cosine_sweep():
    """Figure 4: Accuracy vs Cosine alignment to mu_8."""
    tiny_cos = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_cosine_sweep.csv"))
    small_cos = pd.read_csv(os.path.join(OUTPUT_DIR, "small_cosine_sweep.csv"))
    tiny_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_fraction_summary.csv"))
    small_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "small_fraction_summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    models = [("DeiT-Tiny", tiny_cos, tiny_frac, axes[0]), ("DeiT-Small", small_cos, small_frac, axes[1])]
    colors = {"25%": "#4daf4a", "50%": "#377eb8", "75%": "#ff7f00", "100%": "#e41a1c"}

    for m_name, cos_df, frac_df, ax in models:
        for frac, col in colors.items():
            sub = cos_df[cos_df["fraction"] == frac].sort_values("target_cosine")
            mu8_acc = frac_df[frac_df["fraction"] == frac]["mu8_acc"].values[0] * 100
            cos_vals = sub["target_cosine"].tolist() + [1.0]
            acc_vals = (sub["accuracy"] * 100).tolist() + [mu8_acc]

            ax.plot(cos_vals, acc_vals, marker="s", linewidth=2.0, color=col, label=f"{frac} Replaced")

        ax.set_xlabel(r"Cosine Similarity to $\mu_8$ ($\cos(r_\alpha, \mu_8)$)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name} Cosine Alignment Sweep")
        ax.set_xlim(-0.05, 1.05)
        ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prototype_cosine_sweep.png"))
    plt.close(fig)
    print("Saved: prototype_cosine_sweep.png")


def plot_prototype_sign_flip_sweep():
    """Figure 5: Dose-response of partial sign flips."""
    tiny_sf = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_sign_flip_sweep.csv"))
    small_sf = pd.read_csv(os.path.join(OUTPUT_DIR, "small_sign_flip_sweep.csv"))
    tiny_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_fraction_summary.csv"))
    small_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "small_fraction_summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    models = [("DeiT-Tiny", tiny_sf, tiny_frac, axes[0]), ("DeiT-Small", small_sf, small_frac, axes[1])]
    colors = {"25%": "#4daf4a", "50%": "#377eb8", "75%": "#ff7f00", "100%": "#e41a1c"}

    flip_order = ["sign_flip_25%", "sign_flip_50%", "sign_flip_75%", "sign_flip_100%"]
    x_pct = [0, 25, 50, 75, 100]

    for m_name, sf_df, frac_df, ax in models:
        for frac, col in colors.items():
            sub = sf_df[sf_df["fraction"] == frac].set_index("flip_condition")
            mu8_acc = frac_df[frac_df["fraction"] == frac]["mu8_acc"].values[0] * 100
            accs = [mu8_acc] + [sub.loc[c, "accuracy"] * 100 for c in flip_order]

            ax.plot(x_pct, accs, marker="^", linewidth=2.0, color=col, label=f"{frac} Replaced")

        ax.set_xlabel("Percentage of Feature Coordinates Inverted (Sign-Flipped)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name} Coordinate Sign Inversion")
        ax.set_xticks(x_pct)
        ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prototype_sign_flip_sweep.png"))
    plt.close(fig)
    print("Saved: prototype_sign_flip_sweep.png")


def plot_full_patch_replacement():
    """Figure 6: Complete 100% Patch Replacement Test."""
    tiny_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_fraction_summary.csv"))
    small_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "small_fraction_summary.csv"))
    tiny_comp = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_prototype_comparison.csv"))
    small_comp = pd.read_csv(os.path.join(OUTPUT_DIR, "small_prototype_comparison.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    models = [("DeiT-Tiny", tiny_frac, tiny_comp, axes[0]), ("DeiT-Small", small_frac, small_comp, axes[1])]

    cond_order = [
        ("Clean Baseline", "clean"),
        (r"Block-8 Mean $\mu_8$", "mu8"),
        ("Held-Out Gaussian", "gaussian"),
        (r"Best Wrong Mean ($\mu_7$)", "wrong_7"),
        ("Coordinate Permuted", "coord_perm"),
        (r"Sign Inverted $-\mu_8$", "sign_flip"),
        (r"Orthogonal Cos=0", "cos_0"),
        ("Zero Ablation", "zero")
    ]

    for m_name, frac_df, comp_df, ax in models:
        r100 = frac_df[frac_df["fraction"] == "100%"].iloc[0]
        c100 = comp_df[comp_df["fraction"] == "100%"].set_index("control")

        clean_acc = r100["clean_acc"] * 100
        mu8_acc = r100["mu8_acc"] * 100
        gauss_acc = r100["gaussian_acc"] * 100
        zero_acc = r100["zero_acc"] * 100

        cp_acc = c100.loc["coord_perm_avg", "acc_control"] * 100
        sf_acc = c100.loc["sign_flip_100%", "acc_control"] * 100
        cos0_acc = c100.loc["cosine_0.00_avg", "acc_control"] * 100
        w7_acc = c100.loc["wrong_depth_7", "acc_control"] * 100 if "wrong_depth_7" in c100.index else 0.0

        accs = [clean_acc, mu8_acc, gauss_acc, w7_acc, cp_acc, sf_acc, cos0_acc, zero_acc]
        colors = ["#2b83ba", "#1b9e77", "#7570b3", "#abdda4", "#fdae61", "#d7191c", "#e7298a", "#404040"]

        bars = ax.bar(np.arange(len(accs)), accs, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(np.arange(len(accs)))
        ax.set_xticklabels([c[0] for c in cond_order], rotation=35, ha="right")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name} — 100% Patch Replacement (CLS Untouched)")
        ax.set_ylim(0, 88)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "full_patch_replacement.png"))
    plt.close(fig)
    print("Saved: full_patch_replacement.png")


def plot_cls_vs_patch_replacement():
    """Figure 7 (Optional): Stream Asymmetry Test (100% Patch Replacement vs CLS Replacement)."""
    tiny_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_fraction_summary.csv"))
    small_frac = pd.read_csv(os.path.join(OUTPUT_DIR, "small_fraction_summary.csv"))
    tiny_cls = pd.read_csv(os.path.join(OUTPUT_DIR, "tiny_cls_comparison.csv")).set_index("condition")
    small_cls = pd.read_csv(os.path.join(OUTPUT_DIR, "small_cls_comparison.csv")).set_index("condition")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    models = [("DeiT-Tiny", tiny_frac, tiny_cls, axes[0]), ("DeiT-Small", small_frac, small_cls, axes[1])]

    labels = [
        "Clean Baseline",
        r"100% Patches $\to \mu_8$" + "\n(CLS untouched)",
        r"CLS $\to \mu_{\text{cls}}$" + "\n(Patches untouched)",
        r"CLS $\to$ Cross-Image" + "\n(Patches untouched)",
        r"CLS $\to$ Zero" + "\n(Patches untouched)"
    ]

    for m_name, frac_df, cls_df, ax in models:
        r100 = frac_df[frac_df["fraction"] == "100%"].iloc[0]
        clean_acc = r100["clean_acc"] * 100
        mu8_acc = r100["mu8_acc"] * 100

        cls_mean_acc = cls_df.loc["cls_calib_mean", "accuracy"] * 100
        cls_cross_acc = cls_df.loc["cls_cross_image", "accuracy"] * 100
        cls_zero_acc = cls_df.loc["cls_zero", "accuracy"] * 100

        accs = [clean_acc, mu8_acc, cls_mean_acc, cls_cross_acc, cls_zero_acc]
        colors = ["#2b83ba", "#1b9e77", "#d7191c", "#fdae61", "#404040"]

        bars = ax.bar(np.arange(len(accs)), accs, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(np.arange(len(accs)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"{m_name} — Stream Asymmetry Test")
        ax.set_ylim(0, 88)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cls_vs_patch_replacement.png"))
    plt.close(fig)
    print("Saved: cls_vs_patch_replacement.png")


def generate_all_figures():
    plot_prototype_performance_by_fraction()
    plot_wrong_depth_mean_comparison()
    plot_prototype_scale_sweep()
    plot_prototype_cosine_sweep()
    plot_prototype_sign_flip_sweep()
    plot_full_patch_replacement()
    plot_cls_vs_patch_replacement()


if __name__ == "__main__":
    generate_all_figures()
