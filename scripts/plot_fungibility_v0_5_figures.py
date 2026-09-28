import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_all_fungibility_v0_5_figures(
    manifest_data: Dict[str, Any],
    output_dir: str = "figures/fungibility_v0_5"
):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    tiny_res = manifest_data["tiny_results"]
    small_res = manifest_data["small_results"]

    conditions_ordered = [
        ("zero", "Zero Ablation", "#d62728"),
        ("random_sphere_mean", "Norm-Matched Sphere", "#ff7f0e"),
        ("gaussian_mean", "Layer-Stat Gaussian", "#e377c2"),
        ("feature_shuffle_mean", "Feature-Shuffled Orig", "#8c564b"),
        ("cross_image", "Cross-Image Donor", "#2ca02c"),
        ("norm_matched_cross", "Norm-Matched Cross", "#17becf"),
        ("layer_mean", "Same-Image Mean", "#9467bd"),
        ("norm_matched_mean", "Norm-Matched Mean", "#bcbd22")
    ]

    models = [("DeiT-Tiny", tiny_res), ("DeiT-Small", small_res)]

    # -----------------------------------------------------------------
    # Figure 1: condition_margin_damage.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300, sharey=False)
    for ax_idx, (model_name, res_dict) in enumerate(models):
        ax = axes[ax_idx]
        conds = res_dict["conditions"]
        labels = [lbl for _, lbl, _ in conditions_ordered]
        colors = [col for _, _, col in conditions_ordered]
        
        damages = []
        ci_errs = []
        for c_key, _, _ in conditions_ordered:
            c_info = conds[c_key]
            dmg = c_info["mean_damage"]
            ci = c_info["damage_ci_95"]
            damages.append(dmg)
            ci_errs.append((dmg - ci[0], ci[1] - dmg))

        yerr = np.array(ci_errs).T
        bars = ax.bar(range(len(labels)), damages, yerr=yerr, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=1.0)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9.5, fontweight="medium")
        ax.set_ylabel("True-Class Margin Damage ($Baseline - Interv$)", fontsize=11, fontweight="bold")
        ax.set_title(f"{model_name}: Margin Damage at Block 8", fontsize=12, fontweight="bold")
        
        # Add values on top of bars
        for bar, val in zip(bars, damages):
            height = bar.get_height()
            ax.annotate(f"{val:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("Causal Margin Degradation Under Norm- and Distribution-Matched Nulls", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "condition_margin_damage.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 2: condition_accuracy.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300, sharey=False)
    for ax_idx, (model_name, res_dict) in enumerate(models):
        ax = axes[ax_idx]
        conds = res_dict["conditions"]
        base_acc = res_dict["manifest"]["baseline_accuracy"] * 100.0
        
        labels = [lbl for _, lbl, _ in conditions_ordered]
        colors = [col for _, _, col in conditions_ordered]
        accs = [conds[c_key]["accuracy"] * 100.0 for c_key, _, _ in conditions_ordered]

        bars = ax.bar(range(len(labels)), accs, color=colors, alpha=0.85, edgecolor="black", linewidth=1.0)
        ax.axhline(base_acc, color="blue", linestyle="--", linewidth=1.5, label=f"Clean Baseline ({base_acc:.1f}%)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9.5, fontweight="medium")
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"{model_name}: Accuracy at Block 8", fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", frameon=True)
        ax.set_ylim([max(0, min(accs) - 5), base_acc + 5])

        for bar, val in zip(bars, accs):
            height = bar.get_height()
            ax.annotate(f"{val:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("Top-1 Accuracy Under Null Interventions vs Clean Baseline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "condition_accuracy.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 3: norm_vs_performance.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    for ax_idx, (model_name, res_dict) in enumerate(models):
        ax = axes[ax_idx]
        conds = res_dict["conditions"]
        
        for c_key, lbl, col in conditions_ordered:
            c_info = conds[c_key]
            norm_val = c_info["mean_token_l2"]
            dmg_val = c_info["mean_damage"]
            ax.scatter(norm_val, dmg_val, color=col, s=120, edgecolors="black", linewidth=1.2, label=lbl, zorder=5)
            ax.annotate(lbl, (norm_val, dmg_val), textcoords="offset points", xytext=(6, 2), fontsize=8.5)

        ax.set_xlabel("Mean Token $L_2$ Norm at Replaced Slots", fontsize=11, fontweight="bold")
        ax.set_ylabel("Margin Damage ($Baseline - Interv$)", fontsize=11, fontweight="bold")
        ax.set_title(f"{model_name}: Token Norm vs Margin Damage", fontsize=12, fontweight="bold")
        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)

    plt.suptitle("Testing H2 (Scale Artifact): Token Norm Does NOT Predict Predictive Recovery", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "norm_vs_performance.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 4: distribution_distance_vs_damage.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    for ax_idx, (model_name, res_dict) in enumerate(models):
        ax = axes[ax_idx]
        conds = res_dict["conditions"]
        
        for c_key, lbl, col in conditions_ordered:
            c_info = conds[c_key]
            mahal_val = c_info["mahalanobis_dist"]
            dmg_val = c_info["mean_damage"]
            ax.scatter(mahal_val, dmg_val, color=col, s=120, edgecolors="black", linewidth=1.2, label=lbl, zorder=5)
            ax.annotate(lbl, (mahal_val, dmg_val), textcoords="offset points", xytext=(6, 2), fontsize=8.5)

        ax.set_xlabel("Mean Diagonal Mahalanobis Distance to Layer Manifold", fontsize=11, fontweight="bold")
        ax.set_ylabel("Margin Damage ($Baseline - Interv$)", fontsize=11, fontweight="bold")
        ax.set_title(f"{model_name}: Manifold Distance vs Margin Damage", fontsize=12, fontweight="bold")
        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)

    plt.suptitle("Testing H3 (Distributional Plausibility): Manifold Distance vs Causal Damage", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_distance_vs_damage.png"))
    plt.close()
    print(f"Generated all 4 figures successfully in {output_dir}/")
