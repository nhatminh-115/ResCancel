import os
import json
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_all_fungibility_figures(
    results_manifest: Dict[str, Any],
    output_dir: str = "figures/fungibility_v0"
):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    tiny_results = results_manifest["tiny_results"]["depth_results"]
    small_results = results_manifest["small_results"]["depth_results"]
    depths = sorted([int(k) for k in tiny_results.keys()])

    models = [("DeiT-Tiny", tiny_results, "#1f77b4"), ("DeiT-Small", small_results, "#ff7f0e")]

    # -----------------------------------------------------------------
    # Figure 1: fungibility_gap_by_depth.png
    # -----------------------------------------------------------------
    plt.figure(figsize=(9, 5.5), dpi=300)
    for model_name, res_dict, color in models:
        means = [res_dict[d]["fungibility_gap"]["mean"] for d in depths]
        ci_lows = [res_dict[d]["fungibility_gap"]["mean"] - res_dict[d]["fungibility_gap"]["bootstrap_ci_95"][0] for d in depths]
        ci_highs = [res_dict[d]["fungibility_gap"]["bootstrap_ci_95"][1] - res_dict[d]["fungibility_gap"]["mean"] for d in depths]
        
        plt.errorbar(
            depths, means, yerr=[ci_lows, ci_highs],
            marker="o", linewidth=2.5, markersize=8, capsize=5, capthick=1.5,
            label=f"{model_name}", color=color
        )
        for d, m in zip(depths, means):
            dz = res_dict[d]["fungibility_gap"]["cohens_dz"]
            q = res_dict[d]["fungibility_gap"].get("fdr_p_value", 1.0)
            sig = "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "ns"))
            plt.annotate(
                f"$d_z$={dz:.2f} ({sig})",
                (d, m),
                textcoords="offset points",
                xytext=(0, 10 if model_name == "DeiT-Tiny" else -18),
                ha="center",
                fontsize=9,
                fontweight="bold" if sig != "ns" else "normal"
            )

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="Zero Gap (Damage_cross = Damage_zero)")
    plt.title("Patch Content Fungibility Gap Across Transformer Depth\n$FungibilityGap = Damage_{zero} - Damage_{cross-image}$", fontsize=13, fontweight="bold")
    plt.xlabel("Transformer Block Depth $l$ (Intervention after Block $l$)", fontsize=11, fontweight="bold")
    plt.ylabel("Fungibility Gap (Margin Advantage over Zero)", fontsize=11, fontweight="bold")
    plt.xticks(depths, [f"Block {d}" for d in depths])
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fungibility_gap_by_depth.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 2: margin_damage_by_condition.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=True)
    cond_labels = {
        "zero": ("Zero Ablation", "#d62728", "s--"),
        "layer_mean": ("Layer Mean", "#9467bd", "^-."),
        "cross_same_pos": ("Cross-Image Same-Pos", "#2ca02c", "o-"),
        "cross_rand_pos": ("Cross-Image Rand-Pos", "#8c564b", "x-"),
        "within_shuff": ("Within-Image Shuffle", "#e377c2", "d-")
    }

    for ax_idx, (model_name, res_dict, _) in enumerate(models):
        ax = axes[ax_idx]
        for cond, (lbl, col, style) in cond_labels.items():
            dmg_means = [res_dict[d]["damage"][cond]["mean"] for d in depths]
            ax.plot(depths, dmg_means, style, label=lbl, color=col, linewidth=2, markersize=7)
        ax.axhline(0.0, color="black", linestyle=":", alpha=0.6)
        ax.set_title(f"{model_name}: True-Class Margin Damage", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Depth $l$", fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Margin Damage ($BaselineMargin - ConditionMargin$)", fontsize=11, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"Block {d}" for d in depths])
        ax.legend(frameon=True, loc="upper left", fontsize=9.5)

    plt.suptitle("Causal Degradation Across Intervention Conditions and Depth", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "margin_damage_by_condition.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 3: accuracy_by_condition_depth.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=True)
    for ax_idx, (model_name, res_dict, _) in enumerate(models):
        ax = axes[ax_idx]
        base_acc = results_manifest[f"{'tiny' if 'Tiny' in model_name else 'small'}_results"]["manifest"]["baseline_accuracy"]
        ax.axhline(base_acc * 100, color="blue", linestyle="--", linewidth=1.5, label=f"Clean Baseline ({base_acc*100:.1f}%)")

        for cond, (lbl, col, style) in cond_labels.items():
            accs = [res_dict[d]["conditions"][cond]["accuracy"] * 100 for d in depths]
            ax.plot(depths, accs, style, label=lbl, color=col, linewidth=2, markersize=7)

        ax.set_title(f"{model_name}: Top-1 Accuracy Across Depths", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Depth $l$", fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"Block {d}" for d in depths])
        ax.legend(frameon=True, loc="lower left", fontsize=9.5)

    plt.suptitle("Model Accuracy Under 25% Token Interventions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_condition_depth.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Figure 4: replacement_norm_distribution.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    # Subplot 1: Token L2 norm comparison across depth for DeiT-Small
    ax1 = axes[0]
    for cond, (lbl, col, style) in cond_labels.items():
        norms = [small_results[d]["conditions"][cond]["mean_token_l2"] for d in depths]
        ax1.plot(depths, norms, style, label=lbl, color=col, linewidth=2, markersize=7)
    ax1.set_title("DeiT-Small: Mean Token $L_2$ Norm at Replaced Slots", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Transformer Depth $l$", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Token $L_2$ Norm", fontsize=10, fontweight="bold")
    ax1.set_xticks(depths)
    ax1.set_xticklabels([f"Block {d}" for d in depths])
    ax1.legend(frameon=True, fontsize=9)

    # Subplot 2: Cosine similarity to original token
    ax2 = axes[1]
    for cond, (lbl, col, style) in cond_labels.items():
        if cond == "zero":
            continue
        cos_sims = [small_results[d]["conditions"][cond]["mean_cosine_to_orig"] for d in depths]
        ax2.plot(depths, cos_sims, style, label=lbl, color=col, linewidth=2, markersize=7)
    ax2.set_title("DeiT-Small: Cosine Similarity to Original Patch Token", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Transformer Depth $l$", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Cosine Similarity", fontsize=10, fontweight="bold")
    ax2.set_xticks(depths)
    ax2.set_xticklabels([f"Block {d}" for d in depths])
    ax2.legend(frameon=True, fontsize=9)

    plt.suptitle("Activation Scale and Alignment Checks Across Depths", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "replacement_norm_distribution.png"))
    plt.close()

    # -----------------------------------------------------------------
    # Optional Figure 5: condition_difference_heatmap.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    contrast_keys = [
        ("spatial_slot_specificity", "Spatial Specificity\n(CrossSame - CrossRand)"),
        ("content_vs_spatial_assignment", "Content Specificity\n(WithinShuff - CrossSame)"),
        ("token_vs_layer_mean", "Structured Token\n(CrossSame - LayerMean)")
    ]

    for ax_idx, (model_name, res_dict, _) in enumerate(models):
        ax = axes[ax_idx]
        matrix = np.zeros((len(contrast_keys), len(depths)))
        for r_idx, (c_key, _) in enumerate(contrast_keys):
            for c_col, d in enumerate(depths):
                matrix[r_idx, c_col] = res_dict[d]["contrasts"][c_key]["mean"]
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt="+.3f",
            cmap="vlag",
            center=0.0,
            xticklabels=[f"Block {d}" for d in depths],
            yticklabels=[lbl for _, lbl in contrast_keys],
            cbar=True,
            ax=ax
        )
        ax.set_title(f"{model_name}: Contrast Margin Deltas", fontsize=11, fontweight="bold")
        ax.set_xlabel("Transformer Depth $l$", fontsize=10, fontweight="bold")

    plt.suptitle("Mechanistic Specificity Contrasts Across Depths", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "condition_difference_heatmap.png"))
    plt.close()
    print(f"Successfully generated all 5 figures in {output_dir}/")
