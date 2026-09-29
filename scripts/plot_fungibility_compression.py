"""
scripts/plot_fungibility_compression.py

Publication-quality figures for FUNGIBILITY-TO-COMPRESSION POC:
1. accuracy_vs_tail_tokens.png: Accuracy vs Downstream Sequence Length across 4 models
2. accuracy_vs_total_flops.png: Accuracy vs Total Model GFLOPs (Pareto frontier)
3. accuracy_vs_latency.png: Accuracy vs Measured End-to-End Latency on RTX 5070 GPU
4. equivalence_error.png: Logit error and 100% prediction agreement audit
5. per_model_pareto.png: Multi-panel Pareto efficiency analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


COLOR_CLEAN = "#7f8c8d"       # Gray dashed
COLOR_WEIGHTED = "#2980b9"    # Primary Blue (Weighted Centroid Carrier)
COLOR_UNWEIGHTED = "#e67e22"  # Orange (Unweighted Centroid)
COLOR_IMG_MEAN = "#27ae60"    # Green (Image-Mean Carrier)
COLOR_PRUNED = "#c0392b"      # Red (Random Pruning)

PRIMARY_MODELS = [
    ("deit_tiny", 8, "DeiT-Tiny (Depth 8)"),
    ("deit_small", 8, "DeiT-Small (Depth 8)"),
    ("vit_base", 7, "ViT-B AugReg (Depth 7 - Primary)"),
    ("dinov2", 9, "DINOv2 ViT-S/14 (Depth 9 - Primary)")
]


def plot_accuracy_vs_tail_tokens(df_base: pd.DataFrame, figures_dir: str):
    """
    Figure 1: 4-panel Top-1 Accuracy (%) vs Downstream Patch Tokens (M + 1).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (m_key, depth, title) in zip(axes, PRIMARY_MODELS):
        sub_m = df_base[(df_base["model"] == m_key) & (df_base["depth"] == depth)]

        # Aggregate across mask seeds
        agg = sub_m.groupby(["condition", "downstream_patch_tokens"], as_index=False).agg({"top1_acc": "mean"})

        # Clean baseline
        clean_row = agg[agg["condition"] == "WEIGHTED_CENTROID_CARRIER"].sort_values(by="downstream_patch_tokens", ascending=False).iloc[0]
        clean_acc = clean_row["top1_acc"] * 100.0
        ax.axhline(clean_acc, color=COLOR_CLEAN, linestyle="--", linewidth=1.5, label=f"Clean ({clean_acc:.1f}%)")

        for cond, color, marker, label in [
            ("WEIGHTED_CENTROID_CARRIER", COLOR_WEIGHTED, "o", r"Weighted Centroid Carrier ($s=m$)"),
            ("IMAGE_MEAN_CARRIER", COLOR_IMG_MEAN, "^", r"Image-Mean Carrier ($s=m$)"),
            ("UNWEIGHTED_CENTROID", COLOR_UNWEIGHTED, "s", r"Unweighted Centroid ($s=1$)"),
            ("RANDOM_PRUNING", COLOR_PRUNED, "x", "Random Pruning (No Carrier)")
        ]:
            sub_c = agg[agg["condition"] == cond].sort_values(by="downstream_patch_tokens")
            if len(sub_c) == 0:
                continue
            x = sub_c["downstream_patch_tokens"].values
            y = sub_c["top1_acc"].values * 100.0

            ax.plot(x, y, color=color, marker=marker, markersize=5, linewidth=1.8, label=label)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
        ax.set_xlabel("Downstream Spatial Patch Tokens", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, frameon=True, fontsize=10)

    plt.suptitle("Accuracy vs Downstream Sequence Length: Weighted Carrier vs Matched-Budget Baselines",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "accuracy_vs_tail_tokens.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def plot_accuracy_vs_total_flops(df_base: pd.DataFrame, df_comp: pd.DataFrame, figures_dir: str):
    """
    Figure 2: 4-panel Top-1 Accuracy (%) vs Total Model GFLOPs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (m_key, depth, title) in zip(axes, PRIMARY_MODELS):
        sub_base = df_base[(df_base["model"] == m_key) & (df_base["depth"] == depth)]
        sub_comp = df_comp[(df_comp["model"] == m_key) & (df_comp["depth"] == depth)]

        # Map actual_k to total_comp_gflops
        flop_map = dict(zip(sub_comp["actual_k"], sub_comp["total_comp_gflops"]))
        clean_flops = sub_comp["total_orig_gflops"].values[0]

        agg = sub_base.groupby(["condition", "actual_k"], as_index=False).agg({"top1_acc": "mean"})
        agg["total_gflops"] = agg["actual_k"].map(flop_map)

        # Clean baseline
        clean_row = agg[agg["condition"] == "WEIGHTED_CENTROID_CARRIER"].sort_values(by="actual_k").iloc[0]
        clean_acc = clean_row["top1_acc"] * 100.0
        ax.scatter([clean_flops], [clean_acc], color=COLOR_CLEAN, s=80, zorder=5, label=f"Clean ({clean_flops:.2f} GF)")
        ax.axhline(clean_acc, color=COLOR_CLEAN, linestyle="--", alpha=0.5)

        for cond, color, marker, label in [
            ("WEIGHTED_CENTROID_CARRIER", COLOR_WEIGHTED, "o", "Weighted Centroid Carrier"),
            ("IMAGE_MEAN_CARRIER", COLOR_IMG_MEAN, "^", "Image-Mean Carrier"),
            ("UNWEIGHTED_CENTROID", COLOR_UNWEIGHTED, "s", "Unweighted Centroid (s=1)"),
            ("RANDOM_PRUNING", COLOR_PRUNED, "x", "Random Pruning")
        ]:
            sub_c = agg[agg["condition"] == cond].sort_values(by="total_gflops")
            if len(sub_c) == 0:
                continue
            x = sub_c["total_gflops"].values
            y = sub_c["top1_acc"].values * 100.0

            ax.plot(x, y, color=color, marker=marker, markersize=5, linewidth=1.8, label=label)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
        ax.set_xlabel("Total Model Compute (GFLOPs)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, frameon=True, fontsize=10)

    plt.suptitle("Pareto Efficiency: Accuracy vs Total Model Theoretical GFLOPs",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "accuracy_vs_total_flops.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def plot_accuracy_vs_latency(df_lat: pd.DataFrame, df_base: pd.DataFrame, figures_dir: str):
    """
    Figure 3: Measured End-to-End Latency vs Top-1 Accuracy on RTX 5070 GPU.
    """
    if len(df_lat) == 0:
        print("[Plotting] Latency data empty, skipping Figure 3.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax_idx, bs in enumerate([1, 16]):
        ax = axes[ax_idx]
        sub_lat = df_lat[df_lat["batch_size"] == bs]

        # For each model, plot points
        for m_key, depth, title in PRIMARY_MODELS:
            sub_m_lat = sub_lat[sub_lat["model"] == m_key]
            sub_m_base = df_base[(df_base["model"] == m_key) & (df_base["depth"] == depth)]

            if len(sub_m_lat) == 0:
                continue

            clean_lat = sub_m_lat["clean_latency_ms"].values[0]
            clean_acc = sub_m_base[sub_m_base["actual_k"] == 0]["top1_acc"].mean() * 100.0

            ax.scatter([clean_lat], [clean_acc], color="black", marker="*", s=120, label="Clean Baseline" if m_key == "deit_tiny" else "")

            for cond, color, marker, label in [
                ("WEIGHTED_CENTROID_CARRIER", COLOR_WEIGHTED, "o", "Weighted Centroid"),
                ("IMAGE_MEAN_CARRIER", COLOR_IMG_MEAN, "^", "Image-Mean Carrier"),
                ("RANDOM_PRUNING", COLOR_PRUNED, "x", "Random Pruning")
            ]:
                row_lat = sub_m_lat[sub_m_lat["condition"] == cond]
                if len(row_lat) == 0:
                    continue
                k_val = row_lat["k_replaced"].values[0]
                lat_val = row_lat["compressed_latency_ms"].values[0]

                # Get acc for this k
                acc_val = sub_m_base[(sub_m_base["condition"] == cond) & (sub_m_base["actual_k"] == k_val)]["top1_acc"].mean() * 100.0

                ax.scatter([lat_val], [acc_val], color=color, marker=marker, s=80, label=label if m_key == "deit_tiny" else "")
                ax.annotate(f"{m_key.replace('_', ' ').title()}", (lat_val, acc_val),
                            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        ax.set_title(f"Batch Size = {bs} (RTX 5070 GPU)", fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("Measured End-to-End Latency (ms)", fontsize=11)
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=True, fontsize=11)

    plt.suptitle("Physical Hardware Execution: Measured End-to-End Latency vs Accuracy Retention",
                 fontsize=14, fontweight="bold", y=1.09)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "accuracy_vs_latency.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def plot_equivalence_error(df_equiv: pd.DataFrame, figures_dir: str):
    """
    Figure 4: Audit of Maximum Absolute Logit Difference and Prediction Agreement.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for m_key, depth, title in PRIMARY_MODELS:
        sub_m = df_equiv[(df_equiv["model"] == m_key) & (df_equiv["depth"] == depth)].sort_values(by="actual_fraction")
        if len(sub_m) == 0:
            continue
        x = sub_m["actual_fraction"].values * 100.0
        max_err = sub_m["max_abs_logit_diff"].values
        agree = sub_m["prediction_agreement"].values * 100.0

        ax1.plot(x, max_err, marker="o", markersize=4, label=title)
        ax2.plot(x, agree, marker="s", markersize=4, label=title)

    ax1.axhline(1e-4, color="crimson", linestyle="--", linewidth=1.2, label=r"Protocol Tolerance ($10^{-4}$)")
    ax1.axhline(1e-5, color="darkgreen", linestyle=":", linewidth=1.2, label=r"Ideal Target ($10^{-5}$)")
    ax1.set_yscale("log")
    ax1.set_title("Maximum Absolute Logit Discrepancy", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Replaced Patch Fraction (%)", fontsize=11)
    ax1.set_ylabel(r"$\max |l_{\mathrm{ref}} - l_{\mathrm{comp}}|$ (Log Scale)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.axhline(100.0, color="darkgreen", linestyle="--", linewidth=1.2, label="100% Agreement")
    ax2.set_title("Prediction Agreement Between Compressed and Full Reference", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Replaced Patch Fraction (%)", fontsize=11)
    ax2.set_ylabel("Prediction Agreement (%)", fontsize=11)
    ax2.set_ylim(98.5, 100.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.suptitle("Exact Numerical Equivalence Audit: Weighted Carrier vs Full Uncompressed Reference",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "equivalence_error.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved {out_path}")


def generate_all_compression_figures(output_dir: str, figures_dir: str):
    """
    Master plotting function for compression POC.
    """
    print(f"[Plotting] Loading data from {output_dir}...")
    df_equiv = pd.read_csv(os.path.join(output_dir, "equivalence_results.csv"))
    df_base = pd.read_csv(os.path.join(output_dir, "baseline_comparison.csv"))
    df_comp = pd.read_csv(os.path.join(output_dir, "compute_summary.csv"))
    df_lat = pd.read_csv(os.path.join(output_dir, "latency_summary.csv"))

    print("[Plotting] Generating Figure 1: accuracy_vs_tail_tokens.png...")
    plot_accuracy_vs_tail_tokens(df_base, figures_dir)

    print("[Plotting] Generating Figure 2: accuracy_vs_total_flops.png...")
    plot_accuracy_vs_total_flops(df_base, df_comp, figures_dir)

    print("[Plotting] Generating Figure 3: accuracy_vs_latency.png...")
    plot_accuracy_vs_latency(df_lat, df_base, figures_dir)

    print("[Plotting] Generating Figure 4: equivalence_error.png...")
    plot_equivalence_error(df_equiv, figures_dir)

    print("[Plotting] All compression figures created successfully.")


if __name__ == "__main__":
    generate_all_compression_figures(
        output_dir="outputs/fungibility_compression_poc",
        figures_dir="figures/fungibility_compression_poc"
    )
