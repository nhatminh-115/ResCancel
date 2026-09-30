"""
Render the final, submission-facing figure set for the Patch Content Fungibility paper.

Historical experiment plots are intentionally left untouched. This script reads only frozen
machine-readable outputs and writes a unified publication-style set under:
    figures/paper_final/main/
    figures/paper_final/supp/

Outputs are saved as both high-resolution PNG and vector PDF.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "figures" / "paper_final" / "main"
SUPP_DIR = ROOT / "figures" / "paper_final" / "supp"

# Semantic palette: same meaning everywhere.
C = {
    "clean": "#4C566A",
    "zero": "#C43C39",
    "centroid": "#2F6FB0",
    "gaussian": "#3A8E5B",
    "permute": "#D08C28",
    "sign": "#8E5AA7",
    "shared": "#A65A5A",
    "independent": "#3A8E5B",
    "pc1": "#2F6FB0",
    "pc2": "#79A7D3",
    "random": "#7A7A7A",
    "accent": "#334E68",
    "light": "#D9E2EC",
    "very_light": "#F3F6F8",
    "text": "#1F2933",
}

MODEL_ORDER = ["deit_tiny", "deit_small", "vit_base", "dinov2"]
MODEL_LABEL = {
    "deit_tiny": "DeiT-Tiny",
    "deit_small": "DeiT-Small",
    "vit_base": "ViT-B/16 AugReg",
    "dinov2": "DINOv2 ViT-S/14",
}
MODEL_DEPTH = {"deit_tiny": 8, "deit_small": 8, "vit_base": 7, "dinov2": 9}

MODEL_LONG_TO_KEY = {
    "deit_tiny_patch16_224": "deit_tiny",
    "deit_small_patch16_224": "deit_small",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.2,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.2,
            "figure.titlesize": 11,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB7C4")
    ax.spines["bottom"].set_color("#AAB7C4")
    ax.tick_params(colors=C["text"])
    if grid == "y":
        ax.grid(axis="y", color="#D8E0E7", linewidth=0.65, alpha=0.75, zorder=0)
    elif grid == "both":
        ax.grid(color="#D8E0E7", linewidth=0.65, alpha=0.65, zorder=0)
    else:
        ax.grid(False)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.055,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10.8,
        va="top",
        ha="left",
        color=C["text"],
    )


def save_figure(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[paper-final] {out_base.with_suffix('.png')}")


def _read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _parse_pct(series: pd.Series) -> pd.Series:
    return series.astype(str).str.rstrip("%").astype(float)


def _mean_std(df: pd.DataFrame, value: str) -> Tuple[float, float]:
    vals = pd.to_numeric(df[value], errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan, 0.0
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def _bar_value_labels(ax, bars, fmt="{:.1f}", dy=0.6, fontsize=7.2) -> None:
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=C["text"],
        )


# -----------------------------------------------------------------------------
# Main Figure 1: conceptual schematic
# -----------------------------------------------------------------------------

def _rounded_box(ax, xy, wh, text, fc="white", ec="#9FB3C8", lw=1.1, fontsize=8.5, weight="normal"):
    x, y = xy
    w, h = wh
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=C["text"], fontweight=weight)
    return box


def _arrow(ax, p0, p1, color="#718096", lw=1.2):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw, shrinkA=2, shrinkB=2),
    )


def _token_grid(ax, x, y, rows=3, cols=6, size=0.022, gap=0.007, colors=None, edge="#FFFFFF"):
    if colors is None:
        colors = [C["centroid"]] * (rows * cols)
    k = 0
    for r in range(rows):
        for c in range(cols):
            col = colors[k % len(colors)]
            rect = patches.FancyBboxPatch(
                (x + c * (size + gap), y - r * (size + gap)),
                size,
                size,
                boxstyle="round,pad=0.002,rounding_size=0.003",
                facecolor=col,
                edgecolor=edge,
                linewidth=0.5,
            )
            ax.add_patch(rect)
            k += 1



def plot_main_01_schematic() -> None:
    fig = plt.figure(figsize=(7.25, 4.65))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.12, 1.0],
        hspace=0.24,
        wspace=0.16,
        left=0.04, right=0.99, top=0.98, bottom=0.04,
    )

    # ------------------------------------------------------------------
    # (a) Intervention design
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.005, 0.98, "(a)", fontsize=10.8, fontweight="bold", va="top", color=C["text"])
    ax.text(0.055, 0.98, "Activation substitution isolates what late patch states must preserve",
            fontsize=10.0, fontweight="semibold", va="top", color=C["text"])

    _rounded_box(
        ax, (0.03, 0.33), (0.12, 0.32),
        "Input image", fc=C["very_light"], ec="#B8C5D1", fontsize=8.2, weight="semibold"
    )
    # Small patch strip below the label instead of behind it.
    patch_cols = ["#7AA6C2", "#8EC0A4", "#D4A76A", "#9D8AC7"]
    _token_grid(ax, 0.052, 0.43, rows=1, cols=4, size=0.017, gap=0.006, colors=patch_cols)

    _arrow(ax, (0.155, 0.49), (0.215, 0.49))
    _rounded_box(
        ax, (0.22, 0.33), (0.15, 0.32),
        "Early ViT\nblocks", fc="#EEF4F8", ec="#A8C2D4", fontsize=8.4, weight="semibold"
    )
    _arrow(ax, (0.375, 0.49), (0.435, 0.49))
    _rounded_box(
        ax, (0.44, 0.28), (0.16, 0.42),
        "Late patch state\n$H_\\ell$", fc="#F4F7FA", ec="#A8B8C8", fontsize=8.4, weight="semibold"
    )
    varied = ["#3F78B5", "#5D91C3", "#32659B", "#759DC7", "#4D80B2"]
    _token_grid(ax, 0.466, 0.42, rows=1, cols=5, size=0.017, gap=0.006, colors=varied)

    # Branch into three interventions.
    branch_x = 0.655
    ax.plot([0.605, branch_x], [0.49, 0.49], color="#718096", lw=1.15)
    ax.plot([branch_x, branch_x], [0.23, 0.76], color="#718096", lw=1.05)
    for yy in [0.76, 0.49, 0.23]:
        _arrow(ax, (branch_x, yy), (0.705, yy))

    _rounded_box(
        ax, (0.71, 0.67), (0.18, 0.18),
        "Zero\nreplacement", fc="#FBECEC", ec="#E1A7A5", fontsize=8.1, weight="semibold"
    )
    _rounded_box(
        ax, (0.71, 0.40), (0.18, 0.18),
        "Calibration\nsurrogate", fc="#EAF3FA", ec="#9BC0DE", fontsize=8.1, weight="semibold"
    )
    _rounded_box(
        ax, (0.71, 0.14), (0.18, 0.18),
        "Unmodified\nreference", fc="#EEF7F0", ec="#9EC5AA", fontsize=8.1, weight="semibold"
    )
    ax.text(0.915, 0.76, "destructive", color=C["zero"], fontsize=7.8, fontweight="semibold", va="center")
    ax.text(0.915, 0.49, "often tolerated late", color=C["gaussian"], fontsize=7.8, fontweight="semibold", va="center")
    ax.text(0.915, 0.23, "clean", color=C["clean"], fontsize=7.8, fontweight="semibold", va="center")

    ax.text(
        0.49, 0.055,
        "Upstream image processing is unchanged; [CLS], token slots, sequence length, model weights, and downstream blocks remain fixed.",
        fontsize=7.4, color="#52606D", ha="center", va="bottom",
    )

    # ------------------------------------------------------------------
    # (b) Geometry constraint
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, 0.98, "(b)", fontsize=10.8, fontweight="bold", va="top", color=C["text"])
    ax.text(0.15, 0.98, "Geometry", fontsize=9.6, fontweight="semibold", va="top", color=C["text"])

    vals = np.array([0.25, 0.72, 0.43, 0.88, 0.35, 0.60])
    for i, v in enumerate(vals):
        ax.add_patch(patches.Rectangle(
            (0.08 + i * 0.055, 0.56), 0.032, v * 0.22,
            facecolor=C["centroid"], edgecolor="none"
        ))
    ax.text(0.25, 0.48, "aligned $\\mu_\\ell$", ha="center", fontsize=7.7)
    ax.text(0.25, 0.39, "tolerated", ha="center", color=C["gaussian"], fontsize=8.0, fontweight="semibold")

    perm = vals[[3, 0, 5, 2, 1, 4]]
    for i, v in enumerate(perm):
        ax.add_patch(patches.Rectangle(
            (0.58 + i * 0.055, 0.56), 0.032, v * 0.22,
            facecolor=C["permute"], edgecolor="none"
        ))
    ax.text(0.75, 0.48, "permuted / sign-flipped", ha="center", fontsize=7.5)
    ax.text(0.75, 0.39, "degrades", ha="center", color=C["zero"], fontsize=8.0, fontweight="semibold")

    _rounded_box(
        ax, (0.07, 0.09), (0.86, 0.19),
        "Feature-coordinate identity\nand orientation matter.",
        fc="#F6F8FA", ec="#C3CED8", fontsize=7.0, weight="semibold"
    )

    # ------------------------------------------------------------------
    # (c) Diversity constraint
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, 0.98, "(c)", fontsize=10.8, fontweight="bold", va="top", color=C["text"])
    ax.text(0.15, 0.98, "Diversity", fontsize=9.6, fontweight="semibold", va="top", color=C["text"])

    _token_grid(ax, 0.10, 0.67, rows=2, cols=5, size=0.025, gap=0.011, colors=[C["shared"]] * 10)
    ax.text(0.25, 0.47, "shared state", ha="center", fontsize=7.7)
    ax.text(0.25, 0.38, "collapse", ha="center", color=C["zero"], fontsize=8.0, fontweight="semibold")

    independent_cols = ["#4D8B6A", "#6AA37F", "#357A59", "#80B18F", "#438563"]
    _token_grid(ax, 0.59, 0.67, rows=2, cols=5, size=0.025, gap=0.011, colors=independent_cols)
    ax.text(0.74, 0.47, "independent states", ha="center", fontsize=7.7)
    ax.text(0.74, 0.38, "rescue", ha="center", color=C["gaussian"], fontsize=8.0, fontweight="semibold")

    _rounded_box(
        ax, (0.07, 0.09), (0.86, 0.19),
        "Complete replacement still\nrequires token diversity.",
        fc="#F6F8FA", ec="#C3CED8", fontsize=7.0, weight="semibold"
    )

    # ------------------------------------------------------------------
    # (d) Compression boundary
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, 0.98, "(d)", fontsize=10.8, fontweight="bold", va="top", color=C["text"])
    ax.text(0.15, 0.98, "Boundary", fontsize=9.6, fontweight="semibold", va="top", color=C["text"])

    _rounded_box(
        ax, (0.08, 0.60), (0.84, 0.18),
        "Full sequence\nsurrogates can be tolerated",
        fc="#EAF3FA", ec="#9BC0DE", fontsize=7.6, weight="semibold"
    )
    _arrow(ax, (0.50, 0.59), (0.50, 0.46))
    _rounded_box(
        ax, (0.08, 0.28), (0.84, 0.18),
        "Shorter sequence\nreduces compute",
        fc="#EEF7F0", ec="#9EC5AA", fontsize=7.6, weight="semibold"
    )
    ax.text(0.50, 0.19, "but", ha="center", fontsize=7.2, color="#52606D")
    ax.text(
        0.50, 0.095,
        "matched pruning ≥ synthetic carrier",
        ha="center", fontsize=7.5, color=C["zero"], fontweight="semibold"
    )

    save_figure(fig, MAIN_DIR / "fig01_conceptual_overview")

def _depth_series_deit(path: str) -> Dict[str, np.ndarray]:
    df = _read(path)
    sub = df[df["fraction"].astype(str) == "25%"].copy().sort_values("depth")
    return {
        "x": sub["depth"].to_numpy(float),
        "clean": sub["clean_acc"].to_numpy(float) * 100,
        "zero": sub["zero_acc"].to_numpy(float) * 100,
        "centroid": sub["global_mean_acc"].to_numpy(float) * 100,
        "gaussian": sub["gaussian_acc"].to_numpy(float) * 100,
        "gaussian_sd": np.zeros(len(sub)),
    }


def _depth_series_v1(path: str) -> Dict[str, np.ndarray]:
    df = _read(path)
    clean = float(df[df["condition"] == "CLEAN"]["top1_accuracy"].iloc[0]) * 100
    depths = np.sort(df[df["condition"] != "CLEAN"]["depth"].unique().astype(float))
    def vals(cond):
        out, sd = [], []
        for d in depths:
            s = df[(df["condition"] == cond) & (df["depth"] == d)]
            m, st = _mean_std(s, "top1_accuracy")
            out.append(m * 100)
            sd.append(st * 100)
        return np.asarray(out), np.asarray(sd)
    zero, _ = vals("ZERO")
    centroid, _ = vals("CENTROID")
    gaussian, gaussian_sd = vals("DIAGONAL_GAUSSIAN")
    return {
        "x": depths,
        "clean": np.repeat(clean, len(depths)),
        "zero": zero,
        "centroid": centroid,
        "gaussian": gaussian,
        "gaussian_sd": gaussian_sd,
    }


def plot_main_02_depth_emergence() -> None:
    data = {
        "deit_tiny": _depth_series_deit("outputs/fungibility_v0_6/tiny_depth_fraction_summary.csv"),
        "deit_small": _depth_series_deit("outputs/fungibility_v0_6/small_depth_fraction_summary.csv"),
        "vit_base": _depth_series_v1("outputs/fungibility_v1/vitb_depth_results.csv"),
        "dinov2": _depth_series_v1("outputs/fungibility_v1/dinov2_depth_results.csv"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharey=True)
    axes = axes.ravel()

    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        d = data[model]
        ax.plot(d["x"], d["zero"], color=C["zero"], marker="o", label="Zero")
        ax.plot(d["x"], d["centroid"], color=C["centroid"], marker="s", label="Centroid")
        ax.plot(d["x"], d["gaussian"], color=C["gaussian"], marker="^", label="Diagonal Gaussian")
        if np.nanmax(d["gaussian_sd"]) > 0:
            ax.fill_between(
                d["x"], d["gaussian"] - d["gaussian_sd"], d["gaussian"] + d["gaussian_sd"],
                color=C["gaussian"], alpha=0.14, linewidth=0
            )
        ax.plot(d["x"], d["clean"], color=C["clean"], linestyle="--", linewidth=1.3, label="Clean")
        ax.set_title(MODEL_LABEL[model])
        ax.set_ylim(0, 82)
        ax.set_xticks(d["x"])
        ax.set_xlabel("Intervention depth")
        if i % 2 == 0:
            ax.set_ylabel("Top-1 accuracy (%)")
        style_axis(ax, "y")
        panel_label(ax, f"({chr(97+i)})")

    handles = [
        Line2D([0], [0], color=C["clean"], linestyle="--", linewidth=1.3, label="Clean"),
        Line2D([0], [0], color=C["zero"], marker="o", label="Zero"),
        Line2D([0], [0], color=C["centroid"], marker="s", label="Centroid"),
        Line2D([0], [0], color=C["gaussian"], marker="^", label="Diagonal Gaussian"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.15, hspace=0.34, wspace=0.20)
    save_figure(fig, MAIN_DIR / "fig02_depth_emergence")


# -----------------------------------------------------------------------------
# Main Figure 3: dense replacement dose-response
# -----------------------------------------------------------------------------

def plot_main_03_dense_fraction() -> None:
    df = _read("outputs/fungibility_dense_fraction/summary_across_masks.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        depth = MODEL_DEPTH[model]
        sub = df[(df["model"] == model) & (df["depth"] == depth)].copy()
        clean_rows = sub[np.isclose(sub["actual_fraction"].astype(float), 0.0)]
        clean = float(clean_rows["mean_acc"].iloc[0]) * 100 if len(clean_rows) else np.nan
        ax.axhline(clean, color=C["clean"], linestyle="--", linewidth=1.25, zorder=1)

        for cond, color, label, ls in [
            ("ZERO", C["zero"], "Zero", "-"),
            ("CENTROID", C["centroid"], "Centroid", "-"),
            ("DIAGONAL_GAUSSIAN", C["gaussian"], "Diagonal Gaussian", "-"),
        ]:
            s = sub[sub["condition"] == cond].sort_values("actual_fraction")
            if len(s) == 0:
                continue
            x = s["actual_percent"].astype(float).to_numpy()
            y = s["mean_acc"].astype(float).to_numpy() * 100
            sd = s["sd_acc"].astype(float).fillna(0).to_numpy() * 100
            ax.plot(x, y, color=color, linestyle=ls, label=label)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12, linewidth=0)

        ax.set_title(f"{MODEL_LABEL[model]} · depth {depth}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 82)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        if i >= 2:
            ax.set_xlabel("Spatial patches replaced (%)")
        if i % 2 == 0:
            ax.set_ylabel("Top-1 accuracy (%)")
        style_axis(ax, "both")
        panel_label(ax, f"({chr(97+i)})")

    handles = [
        Line2D([0], [0], color=C["clean"], linestyle="--", linewidth=1.25, label="Clean"),
        Line2D([0], [0], color=C["zero"], label="Zero"),
        Line2D([0], [0], color=C["centroid"], label="Centroid"),
        Line2D([0], [0], color=C["gaussian"], label="Diagonal Gaussian"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.15, hspace=0.34, wspace=0.20)
    save_figure(fig, MAIN_DIR / "fig03_dense_fraction")


# -----------------------------------------------------------------------------
# Main Figure 4: geometry controls
# -----------------------------------------------------------------------------

def _geometry_deit(model: str) -> Dict[str, np.ndarray]:
    prefix = "tiny" if model == "deit_tiny" else "small"
    frac = _read(f"outputs/fungibility_v0_7/{prefix}_fraction_summary.csv")
    comp = _read(f"outputs/fungibility_v0_7/{prefix}_prototype_comparison.csv")
    sign = _read(f"outputs/fungibility_v0_7/{prefix}_sign_flip_sweep.csv")
    fractions = np.array([25.0, 50.0, 75.0])

    centroid, zero, perm_m, perm_s, signv = [], [], [], [], []
    for f in fractions:
        fs = f"{int(f)}%"
        r = frac[frac["fraction"].astype(str) == fs].iloc[0]
        centroid.append(float(r["mu8_acc"]) * 100)
        zero.append(float(r["zero_acc"]) * 100)

        p = comp[
            (comp["family"] == "coord_perm")
            & (comp["fraction"].astype(str) == fs)
            & (comp["control"].astype(str).str.contains("seed_"))
        ]
        pm, ps = _mean_std(p, "acc_control")
        perm_m.append(pm * 100)
        perm_s.append(ps * 100)

        s = sign[
            (sign["fraction"].astype(str) == fs)
            & (sign["flip_condition"].astype(str) == "sign_flip_100%")
        ]
        signv.append(float(s["accuracy"].iloc[0]) * 100)

    return {
        "x": fractions,
        "centroid": np.asarray(centroid),
        "zero": np.asarray(zero),
        "perm": np.asarray(perm_m),
        "perm_sd": np.asarray(perm_s),
        "sign": np.asarray(signv),
    }


def _geometry_v1(model: str) -> Dict[str, np.ndarray]:
    stem = "vitb" if model == "vit_base" else "dinov2"
    g = _read(f"outputs/fungibility_v1/{stem}_geometry_results.csv")
    fdf = _read(f"outputs/fungibility_v1/{stem}_fraction_results.csv")
    fractions = np.array([25.0, 50.0, 75.0])
    centroid, zero, perm_m, perm_s, signv = [], [], [], [], []
    for f in fractions:
        ff = f / 100.0
        c = g[(np.isclose(g["fraction"].astype(float), ff)) & (g["condition"] == "CENTROID")]
        centroid.append(float(c["top1_accuracy"].iloc[0]) * 100)

        z = fdf[(np.isclose(fdf["fraction"].astype(float), ff)) & (fdf["condition"] == "ZERO")]
        zero.append(float(z["top1_accuracy"].iloc[0]) * 100)

        p = g[(np.isclose(g["fraction"].astype(float), ff)) & (g["condition"] == "COORDINATE_PERMUTED_CENTROID")]
        pm, ps = _mean_std(p, "top1_accuracy")
        perm_m.append(pm * 100)
        perm_s.append(ps * 100)

        s = g[(np.isclose(g["fraction"].astype(float), ff)) & (g["condition"] == "SIGN_FLIPPED_CENTROID")]
        signv.append(float(s["top1_accuracy"].iloc[0]) * 100)

    return {
        "x": fractions,
        "centroid": np.asarray(centroid),
        "zero": np.asarray(zero),
        "perm": np.asarray(perm_m),
        "perm_sd": np.asarray(perm_s),
        "sign": np.asarray(signv),
    }


def plot_main_04_geometry() -> None:
    data = {
        "deit_tiny": _geometry_deit("deit_tiny"),
        "deit_small": _geometry_deit("deit_small"),
        "vit_base": _geometry_v1("vit_base"),
        "dinov2": _geometry_v1("dinov2"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        d = data[model]
        ax.plot(d["x"], d["centroid"], color=C["centroid"], marker="s", label="Centroid")
        ax.errorbar(
            d["x"], d["perm"], yerr=d["perm_sd"], color=C["permute"], marker="D",
            capsize=2.5, linewidth=1.6, label="Coordinate permutation"
        )
        ax.plot(d["x"], d["sign"], color=C["sign"], marker="x", linestyle=":", label="Sign inversion")
        ax.plot(d["x"], d["zero"], color=C["zero"], marker="o", linestyle="--", linewidth=1.4, label="Zero")
        ax.set_title(MODEL_LABEL[model])
        ax.set_xticks([25, 50, 75])
        ax.set_ylim(0, 82)
        if i >= 2:
            ax.set_xlabel("Spatial patches replaced (%)")
        if i % 2 == 0:
            ax.set_ylabel("Top-1 accuracy (%)")
        style_axis(ax, "y")
        panel_label(ax, f"({chr(97+i)})")

    handles = [
        Line2D([0], [0], color=C["centroid"], marker="s", label="Centroid"),
        Line2D([0], [0], color=C["permute"], marker="D", label="Coordinate permutation"),
        Line2D([0], [0], color=C["sign"], marker="x", linestyle=":", label="Sign inversion"),
        Line2D([0], [0], color=C["zero"], marker="o", linestyle="--", label="Zero"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.16, hspace=0.34, wspace=0.20)
    save_figure(fig, MAIN_DIR / "fig04_geometry_constraint")


# -----------------------------------------------------------------------------
# Main Figure 5: diversity constraint
# -----------------------------------------------------------------------------

def _diversity_summary() -> pd.DataFrame:
    rows = []

    old = _read("outputs/fungibility_v0_8/shared_vs_independent_results.csv")
    for long_model, key in MODEL_LONG_TO_KEY.items():
        s = old[old["model"] == long_model]
        static = s[s["condition_type"] == "static_centroid"]
        shared = s[s["condition_type"] == "shared_noise"]
        indep = s[s["condition_type"] == "independent_noise"]
        for name, sub in [("Static centroid", static), ("Shared Gaussian", shared), ("Independent Gaussian", indep)]:
            am, asd = _mean_std(sub, "accuracy")
            mm, msd = _mean_std(sub, "mean_margin")
            rows.append(dict(model=key, condition=name, acc=am*100, acc_sd=asd*100, margin=mm, margin_sd=msd))

    for key, stem in [("vit_base", "vitb"), ("dinov2", "dinov2")]:
        d = _read(f"outputs/fungibility_v1/{stem}_diversity_results.csv")
        mapping = {
            "STATIC_CENTROID": "Static centroid",
            "SHARED_GAUSSIAN": "Shared Gaussian",
            "INDEPENDENT_GAUSSIAN": "Independent Gaussian",
        }
        for raw, name in mapping.items():
            sub = d[d["condition"] == raw]
            am, asd = _mean_std(sub, "top1_accuracy")
            mm, msd = _mean_std(sub, "mean_margin")
            rows.append(dict(model=key, condition=name, acc=am*100, acc_sd=asd*100, margin=mm, margin_sd=msd))
    return pd.DataFrame(rows)


def plot_main_05_diversity() -> None:
    df = _diversity_summary()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 3.35), gridspec_kw={"width_ratios": [1.65, 1.0]})

    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    conditions = [
        ("Static centroid", C["clean"], ".."),
        ("Shared Gaussian", C["shared"], "////"),
        ("Independent Gaussian", C["independent"], ""),
    ]
    for j, (cond, color, hatch) in enumerate(conditions):
        vals, errs = [], []
        for m in MODEL_ORDER:
            r = df[(df["model"] == m) & (df["condition"] == cond)].iloc[0]
            vals.append(r["acc"])
            errs.append(r["acc_sd"])
        bars = ax1.bar(
            x + (j - 1) * width, vals, width=width, yerr=errs, capsize=2.2,
            color=color, alpha=0.92, edgecolor="white", linewidth=0.6, hatch=hatch, label=cond, zorder=3
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Tiny", "Small", "ViT-B", "DINOv2"])
    ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.set_ylim(0, 52)
    ax1.legend(frameon=False, loc="upper left")
    style_axis(ax1, "y")
    panel_label(ax1, "(a)")
    ax1.set_title("Complete patch-stream replacement")

    gains = []
    for m in MODEL_ORDER:
        sh = float(df[(df["model"] == m) & (df["condition"] == "Shared Gaussian")]["margin"].iloc[0])
        ind = float(df[(df["model"] == m) & (df["condition"] == "Independent Gaussian")]["margin"].iloc[0])
        gains.append(ind - sh)
    bars = ax2.bar(x, gains, color=C["independent"], width=0.62, zorder=3)
    ax2.axhline(0, color=C["clean"], linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Tiny", "Small", "ViT-B", "DINOv2"], rotation=25, ha="right")
    ax2.set_ylabel("Independent − shared\ntrue-class margin")
    style_axis(ax2, "y")
    panel_label(ax2, "(b)")
    ax2.set_title("Diversity gain")
    _bar_value_labels(ax2, bars, fmt="+{:.2f}", dy=0.04, fontsize=7.0)

    fig.subplots_adjust(wspace=0.34, bottom=0.20)
    save_figure(fig, MAIN_DIR / "fig05_diversity_constraint")


# -----------------------------------------------------------------------------
# Main Figure 6: learned low-dimensional variation
# -----------------------------------------------------------------------------

def _lowd_summary() -> pd.DataFrame:
    rows = []

    pc = _read("outputs/fungibility_v0_9/pc_identity_results.csv")
    rand = _read("outputs/fungibility_v0_9/random_direction_results.csv")
    div = _read("outputs/fungibility_v0_8/shared_vs_independent_results.csv")
    for long_model, key in MODEL_LONG_TO_KEY.items():
        static = div[(div["model"] == long_model) & (div["condition_type"] == "static_centroid")]
        sam, sasd = _mean_std(static, "accuracy")
        smm, smsd = _mean_std(static, "mean_margin")
        rows.append(dict(model=key, condition="Static centroid", acc=sam*100, acc_sd=sasd*100, margin=smm, margin_sd=smsd))

        for idx, name in [(1, "Natural PC1"), (2, "Natural PC2")]:
            s = pc[(pc["model"] == long_model) & (pc["condition_type"] == "natural") & (pc["pc_index"] == idx)]
            am, asd = _mean_std(s, "accuracy")
            mm, msd = _mean_std(s, "mean_margin")
            rows.append(dict(model=key, condition=name, acc=am*100, acc_sd=asd*100, margin=mm, margin_sd=msd))

        r = rand[rand["model"] == long_model]
        am, asd = _mean_std(r, "accuracy")
        mm, msd = _mean_std(r, "mean_margin")
        rows.append(dict(model=key, condition="Random 1D", acc=am*100, acc_sd=asd*100, margin=mm, margin_sd=msd))

    for key, stem in [("vit_base", "vitb"), ("dinov2", "dinov2")]:
        d = _read(f"outputs/fungibility_v1/{stem}_1d_results.csv")
        mapping = {
            "STATIC_CENTROID": "Static centroid",
            "NATURAL_PC1": "Natural PC1",
            "NATURAL_PC2": "Natural PC2",
            "RANDOM_1D": "Random 1D",
        }
        for raw, name in mapping.items():
            s = d[d["condition"] == raw]
            am, asd = _mean_std(s, "top1_accuracy")
            mm, msd = _mean_std(s, "mean_margin")
            rows.append(dict(model=key, condition=name, acc=am*100, acc_sd=asd*100, margin=mm, margin_sd=msd))
    return pd.DataFrame(rows)


def plot_main_06_lowd() -> None:
    df = _lowd_summary()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 3.35), gridspec_kw={"width_ratios": [1.7, 1.0]})

    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    conditions = [
        ("Static centroid", C["clean"]),
        ("Natural PC1", C["pc1"]),
        ("Natural PC2", C["pc2"]),
        ("Random 1D", C["random"]),
    ]
    for j, (cond, color) in enumerate(conditions):
        vals, errs = [], []
        for m in MODEL_ORDER:
            r = df[(df["model"] == m) & (df["condition"] == cond)].iloc[0]
            vals.append(r["acc"])
            errs.append(r["acc_sd"])
        ax1.bar(
            x + (j - 1.5) * width, vals, width=width, yerr=errs, capsize=2.0,
            color=color, edgecolor="white", linewidth=0.5, label=cond, zorder=3
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Tiny", "Small", "ViT-B", "DINOv2"])
    ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.set_ylim(0, 48)
    ax1.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax1, "y")
    panel_label(ax1, "(a)")
    ax1.set_title("Accuracy under 100% replacement")

    gains = []
    for m in MODEL_ORDER:
        pc1 = float(df[(df["model"] == m) & (df["condition"] == "Natural PC1")]["margin"].iloc[0])
        rnd = float(df[(df["model"] == m) & (df["condition"] == "Random 1D")]["margin"].iloc[0])
        gains.append(pc1 - rnd)
    bars = ax2.bar(x, gains, color=C["pc1"], width=0.62, zorder=3)
    ax2.axhline(0, color=C["clean"], linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Tiny", "Small", "ViT-B", "DINOv2"], rotation=25, ha="right")
    ax2.set_ylabel("PC1 − random\ntrue-class margin")
    style_axis(ax2, "y")
    panel_label(ax2, "(b)")
    ax2.set_title("PC1 vs random")
    for bar, val in zip(bars, gains):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            val + (0.04 if val >= 0 else -0.05),
            f"{val:+.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=7.0,
        )

    fig.subplots_adjust(wspace=0.34, bottom=0.20)
    save_figure(fig, MAIN_DIR / "fig06_lowdim_direction")


# -----------------------------------------------------------------------------
# Supplement S1: mask-seed robustness
# -----------------------------------------------------------------------------

def plot_supp_s1_mask_robustness() -> None:
    seed = _read("outputs/fungibility_dense_fraction/summary_by_mask_seed.csv")
    across = _read("outputs/fungibility_dense_fraction/summary_across_masks.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.1), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        depth = MODEL_DEPTH[model]
        s = seed[(seed["model"] == model) & (seed["depth"] == depth) & (seed["condition"] == "CENTROID")]
        for _, ss in s.groupby("mask_seed"):
            ss = ss.sort_values("actual_fraction")
            ax.plot(ss["actual_percent"], ss["top1_acc"] * 100, color="#AAB7C4", linewidth=0.9, alpha=0.55)

        m = across[(across["model"] == model) & (across["depth"] == depth) & (across["condition"] == "CENTROID")].sort_values("actual_fraction")
        z = across[(across["model"] == model) & (across["depth"] == depth) & (across["condition"] == "ZERO")].sort_values("actual_fraction")
        ax.plot(m["actual_percent"], m["mean_acc"] * 100, color=C["centroid"], linewidth=2.2, label="Centroid mean")
        ax.plot(z["actual_percent"], z["mean_acc"] * 100, color=C["zero"], linestyle="--", linewidth=1.4, label="Zero mean")

        ax.set_title(f"{MODEL_LABEL[model]} · depth {depth}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 82)
        if i >= 2:
            ax.set_xlabel("Spatial patches replaced (%)")
        if i % 2 == 0:
            ax.set_ylabel("Top-1 accuracy (%)")
        style_axis(ax, "both")
        panel_label(ax, f"({chr(97+i)})")

    handles = [
        Line2D([0], [0], color="#AAB7C4", linewidth=1, label="Individual mask seed"),
        Line2D([0], [0], color=C["centroid"], linewidth=2.2, label="Centroid mean"),
        Line2D([0], [0], color=C["zero"], linestyle="--", label="Zero mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.15, hspace=0.34, wspace=0.20)
    save_figure(fig, SUPP_DIR / "figS01_mask_robustness")


# -----------------------------------------------------------------------------
# Supplement S2: retention thresholds
# -----------------------------------------------------------------------------

def plot_supp_s2_thresholds() -> None:
    df = _read("outputs/fungibility_dense_fraction/threshold_crossings.csv")
    mdf = df[df["mask_seed"].astype(str) == "MEAN"].copy()
    metrics = ["F95", "F90", "F80"]
    titles = ["95% retention", "90% retention", "80% retention"]
    conds = [("ZERO", C["zero"], "Zero"), ("CENTROID", C["centroid"], "Centroid"), ("DIAGONAL_GAUSSIAN", C["gaussian"], "Gaussian")]

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    for k, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
        for j, (cond, color, label) in enumerate(conds):
            vals, errs = [], []
            for m in MODEL_ORDER:
                r = mdf[(mdf["model"] == m) & (mdf["condition"] == cond) & (mdf["is_primary"] == True)]
                if len(r):
                    vals.append(float(r[metric].iloc[0]) * 100)
                    errs.append(float(r[f"{metric}_sd"].iloc[0]) * 100)
                else:
                    vals.append(np.nan)
                    errs.append(0.0)
            ax.bar(x + (j-1)*width, vals, width=width, yerr=errs, capsize=1.8, color=color, label=label, zorder=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["Tiny", "Small", "ViT-B", "DINO"], rotation=30, ha="right")
        ax.set_ylim(0, 100)
        if k == 0:
            ax.set_ylabel("Max replacement fraction (%)")
        style_axis(ax, "y")
        panel_label(ax, f"({chr(97+k)})")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.6)
    fig.subplots_adjust(wspace=0.16, bottom=0.27)
    save_figure(fig, SUPP_DIR / "figS02_retention_thresholds")


# -----------------------------------------------------------------------------
# Supplement S3: natural vs energy-matched PCA rank
# -----------------------------------------------------------------------------

def plot_supp_s3_pca_rank() -> None:
    df = _read("outputs/fungibility_v0_9/natural_vs_energy_matched_rank.csv")
    models = [("deit_tiny_patch16_224", "DeiT-Tiny"), ("deit_small_patch16_224", "DeiT-Small")]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), sharey=False)
    for i, (ax, (m, title)) in enumerate(zip(axes, models)):
        d = df[df["model"] == m]
        for formulation, color, label, marker in [
            ("natural", C["pc1"], "Natural PCA", "o"),
            ("energy_matched", C["permute"], "Energy-matched PCA", "s"),
        ]:
            g = d[d["formulation"] == formulation].groupby("rank")["accuracy"].agg(["mean", "std"]).reset_index()
            x = g["rank"].astype(int).to_numpy()
            y = g["mean"].to_numpy() * 100
            sd = g["std"].fillna(0).to_numpy() * 100
            ax.plot(x, y, color=color, marker=marker, label=label)
            ax.fill_between(x, y-sd, y+sd, color=color, alpha=0.12, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1,2,4,8,16,32,64])
        ax.set_xticklabels(["1","2","4","8","16","32","64"])
        ax.set_xlabel("PCA rank")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_title(title)
        style_axis(ax, "both")
        panel_label(ax, f"({chr(97+i)})")
    axes[0].legend(frameon=False, loc="best")
    fig.subplots_adjust(wspace=0.28, bottom=0.20)
    save_figure(fig, SUPP_DIR / "figS03_pca_rank")


# -----------------------------------------------------------------------------
# Supplement S4: PC1 amplitude sensitivity
# -----------------------------------------------------------------------------

def plot_supp_s4_pc1_amplitude() -> None:
    df = _read("outputs/fungibility_v0_9/pc1_scale_sweep.csv")
    models = [("deit_tiny_patch16_224", "DeiT-Tiny"), ("deit_small_patch16_224", "DeiT-Small")]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), sharey=False)

    for i, (ax, (m, title)) in enumerate(zip(axes, models)):
        d = df[df["model"] == m].copy()
        # Standard numeric sweep.
        numeric = d[d["scale_label"].astype(str) != "E_MATCH"].groupby("scale_multiplier")["accuracy"].agg(["mean", "std"]).reset_index().sort_values("scale_multiplier")
        ax.errorbar(
            numeric["scale_multiplier"], numeric["mean"]*100, yerr=numeric["std"].fillna(0)*100,
            color=C["pc1"], marker="o", capsize=2.2, label="PC1 scale sweep"
        )
        em = d[d["scale_label"].astype(str) == "E_MATCH"]
        if len(em):
            xm = float(em["scale_multiplier"].iloc[0])
            ym = float(em["accuracy"].mean()) * 100
            ax.scatter([xm], [ym], marker="D", s=34, color=C["permute"], zorder=5, label="Energy-matched scale")
        ax.axvline(1.0, color=C["clean"], linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xticks([0.25,0.5,1,2,4])
        ax.set_xlabel("PC1 amplitude multiplier")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_title(title)
        style_axis(ax, "both")
        panel_label(ax, f"({chr(97+i)})")
    axes[0].legend(frameon=False, loc="best")
    fig.subplots_adjust(wspace=0.28, bottom=0.20)
    save_figure(fig, SUPP_DIR / "figS04_pc1_amplitude")


# -----------------------------------------------------------------------------
# Supplement S5: effective-rank propagation
# -----------------------------------------------------------------------------

def plot_supp_s5_rank_propagation() -> None:
    df = _read("outputs/fungibility_v0_9/rank_propagation.csv")
    models = [("deit_tiny_patch16_224", "DeiT-Tiny"), ("deit_small_patch16_224", "DeiT-Small")]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), sharey=False)
    conds = [
        ("clean", C["clean"], "Clean", "--"),
        ("static_centroid", C["zero"], "Static centroid", ":"),
        ("natural_pca_rank_1_seed_16001", C["pc1"], "Natural rank-1", "-"),
        ("diagonal_gaussian_seed_16001", C["gaussian"], "Diagonal Gaussian", "-"),
    ]
    for i, (ax, (m, title)) in enumerate(zip(axes, models)):
        d = df[df["model"] == m]
        for cond, color, label, ls in conds:
            s = d[d["condition"] == cond].sort_values("block_depth")
            if len(s):
                ax.plot(s["block_depth"], s["effective_rank"], color=color, linestyle=ls, marker="o", label=label)
        ax.set_xlabel("Downstream block depth")
        ax.set_ylabel("Effective rank")
        ax.set_xticks([8,9,10,11])
        ax.set_title(title)
        style_axis(ax, "y")
        panel_label(ax, f"({chr(97+i)})")
    axes[0].legend(frameon=False, fontsize=7.7, loc="upper right")
    fig.subplots_adjust(wspace=0.28, bottom=0.20)
    save_figure(fig, SUPP_DIR / "figS05_rank_propagation")


# -----------------------------------------------------------------------------
# Supplement S6: exact equivalence audit
# -----------------------------------------------------------------------------

def plot_supp_s6_equivalence() -> None:
    df = _read("outputs/fungibility_compression_poc/equivalence_results.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 3.0))
    model_colors = {
        "deit_tiny": "#4E79A7",
        "deit_small": "#59A14F",
        "vit_base": "#F28E2B",
        "dinov2": "#B07AA1",
    }
    for m in MODEL_ORDER:
        d = df[df["model"] == m].sort_values("actual_fraction")
        x = d["actual_fraction"] * 100
        ax1.plot(x, d["max_abs_logit_diff"], color=model_colors[m], marker="o", label=MODEL_LABEL[m])
        ax2.plot(x, d["prediction_agreement"] * 100, color=model_colors[m], marker="o", label=MODEL_LABEL[m])

    ax1.axhline(1e-4, color=C["zero"], linestyle="--", linewidth=1.0, label="$10^{-4}$ tolerance")
    ax1.set_yscale("log")
    ax1.set_xlabel("Replaced patches (%)")
    ax1.set_ylabel("Max absolute logit error")
    ax1.set_title("Numerical discrepancy")
    style_axis(ax1, "both")
    panel_label(ax1, "(a)")

    ax2.axhline(100, color=C["clean"], linestyle="--", linewidth=1.0)
    ax2.set_ylim(99.94, 100.02)
    ax2.set_yticks([99.95, 100.00])
    ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax2.set_xlabel("Replaced patches (%)")
    ax2.set_ylabel("Prediction agreement (%)")
    ax2.set_title("Argmax agreement")
    style_axis(ax2, "both")
    panel_label(ax2, "(b)")

    handles, labels = ax1.get_legend_handles_labels()
    # exclude tolerance from common model legend
    fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.30, bottom=0.23)
    save_figure(fig, SUPP_DIR / "figS06_carrier_equivalence")


# -----------------------------------------------------------------------------
# Supplement S7: accuracy vs downstream token count
# -----------------------------------------------------------------------------

def plot_supp_s7_token_count() -> None:
    df = _read("outputs/fungibility_compression_poc/baseline_comparison.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.1), sharex=False, sharey=False)
    axes = axes.ravel()
    conds = [
        ("WEIGHTED_CENTROID_CARRIER", C["centroid"], "Weighted carrier", "o"),
        ("UNWEIGHTED_CENTROID", C["permute"], "Unweighted centroid", "s"),
        ("IMAGE_MEAN_CARRIER", C["gaussian"], "Image-mean carrier", "^"),
        ("RANDOM_PRUNING", C["zero"], "Random pruning", "x"),
    ]
    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        d = df[(df["model"] == model) & (df["depth"] == MODEL_DEPTH[model])]
        agg = d.groupby(["condition", "downstream_patch_tokens"])["top1_acc"].agg(["mean", "std"]).reset_index()
        for cond, color, label, marker in conds:
            s = agg[agg["condition"] == cond].sort_values("downstream_patch_tokens")
            if len(s):
                ax.plot(s["downstream_patch_tokens"], s["mean"]*100, color=color, marker=marker, label=label)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel("Downstream spatial tokens")
        ax.set_ylabel("Top-1 accuracy (%)")
        style_axis(ax, "both")
        panel_label(ax, f"({chr(97+i)})")
    handles = [Line2D([0],[0], color=c, marker=m, label=l) for _,c,l,m in conds]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.16, hspace=0.36, wspace=0.24)
    save_figure(fig, SUPP_DIR / "figS07_accuracy_vs_tokens")


# -----------------------------------------------------------------------------
# Supplement S8: accuracy vs latency
# -----------------------------------------------------------------------------

def plot_supp_s8_latency() -> None:
    lat = _read("outputs/fungibility_compression_poc/latency_summary.csv")
    base = _read("outputs/fungibility_compression_poc/baseline_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), sharey=False)
    conds = [
        ("WEIGHTED_CENTROID_CARRIER", C["centroid"], "Weighted carrier"),
        ("UNWEIGHTED_CENTROID", C["permute"], "Unweighted centroid"),
        ("IMAGE_MEAN_CARRIER", C["gaussian"], "Image-mean carrier"),
        ("RANDOM_PRUNING", C["zero"], "Random pruning"),
    ]
    markers = {"deit_tiny":"o", "deit_small":"s", "vit_base":"D", "dinov2":"^"}

    for i, (ax, bs) in enumerate(zip(axes, [1,16])):
        for model in MODEL_ORDER:
            lm = lat[(lat["model"] == model) & (lat["batch_size"] == bs)]
            bm = base[(base["model"] == model) & (base["depth"] == MODEL_DEPTH[model])]
            if len(lm) == 0 or len(bm) == 0:
                continue
            clean_lat = float(lm["clean_latency_ms"].iloc[0])
            clean_acc = float(bm[bm["actual_k"] == 0]["top1_acc"].mean()) * 100
            ax.scatter(clean_lat, clean_acc, marker=markers[model], s=38, facecolors="white", edgecolors=C["clean"], linewidths=1.1, zorder=5)
            for cond, color, _ in conds:
                r = lm[lm["condition"] == cond]
                if len(r) == 0:
                    continue
                k = int(r["k_replaced"].iloc[0])
                acc = bm[(bm["condition"] == cond) & (bm["actual_k"] == k)]["top1_acc"]
                if len(acc) == 0:
                    continue
                ax.scatter(float(r["compressed_latency_ms"].iloc[0]), float(acc.mean())*100,
                           marker=markers[model], s=34, color=color, zorder=4)
        ax.set_title(f"Batch size {bs}")
        ax.set_xlabel("End-to-end latency (ms)")
        ax.set_ylabel("Top-1 accuracy (%)")
        style_axis(ax, "both")
        panel_label(ax, f"({chr(97+i)})")

    cond_handles = [Line2D([0],[0], marker="o", linestyle="", color=color, label=label) for _,color,label in conds]
    model_handles = [Line2D([0],[0], marker=markers[m], linestyle="", markerfacecolor="white",
                            markeredgecolor=C["clean"], color=C["clean"], label=MODEL_LABEL[m]) for m in MODEL_ORDER]
    leg1 = axes[0].legend(handles=cond_handles, frameon=False, fontsize=7.0, loc="lower left")
    axes[0].add_artist(leg1)
    axes[1].legend(handles=model_handles, frameon=False, fontsize=7.0, loc="lower left")
    fig.subplots_adjust(wspace=0.30, bottom=0.20)
    save_figure(fig, SUPP_DIR / "figS08_accuracy_vs_latency")


# -----------------------------------------------------------------------------
# Supplement S9: geometry-bank falsification
# -----------------------------------------------------------------------------

def plot_supp_s9_geometry_bank() -> None:
    df = _read("outputs/fungibility_geometry_bank/delta_vs_pruning.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), sharey=True)
    axes = axes.ravel()
    family = [
        ("REAL_PLUS_CENTROID", 1, C["centroid"], "Centroid K=1", "o"),
        ("REAL_PLUS_PCA_BANK", 4, "#D08C28", "PCA K=4", "s"),
        ("REAL_PLUS_PCA_BANK", 8, "#B56A2D", "PCA K=8", "D"),
        ("REAL_PLUS_PCA_BANK", 16, "#8C4C20", "PCA K=16", "^"),
        ("REAL_PLUS_KMEANS_BANK", 4, "#8E5AA7", "K-means K=4", "s"),
        ("REAL_PLUS_KMEANS_BANK", 8, "#6F4786", "K-means K=8", "D"),
        ("REAL_PLUS_KMEANS_BANK", 16, "#533665", "K-means K=16", "^"),
    ]
    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        d = df[df["model"] == model]
        for cond, k, color, label, marker in family:
            s = d[(d["condition"] == cond) & (d["K_synth"] == k)].sort_values("budget_B")
            if len(s):
                ax.plot(s["budget_B"], s["mean_delta_acc"]*100, color=color, marker=marker,
                        linewidth=1.25, markersize=3.8, label=label)
        ax.axhline(0, color=C["clean"], linewidth=1.0)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel("Downstream token budget")
        if i % 2 == 0:
            ax.set_ylabel("Δ top-1 vs random pruning (pp)")
        style_axis(ax, "both")
        panel_label(ax, f"({chr(97+i)})")
    axes[0].legend(frameon=False, fontsize=6.8, ncol=2, loc="lower left")
    fig.subplots_adjust(hspace=0.36, wspace=0.20)
    save_figure(fig, SUPP_DIR / "figS09_geometry_bank_vs_pruning")


def generate_all() -> None:
    apply_style()
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    plot_main_01_schematic()
    plot_main_02_depth_emergence()
    plot_main_03_dense_fraction()
    plot_main_04_geometry()
    plot_main_05_diversity()
    plot_main_06_lowd()

    plot_supp_s1_mask_robustness()
    plot_supp_s2_thresholds()
    plot_supp_s3_pca_rank()
    plot_supp_s4_pc1_amplitude()
    plot_supp_s5_rank_propagation()
    plot_supp_s6_equivalence()
    plot_supp_s7_token_count()
    plot_supp_s8_latency()
    plot_supp_s9_geometry_bank()


if __name__ == "__main__":
    generate_all()
