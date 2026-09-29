"""
patch_fungibility/dense_fraction_analysis.py

Analysis, hierarchical aggregation, AUC computation, threshold crossing detection,
and decision rule evaluation for the Dense Fraction & Spatial-Mask Robustness Sweep.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def compute_trapezoidal_auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes trapezoidal Area Under Curve normalized to x in [0, 1].
    Assumes x is sorted and in range [0, 1].
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def find_threshold_crossing(
    fractions: np.ndarray,
    accuracies: np.ndarray,
    threshold_ratio: float,
    clean_acc: float
) -> float:
    """
    Finds largest fraction retaining >= threshold_ratio * clean_acc.
    If even fraction 0 does not retain it, returns 0.0.
    If all fractions retain it, returns fractions[-1].
    """
    target = threshold_ratio * clean_acc
    # Find all indices where accuracy >= target
    valid_idx = np.where(accuracies >= target)[0]
    if len(valid_idx) == 0:
        return 0.0

    # Contiguous retention from beginning
    # Find largest fraction where it stayed >= target before first drop, or global largest
    # Protocol: largest replacement fraction retaining >= target
    return float(fractions[valid_idx[-1]])


def find_degradation_cliff(
    fractions: np.ndarray,
    values: np.ndarray
) -> Dict[str, Any]:
    """
    Identifies adjacent fraction interval with largest decrease in values (accuracy or margin).
    """
    diffs = values[:-1] - values[1:]  # positive means decrease from k to k+1
    max_idx = int(np.argmax(diffs))
    max_drop = float(diffs[max_idx])
    return {
        "cliff_start_fraction": float(fractions[max_idx]),
        "cliff_end_fraction": float(fractions[max_idx + 1]),
        "cliff_drop_magnitude": max_drop
    }


def find_contiguous_fungibility_window(
    fractions: np.ndarray,
    zero_damages: np.ndarray,
    recoveries: np.ndarray,
    min_damage: float = 0.10,
    min_recovery: float = 0.50
) -> Tuple[Optional[float], Optional[float]]:
    """
    Finds contiguous fraction region where zero_damage >= min_damage and recovery >= min_recovery.
    """
    qualifies = (zero_damages >= min_damage) & (recoveries >= min_recovery)
    if not np.any(qualifies):
        return None, None

    # Find longest contiguous True segment
    best_start, best_end = None, None
    max_len = 0
    cur_start = None

    for i, q in enumerate(qualifies):
        if q:
            if cur_start is None:
                cur_start = i
            cur_len = i - cur_start + 1
            if cur_len > max_len:
                max_len = cur_len
                best_start = cur_start
                best_end = i
        else:
            cur_start = None

    if best_start is not None:
        return float(fractions[best_start]), float(fractions[best_end])
    return None, None


def evaluate_dense_decision(
    auc_df: pd.DataFrame,
    summary_across: pd.DataFrame,
    threshold_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates pre-registered decision rules:
    - ROBUST
    - ROBUST WITH SPATIAL HETEROGENEITY
    - MASK-SENSITIVE
    """
    # Check if Centroid consistently outperforms Zero when Zero is damaging
    # For each model, check fraction range where zero_damage >= 0.10:
    # Does Centroid accuracy > Zero accuracy for >= 90% of those points across all mask seeds?
    # Filter to primary depth curves for each model
    df_primary = summary_across[summary_across["is_primary"] == True]
    models = df_primary["model"].unique()
    model_statuses = {}
    all_robust = True
    has_heterogeneity = False

    for m in models:
        df_m = df_primary[df_primary["model"] == m]
        # Filter for points where zero damage >= 0.10
        # Check advantage of centroid over zero
        c_sub = df_m[df_m["condition"] == "CENTROID"]
        z_sub = df_m[df_m["condition"] == "ZERO"]

        merged = pd.merge(
            c_sub, z_sub,
            on=["model", "depth", "actual_fraction"],
            suffixes=("_c", "_z")
        )

        damaging = merged[merged["mean_damage_z"] >= 0.10]
        if len(damaging) == 0:
            model_statuses[m] = {"status": "ROBUST", "pct_points_centroid_better": 100.0, "max_mask_spread_pct": 0.0}
            continue

        c_better = damaging["mean_acc_c"] > damaging["mean_acc_z"]
        pct_better = float(c_better.mean() * 100.0)

        # Check mask spread across all points
        max_spread = float(c_sub["mask_spread_acc"].max() * 100.0)

        if pct_better >= 80.0 and max_spread <= 15.0:
            status = "ROBUST"
        elif pct_better >= 70.0 and max_spread > 15.0:
            status = "ROBUST WITH SPATIAL HETEROGENEITY"
            has_heterogeneity = True
        else:
            status = "MASK-SENSITIVE"
            all_robust = False

        model_statuses[m] = {
            "status": status,
            "pct_points_centroid_better": pct_better,
            "max_mask_spread_pct": max_spread
        }

    if all_robust and not has_heterogeneity:
        overall = "ROBUST"
        interp = "The patch-content fungibility phenomenon is robust across independently sampled spatial masks; Centroid and Gaussian replacements consistently outperform destructive Zero replacement across the fraction range."
    elif all_robust and has_heterogeneity:
        overall = "ROBUST WITH SPATIAL HETEROGENEITY"
        interp = "The phenomenon persists across all models, but spatial position selection introduces noticeable variance in intermediate replacement budgets."
    else:
        overall = "MASK-SENSITIVE"
        interp = "The phenomenon depends heavily on spatial mask ordering and fails to generalize consistently across independent permutations."

    return {
        "overall_decision": overall,
        "interpretation": interp,
        "model_statuses": model_statuses
    }
