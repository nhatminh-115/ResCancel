"""
patch_fungibility/v1_decision.py

Statistical testing, FDR corrections, signature evaluations, and outcome classification
for Patch Fungibility V1:
- Paired statistical comparison (Bootstrap CI, Paired t-test, Wilcoxon, Cohen's dz, McNemar)
- Benjamini-Hochberg FDR correction within families
- Signature A (Content Fungibility)
- Signature B (Geometry Constraint)
- Signature C (Token Diversity)
- Pre-registered Outcomes: OUTCOME A, OUTCOME B, OUTCOME C, OUTCOME D
- Paper Claim Levels: LEVEL 1, LEVEL 2, LEVEL 3
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_mcnemar_exact(corr_a: np.ndarray, corr_b: np.ndarray) -> Tuple[int, int, float]:
    """
    Computes exact two-sided binomial McNemar test between binary outcome arrays.
    b: a correct, b incorrect
    c: a incorrect, b correct
    """
    b = int(np.sum((corr_a == 1) & (corr_b == 0)))
    c = int(np.sum((corr_a == 0) & (corr_b == 1)))
    n_discordant = b + c
    if n_discordant == 0:
        return b, c, 1.0
    res = stats.binomtest(b, n_discordant, p=0.5, alternative="two-sided")
    return b, c, float(res.pvalue)


def compute_paired_statistics(
    margins_a: np.ndarray,
    margins_b: np.ndarray,
    corrects_a: np.ndarray,
    corrects_b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired statistical tests for condition A vs condition B:
    diff = margins_a - margins_b (positive means A has higher margin than B).
      - mean margin diff
      - 95% bootstrap CI
      - paired t-test
      - Wilcoxon signed-rank test
      - Cohen's dz
      - McNemar test on accuracy
    """
    diffs = margins_a - margins_b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    n = len(diffs)

    cohen_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    if np.all(np.abs(diffs) < 1e-12) or std_diff < 1e-12:
        t_stat, t_pval = 0.0, 1.0
        w_stat, w_pval = 0.0, 1.0
    else:
        t_res = stats.ttest_rel(margins_a, margins_b)
        t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)
        try:
            w_res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = 0.0, 1.0

    rng = np.random.RandomState(seed)
    boot_indices = rng.choice(n, size=(n_boot, n), replace=True)
    boot_means = np.mean(diffs[boot_indices], axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    b_disc, c_disc, mcnemar_p = compute_mcnemar_exact(corrects_a, corrects_b)
    acc_a = float(np.mean(corrects_a))
    acc_b = float(np.mean(corrects_b))

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "t_pval": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pval": w_pval,
        "cohen_dz": cohen_dz,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "acc_diff": acc_a - acc_b,
        "mcnemar_b": b_disc,
        "mcnemar_c": c_disc,
        "mcnemar_pval": mcnemar_p
    }


def apply_bh_fdr(p_values: List[float]) -> List[float]:
    """
    Applies Benjamini-Hochberg False Discovery Rate correction to a list of p-values.
    """
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adjusted = np.zeros(m)

    for i in range(m - 1, -1, -1):
        rank = i + 1
        adj = sorted_p[i] * m / rank
        if i < m - 1:
            adj = min(adj, adjusted[sorted_indices[i + 1]])
        adjusted[sorted_indices[i]] = min(adj, 1.0)

    return adjusted.tolist()


def evaluate_model_signatures(
    model_name: str,
    depth_results: pd.DataFrame,
    fraction_geometry_results: pd.DataFrame,
    diversity_results: pd.DataFrame,
    image_results: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Evaluates Signature A (Content Fungibility), Signature B (Geometry Constraint),
    and Signature C (Token Diversity) for a specific model according to protocol rules.
    """
    res = {
        "model_name": model_name,
        "signature_a": False,
        "signature_a_details": {},
        "signature_b": False,
        "signature_b_details": {},
        "signature_c": False,
        "signature_c_details": {},
        "secondary_depth_candidate": None
    }

    # Clean baseline margin & acc
    clean_df = image_results["CLEAN"]
    clean_margins = clean_df["true_class_margin"].values
    clean_corrects = clean_df["correctness"].values
    clean_acc = float(np.mean(clean_corrects))

    # --- SIGNATURE A: Content Fungibility in late depth {7, 8, 9} ---
    # Damage_zero >= 0.10 and Recovery >= 0.50, bootstrap CI for (Margin_repl - Margin_zero) > 0
    sig_a_candidates = {}
    best_advantage = -1e9
    best_adapted_depth = None

    for d in [5, 7, 8, 9, 10]:
        z_key = f"ZERO_depth_{d}"
        c_key = f"CENTROID_depth_{d}"
        g_key_mean = f"GAUSSIAN_depth_{d}_mean"  # or check seeds

        if z_key not in image_results:
            continue

        z_df = image_results[z_key]
        z_margins = z_df["true_class_margin"].values
        z_damage = float(np.mean(clean_margins - z_margins))

        c_df = image_results[c_key]
        c_margins = c_df["true_class_margin"].values
        c_damage = float(np.mean(clean_margins - c_margins))

        c_recovery = (z_damage - c_damage) / z_damage if z_damage > 1e-6 else 0.0

        # Also check Gaussian (using seed 22001 or mean across seeds)
        g_key = f"GAUSSIAN_depth_{d}_seed_22001"
        if g_key in image_results:
            g_df = image_results[g_key]
            g_margins = g_df["true_class_margin"].values
            g_damage = float(np.mean(clean_margins - g_margins))
            g_recovery = (z_damage - g_damage) / z_damage if z_damage > 1e-6 else 0.0
        else:
            g_recovery = 0.0
            g_margins = None

        # Check secondary depth selection rule among {7, 8, 9, 10}:
        # max Fungibility Advantage = Damage_zero - Damage_centroid subject to Damage_zero >= 0.10
        if d in [7, 8, 9, 10] and z_damage >= 0.10:
            advantage = z_damage - c_damage
            if advantage > best_advantage:
                best_advantage = advantage
                best_adapted_depth = d

        # Check Signature A criteria at depth 7, 8, or 9
        if d in [7, 8, 9]:
            # Statistical test for Centroid vs Zero
            stats_c_vs_z = compute_paired_statistics(
                c_margins, z_margins, c_df["correctness"].values, z_df["correctness"].values
            )
            # Recovery >= 50% and CI lower > 0
            centroid_qualifies = (z_damage >= 0.10) and (c_recovery >= 0.50) and (stats_c_vs_z["ci_lower"] > 0)

            gaussian_qualifies = False
            if g_margins is not None:
                stats_g_vs_z = compute_paired_statistics(
                    g_margins, z_margins, g_df["correctness"].values, z_df["correctness"].values
                )
                gaussian_qualifies = (z_damage >= 0.10) and (g_recovery >= 0.50) and (stats_g_vs_z["ci_lower"] > 0)

            qualifies = centroid_qualifies or gaussian_qualifies
            sig_a_candidates[d] = {
                "depth": d,
                "damage_zero": z_damage,
                "damage_centroid": c_damage,
                "recovery_centroid": c_recovery,
                "centroid_ci_lower": stats_c_vs_z["ci_lower"],
                "centroid_qualifies": centroid_qualifies,
                "gaussian_qualifies": gaussian_qualifies,
                "qualifies": qualifies
            }
            if qualifies:
                res["signature_a"] = True

    res["signature_a_details"] = sig_a_candidates
    res["secondary_depth_candidate"] = best_adapted_depth

    # --- SIGNATURE B: Geometry Constraint at Depth 8 (or adapted depth) ---
    # Correct centroid must outperform coordinate-permuted OR sign-flipped centroid with dz >= 0.20
    cent_d8_key = "CENTROID_d8_f0.25" if "CENTROID_d8_f0.25" in image_results else "CENTROID_depth_8"
    cent_d8_df = image_results[cent_d8_key]
    cent_d8_margins = cent_d8_df["true_class_margin"].values
    cent_d8_corrects = cent_d8_df["correctness"].values

    sign_flip_key = "SIGN_FLIPPED_CENTROID_d8_f0.25" if "SIGN_FLIPPED_CENTROID_d8_f0.25" in image_results else "SIGN_FLIPPED_CENTROID_depth_8"
    sign_flip_df = image_results[sign_flip_key]
    stats_c_vs_sign = compute_paired_statistics(
        cent_d8_margins, sign_flip_df["true_class_margin"].values,
        cent_d8_corrects, sign_flip_df["correctness"].values
    )

    # Permuted centroid seeds
    perm_dzes = []
    perm_acc_diffs = []
    for s in [23001, 23002, 23003]:
        pk = f"COORD_PERM_d8_f0.25_s{s}" if f"COORD_PERM_d8_f0.25_s{s}" in image_results else f"COORDINATE_PERMUTED_CENTROID_depth_8_s{s}"
        if pk in image_results:
            p_df = image_results[pk]
            p_stats = compute_paired_statistics(
                cent_d8_margins, p_df["true_class_margin"].values,
                cent_d8_corrects, p_df["correctness"].values
            )
            perm_dzes.append(p_stats["cohen_dz"])
            perm_acc_diffs.append(p_stats["acc_diff"])

    mean_perm_dz = float(np.mean(perm_dzes)) if perm_dzes else 0.0
    sign_dz = stats_c_vs_sign["cohen_dz"]

    sig_b_passes = (sign_dz >= 0.20) or (mean_perm_dz >= 0.20)
    res["signature_b"] = bool(sig_b_passes)
    res["signature_b_details"] = {
        "sign_flip_dz": sign_dz,
        "sign_flip_acc_diff": stats_c_vs_sign["acc_diff"],
        "mean_permuted_dz": mean_perm_dz,
        "permuted_acc_diffs": perm_acc_diffs,
        "passed": bool(sig_b_passes)
    }

    # --- SIGNATURE C: Token Diversity at 100% Replacement ---
    # Independent Gaussian must outperform Shared Gaussian with dz >= 0.20
    # and accuracy gain >= 5 percentage points OR equivalently strong margin effect (mean margin diff >= 0.25)
    # Compare seed-by-seed then average
    div_dzes = []
    div_acc_diffs = []
    div_mean_margin_diffs = []

    for s in [24001, 24002, 24003, 24004, 24005]:
        indep_k = f"INDEPENDENT_GAUSSIAN_s{s}"
        shared_k = f"SHARED_GAUSSIAN_s{s}"
        if indep_k in image_results and shared_k in image_results:
            i_df = image_results[indep_k]
            s_df = image_results[shared_k]
            d_stats = compute_paired_statistics(
                i_df["true_class_margin"].values, s_df["true_class_margin"].values,
                i_df["correctness"].values, s_df["correctness"].values
            )
            div_dzes.append(d_stats["cohen_dz"])
            div_acc_diffs.append(d_stats["acc_diff"])
            div_mean_margin_diffs.append(d_stats["mean_diff"])

    mean_div_dz = float(np.mean(div_dzes)) if div_dzes else 0.0
    mean_div_acc_gain = float(np.mean(div_acc_diffs)) if div_acc_diffs else 0.0
    mean_div_margin_diff = float(np.mean(div_mean_margin_diffs)) if div_mean_margin_diffs else 0.0

    sig_c_passes = (mean_div_dz >= 0.20) and ((mean_div_acc_gain >= 0.05) or (mean_div_margin_diff >= 0.25))
    res["signature_c"] = bool(sig_c_passes)
    res["signature_c_details"] = {
        "mean_cohen_dz": mean_div_dz,
        "mean_acc_gain": mean_div_acc_gain,
        "mean_margin_diff": mean_div_margin_diff,
        "seed_dzes": div_dzes,
        "seed_acc_gains": div_acc_diffs,
        "passed": bool(sig_c_passes)
    }

    return res


def evaluate_v1_decision(
    vitb_eval: Dict[str, Any],
    dinov2_eval: Dict[str, Any],
    validation_status: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates final V1 Outcome (A, B, C, D) and Paper Claim Level (1, 2, 3).
    """
    # Check if implementation validation passed
    all_validation_passed = validation_status.get("all_passed", False)
    if not all_validation_passed:
        return {
            "outcome": "OUTCOME D — UNINTERPRETABLE",
            "supported_level": "LEVEL 0 — UNINTERPRETABLE",
            "ready_for_paper": False,
            "interpretation": "Baseline manual forwards fail verification or implementation mismatch prevents causal interpretation."
        }

    vitb_a = vitb_eval["signature_a"]
    vitb_b = vitb_eval["signature_b"]
    vitb_c = vitb_eval["signature_c"]

    dino_a = dinov2_eval["signature_a"]
    dino_b = dinov2_eval["signature_b"]
    dino_c = dinov2_eval["signature_c"]

    vitb_all = vitb_a and vitb_b and vitb_c
    dino_all = dino_a and dino_b and dino_c

    if vitb_all and dino_all:
        outcome = "OUTCOME A — BROAD GENERALIZATION"
        supported_level = "LEVEL 3: Late patch-content fungibility generalizes across supervised and self-supervised ViTs."
        ready_for_paper = True
        interpretation = (
            "Late-layer patch content fungibility with geometric and diversity constraints "
            "generalizes across supervised and self-supervised Vision Transformers."
        )
    elif vitb_all or dino_all or (vitb_a and dino_a):
        outcome = "OUTCOME B — PARTIAL GENERALIZATION"
        ready_for_paper = True
        if vitb_all and not dino_all:
            supported_level = "LEVEL 2: Supervised ViTs show late patch-content fungibility."
            interpretation = "The phenomenon generalizes beyond DeiT to Supervised ViT-B, but self-supervised DINOv2 shows different downstream constraints."
        elif dino_all and not vitb_all:
            supported_level = "LEVEL 2: Self-supervised ViTs show late patch-content fungibility."
            interpretation = "The phenomenon generalizes to DINOv2, but Supervised ViT-B shows different constraints."
        else:
            supported_level = "LEVEL 2: Supervised and self-supervised ViTs show content fungibility, but specific constraints vary."
            interpretation = "The phenomenon generalizes beyond DeiT but downstream geometry/diversity constraints depend on training regime."
    elif (not vitb_a) and (not dino_a):
        outcome = "OUTCOME C — DEIT-FAMILY PHENOMENON"
        supported_level = "LEVEL 1: DeiT patch content is fungible."
        ready_for_paper = True
        interpretation = "The previously observed phenomenon is substantially specific to DeiT-style representations."
    else:
        outcome = "OUTCOME B — PARTIAL GENERALIZATION"
        supported_level = "LEVEL 2: Supervised ViTs show late patch-content fungibility."
        ready_for_paper = True
        interpretation = "Partial generalization observed across models."

    return {
        "outcome": outcome,
        "supported_level": supported_level,
        "ready_for_paper": ready_for_paper,
        "interpretation": interpretation,
        "vitb_signatures": {"A": vitb_a, "B": vitb_b, "C": vitb_c},
        "dinov2_signatures": {"A": dino_a, "B": dino_b, "C": dino_c}
    }
