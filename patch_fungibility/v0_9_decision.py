import math
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
    base_margins: np.ndarray,
    cond_margins: np.ndarray,
    base_corrects: np.ndarray,
    cond_corrects: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired statistical tests between baseline and condition:
      - mean margin diff (base - cond)
      - 95% bootstrap CI
      - paired t-test
      - Wilcoxon signed-rank test
      - Cohen's dz
      - McNemar test
    """
    diffs = base_margins - cond_margins
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    n = len(diffs)

    cohen_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    if np.all(diffs == 0) or std_diff < 1e-12:
        t_stat, t_pval = 0.0, 1.0
        w_stat, w_pval = 0.0, 1.0
    else:
        t_stat, t_pval = stats.ttest_rel(base_margins, cond_margins)
        try:
            w_stat, w_pval = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        except Exception:
            w_stat, w_pval = 0.0, 1.0

    rng = np.random.RandomState(seed)
    boot_indices = rng.choice(n, size=(n_boot, n), replace=True)
    boot_means = np.mean(diffs[boot_indices], axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    b_disc, c_disc, mcnemar_p = compute_mcnemar_exact(base_corrects, cond_corrects)

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": float(t_stat),
        "t_pval": float(t_pval),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pval": float(w_pval),
        "cohen_dz": cohen_dz,
        "mcnemar_b": b_disc,
        "mcnemar_c": c_disc,
        "mcnemar_pval": mcnemar_p
    }


def evaluate_v0_9_decision(
    tiny_summary: Dict[str, Any],
    small_summary: Dict[str, Any],
    tiny_propagation: Dict[str, Any],
    small_propagation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates pre-registered decision rules for Outcome A, B, C, D
    and independently classifies rank propagation (RANK-EXPANSION vs RANK-PRESERVING).
    """
    # Tiny metrics
    t_cent = tiny_summary["centroid_acc"]
    t_gauss = tiny_summary["gaussian_acc"]
    t_delta_full = t_gauss - t_cent
    t_target_90 = t_cent + 0.90 * t_delta_full

    t_nat_pca = tiny_summary["natural_pca_accs"]  # dict rank -> acc
    t_em_pca = tiny_summary["energy_matched_pca_accs"]
    t_pc_nat = tiny_summary["pc_identity_natural"]  # dict k -> acc
    t_pc_mat = tiny_summary["pc_identity_matched"]
    t_rand_1d = tiny_summary["random_1d_acc"]

    # Small metrics
    s_cent = small_summary["centroid_acc"]
    s_gauss = small_summary["gaussian_acc"]
    s_delta_full = s_gauss - s_cent
    s_target_90 = s_cent + 0.90 * s_delta_full

    s_nat_pca = small_summary["natural_pca_accs"]
    s_em_pca = small_summary["energy_matched_pca_accs"]
    s_pc_nat = small_summary["pc_identity_natural"]
    s_pc_mat = small_summary["pc_identity_matched"]
    s_rand_1d = small_summary["random_1d_acc"]

    # Check Small saturation at low ranks (R <= 4)
    s_r4_reaches_90 = (s_nat_pca.get(1, 0.0) >= s_target_90) or (s_nat_pca.get(2, 0.0) >= s_target_90) or (s_nat_pca.get(4, 0.0) >= s_target_90)

    # Check Tiny saturation at low ranks (R <= 8)
    t_r8_reaches_90 = any(t_nat_pca.get(r, 0.0) >= t_target_90 for r in [1, 2, 4, 8])

    # Check if Natural PCA substantially improved Tiny over Energy-Matched at low ranks
    t_low_rank_gain = np.mean([t_nat_pca.get(r, 0.0) - t_em_pca.get(r, 0.0) for r in [1, 2, 4, 8]])

    # Check PC1 vs PC2..16 and Random 1D advantage
    t_pc1_adv_rand = t_pc_nat.get(1, 0.0) - t_rand_1d
    s_pc1_adv_rand = s_pc_nat.get(1, 0.0) - s_rand_1d
    t_pc1_adv_pc2 = t_pc_nat.get(1, 0.0) - t_pc_nat.get(2, 0.0)
    s_pc1_adv_pc2 = s_pc_nat.get(1, 0.0) - s_pc_nat.get(2, 0.0)

    pc1_dominates_both = (t_pc1_adv_rand > 0.03 and s_pc1_adv_rand > 0.05 and t_pc1_adv_pc2 > 0.02 and s_pc1_adv_pc2 > 0.05)

    # Decision rule evaluation
    if t_r8_reaches_90 and t_low_rank_gain > 0.05:
        verdict = "OUTCOME_A"
        desc = "OUTCOME A — V0.8 ARCHITECTURAL DICHOTOMY WAS AN ENERGY ARTIFACT: Natural-energy low-rank PCA rescues DeiT-Tiny, reaching near-full Gaussian performance at low R."
    elif s_r4_reaches_90 and not t_r8_reaches_90:
        verdict = "OUTCOME_B"
        desc = "OUTCOME B — TRUE ARCHITECTURAL DICHOTOMY: Even with natural unscaled variance, DeiT-Small exhibits low-rank sufficiency (R<=4) while DeiT-Tiny fails to saturate at low ranks."
    elif pc1_dominates_both:
        verdict = "OUTCOME_C"
        desc = "OUTCOME C — SPECIFIC DOMINANT DIRECTION: In both models, PC1 substantially outperforms other principal components and matched random 1D directions."
    else:
        verdict = "OUTCOME_D"
        desc = "OUTCOME D — GENERIC LOW-DIMENSIONAL SEED: Multiple single directions or random 1D directions produce comparable recovery once natural amplitude is used."

    # Rank propagation classification for Natural PCA Rank 1
    t_r_inj = tiny_propagation["r_eff_depth8"]
    t_r_b11 = tiny_propagation["r_eff_block11"]
    s_r_inj = small_propagation["r_eff_depth8"]
    s_r_b11 = small_propagation["r_eff_block11"]

    if (t_r_inj <= 1.5 and t_r_b11 >= 2.5) or (s_r_inj <= 1.5 and s_r_b11 >= 2.5):
        prop_verdict = "RANK-EXPANSION"
        prop_desc = "RANK-EXPANSION: One-dimensional input variation (r_eff <= 1.5 at injection) expands substantially into higher-dimensional patch representations downstream (r_eff >= 2.5 at Block 11)."
    else:
        prop_verdict = "RANK-PRESERVING"
        prop_desc = "RANK-PRESERVING: Effective patch representation rank remains close to 1 throughout downstream Blocks 9-11."

    return {
        "verdict": verdict,
        "verdict_description": desc,
        "rank_propagation_verdict": prop_verdict,
        "rank_propagation_description": prop_desc,
        "tiny": {
            "r8_reaches_90": bool(t_r8_reaches_90),
            "target_90_acc": float(t_target_90),
            "low_rank_gain_natural_vs_em": float(t_low_rank_gain),
            "pc1_vs_rand_diff": float(t_pc1_adv_rand),
            "pc1_vs_pc2_diff": float(t_pc1_adv_pc2),
            "r_eff_depth8": float(t_r_inj),
            "r_eff_block11": float(t_r_b11)
        },
        "small": {
            "r4_reaches_90": bool(s_r4_reaches_90),
            "target_90_acc": float(s_target_90),
            "pc1_vs_rand_diff": float(s_pc1_adv_rand),
            "pc1_vs_pc2_diff": float(s_pc1_adv_pc2),
            "r_eff_depth8": float(s_r_inj),
            "r_eff_block11": float(s_r_b11)
        }
    }
