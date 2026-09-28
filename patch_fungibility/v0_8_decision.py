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
    # Two-sided binomial test under p=0.5
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
    Computes paired statistical tests between clean baseline and condition:
      - mean margin diff (base - cond)
      - 95% bootstrap CI
      - paired t-test
      - Wilcoxon signed-rank test
      - Cohen's dz
      - McNemar test
    """
    diffs = base_margins - cond_margins  # Positive means baseline is better (damage)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    n = len(diffs)

    cohen_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    # Paired t-test and Wilcoxon signed rank test
    if np.all(diffs == 0) or std_diff < 1e-12:
        t_stat, t_pval = 0.0, 1.0
        w_stat, w_pval = 0.0, 1.0
    else:
        t_stat, t_pval = stats.ttest_rel(base_margins, cond_margins)
        try:
            w_stat, w_pval = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        except Exception:
            w_stat, w_pval = 0.0, 1.0

    # 10,000 bootstrap CI of mean difference
    rng = np.random.RandomState(seed)
    boot_indices = rng.choice(n, size=(n_boot, n), replace=True)
    boot_means = np.mean(diffs[boot_indices], axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # McNemar test
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


def compute_r95_and_k95(
    centroid_acc: float,
    gaussian_acc: float,
    rank_accs: Dict[int, float],
    k_accs: Dict[int, float]
) -> Tuple[Optional[int], Optional[int], float]:
    """
    Computes descriptive saturation thresholds:
      R_95: Smallest tested rank reaching >= 95% of full-Gaussian recovery over static centroid.
      K_95: Smallest tested K reaching >= 95% of full-Gaussian recovery over static centroid.
    """
    delta_full = gaussian_acc - centroid_acc
    if delta_full <= 0:
        return None, None, 0.0

    target_95 = centroid_acc + 0.95 * delta_full

    r_95 = None
    for r in sorted(rank_accs.keys()):
        if rank_accs[r] >= target_95:
            r_95 = r
            break

    k_95 = None
    for k in sorted(k_accs.keys()):
        if k_accs[k] >= target_95:
            k_95 = k
            break

    return r_95, k_95, float(target_95)


def evaluate_v0_8_decision(
    tiny_summary: Dict[str, Any],
    small_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates pre-registered decision rules for Outcome A, B, C, D across both architectures.
    """
    # Key comparison metrics for Tiny
    t_cent = tiny_summary["centroid_acc"]
    t_gauss = tiny_summary["gaussian_acc"]
    t_shared = tiny_summary["shared_acc"]
    t_indep = tiny_summary["indep_acc"]
    t_iso = tiny_summary["iso_acc"]
    t_pca = tiny_summary["pca_accs"]  # dict rank -> acc
    t_rand = tiny_summary["rand_accs"]  # dict rank -> acc
    t_k = tiny_summary["grouped_k_accs"]  # dict k -> acc

    # Key comparison metrics for Small
    s_cent = small_summary["centroid_acc"]
    s_gauss = small_summary["gaussian_acc"]
    s_shared = small_summary["shared_acc"]
    s_indep = small_summary["indep_acc"]
    s_iso = small_summary["iso_acc"]
    s_pca = small_summary["pca_accs"]
    s_rand = small_summary["rand_accs"]
    s_k = small_summary["grouped_k_accs"]

    # 1. Independent >> Shared check
    t_indep_gain = t_indep - t_shared
    s_indep_gain = s_indep - s_shared
    indep_substantially_better = (t_indep_gain >= 0.05) and (s_indep_gain >= 0.10)

    # 2. Monotonic or near-monotonic rank recovery
    # Check if max(PCA ranks) > PCA rank 1 and monotonic trend
    t_pca_sorted = [t_pca[r] for r in sorted(t_pca.keys())]
    s_pca_sorted = [s_pca[r] for r in sorted(s_pca.keys())]
    t_rank_improves = t_pca_sorted[-1] > t_pca_sorted[0] + 0.05
    s_rank_improves = s_pca_sorted[-1] > s_pca_sorted[0] + 0.10

    # 3. PCA vs Random at low/intermediate ranks (e.g. ranks 1, 2, 4, 8)
    t_pca_vs_rand_diffs = [t_pca[r] - t_rand[r] for r in [1, 2, 4, 8] if r in t_pca and r in t_rand]
    s_pca_vs_rand_diffs = [s_pca[r] - s_rand[r] for r in [1, 2, 4, 8] if r in s_pca and r in s_rand]
    pca_beats_rand = (np.mean(t_pca_vs_rand_diffs) > 0.02) or (np.mean(s_pca_vs_rand_diffs) > 0.02)
    rand_matches_pca = (abs(np.mean(t_pca_vs_rand_diffs)) <= 0.02) and (abs(np.mean(s_pca_vs_rand_diffs)) <= 0.02)

    # 4. Diagonal Gaussian vs Isotropic
    t_diag_vs_iso = t_gauss - t_iso
    s_diag_vs_iso = s_gauss - s_iso
    diag_beats_iso = (t_diag_vs_iso > 0.02) or (s_diag_vs_iso > 0.02)
    iso_matches_diag = (abs(t_diag_vs_iso) <= 0.02) and (abs(s_diag_vs_iso) <= 0.02)

    # 5. Low-rank saturation (R_95 <= 16) vs High-rank requirement (R_95 >= 64 or None)
    t_r95, t_k95, t_t95 = compute_r95_and_k95(t_cent, t_gauss, t_pca, t_k)
    s_r95, s_k95, s_t95 = compute_r95_and_k95(s_cent, s_gauss, s_pca, s_k)

    # Decision logic
    if not indep_substantially_better and not t_rank_improves and not s_rank_improves:
        verdict = "OUTCOME_A"
        desc = "OUTCOME A — DIVERSITY DOES NOT EXPLAIN FAILURE: Independent noise does not outperform shared noise, and rank/K increases do not rescue performance."
    elif indep_substantially_better and rand_matches_pca and iso_matches_diag:
        verdict = "OUTCOME_B"
        desc = "OUTCOME B — GENERIC TOKEN DIVERSITY: Downstream attention requires token diversity, but random subspaces and isotropic noise perform similarly to learned PCA geometry."
    elif indep_substantially_better and (pca_beats_rand or diag_beats_iso or (t_r95 is not None and t_r95 <= 16 and s_r95 is not None and s_r95 <= 16)):
        if (t_r95 is not None and t_r95 <= 32) and (s_r95 is not None and s_r95 <= 32):
            verdict = "OUTCOME_C"
            desc = "OUTCOME C — STRUCTURED LOW-RANK DIVERSITY: Token diversity is strictly required, and downstream computation recovers at a structured low-dimensional subspace of late-layer variation."
        else:
            verdict = "OUTCOME_D"
            desc = "OUTCOME D — HIGH-DIMENSIONAL STRUCTURE REQUIRED: Diversity matters, but recovery requires high-dimensional rank."
    else:
        # Check Outcome D vs C
        if (t_r95 is None or t_r95 >= 64) or (s_r95 is None or s_r95 >= 64):
            verdict = "OUTCOME_D"
            desc = "OUTCOME D — HIGH-DIMENSIONAL STRUCTURE REQUIRED: Low-rank conditions remain substantially degraded; high-dimensional variation required."
        else:
            verdict = "OUTCOME_C"
            desc = "OUTCOME C — STRUCTURED LOW-RANK DIVERSITY: Token-level diversity rescues performance in a structured low-rank regime."

    return {
        "verdict": verdict,
        "verdict_description": desc,
        "tiny": {
            "r_95": t_r95,
            "k_95": t_k95,
            "target_95_acc": t_t95,
            "indep_gain": float(t_indep_gain),
            "diag_vs_iso": float(t_diag_vs_iso),
            "pca_vs_rand_mean_diff": float(np.mean(t_pca_vs_rand_diffs)) if t_pca_vs_rand_diffs else 0.0
        },
        "small": {
            "r_95": s_r95,
            "k_95": s_k95,
            "target_95_acc": s_t95,
            "indep_gain": float(s_indep_gain),
            "diag_vs_iso": float(s_diag_vs_iso),
            "pca_vs_rand_mean_diff": float(np.mean(s_pca_vs_rand_diffs)) if s_pca_vs_rand_diffs else 0.0
        }
    }
