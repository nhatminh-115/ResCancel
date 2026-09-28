from typing import Dict, List, Tuple, Any
import numpy as np
import scipy.stats as stats


def compute_paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Computes a percentile bootstrap confidence interval for the mean paired difference.
    """
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_indices = rng.randint(0, n, size=(n_resamples, n))
    boot_means = np.mean(diffs[boot_indices], axis=1)
    
    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return ci_lower, ci_upper


def evaluate_paired_margin_diff(
    margins_a: np.ndarray,
    margins_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates paired difference between Condition A and Condition B: diff = A - B.
    For FungibilityGap: A = CrossImageSamePosition, B = Zero.
    diff = CrossImageSamePosition - Zero = Damage_zero - Damage_cross.
    """
    assert len(margins_a) == len(margins_b), "Sample sizes must match"
    n = len(margins_a)
    diffs = margins_a - margins_b

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    cohens_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(margins_a, margins_b)
    t_stat = float(t_stat) if not np.isnan(t_stat) else 0.0
    t_pval = float(t_pval) if not np.isnan(t_pval) else 1.0

    # Wilcoxon signed-rank test
    try:
        w_stat, w_pval = stats.wilcoxon(diffs)
        w_stat = float(w_stat)
        w_pval = float(w_pval)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    # Bootstrap 95% CI
    ci_lower, ci_upper = compute_paired_bootstrap_ci(
        diffs, n_resamples=n_bootstrap, ci_level=0.95, seed=seed
    )

    return {
        "n_samples": n,
        "mean": mean_diff,
        "median": median_diff,
        "std": std_diff,
        "cohens_dz": cohens_dz,
        "t_stat": t_stat,
        "p_value": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_pval,
        "bootstrap_ci_95": [ci_lower, ci_upper]
    }


def evaluate_paired_accuracy(
    correct_a: np.ndarray,
    correct_b: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates paired 0/1 accuracy comparison and exact McNemar test between A and B.
    """
    assert len(correct_a) == len(correct_b), "Sample sizes must match"
    n = len(correct_a)
    acc_a = float(np.mean(correct_a))
    acc_b = float(np.mean(correct_b))
    acc_diff = acc_a - acc_b

    both_correct = int(np.sum((correct_a == 1) & (correct_b == 1)))
    a_only = int(np.sum((correct_a == 1) & (correct_b == 0)))
    b_only = int(np.sum((correct_a == 0) & (correct_b == 1)))
    both_incorrect = int(np.sum((correct_a == 0) & (correct_b == 0)))

    discordant = a_only + b_only
    if discordant > 0:
        binom_res = stats.binomtest(a_only, discordant, p=0.5, alternative="two-sided")
        exact_pval = float(binom_res.pvalue)
    else:
        exact_pval = 1.0

    return {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "acc_diff": acc_diff,
        "a_only": a_only,
        "b_only": b_only,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "discordant": discordant,
        "exact_mcnemar_p_value": exact_pval
    }


def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """
    Applies the Benjamini-Hochberg FDR correction to a list of p-values.
    Returns adjusted p-values (q-values) preserving input order.
    """
    m = len(p_values)
    if m == 0:
        return []
    
    # Sort with original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    orig_indices = [p[0] for p in sorted_pairs]
    sorted_p = [p[1] for p in sorted_pairs]

    q_values = [0.0] * m
    # Step-up calculation: q_k = min_{j >= k} (p_j * m / (j + 1))
    running_min = 1.0
    for k in range(m - 1, -1, -1):
        rank = k + 1
        val = min(1.0, sorted_p[k] * m / rank)
        running_min = min(running_min, val)
        q_values[k] = running_min

    # Reorder to original indices
    adjusted = [0.0] * m
    for orig_idx, q in zip(orig_indices, q_values):
        adjusted[orig_idx] = float(q)
    return adjusted
