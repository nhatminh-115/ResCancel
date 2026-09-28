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


def evaluate_paired_comparison(
    margins_a: np.ndarray,
    margins_b: np.ndarray,
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    label_a: str = "Tokenwise",
    label_b: str = "Global",
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates paired comparison between Policy A (e.g. Tokenwise) and Policy B (e.g. Global).
    diff = A - B. Positive difference indicates Policy A outperforms Policy B.
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

    # Accuracy and McNemar test
    acc_a = float(np.mean(correct_a))
    acc_b = float(np.mean(correct_b))
    acc_diff = acc_a - acc_b

    # 2x2 contingency table:
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    both_correct = int(np.sum((correct_a == 1) & (correct_b == 1)))
    a_only = int(np.sum((correct_a == 1) & (correct_b == 0)))
    b_only = int(np.sum((correct_a == 0) & (correct_b == 1)))
    both_incorrect = int(np.sum((correct_a == 0) & (correct_b == 0)))

    discordant = a_only + b_only
    if discordant > 0:
        # Continuity-corrected McNemar chi-square
        mcnemar_stat = float(((abs(a_only - b_only) - 1.0) ** 2) / discordant)
        mcnemar_pval = float(stats.chi2.sf(mcnemar_stat, df=1))
        # Exact binomial test
        binom_res = stats.binomtest(a_only, discordant, p=0.5, alternative="two-sided")
        exact_pval = float(binom_res.pvalue)
    else:
        mcnemar_stat = 0.0
        mcnemar_pval = 1.0
        exact_pval = 1.0

    risk_diff = float((a_only - b_only) / n)

    return {
        "comparison": f"{label_a}_vs_{label_b}",
        "n_samples": n,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "std_diff": std_diff,
        "cohens_dz": cohens_dz,
        "t_stat": t_stat,
        "p_value": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_pval,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "acc_a": acc_a,
        "acc_b": acc_b,
        "acc_diff": acc_diff,
        "contingency_table": {
            "both_correct": both_correct,
            f"{label_a}_only": a_only,
            f"{label_b}_only": b_only,
            "both_incorrect": both_incorrect
        },
        "risk_diff": risk_diff,
        "mcnemar_stat": mcnemar_stat,
        "mcnemar_p_value": mcnemar_pval,
        "exact_mcnemar_p_value": exact_pval
    }


def summarize_spatial_diversity(
    diversity_records: List[Dict[str, Any]],
    num_candidate_layers: int = 8
) -> Dict[str, Any]:
    """
    Summarizes spatial heterogeneity metrics across all evaluated images.
    """
    n_images = len(diversity_records)
    layer_counts_matrix = np.array([r["layer_counts"] for r in diversity_records]) # (N, num_candidates)
    total_token_counts = np.sum(layer_counts_matrix, axis=0) # (num_candidates,)
    total_tokens = float(np.sum(total_token_counts))
    overall_layer_probs = (total_token_counts / total_tokens).tolist()

    entropies = np.array([r["entropy"] for r in diversity_records])
    dominant_fracs = np.array([r["dominant_fraction"] for r in diversity_records])
    matching_global_fracs = np.array([r["fraction_matching_global"] for r in diversity_records])
    distinct_layers = np.array([r["num_distinct_layers"] for r in diversity_records])

    return {
        "n_images": n_images,
        "total_tokens_evaluated": int(total_tokens),
        "overall_layer_counts": total_token_counts.tolist(),
        "overall_layer_proportions": overall_layer_probs,
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies, ddof=1)),
        "entropy_median": float(np.median(entropies)),
        "dominant_fraction_mean": float(np.mean(dominant_fracs)),
        "dominant_fraction_std": float(np.std(dominant_fracs, ddof=1)),
        "fraction_matching_global_mean": float(np.mean(matching_global_fracs)),
        "fraction_matching_global_std": float(np.std(matching_global_fracs, ddof=1)),
        "distinct_layers_mean": float(np.mean(distinct_layers)),
        "distinct_layers_std": float(np.std(distinct_layers, ddof=1))
    }


def evaluate_decision_rule(
    tiny_token_vs_global: Dict[str, Any],
    small_token_vs_global: Dict[str, Any],
    tiny_global_vs_base: Dict[str, Any],
    small_global_vs_base: Dict[str, Any],
    tiny_diversity: Dict[str, Any],
    small_diversity: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Strictly applies the pre-registered V0 decision rules to classify the outcome into
    Outcome A, Outcome B, or Outcome C.
    """
    tiny_mean_diff = tiny_token_vs_global["mean_diff"]
    small_mean_diff = small_token_vs_global["mean_diff"]

    tiny_ci = tiny_token_vs_global["bootstrap_ci_95"]
    small_ci = small_token_vs_global["bootstrap_ci_95"]

    tiny_dz = tiny_token_vs_global["cohens_dz"]
    small_dz = small_token_vs_global["cohens_dz"]

    tiny_acc_diff = tiny_token_vs_global["acc_diff"]
    small_acc_diff = small_token_vs_global["acc_diff"]

    tiny_dom_frac = tiny_diversity["dominant_fraction_mean"]
    small_dom_frac = small_diversity["dominant_fraction_mean"]

    tiny_entropy = tiny_diversity["entropy_mean"]
    small_entropy = small_diversity["entropy_mean"]

    # Check Outcome C (INTERESTING) criteria:
    # 1. Tokenwise beats Global on BOTH architectures (mean_diff > 0)
    # 2. Paired bootstrap 95% CI strictly above zero on both
    # 3. Paired effect size dz >= 0.20 on both
    # 4. Tokenwise does not reduce clean accuracy by >0.5% relative to baseline
    # 5. Genuine spatial diversity (entropy >= 1.0 and dominant fraction < 75%)
    passes_c1 = (tiny_mean_diff > 0.0) and (small_mean_diff > 0.0)
    passes_c2 = (tiny_ci[0] > 0.0) and (small_ci[0] > 0.0)
    passes_c3 = (tiny_dz >= 0.20) and (small_dz >= 0.20)
    passes_c4 = (tiny_acc_diff >= -0.005) and (small_acc_diff >= -0.005)
    passes_c5 = (tiny_entropy >= 1.0 and small_entropy >= 1.0) and (tiny_dom_frac < 0.75 and small_dom_frac < 0.75)

    is_outcome_c = passes_c1 and passes_c2 and passes_c3 and passes_c4 and passes_c5

    # Check Outcome B (GLOBAL DELTA REUSE ONLY):
    # Global improves over baseline, but Tokenwise does NOT materially beat Global
    global_beats_base = (tiny_global_vs_base["bootstrap_ci_95"][0] > 0.0) and (small_global_vs_base["bootstrap_ci_95"][0] > 0.0)
    tokenwise_not_superior = (not passes_c1) or (not passes_c2) or (tiny_dz < 0.20 and small_dz < 0.20)

    rationale = []
    if is_outcome_c:
        decision = "OUTCOME C - INTERESTING: Evidence supports spatially selective historical delta routing."
        rationale.append(f"Tokenwise beats Global on Tiny (Delta_m = {tiny_mean_diff:+.4f}, 95% CI [{tiny_ci[0]:.4f}, {tiny_ci[1]:.4f}], dz = {tiny_dz:.3f}).")
        rationale.append(f"Tokenwise beats Global on Small (Delta_m = {small_mean_diff:+.4f}, 95% CI [{small_ci[0]:.4f}, {small_ci[1]:.4f}], dz = {small_dz:.3f}).")
        rationale.append(f"Genuine spatial diversity confirmed (Tiny Entropy = {tiny_entropy:.2f}, Small Entropy = {small_entropy:.2f}).")
    elif global_beats_base and tokenwise_not_superior:
        decision = "OUTCOME B - GLOBAL DELTA REUSE ONLY: Historical delta reuse helps, but per-token spatial routing is unhelpful."
        rationale.append("Global oracle delta reuse reliably improves over baseline on both models.")
        rationale.append(f"However, tokenwise routing provides no material advantage over global selection (Tiny dz = {tiny_dz:.3f}, Small dz = {small_dz:.3f}).")
    else:
        decision = "OUTCOME A - KILL: No evidence of useful per-token spatial delta reuse advantage."
        if not passes_c1:
            rationale.append(f"Tokenwise oracle fails to beat global oracle on both architectures (Tiny Delta_m = {tiny_mean_diff:+.4f}, Small Delta_m = {small_mean_diff:+.4f}).")
        if not passes_c2:
            rationale.append(f"Bootstrap 95% CIs do not strictly clear zero on both models (Tiny CI: [{tiny_ci[0]:.4f}, {tiny_ci[1]:.4f}], Small CI: [{small_ci[0]:.4f}, {small_ci[1]:.4f}]).")
        if not passes_c3:
            rationale.append(f"Effect size threshold dz >= 0.20 not satisfied (Tiny dz = {tiny_dz:.3f}, Small dz = {small_dz:.3f}).")
        if not passes_c5:
            rationale.append("Spatial selection collapses towards global selection or low entropy.")

    return {
        "decision": decision,
        "outcome": "OUTCOME C" if is_outcome_c else ("OUTCOME B" if (global_beats_base and tokenwise_not_superior) else "OUTCOME A"),
        "passes_c1_positive_diff": passes_c1,
        "passes_c2_ci_above_zero": passes_c2,
        "passes_c3_effect_size_dz": passes_c3,
        "passes_c4_clean_accuracy_safe": passes_c4,
        "passes_c5_spatial_diversity": passes_c5,
        "rationale": rationale
    }
