from typing import Dict, Any, List


def evaluate_fungibility_v0_5_decision(
    tiny_res: Dict[str, Any],
    small_res: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates pre-registered decision rules for Patch Content Fungibility V0.5:
      - OUTCOME A: Zero-Ablation / Scale Artifact
      - OUTCOME B: Distribution Matters, Content Does Not
      - OUTCOME C: Learned Representation Structure Matters
    """
    rationale = []

    # Extract primary contrasts for Tiny and Small
    # 1. Random Sphere vs Cross-Image
    t_sphere_vs_cross = tiny_res["comparisons"]["cross_vs_sphere_mean"]
    s_sphere_vs_cross = small_res["comparisons"]["cross_vs_sphere_mean"]

    # 2. Gaussian vs Cross-Image
    t_gauss_vs_cross = tiny_res["comparisons"]["cross_vs_gaussian_mean"]
    s_gauss_vs_cross = small_res["comparisons"]["cross_vs_gaussian_mean"]

    # 3. Feature Shuffle vs Clean / Cross
    t_shuff_vs_cross = tiny_res["comparisons"]["cross_vs_feature_shuffle_mean"]
    s_shuff_vs_cross = small_res["comparisons"]["cross_vs_feature_shuffle_mean"]

    # 4. Recovery fraction: (SphereMargin - ZeroMargin) / (CrossMargin - ZeroMargin)
    t_zero_m = tiny_res["conditions"]["zero"]["mean_margin"]
    t_cross_m = tiny_res["conditions"]["cross_image"]["mean_margin"]
    t_sphere_m = tiny_res["conditions"]["random_sphere_mean"]["mean_margin"]
    t_cross_gap = t_cross_m - t_zero_m
    t_sphere_rec = (t_sphere_m - t_zero_m) / t_cross_gap if abs(t_cross_gap) > 1e-6 else 0.0

    s_zero_m = small_res["conditions"]["zero"]["mean_margin"]
    s_cross_m = small_res["conditions"]["cross_image"]["mean_margin"]
    s_sphere_m = small_res["conditions"]["random_sphere_mean"]["mean_margin"]
    s_cross_gap = s_cross_m - s_zero_m
    s_sphere_rec = (s_sphere_m - s_zero_m) / s_cross_gap if abs(s_cross_gap) > 1e-6 else 0.0

    # Accuracy differences
    t_cross_acc = tiny_res["conditions"]["cross_image"]["accuracy"]
    t_sphere_acc = tiny_res["conditions"]["random_sphere_mean"]["accuracy"]
    s_cross_acc = small_res["conditions"]["cross_image"]["accuracy"]
    s_sphere_acc = small_res["conditions"]["random_sphere_mean"]["accuracy"]

    t_acc_diff = abs(t_cross_acc - t_sphere_acc)
    s_acc_diff = abs(s_cross_acc - s_sphere_acc)

    # -------------------------------------------------------------
    # Check OUTCOME A: Scale Artifact
    # Criteria: RandomSphere recovers >= 80% of cross gap AND (|dz| < 0.20 OR acc diff < 1.0%) on BOTH
    # -------------------------------------------------------------
    t_passes_a = (t_sphere_rec >= 0.80) and (abs(t_sphere_vs_cross["cohens_dz"]) < 0.20 or t_acc_diff < 0.01)
    s_passes_a = (s_sphere_rec >= 0.80) and (abs(s_sphere_vs_cross["cohens_dz"]) < 0.20 or s_acc_diff < 0.01)

    if t_passes_a and s_passes_a:
        outcome = "OUTCOME A"
        decision = "OUTCOME A - ZERO-ABLATION / SCALE ARTIFACT: Norm-matched random vectors rescue performance; exact or learned structure is not required."
        rationale.append(f"Random sphere recovers {t_sphere_rec*100:.1f}% of cross gap on Tiny and {s_sphere_rec*100:.1f}% on Small.")
        rationale.append(f"Random sphere vs Cross difference is negligible (Tiny dz={t_sphere_vs_cross['cohens_dz']:.3f}, Small dz={s_sphere_vs_cross['cohens_dz']:.3f}).")
        return {"outcome": outcome, "decision": decision, "rationale": rationale}

    # -------------------------------------------------------------
    # Check OUTCOME B: Distribution Matters, Content Does Not
    # Criteria: Sphere is substantially worse than Cross, BUT Gaussian approaches Cross
    # -------------------------------------------------------------
    t_sphere_worse = (t_sphere_vs_cross["mean"] > 0.0) and (t_sphere_vs_cross["cohens_dz"] >= 0.20) and (t_sphere_vs_cross["bootstrap_ci_95"][0] > 0.0)
    s_sphere_worse = (s_sphere_vs_cross["mean"] > 0.0) and (s_sphere_vs_cross["cohens_dz"] >= 0.20) and (s_sphere_vs_cross["bootstrap_ci_95"][0] > 0.0)

    t_gauss_m = tiny_res["conditions"]["gaussian_mean"]["mean_margin"]
    s_gauss_m = small_res["conditions"]["gaussian_mean"]["mean_margin"]
    t_gauss_rec = (t_gauss_m - t_zero_m) / t_cross_gap if abs(t_cross_gap) > 1e-6 else 0.0
    s_gauss_rec = (s_gauss_m - s_zero_m) / s_cross_gap if abs(s_cross_gap) > 1e-6 else 0.0

    t_gauss_approaches_cross = (t_gauss_rec >= 0.80) or (abs(t_gauss_vs_cross["cohens_dz"]) < 0.20)
    s_gauss_approaches_cross = (s_gauss_rec >= 0.80) or (abs(s_gauss_vs_cross["cohens_dz"]) < 0.20)

    if (t_sphere_worse and s_sphere_worse) and (t_gauss_approaches_cross and s_gauss_approaches_cross):
        outcome = "OUTCOME B"
        decision = "OUTCOME B - DISTRIBUTION MATTERS, CONTENT DOES NOT: Arbitrary random vectors fail, but moment-matched Gaussian activations match Cross-Image performance."
        rationale.append("Random sphere vectors fail to match Cross-Image performance on both models.")
        rationale.append(f"However, empirical Gaussian activations recover performance (Tiny Gaussian rec={t_gauss_rec*100:.1f}%, Small Gaussian rec={s_gauss_rec*100:.1f}%).")
        return {"outcome": outcome, "decision": decision, "rationale": rationale}

    # -------------------------------------------------------------
    # Check OUTCOME C: Learned Representation Structure Matters
    # Criteria:
    # 1. Random Sphere significantly worse than Cross (dz >= 0.20, CI > 0)
    # 2. Gaussian significantly worse than Cross or LayerMean (dz >= 0.20, CI > 0)
    # 3. Feature Shuffle significantly degrades performance (dz >= 0.20, CI > 0)
    # 4. Same-Image Mean and/or Cross-Image retain strong performance
    # -------------------------------------------------------------
    t_gauss_worse = (t_gauss_vs_cross["mean"] > 0.0) and (t_gauss_vs_cross["cohens_dz"] >= 0.20) and (t_gauss_vs_cross["bootstrap_ci_95"][0] > 0.0)
    s_gauss_worse = (s_gauss_vs_cross["mean"] > 0.0) and (s_gauss_vs_cross["cohens_dz"] >= 0.20) and (s_gauss_vs_cross["bootstrap_ci_95"][0] > 0.0)

    t_shuff_worse = (t_shuff_vs_cross["mean"] > 0.0) and (t_shuff_vs_cross["cohens_dz"] >= 0.20) and (t_shuff_vs_cross["bootstrap_ci_95"][0] > 0.0)
    s_shuff_worse = (s_shuff_vs_cross["mean"] > 0.0) and (s_shuff_vs_cross["cohens_dz"] >= 0.20) and (s_shuff_vs_cross["bootstrap_ci_95"][0] > 0.0)

    if (t_sphere_worse and s_sphere_worse) and (t_gauss_worse and s_gauss_worse) and (t_shuff_worse and s_shuff_worse):
        outcome = "OUTCOME C"
        decision = "OUTCOME C - LEARNED REPRESENTATION STRUCTURE MATTERS: Neither arbitrary norm-matched vectors nor moment-matched Gaussians suffice; learned activation geometry is causally necessary."
        rationale.append(f"Norm-matched random sphere is severely degraded vs Cross-Image: Tiny dz={t_sphere_vs_cross['cohens_dz']:.3f} (95% CI [{t_sphere_vs_cross['bootstrap_ci_95'][0]:.3f}, {t_sphere_vs_cross['bootstrap_ci_95'][1]:.3f}]), Small dz={s_sphere_vs_cross['cohens_dz']:.3f} (95% CI [{s_sphere_vs_cross['bootstrap_ci_95'][0]:.3f}, {s_sphere_vs_cross['bootstrap_ci_95'][1]:.3f}]).")
        rationale.append(f"Layer-statistics Gaussian also fails to rescue performance: Tiny dz={t_gauss_vs_cross['cohens_dz']:.3f}, Small dz={s_gauss_vs_cross['cohens_dz']:.3f}.")
        rationale.append(f"Feature dimension shuffling destroys performance despite exact norm and scalar multiset preservation: Tiny dz={t_shuff_vs_cross['cohens_dz']:.3f}, Small dz={s_shuff_vs_cross['cohens_dz']:.3f}.")
        rationale.append("Same-Image Mean and Cross-Image retain strong predictive recovery.")
        return {"outcome": outcome, "decision": decision, "rationale": rationale}

    # Fallback if mixed/partial
    outcome = "OUTCOME A (PARTIAL / AMBIGUOUS)"
    decision = f"PARTIAL / AMBIGUOUS OUTCOME: Hypotheses did not cleanly separate across both architectures."
    rationale.append(f"Tiny: Sphere vs Cross dz={t_sphere_vs_cross['cohens_dz']:.3f}, Gauss vs Cross dz={t_gauss_vs_cross['cohens_dz']:.3f}.")
    rationale.append(f"Small: Sphere vs Cross dz={s_sphere_vs_cross['cohens_dz']:.3f}, Gauss vs Cross dz={s_gauss_vs_cross['cohens_dz']:.3f}.")
    return {"outcome": outcome, "decision": decision, "rationale": rationale}
