from typing import Dict, Any, List


def evaluate_fungibility_v0_6_decision(
    tiny_res: Dict[str, Any],
    small_res: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates pre-registered decision rules for Patch Content Fungibility V0.6:
      - OUTCOME A: Does Not Generalize / Kill
      - OUTCOME B: Local Robust Phenomenon
      - OUTCOME C: Depth-Transition Phenomenon
    """
    rationale = []

    # 1. Primary Test: Depth 8, Fraction 25% Held-Out Gaussian Replication
    t_d8_f25 = tiny_res["depth_fraction_results"][8]["25%"]
    s_d8_f25 = small_res["depth_fraction_results"][8]["25%"]

    t_gauss_vs_zero = t_d8_f25["comparisons"]["gaussian_vs_zero"]
    s_gauss_vs_zero = s_d8_f25["comparisons"]["gaussian_vs_zero"]

    t_gauss_vs_sphere = t_d8_f25["comparisons"]["gaussian_vs_sphere"]
    s_gauss_vs_sphere = s_d8_f25["comparisons"]["gaussian_vs_sphere"]

    t_replicates_d8 = (t_gauss_vs_zero["mean"] > 0.0) and (t_gauss_vs_zero["cohens_dz"] >= 0.20) and (t_gauss_vs_zero["bootstrap_ci_95"][0] > 0.0)
    s_replicates_d8 = (s_gauss_vs_zero["mean"] > 0.0) and (s_gauss_vs_zero["cohens_dz"] >= 0.20) and (s_gauss_vs_zero["bootstrap_ci_95"][0] > 0.0)

    # Check OUTCOME A
    if not (t_replicates_d8 and s_replicates_d8):
        outcome = "OUTCOME A"
        decision = "OUTCOME A - DOES NOT GENERALIZE / KILL: Held-out Gaussian replacement fails to replicate Depth-8 recovery under disjoint calibration data."
        rationale.append(f"Depth-8, 25% Gaussian vs Zero failed to replicate (Tiny dz={t_gauss_vs_zero['cohens_dz']:.3f}, Small dz={s_gauss_vs_zero['cohens_dz']:.3f}).")
        return {"outcome": outcome, "decision": decision, "rationale": rationale}

    # 2. Check Coherent Depth Region:
    # Does Gaussian recovery extend across neighboring depths (e.g. depths 7, 8, 9)?
    # And are early depths (5, 6) more sensitive, while late depth (10) approaches token irrelevance?
    def evaluate_depth_transition(model_res: Dict[str, Any]):
        df_res = model_res["depth_fraction_results"]
        # Check depth 7, 8, 9 show positive Gaussian recovery
        d7_ok = df_res[7]["25%"]["comparisons"]["gaussian_vs_zero"]["mean"] > 0.0
        d8_ok = df_res[8]["25%"]["comparisons"]["gaussian_vs_zero"]["mean"] > 0.0
        d9_ok = df_res[9]["25%"]["comparisons"]["gaussian_vs_zero"]["mean"] > 0.0
        # Check depth 10 has near-zero damage (irrelevance)
        d10_irrelevant = df_res[10]["25%"]["conditions"]["zero"]["mean_damage"] < 0.15
        # Check early depths (5 or 6) show sensitivity
        early_sensitive = df_res[5]["25%"]["conditions"]["zero"]["mean_damage"] > 0.20
        # Check budget scaling: 10% damage <= 25% damage <= 50% damage
        f10_dmg = df_res[8]["10%"]["conditions"]["gaussian_mean"]["mean_damage"]
        f25_dmg = df_res[8]["25%"]["conditions"]["gaussian_mean"]["mean_damage"]
        f50_dmg = df_res[8]["50%"]["conditions"]["gaussian_mean"]["mean_damage"]
        budget_scaling = (f10_dmg <= f25_dmg + 0.10) and (f25_dmg <= f50_dmg + 0.10)
        return d7_ok and d8_ok and d9_ok and d10_irrelevant and early_sensitive and budget_scaling

    t_transition_ok = evaluate_depth_transition(tiny_res)
    s_transition_ok = evaluate_depth_transition(small_res)

    t_sphere_ok = t_gauss_vs_sphere["mean"] > 0.0 and t_gauss_vs_sphere["cohens_dz"] >= 0.20
    s_sphere_ok = s_gauss_vs_sphere["mean"] > 0.0 and s_gauss_vs_sphere["cohens_dz"] >= 0.20

    if t_transition_ok and s_transition_ok and t_sphere_ok and s_sphere_ok:
        outcome = "OUTCOME C"
        decision = "OUTCOME C - DEPTH-TRANSITION PHENOMENON: Held-out distribution-matched Gaussian replacements robustly rescue predictive performance across a coherent mid-to-late depth window with interpretable budget scaling."
        rationale.append(f"Depth-8 held-out replication strongly confirmed: Tiny (dz={t_gauss_vs_zero['cohens_dz']:.3f}, 95% CI [{t_gauss_vs_zero['bootstrap_ci_95'][0]:.3f}, {t_gauss_vs_zero['bootstrap_ci_95'][1]:.3f}]) and Small (dz={s_gauss_vs_zero['cohens_dz']:.3f}, 95% CI [{s_gauss_vs_zero['bootstrap_ci_95'][0]:.3f}, {s_gauss_vs_zero['bootstrap_ci_95'][1]:.3f}]).")
        rationale.append(f"Gaussian decisively outperforms Random Sphere across models (Tiny dz={t_gauss_vs_sphere['cohens_dz']:.3f}, Small dz={s_gauss_vs_sphere['cohens_dz']:.3f}).")
        rationale.append("Coherent multi-layer transition window confirmed: earlier intermediate depths show content sensitivity, mid-to-late blocks show broad Gaussian tolerance, and depth 10 approaches token irrelevance.")
        rationale.append("Interpretable budget scaling confirmed across 10%, 25%, and 50% replacement fractions.")
        return {"outcome": outcome, "decision": decision, "rationale": rationale}
    else:
        outcome = "OUTCOME B"
        decision = "OUTCOME B - LOCAL ROBUST PHENOMENON: Held-out Gaussian recovery replicates at Block 8, but depth transition or budget scaling is localized or irregular."
        rationale.append(f"Depth-8 held-out replication confirmed (Tiny dz={t_gauss_vs_zero['cohens_dz']:.3f}, Small dz={s_gauss_vs_zero['cohens_dz']:.3f}).")
        rationale.append(f"Transition criteria: Tiny transition={t_transition_ok}, Small transition={s_transition_ok}, Tiny sphere check={t_sphere_ok}, Small sphere check={s_sphere_ok}.")
        return {"outcome": outcome, "decision": decision, "rationale": rationale}
