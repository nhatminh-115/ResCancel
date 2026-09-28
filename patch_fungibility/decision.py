from typing import Dict, List, Any


def evaluate_fungibility_decision(
    tiny_depth_results: Dict[int, Dict[str, Any]],
    small_depth_results: Dict[int, Dict[str, Any]],
    tested_depths: List[int] = [2, 4, 6, 8, 10]
) -> Dict[str, Any]:
    """
    Evaluates the pre-registered decision rules for Patch Content Fungibility V0.
    Classifies the outcome into:
      - OUTCOME C: CONTENT FUNGIBILITY
      - OUTCOME B: TOKEN IRRELEVANCE ONLY
      - OUTCOME A: KILL
    """
    passed_depths_tiny = []
    passed_depths_small = []
    token_irrelevant_depths_tiny = []
    token_irrelevant_depths_small = []

    per_depth_eval = {}

    for d in tested_depths:
        t_res = tiny_depth_results[d]
        s_res = small_depth_results[d]

        t_gap = t_res["fungibility_gap"]
        s_gap = s_res["fungibility_gap"]

        t_zero_damage = t_res["damage"]["zero"]["mean"]
        s_zero_damage = s_res["damage"]["zero"]["mean"]
        t_cross_damage = t_res["damage"]["cross_same_pos"]["mean"]
        s_cross_damage = s_res["damage"]["cross_same_pos"]["mean"]

        # Check token irrelevance: both zero and cross damage are near zero (< 0.10 margin drop)
        t_irrelevant = (t_zero_damage < 0.10) and (t_cross_damage < 0.10)
        s_irrelevant = (s_zero_damage < 0.10) and (s_cross_damage < 0.10)

        if t_irrelevant:
            token_irrelevant_depths_tiny.append(d)
        if s_irrelevant:
            token_irrelevant_depths_small.append(d)

        # Conditions for Outcome C at depth d:
        # 1. Zero causes meaningful degradation (> 0.20 margin drop, bootstrap CI > 0)
        t_zero_meaningful = (t_zero_damage >= 0.20) and (t_res["damage"]["zero"]["ci_95"][0] > 0.0)
        s_zero_meaningful = (s_zero_damage >= 0.20) and (s_res["damage"]["zero"]["ci_95"][0] > 0.0)

        # 2. FungibilityGap bootstrap CI strictly > 0
        t_ci_pos = t_gap["bootstrap_ci_95"][0] > 0.0
        s_ci_pos = s_gap["bootstrap_ci_95"][0] > 0.0

        # 3. Effect size dz >= 0.20
        t_dz_ok = t_gap["cohens_dz"] >= 0.20
        s_dz_ok = s_gap["cohens_dz"] >= 0.20

        # 4. Effect survives BH-FDR (q < 0.05)
        t_fdr_ok = t_gap["fdr_p_value"] < 0.05
        s_fdr_ok = s_gap["fdr_p_value"] < 0.05

        # 5. Not token irrelevant
        t_pass = t_zero_meaningful and t_ci_pos and t_dz_ok and t_fdr_ok and (not t_irrelevant)
        s_pass = s_zero_meaningful and s_ci_pos and s_dz_ok and s_fdr_ok and (not s_irrelevant)

        if t_pass:
            passed_depths_tiny.append(d)
        if s_pass:
            passed_depths_small.append(d)

        per_depth_eval[d] = {
            "tiny_passed": bool(t_pass),
            "small_passed": bool(s_pass),
            "both_passed": bool(t_pass and s_pass),
            "tiny_gap_dz": t_gap["cohens_dz"],
            "small_gap_dz": s_gap["cohens_dz"],
            "tiny_gap_ci": t_gap["bootstrap_ci_95"],
            "small_gap_ci": s_gap["bootstrap_ci_95"],
            "tiny_fdr_q": t_gap["fdr_p_value"],
            "small_fdr_q": s_gap["fdr_p_value"],
            "tiny_zero_damage": t_zero_damage,
            "small_zero_damage": s_zero_damage,
            "tiny_cross_damage": t_cross_damage,
            "small_cross_damage": s_cross_damage
        }

    # Replicated depths where BOTH architectures satisfy Outcome C criteria
    replicated_fungible_depths = [d for d in tested_depths if per_depth_eval[d]["both_passed"]]

    rationale = []
    if len(replicated_fungible_depths) > 0:
        outcome = "OUTCOME C"
        decision = "OUTCOME C - CONTENT FUNGIBILITY: Evidence supports patch-token content fungibility across replicated depths."
        for d in replicated_fungible_depths:
            t_eval = per_depth_eval[d]
            rationale.append(
                f"Depth {d}: Tiny (dz={t_eval['tiny_gap_dz']:.3f}, 95% CI [{t_eval['tiny_gap_ci'][0]:.3f}, {t_eval['tiny_gap_ci'][1]:.3f}], q={t_eval['tiny_fdr_q']:.4e}) "
                f"and Small (dz={t_eval['small_gap_dz']:.3f}, 95% CI [{t_eval['small_gap_ci'][0]:.3f}, {t_eval['small_gap_ci'][1]:.3f}], q={t_eval['small_fdr_q']:.4e}) "
                f"both demonstrate significant positive FungibilityGap with meaningful zero degradation."
            )
    else:
        # Check if the only reason for low cross-image damage is token irrelevance
        all_late_irrelevant = (10 in token_irrelevant_depths_tiny) and (10 in token_irrelevant_depths_small)
        any_partial_pass = len(passed_depths_tiny) > 0 or len(passed_depths_small) > 0

        if all_late_irrelevant and not any_partial_pass:
            outcome = "OUTCOME B"
            decision = "OUTCOME B - TOKEN IRRELEVANCE ONLY: Patch tokens become causally unimportant at late depths without content fungibility."
            rationale.append(f"Late depth(s) show Zero ≈ Baseline and Cross-Image ≈ Baseline (Tiny irrelevance at {token_irrelevant_depths_tiny}, Small at {token_irrelevant_depths_small}).")
            rationale.append("Tokens simply become causally inert rather than content-fungible.")
        else:
            outcome = "OUTCOME A"
            decision = "OUTCOME A - KILL: No evidence that patch-token slots become content-fungible across depth."
            rationale.append(f"No tested depth satisfied all pre-registered criteria on both architectures simultaneously.")
            if any_partial_pass:
                rationale.append(f"Single-model partial candidate depths (Tiny: {passed_depths_tiny}, Small: {passed_depths_small}) failed cross-architecture replication.")
            else:
                rationale.append("FungibilityGap failed dz >= 0.20, bootstrap CI > 0, or BH-FDR q < 0.05 across all depths on one or both models.")

    return {
        "outcome": outcome,
        "decision": decision,
        "replicated_fungible_depths": replicated_fungible_depths,
        "passed_depths_tiny": passed_depths_tiny,
        "passed_depths_small": passed_depths_small,
        "per_depth_eval": per_depth_eval,
        "rationale": rationale
    }
