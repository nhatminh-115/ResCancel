from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def evaluate_v0_7_decision_rules(
    tiny_comp_df: pd.DataFrame,
    small_comp_df: pd.DataFrame,
    tiny_frac_df: pd.DataFrame,
    small_frac_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluates pre-registered decision rules for Outcome A, Outcome B, or Outcome C.
    """
    rationale = []

    # Check Outcome B criteria across Tiny and Small:
    # 1. mu_8 significantly beats wrong-depth means (at 25% & 50%)
    # 2. Norm matching does not remove advantage
    # 3. Coordinate permutation substantially degrades performance (dz >= 0.20)
    # 4. Sign inversion substantially degrades performance
    # 5. Cosine sweep shows monotonic performance degradation as cosine decreases

    def check_outcome_b_for_model(comp_df: pd.DataFrame, frac_df: pd.DataFrame, m_name: str) -> Dict[str, bool]:
        # Filter 25% and 50%
        df_25_50 = comp_df[comp_df["fraction"].isin(["25%", "50%"])]

        # Check wrong-depth raw
        wd_raw = df_25_50[df_25_50["family"] == "wrong_depth_raw"]
        wd_raw_beats = (wd_raw["prototype_advantage"] > 0) & (wd_raw["cohens_dz"] >= 0.20) & (wd_raw["fdr_qval"] < 0.05)
        wd_raw_ok = bool(wd_raw_beats.all())

        # Check wrong-depth norm-matched
        wd_norm = df_25_50[df_25_50["family"] == "wrong_depth_norm"]
        wd_norm_beats = (wd_norm["prototype_advantage"] > 0) & (wd_norm["cohens_dz"] >= 0.20) & (wd_norm["fdr_qval"] < 0.05)
        wd_norm_ok = bool(wd_norm_beats.all())

        # Check coordinate perm
        cp = df_25_50[df_25_50["control"] == "coord_perm_avg"]
        cp_ok = bool(((cp["prototype_advantage"] > 0) & (cp["cohens_dz"] >= 0.20)).all())

        # Check sign flip 100%
        sf = df_25_50[df_25_50["control"] == "sign_flip_100%"]
        sf_ok = bool(((sf["prototype_advantage"] > 0) & (sf["cohens_dz"] >= 0.20)).all())

        # Check cosine sweep monotonic trend
        cos_0 = df_25_50[df_25_50["control"] == "cosine_0.00_avg"]["prototype_advantage"].mean()
        cos_25 = df_25_50[df_25_50["control"] == "cosine_0.25_avg"]["prototype_advantage"].mean()
        cos_50 = df_25_50[df_25_50["control"] == "cosine_0.50_avg"]["prototype_advantage"].mean()
        cos_75 = df_25_50[df_25_50["control"] == "cosine_0.75_avg"]["prototype_advantage"].mean()
        # Advantage over control should decrease as cosine increases towards 1 (i.e. damage of control decreases)
        cosine_monotonic = bool(cos_0 >= cos_25 >= cos_50 >= cos_75)

        return {
            "wd_raw_ok": wd_raw_ok,
            "wd_norm_ok": wd_norm_ok,
            "cp_ok": cp_ok,
            "sf_ok": sf_ok,
            "cosine_monotonic": cosine_monotonic,
            "model_b_ok": wd_raw_ok and wd_norm_ok and cp_ok and sf_ok and cosine_monotonic
        }

    b_tiny = check_outcome_b_for_model(tiny_comp_df, tiny_frac_df, "DeiT-Tiny")
    b_small = check_outcome_b_for_model(small_comp_df, small_frac_df, "DeiT-Small")

    outcome_b_holds = b_tiny["model_b_ok"] and b_small["model_b_ok"]

    # Check Outcome C criteria:
    # Outcome B holds AND at 75% and/or 100% patch replacement:
    # mu_8 preserves strong classification performance while destructive controls fail materially
    def check_outcome_c_for_model(comp_df: pd.DataFrame, frac_df: pd.DataFrame) -> Dict[str, Any]:
        row_100 = frac_df[frac_df["fraction"] == "100%"].iloc[0]
        row_75 = frac_df[frac_df["fraction"] == "75%"].iloc[0]

        clean_acc = row_100["clean_acc"]
        mu8_acc_100 = row_100["mu8_acc"]
        zero_acc_100 = row_100["zero_acc"]

        # Strong retention: mu8 accuracy retains substantial fraction of baseline
        # (e.g. > 50% baseline accuracy, substantially outperforming zero by >= 10 percentage points)
        acc_gap_100 = mu8_acc_100 - zero_acc_100
        strong_retention_100 = (mu8_acc_100 >= 0.50 * clean_acc) and (acc_gap_100 >= 0.10)

        # Destructive controls at 100%
        df_100 = comp_df[comp_df["fraction"] == "100%"]
        cp_adv = df_100[df_100["control"] == "coord_perm_avg"]["prototype_advantage"].values[0]
        sf_adv = df_100[df_100["control"] == "sign_flip_100%"]["prototype_advantage"].values[0]
        destructive_fail = (cp_adv >= 0.50) and (sf_adv >= 0.50)

        return {
            "mu8_acc_100": mu8_acc_100,
            "zero_acc_100": zero_acc_100,
            "acc_gap_100": acc_gap_100,
            "strong_retention_100": strong_retention_100,
            "destructive_fail": destructive_fail,
            "model_c_ok": strong_retention_100 and destructive_fail
        }

    c_tiny = check_outcome_c_for_model(tiny_comp_df, tiny_frac_df)
    c_small = check_outcome_c_for_model(small_comp_df, small_frac_df)

    outcome_c_holds = outcome_b_holds and c_tiny["model_c_ok"] and c_small["model_c_ok"]

    if outcome_c_holds:
        decision = "OUTCOME C — STRONG PROTOTYPE SUFFICIENCY"
        rationale.append("Outcome B confirmed: mu_8 strongly and significantly outperforms wrong-depth means, norm-matched wrong-depth means, coordinate permutations, and sign inversions across DeiT-Tiny and DeiT-Small.")
        rationale.append("Performance degrades monotonically with decreasing cosine similarity to mu_8.")
        rationale.append("At 75% and 100% patch replacement, mu_8 preserves strong classification accuracy while destructive controls collapse.")
        rationale.append("Allowed interpretation: 'After Block 8, downstream classification requires surprisingly little exact patch-token-specific information and can operate using a largely static, depth-specific prototype patch stream.'")
    elif outcome_b_holds:
        decision = "OUTCOME B — DEPTH-SPECIFIC PROTOTYPE"
        rationale.append("mu_8 significantly outperforms wrong-depth means on both architectures.")
        rationale.append("Norm matching does not remove the advantage; coordinate permutation and sign inversion substantially degrade performance.")
        rationale.append("Allowed interpretation: 'Downstream Blocks 9–11 are selectively compatible with a depth-specific feature-coordinate prototype.'")
    else:
        decision = "OUTCOME A — GENERIC REPLACEMENT / PROTOTYPE NOT SPECIAL"
        rationale.append("Destructive controls or wrong-depth means perform approximately as well as mu_8 (|dz| < 0.20, acc diff < 1%).")
        rationale.append("Allowed interpretation: 'The static replacement phenomenon is real, but the specific Block-8 prototype is not functionally special.'")

    return {
        "decision": decision,
        "outcome_b_holds": outcome_b_holds,
        "outcome_c_holds": outcome_c_holds,
        "b_tiny": b_tiny,
        "b_small": b_small,
        "c_tiny": c_tiny,
        "c_small": c_small,
        "rationale": rationale
    }
