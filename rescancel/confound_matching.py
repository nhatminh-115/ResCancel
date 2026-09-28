import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar


def match_controls(
    df: pd.DataFrame,
    is_extreme_col: str = "is_extreme",
    token_type: str = "cls",
    caliper_x_pct: float = 0.15,
    caliper_delta_pct: float = 0.15,
    caliper_margin: float = 0.50,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constructs matched control pairs for high-cancellation events adhering strictly
    to the frozen calipers in docs/V0_PROTOCOL.md:
      1. Exact match on layer
      2. Exact match on site
      3. Exact match on token_type (default: 'cls')
      4. |x_ctrl - x_ext| / x_ext <= 0.15
      5. |delta_ctrl - delta_ext| / delta_ext <= 0.15
      6. |margin_ctrl - margin_ext| <= 0.50
      
    Among all eligible controls for a given extreme event, chooses the nearest by
    standardized caliper distance:
      dist = ((x_c - x_e) / (0.15 * x_e))^2 + ((d_c - d_e) / (0.15 * d_e))^2 + ((m_c - m_e) / 0.50)^2
      
    1:1 matching without replacement. If no eligible control satisfies all calipers,
    the extreme event remains unmatched.
    """
    # Filter to requested token_type (CLS for primary analysis)
    if "token_type" in df.columns:
        subset_df = df[df["token_type"] == token_type].copy()
    else:
        subset_df = df.copy()

    extreme_df = subset_df[subset_df[is_extreme_col] == True].copy()
    control_candidates_df = subset_df[subset_df[is_extreme_col] == False].copy()

    # Filter controls to low/normal cancellation to avoid borderline events
    if "cos_sim" in control_candidates_df.columns:
        control_candidates_df = control_candidates_df[control_candidates_df["cos_sim"] > -0.20].copy()

    total_extreme = len(extreme_df)
    if total_extreme == 0 or len(control_candidates_df) == 0:
        return pd.DataFrame(), {
            "total_extreme_candidates": total_extreme,
            "matched_pairs": 0,
            "matching_rate": 0.0,
            "protocol_violations": 0
        }

    matched_rows = []
    pair_id = 0
    used_control_indices = set()
    protocol_violations = 0

    # Group by exact discrete variables (layer, site)
    group_cols = [c for c in ["layer", "site"] if c in subset_df.columns]
    grouped_extreme = extreme_df.groupby(group_cols)
    grouped_control = control_candidates_df.groupby(group_cols)

    for key, ext_group in grouped_extreme:
        if key not in grouped_control.groups:
            continue
        ctrl_group = grouped_control.get_group(key)

        for ext_idx, ext_row in ext_group.iterrows():
            x_e = ext_row["x_norm"]
            d_e = ext_row["delta_norm"]
            m_e = ext_row["margin"]

            # Filter controls available and within all three calipers simultaneously
            avail_ctrl = ctrl_group.loc[~ctrl_group.index.isin(used_control_indices)]
            if len(avail_ctrl) == 0:
                break

            x_c = avail_ctrl["x_norm"].values
            d_c = avail_ctrl["delta_norm"].values
            m_c = avail_ctrl["margin"].values

            # Caliper masks
            mask_x = np.abs(x_c - x_e) / (np.abs(x_e) + 1e-8) <= caliper_x_pct
            mask_d = np.abs(d_c - d_e) / (np.abs(d_e) + 1e-8) <= caliper_delta_pct
            mask_m = np.abs(m_c - m_e) <= caliper_margin

            eligible_mask = mask_x & mask_d & mask_m
            eligible_indices = avail_ctrl.index[eligible_mask]

            if len(eligible_indices) == 0:
                continue  # No control meets all frozen calipers simultaneously

            # Deterministic distance among strictly eligible controls
            x_diff_norm = (x_c[eligible_mask] - x_e) / (caliper_x_pct * x_e + 1e-8)
            d_diff_norm = (d_c[eligible_mask] - d_e) / (caliper_delta_pct * d_e + 1e-8)
            m_diff_norm = (m_c[eligible_mask] - m_e) / caliper_margin
            dists = x_diff_norm ** 2 + d_diff_norm ** 2 + m_diff_norm ** 2

            best_match_idx = eligible_indices[np.argmin(dists)]
            ctrl_row = avail_ctrl.loc[best_match_idx]
            used_control_indices.add(best_match_idx)

            # Programmatic assertion checks per pair
            x_violation = abs(ctrl_row["x_norm"] - x_e) / (abs(x_e) + 1e-8) > caliper_x_pct
            d_violation = abs(ctrl_row["delta_norm"] - d_e) / (abs(d_e) + 1e-8) > caliper_delta_pct
            m_violation = abs(ctrl_row["margin"] - m_e) > caliper_margin
            layer_violation = ctrl_row["layer"] != ext_row["layer"] if "layer" in ext_row else False
            site_violation = ctrl_row["site"] != ext_row["site"] if "site" in ext_row else False

            if x_violation or d_violation or m_violation or layer_violation or site_violation:
                protocol_violations += 1

            row_ext_dict = ext_row.to_dict()
            row_ext_dict["pair_id"] = pair_id
            row_ext_dict["group"] = "extreme"
            row_ext_dict["matching_dist"] = float(np.min(dists))

            row_ctrl_dict = ctrl_row.to_dict()
            row_ctrl_dict["pair_id"] = pair_id
            row_ctrl_dict["group"] = "control"
            row_ctrl_dict["matching_dist"] = float(np.min(dists))

            matched_rows.append(row_ext_dict)
            matched_rows.append(row_ctrl_dict)
            pair_id += 1

    matched_df = pd.DataFrame(matched_rows)
    stats_summary = {
        "token_type": token_type,
        "total_extreme_candidates": total_extreme,
        "matched_pairs": pair_id,
        "matching_rate": pair_id / max(total_extreme, 1),
        "protocol_violations": protocol_violations
    }

    # Strict assertion
    assert protocol_violations == 0, f"Protocol violations found in matching: {protocol_violations}"
    return matched_df, stats_summary


def compute_matched_margin_effect(
    matched_df: pd.DataFrame,
    outcome_col: str = "margin",
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired difference, paired t-test, and 10,000-resample bootstrap 95% CI
    for continuous logit margin between extreme cancellation and matched controls.
    """
    if len(matched_df) == 0 or "pair_id" not in matched_df.columns:
        return {
            "n_pairs": 0,
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0
        }

    pivoted = matched_df.pivot(index="pair_id", columns="group", values=outcome_col).dropna()
    if len(pivoted) < 2:
        return {
            "n_pairs": len(pivoted),
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0
        }

    diffs = (pivoted["extreme"] - pivoted["control"]).values
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    t_res = stats.ttest_rel(pivoted["extreme"], pivoted["control"])
    cohens_d = mean_diff / (std_diff + 1e-8)

    # Bootstrap 95% CI for paired mean difference
    rng = np.random.RandomState(seed)
    boot_means = rng.choice(diffs, size=(n_bootstrap, len(diffs)), replace=True).mean(axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "n_pairs": len(pivoted),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": float(t_res.statistic),
        "p_value": float(t_res.pvalue),
        "cohens_d": float(cohens_d)
    }


def compute_matched_binary_fragility_effect(
    matched_df: pd.DataFrame,
    outcome_col: str = "flipped"
) -> Dict[str, Any]:
    """
    Computes McNemar's test, paired risk difference, and 95% CI for binary fragility (flipped).
    """
    if len(matched_df) == 0 or "pair_id" not in matched_df.columns:
        return {
            "n_pairs": 0,
            "risk_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "mcnemar_stat": 0.0,
            "mcnemar_p_value": 1.0,
            "paired_odds_ratio": 1.0
        }

    pivoted = matched_df.pivot(index="pair_id", columns="group", values=outcome_col).dropna().astype(int)
    n_pairs = len(pivoted)
    if n_pairs < 2:
        return {
            "n_pairs": n_pairs,
            "risk_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "mcnemar_stat": 0.0,
            "mcnemar_p_value": 1.0,
            "paired_odds_ratio": 1.0
        }

    # 2x2 contingency table: rows=extreme (1/0), cols=control (1/0)
    # n11: ext=1, ctrl=1; n10: ext=1, ctrl=0; n01: ext=0, ctrl=1; n00: ext=0, ctrl=0
    n11 = int(((pivoted["extreme"] == 1) & (pivoted["control"] == 1)).sum())
    n10 = int(((pivoted["extreme"] == 1) & (pivoted["control"] == 0)).sum())
    n01 = int(((pivoted["extreme"] == 0) & (pivoted["control"] == 1)).sum())
    n00 = int(((pivoted["extreme"] == 0) & (pivoted["control"] == 0)).sum())

    table = [[n11, n10], [n01, n00]]

    # McNemar's test with continuity correction (or exact binomial if discordant < 25)
    discordant = n10 + n01
    if discordant == 0:
        mcnemar_stat = 0.0
        mcnemar_p = 1.0
    elif discordant < 25:
        # Exact binomial test
        mcnemar_stat = float(min(n10, n01))
        mcnemar_p = float(stats.binomtest(min(n10, n01), discordant, 0.5).pvalue)
    else:
        mcnemar_res = mcnemar(table, exact=False, correction=True)
        mcnemar_stat = float(mcnemar_res.statistic)
        mcnemar_p = float(mcnemar_res.pvalue)

    # Paired risk difference: P(ext=1) - P(ctrl=1) = (n10 - n01) / n_pairs
    risk_diff = (n10 - n01) / n_pairs
    # Asymptotic standard error for paired risk difference
    se_rd = np.sqrt(max(discordant - (n10 - n01) ** 2 / n_pairs, 0.0)) / n_pairs
    ci_lower = float(risk_diff - 1.96 * se_rd)
    ci_upper = float(risk_diff + 1.96 * se_rd)

    paired_odds_ratio = float((n10 + 1e-8) / (n01 + 1e-8)) if discordant > 0 else 1.0

    return {
        "n_pairs": n_pairs,
        "n11_both_flipped": n11,
        "n10_extreme_only_flipped": n10,
        "n01_control_only_flipped": n01,
        "n00_neither_flipped": n00,
        "risk_diff": float(risk_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mcnemar_stat": mcnemar_stat,
        "mcnemar_p_value": mcnemar_p,
        "paired_odds_ratio": paired_odds_ratio
    }


def run_image_level_logistic_regression(
    img_df: pd.DataFrame,
    cls_event_df: pd.DataFrame,
    target_col: str = "any_flipped"
) -> Dict[str, Any]:
    """
    Fits image-level multivariable logistic regression respecting independent observations (N=1000).
    Regresses image-level fragility (any_flipped) on:
      - has_layer0_attn_extreme: binary indicator (NOT standardized)
      - clean_margin: continuous logit margin (standardized)
      - mean_x_norm: continuous feature norm (standardized)
      - mean_delta_norm: continuous update norm (standardized)
      
    Uses statsmodels.api.Logit for exact standard errors, p-values, and 95% CIs.
    """
    # 1. Construct image-level cancellation features from CLS events
    l0_attn = cls_event_df[(cls_event_df["layer"] == 0) & (cls_event_df["site"] == "attn")]
    l0_extreme_samples = set(l0_attn[l0_attn["is_extreme"] == True]["sample_id"])
    any_extreme_samples = set(cls_event_df[cls_event_df["is_extreme"] == True]["sample_id"])

    cls_counts = cls_event_df[cls_event_df["is_extreme"] == True].groupby("sample_id").size().to_dict()
    mean_x = cls_event_df.groupby("sample_id")["x_norm"].mean().to_dict()
    mean_d = cls_event_df.groupby("sample_id")["delta_norm"].mean().to_dict()

    analysis_df = img_df.copy()
    analysis_df["has_l0_attn_extreme"] = analysis_df["sample_id"].apply(lambda s: 1 if s in l0_extreme_samples else 0)
    analysis_df["has_any_cls_extreme"] = analysis_df["sample_id"].apply(lambda s: 1 if s in any_extreme_samples else 0)
    analysis_df["num_cls_extreme"] = analysis_df["sample_id"].apply(lambda s: cls_counts.get(s, 0))
    analysis_df["mean_x_norm"] = analysis_df["sample_id"].apply(lambda s: mean_x.get(s, 0.0))
    analysis_df["mean_delta_norm"] = analysis_df["sample_id"].apply(lambda s: mean_d.get(s, 0.0))

    # Standardize continuous covariates only (do NOT standardize binary indicators)
    m_mean, m_std = analysis_df["clean_margin"].mean(), max(analysis_df["clean_margin"].std(), 1e-6)
    x_mean, x_std = analysis_df["mean_x_norm"].mean(), max(analysis_df["mean_x_norm"].std(), 1e-6)
    d_mean, d_std = analysis_df["mean_delta_norm"].mean(), max(analysis_df["mean_delta_norm"].std(), 1e-6)

    analysis_df["margin_std"] = (analysis_df["clean_margin"] - m_mean) / m_std
    analysis_df["x_norm_std"] = (analysis_df["mean_x_norm"] - x_mean) / x_std
    analysis_df["delta_norm_std"] = (analysis_df["mean_delta_norm"] - d_mean) / d_std

    y = analysis_df[target_col].values.astype(float)
    X = analysis_df[["has_l0_attn_extreme", "margin_std", "x_norm_std", "delta_norm_std"]].copy()
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    res = logit_model.fit(disp=False)

    coef = float(res.params["has_l0_attn_extreme"])
    se = float(res.bse["has_l0_attn_extreme"])
    z_stat = float(res.tvalues["has_l0_attn_extreme"])
    p_val = float(res.pvalues["has_l0_attn_extreme"])
    ci_lower = float(res.conf_int().loc["has_l0_attn_extreme", 0])
    ci_upper = float(res.conf_int().loc["has_l0_attn_extreme", 1])

    odds_ratio = float(np.exp(coef))
    odds_ratio_ci_lower = float(np.exp(ci_lower))
    odds_ratio_ci_upper = float(np.exp(ci_upper))

    return {
        "n_independent_images": len(analysis_df),
        "target_col": target_col,
        "predictor": "has_l0_attn_extreme",
        "coef": coef,
        "std_err": se,
        "z_stat": z_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_lower": odds_ratio_ci_lower,
        "odds_ratio_ci_upper": odds_ratio_ci_upper,
        "aic": float(res.aic),
        "prsquared": float(res.prsquared)
    }
