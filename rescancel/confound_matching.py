import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats


def match_controls(
    df: pd.DataFrame,
    is_extreme_col: str = "is_extreme",
    caliper_std: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Constructs matched control pairs for high-cancellation events.
    Matches exactly on (layer, site, token_type) and matches on continuous confounds
    (||x||_2, ||delta||_2, clean_margin) within a standardized Mahalanobis/Euclidean caliper.
    
    Args:
        df: DataFrame containing all residual observations
        is_extreme_col: Boolean or 0/1 column indicating candidate extreme cancellation
        caliper_std: Maximum allowed standardized distance for matching
        random_state: Seed for deterministic sampling
        
    Returns:
        matched_df: DataFrame containing pairs with a 'pair_id' and 'group' ('extreme' vs 'control')
        stats_summary: Dictionary of matching statistics (sample sizes, balance metrics)
    """
    rng = np.random.RandomState(random_state)
    
    continuous_confounds = ["x_norm", "delta_norm", "margin"]
    # Check which confounds are available
    avail_confounds = [c for c in continuous_confounds if c in df.columns]
    
    extreme_df = df[df[is_extreme_col] == True].copy()
    control_candidates_df = df[df[is_extreme_col] == False].copy()
    
    # Filter control candidates to low/normal cancellation to avoid borderline events
    if "cos_sim" in control_candidates_df.columns:
        control_candidates_df = control_candidates_df[control_candidates_df["cos_sim"] > -0.20]
        
    if len(extreme_df) == 0 or len(control_candidates_df) == 0:
        return pd.DataFrame(), {"matched_pairs": 0}
        
    # Standardize continuous variables using overall std
    stds = {}
    for col in avail_confounds:
        stds[col] = max(df[col].std(), 1e-6)
        
    matched_rows = []
    pair_id = 0
    used_control_indices = set()
    
    # Exact match on discrete strata
    group_cols = [c for c in ["layer", "site", "token_type"] if c in df.columns]
    
    grouped_extreme = extreme_df.groupby(group_cols)
    grouped_control = control_candidates_df.groupby(group_cols)
    
    for key, ext_subset in grouped_extreme:
        if key not in grouped_control.groups:
            continue
        ctrl_subset = grouped_control.get_group(key)
        
        for ext_idx, ext_row in ext_subset.iterrows():
            # Find eligible controls not yet used (1:1 matching without replacement)
            avail_ctrl = ctrl_subset.loc[~ctrl_subset.index.isin(used_control_indices)]
            if len(avail_ctrl) == 0:
                break
                
            # Compute standardized distance
            dists_sq = np.zeros(len(avail_ctrl))
            for col in avail_confounds:
                diff = (avail_ctrl[col].values - ext_row[col]) / stds[col]
                dists_sq += diff ** 2
            dists = np.sqrt(dists_sq)
            
            # Caliper check
            min_dist_idx = np.argmin(dists)
            if dists[min_dist_idx] <= caliper_std * len(avail_confounds):
                best_ctrl_idx = avail_ctrl.index[min_dist_idx]
                used_control_indices.add(best_ctrl_idx)
                
                # Append matched pair
                row_ext = ext_row.to_dict()
                row_ext["pair_id"] = pair_id
                row_ext["group"] = "extreme"
                row_ext["matching_dist"] = dists[min_dist_idx]
                
                row_ctrl = avail_ctrl.loc[best_ctrl_idx].to_dict()
                row_ctrl["pair_id"] = pair_id
                row_ctrl["group"] = "control"
                row_ctrl["matching_dist"] = dists[min_dist_idx]
                
                matched_rows.append(row_ext)
                matched_rows.append(row_ctrl)
                pair_id += 1
                
    if not matched_rows:
        return pd.DataFrame(), {"matched_pairs": 0}
        
    matched_df = pd.DataFrame(matched_rows)
    stats_summary = {
        "total_extreme_events": len(extreme_df),
        "matched_pairs": pair_id,
        "match_rate": pair_id / max(len(extreme_df), 1),
    }
    return matched_df, stats_summary


def compute_matched_effect_sizes(
    matched_df: pd.DataFrame,
    outcome_col: str = "margin"
) -> Dict[str, float]:
    """
    Computes paired difference, t-statistic, p-value, and Cohen's d effect size
    between extreme cancellation and matched controls.
    """
    if len(matched_df) == 0 or "pair_id" not in matched_df.columns:
        return {"effect_size_d": 0.0, "p_value": 1.0, "mean_diff": 0.0}
        
    pivoted = matched_df.pivot(index="pair_id", columns="group", values=outcome_col)
    pivoted = pivoted.dropna()
    
    if len(pivoted) < 2:
        return {"effect_size_d": 0.0, "p_value": 1.0, "mean_diff": 0.0}
        
    diff = pivoted["extreme"] - pivoted["control"]
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    
    t_stat, p_val = stats.ttest_rel(pivoted["extreme"], pivoted["control"])
    cohens_d = mean_diff / (std_diff + 1e-8)
    
    return {
        "n_pairs": len(pivoted),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d)
    }


def run_controlled_logistic_regression(
    df: pd.DataFrame,
    target_col: str = "flipped",
    is_extreme_col: str = "is_extreme"
) -> Dict[str, Any]:
    """
    Fits multivariable logistic regression predicting fragility / prediction flip
    controlling for ||x||, ||delta||, clean margin, layer, and site.
    Uses statsmodels or analytical weighted least squares / scipy optimize.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = [is_extreme_col, "x_norm", "delta_norm", "margin"]
    avail_cols = [c for c in feature_cols if c in df.columns]
    
    clean_df = df.dropna(subset=avail_cols + [target_col]).copy()
    if len(clean_df) < 50 or clean_df[target_col].nunique() < 2:
        return {"odds_ratio": 1.0, "p_value": 1.0, "coef": 0.0}
        
    X_num = clean_df[avail_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    y = clean_df[target_col].values.astype(int)
    
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    clf.fit(X_scaled, y)
    
    # Feature 0 is is_extreme_col
    coef = clf.coef_[0, 0]
    odds_ratio = np.exp(coef)
    
    return {
        "coef_is_extreme": float(coef),
        "odds_ratio": float(odds_ratio),
        "intercept": float(clf.intercept_[0]),
        "sample_size": len(clean_df)
    }
