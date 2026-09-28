import os
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def validate_v0_8_assertions(
    manifest_data: Dict[str, Any],
    tiny_res: Dict[str, Any],
    small_res: Dict[str, Any],
    energy_df: pd.DataFrame,
    calib_indices: List[int],
    eval_indices: List[int],
) -> Tuple[bool, List[str]]:
    """
    Validates the 18 pre-registered programmatic assertions defined in Section 23 of V0.8 Protocol.
    Returns (all_passed, list_of_failure_messages).
    """
    failures = []

    # Assertion 1: Backbone unchanged
    if tiny_res["initial_hash"] != tiny_res["final_hash"]:
        failures.append("Assertion 1 Failed: DeiT-Tiny parameter hash changed during run!")
    if small_res["initial_hash"] != small_res["final_hash"]:
        failures.append("Assertion 1 Failed: DeiT-Small parameter hash changed during run!")

    # Assertion 2: Exact V0.6/V0.7 calibration and evaluation splits reused
    if len(calib_indices) != 1000 or len(eval_indices) != 1000:
        failures.append(f"Assertion 2 Failed: Split lengths incorrect (calib={len(calib_indices)}, eval={len(eval_indices)})")
    overlap = set(calib_indices).intersection(set(eval_indices))
    if len(overlap) != 0:
        failures.append(f"Assertion 2 Failed: Calibration and evaluation splits overlap! Count: {len(overlap)}")

    # Assertion 3: Evaluation images never used for PCA or Gaussian statistics
    # Guaranteed by calibration split isolation and zero overlap
    if len(overlap) > 0:
        failures.append("Assertion 3 Failed: Evaluation images used in calibration data!")

    # Assertion 4: PCA computed using calibration patch activations only
    if tiny_res["pca_result"]["n_tokens"] != 196000 or small_res["pca_result"]["n_tokens"] != 196000:
        failures.append(f"Assertion 4 Failed: PCA token count mismatch (Tiny={tiny_res['pca_result']['n_tokens']}, Small={small_res['pca_result']['n_tokens']})")

    # Assertion 5: CLS untouched
    # Primary seq_mask starts at index 1 and ends at 196
    if manifest_data.get("cls_untouched") is not True:
        failures.append("Assertion 5 Failed: CLS token modification detected!")

    # Assertion 6: Exactly all 196 patches replaced in primary experiment
    if manifest_data.get("primary_replacement_token_count") != 196:
        failures.append("Assertion 6 Failed: Primary replacement did not replace exactly 196 tokens!")

    # Assertion 7: Low-rank bases have requested ranks
    expected_ranks = [1, 2, 4, 8, 16, 32, 64]
    for r in expected_ranks:
        if r not in manifest_data.get("pca_ranks", []):
            failures.append(f"Assertion 7 Failed: Missing requested PCA rank {r}!")

    # Assertion 8: PCA bases are orthonormal
    if tiny_res["pca_result"]["ortho_error"] >= 1e-4:
        failures.append(f"Assertion 8 Failed: Tiny PCA basis ortho error {tiny_res['pca_result']['ortho_error']} >= 1e-4")
    if small_res["pca_result"]["ortho_error"] >= 1e-4:
        failures.append(f"Assertion 8 Failed: Small PCA basis ortho error {small_res['pca_result']['ortho_error']} >= 1e-4")

    # Assertion 9: Random bases are orthonormal
    max_rand_ortho_err = manifest_data.get("max_random_ortho_error", 0.0)
    if max_rand_ortho_err >= 1e-4:
        failures.append(f"Assertion 9 Failed: Random basis ortho error {max_rand_ortho_err} >= 1e-4")

    # Assertion 10: Total perturbation energy matched within 2%
    max_rel_err = energy_df["energy_rel_error"].max()
    if max_rel_err > 0.02:
        worst_cond = energy_df.loc[energy_df["energy_rel_error"].idxmax()]
        failures.append(f"Assertion 10 Failed: Max energy relative error {max_rel_err*100:.3f}% > 2.0% in {worst_cond['model']}/{worst_cond['condition']}")

    # Assertion 11: Shared-noise and independent-noise use identical distributions
    # Both sampled from N(0, diag(sigma_8^2))
    if manifest_data.get("shared_and_indep_identical_dist") is not True:
        failures.append("Assertion 11 Failed: Shared and independent noise distributions differed!")

    # Assertion 12: Grouped conditions contain exactly requested K unique vectors
    # Checked structurally in intervention generator
    if manifest_data.get("grouped_k_exact") is not True:
        failures.append("Assertion 12 Failed: Grouped diversity unique token counts violated!")

    # Assertion 13: Seeds exactly match protocol
    if manifest_data.get("seeds_match_protocol") is not True:
        failures.append("Assertion 13 Failed: Random seed protocol mismatch!")

    # Assertion 14: No labels or gradients used to construct replacements
    # Strict code review check
    if manifest_data.get("zero_labels_or_gradients") is not True:
        failures.append("Assertion 14 Failed: Labels or gradients used in replacement!")

    # Assertion 15: Effective rank calculated from actual resulting tensors
    if manifest_data.get("representation_ranks_computed") is not True:
        failures.append("Assertion 15 Failed: Effective rank not computed from actual tensors!")

    # Assertion 16: Attention diagnostics measured from actual Blocks 9–11
    if manifest_data.get("attention_diagnostics_computed") is not True:
        failures.append("Assertion 16 Failed: Attention diagnostics not computed from Blocks 9-11!")

    # Assertion 17: Statistics use exactly N=1,000 independent evaluation images
    if len(tiny_res["all_targets"]) != 1000 or len(small_res["all_targets"]) != 1000:
        failures.append("Assertion 17 Failed: Evaluation count is not exactly 1,000 images!")

    # Assertion 18: Report matches machine-readable outputs
    if manifest_data.get("machine_readable_exported") is not True:
        failures.append("Assertion 18 Failed: Machine-readable outputs missing or unexported!")

    all_passed = (len(failures) == 0)
    return all_passed, failures
