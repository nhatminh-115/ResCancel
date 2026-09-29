import os
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def validate_v0_9_assertions(
    manifest_data: Dict[str, Any],
    tiny_res: Dict[str, Any],
    small_res: Dict[str, Any],
    calib_indices: List[int],
    eval_indices: List[int],
) -> Tuple[bool, List[str]]:
    """
    Validates all 18 pre-registered programmatic assertions defined in Section 4 of V0.9 Protocol.
    Returns (all_passed, list_of_failure_messages).
    """
    failures = []

    # Assertion 1: Backbone unchanged
    if tiny_res["initial_hash"] != tiny_res["final_hash"]:
        failures.append("Assertion 1 Failed: DeiT-Tiny parameter hash changed during run!")
    if small_res["initial_hash"] != small_res["final_hash"]:
        failures.append("Assertion 1 Failed: DeiT-Small parameter hash changed during run!")

    # Assertion 2: Split isolation (N=1,000 calib, N=1,000 eval, zero overlap)
    if len(calib_indices) != 1000 or len(eval_indices) != 1000:
        failures.append(f"Assertion 2 Failed: Split lengths incorrect (calib={len(calib_indices)}, eval={len(eval_indices)})")
    overlap = set(calib_indices).intersection(set(eval_indices))
    if len(overlap) != 0:
        failures.append(f"Assertion 2 Failed: Calibration and evaluation splits overlap! Count: {len(overlap)}")

    # Assertion 3: No evaluation leakage
    if len(overlap) > 0:
        failures.append("Assertion 3 Failed: Evaluation images used in calibration statistics!")

    # Assertion 4: Calibration PCA integrity
    if tiny_res["pca_result"]["n_tokens"] != 196000 or small_res["pca_result"]["n_tokens"] != 196000:
        failures.append(f"Assertion 4 Failed: Calibration token count mismatch (Tiny={tiny_res['pca_result']['n_tokens']}, Small={small_res['pca_result']['n_tokens']})")

    # Assertion 5: CLS untouched
    if manifest_data.get("cls_untouched") is not True:
        failures.append("Assertion 5 Failed: CLS token modification detected!")

    # Assertion 6: Exactly 196 spatial patches replaced
    if manifest_data.get("replacement_token_count") != 196:
        failures.append("Assertion 6 Failed: Spatial replacement token count is not exactly 196!")

    # Assertion 7: Natural PCA coefficients use unscaled empirical eigenvalues
    if manifest_data.get("natural_pca_unscaled") is not True:
        failures.append("Assertion 7 Failed: Natural PCA coefficients were rescaled!")

    # Assertion 8: Energy-matched reference reproduces V0.8 formulation
    if manifest_data.get("energy_matched_reproduced") is not True:
        failures.append("Assertion 8 Failed: Energy-matched PCA formulation mismatch!")

    # Assertion 9: Amplitude factors match protocol exactly
    expected_scales = [0.25, 0.5, 1.0, 2.0, 4.0, "E_MATCH"]
    actual_scales = manifest_data.get("pc1_scale_multipliers", [])
    if actual_scales != expected_scales:
        failures.append(f"Assertion 9 Failed: PC1 scale multipliers mismatch! Expected {expected_scales}, got {actual_scales}")

    # Assertion 10: Random unit vectors have ||u||_2 = 1.0 +- 1e-4
    max_u_err = manifest_data.get("max_random_u_norm_error", 0.0)
    if max_u_err > 1e-4:
        failures.append(f"Assertion 10 Failed: Random vector norm error {max_u_err} > 1e-4")

    # Assertion 11: Random 1D variance exactly equals lambda_1
    if manifest_data.get("random_1d_variance_matched") is not True:
        failures.append("Assertion 11 Failed: Random 1D variance did not equal lambda_1!")

    # Assertion 12: All PCA vectors orthonormal (max Gram error < 1e-4)
    if tiny_res["pca_result"]["ortho_error"] >= 1e-4:
        failures.append(f"Assertion 12 Failed: Tiny PCA basis ortho error {tiny_res['pca_result']['ortho_error']} >= 1e-4")
    if small_res["pca_result"]["ortho_error"] >= 1e-4:
        failures.append(f"Assertion 12 Failed: Small PCA basis ortho error {small_res['pca_result']['ortho_error']} >= 1e-4")

    # Assertion 13: Actual perturbation energy logged for every condition
    if manifest_data.get("perturbation_energy_logged") is not True:
        failures.append("Assertion 13 Failed: Perturbation energy not logged for all conditions!")

    # Assertion 14: Per-image rank computation before averaging
    if manifest_data.get("per_image_rank_computed") is not True:
        failures.append("Assertion 14 Failed: Representation rank was averaged before per-image computation!")

    # Assertion 15: Attention diagnostics measured from actual forward pass
    if manifest_data.get("attention_diagnostics_measured") is not True:
        failures.append("Assertion 15 Failed: Attention diagnostics not measured from Blocks 9-11!")

    # Assertion 16: Seeds protocol compliant
    if manifest_data.get("seeds_protocol_compliant") is not True:
        failures.append("Assertion 16 Failed: Seed protocol mismatch detected!")

    # Assertion 17: No labels or gradients used
    if manifest_data.get("zero_labels_or_gradients") is not True:
        failures.append("Assertion 17 Failed: Labels or gradients used in replacement!")

    # Assertion 18: Exactly N=1,000 independent evaluation images
    if len(tiny_res["all_targets"]) != 1000 or len(small_res["all_targets"]) != 1000:
        failures.append("Assertion 18 Failed: Evaluation set is not exactly 1,000 images!")

    all_passed = (len(failures) == 0)
    return all_passed, failures
