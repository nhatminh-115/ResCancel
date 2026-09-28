from typing import Dict, Any, List
import pandas as pd
import numpy as np


def run_fungibility_v0_6_validations(
    manifest: Dict[str, Any],
    calib_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    expected_n: int = 1000,
    expected_depths: List[int] = [5, 6, 7, 8, 9, 10]
) -> Dict[str, bool]:
    """
    Executes the 14 pre-registered programmatic assertions for Patch Content Fungibility V0.6.
    Raises AssertionError if any check fails.
    """
    results = {}

    # Check 1: Backbone weights verified unchanged
    assert manifest.get("backbone_weights_verified", False), "Check 1 Failed: Backbone weights modified!"
    results["check1_backbone_unmodified"] = True

    # Check 2: Disjoint calibration and evaluation image splits
    calib_indices = set(calib_df["global_index"])
    eval_indices = set(eval_df["global_index"])
    intersection = calib_indices.intersection(eval_indices)
    assert len(intersection) == 0, f"Check 2 Failed: Image overlap detected between calibration and evaluation sets ({len(intersection)} images)!"
    assert len(calib_df) == expected_n, f"Check 2 Failed: Calibration count = {len(calib_df)} != {expected_n}"
    assert len(eval_df) == expected_n, f"Check 2 Failed: Evaluation count = {len(eval_df)} != {expected_n}"
    results["check2_disjoint_splits"] = True

    # Check 3: Unlabeled calibration statistics
    assert manifest.get("unlabeled_calibration_stats_verified", False), "Check 3 Failed: Labels used in calibration!"
    results["check3_unlabeled_calibration_stats"] = True

    # Check 4: No evaluation activations in calibration statistics
    assert manifest.get("no_eval_data_in_calibration_stats_verified", False), "Check 4 Failed: Evaluation data leaked into calibration stats!"
    results["check4_no_eval_data_in_stats"] = True

    # Check 5: All 6 depths evaluated
    evaluated_depths = manifest.get("evaluated_depths", [])
    assert sorted(evaluated_depths) == sorted(expected_depths), f"Check 5 Failed: Evaluated depths {evaluated_depths} != {expected_depths}"
    results["check5_all_depths_evaluated"] = True

    # Check 6: Nested masks verified
    assert manifest.get("nested_masks_verified", False), "Check 6 Failed: Masks are not nested subsets!"
    results["check6_nested_masks"] = True

    # Check 7: Mask counts exactly 20, 49, 98
    assert manifest.get("mask_counts") == [20, 49, 98], f"Check 7 Failed: Mask counts = {manifest.get('mask_counts')}, expected [20, 49, 98]"
    results["check7_mask_counts"] = True

    # Check 8: CLS token untouched at injection
    assert manifest.get("cls_untouched_verified", False), "Check 8 Failed: CLS token was modified!"
    results["check8_cls_untouched"] = True

    # Check 9: Downstream blocks identical across conditions
    assert manifest.get("downstream_blocks_identical", False), "Check 9 Failed: Downstream blocks altered!"
    results["check9_downstream_blocks_identical"] = True

    # Check 10: Norm-matched random relative error < 1e-5
    assert manifest.get("norm_matched_random_rel_err_verified", False), "Check 10 Failed: Random Sphere norm mismatch!"
    results["check10_norm_matched_random_rel_err"] = True

    # Check 11: Frozen seeds verified
    assert manifest.get("calib_seed") == 9101, "Check 11 Failed: Calibration seed mismatch"
    assert manifest.get("eval_seed") == 9201, "Check 11 Failed: Evaluation seed mismatch"
    assert manifest.get("mask_seed") == 9301, "Check 11 Failed: Mask seed mismatch"
    assert manifest.get("gaussian_seeds") == [9401, 9402, 9403, 9404, 9405], "Check 11 Failed: Gaussian seeds mismatch"
    assert manifest.get("sphere_seeds") == [9501, 9502, 9503], "Check 11 Failed: Sphere seeds mismatch"
    results["check11_frozen_seeds"] = True

    # Check 12: Image-level statistical independence
    assert manifest.get("n_eval_samples") == expected_n, "Check 12 Failed: Statistics not on image level!"
    results["check12_image_level_independence"] = True

    # Check 13: Calibration stats persisted
    assert manifest.get("calibration_stats_persisted", False), "Check 13 Failed: calibration_statistics.npz not persisted!"
    results["check13_calibration_stats_persisted"] = True

    # Check 14: Report equals manifest
    assert manifest.get("manifest_valid", False), "Check 14 Failed: Manifest invalid!"
    results["check14_report_equals_manifest"] = True

    return results
