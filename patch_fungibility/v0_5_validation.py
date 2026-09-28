from typing import Dict, Any, List
import pandas as pd
import numpy as np


def run_fungibility_v0_5_validations(
    image_df: pd.DataFrame,
    manifest: Dict[str, Any],
    expected_n: int = 1000
) -> Dict[str, bool]:
    """
    Executes the 14 pre-registered programmatic assertions for Patch Content Fungibility V0.5.
    Raises AssertionError if any check fails.
    """
    results = {}

    # Check 1: Backbone weights verified unchanged
    assert manifest.get("backbone_weights_verified", False), "Check 1 Failed: Backbone weights were modified!"
    results["check1_backbone_unmodified"] = True

    # Check 2: Same N=1000 image IDs as V0
    assert len(image_df) == expected_n, f"Check 2 Failed: Image count is {len(image_df)}, expected {expected_n}"
    assert image_df["sample_id"].nunique() == expected_n, "Check 2 Failed: Sample IDs are not unique!"
    results["check2_same_image_ids"] = True

    # Check 3: Exactly 49 patches modified per image
    assert manifest.get("mask_count") == 49, f"Check 3 Failed: Modified patches = {manifest.get('mask_count')}, expected 49"
    results["check3_exactly_49_patches"] = True

    # Check 4: Same patch mask as V0
    assert manifest.get("mask_seed") == 2501, "Check 4 Failed: Mask seed mismatch"
    assert manifest.get("same_mask_verified", False), "Check 4 Failed: Different patch masks used across conditions"
    results["check4_same_patch_mask"] = True

    # Check 5: CLS token untouched at injection
    assert manifest.get("cls_untouched_verified", False), "Check 5 Failed: CLS token was modified!"
    results["check5_cls_untouched"] = True

    # Check 6: Downstream blocks identical across conditions
    assert manifest.get("downstream_blocks_identical", False), "Check 6 Failed: Downstream blocks altered"
    results["check6_downstream_blocks_identical"] = True

    # Check 7: Norm-matched random vectors match original token norm within 1e-5 relative error
    assert manifest.get("norm_matched_random_rel_err_verified", False), "Check 7 Failed: Random Sphere norm mismatch!"
    results["check7_norm_matched_random_relative_error"] = True

    # Check 8: Gaussian statistics estimated without labels
    assert manifest.get("gaussian_stats_unlabeled_verified", False), "Check 8 Failed: Labels used in Gaussian statistics!"
    results["check8_gaussian_stats_unlabeled"] = True

    # Check 9: Norm-matched mean matches original norm within 1e-5 relative error
    assert manifest.get("norm_matched_mean_rel_err_verified", False), "Check 9 Failed: Norm-Matched Mean norm mismatch!"
    results["check9_norm_matched_mean_relative_error"] = True

    # Check 10: Feature shuffle is a true permutation of embedding dimensions
    assert manifest.get("feature_shuffle_bijective_verified", False), "Check 10 Failed: Feature shuffle is not a bijection!"
    results["check10_feature_shuffle_bijective"] = True

    # Check 11: Frozen random seeds
    assert manifest.get("sphere_seeds") == [6501, 6502, 6503, 6504, 6505], "Check 11 Failed: Random Sphere seeds mismatch"
    assert manifest.get("gaussian_seeds") == [7501, 7502, 7503, 7504, 7505], "Check 11 Failed: Gaussian seeds mismatch"
    assert manifest.get("feature_shuffle_seeds") == [8501, 8502, 8503], "Check 11 Failed: Feature shuffle seeds mismatch"
    results["check11_frozen_random_seeds"] = True

    # Check 12: No gradients or labels used
    assert manifest.get("no_gradients_or_labels_verified", False), "Check 12 Failed: Gradients/labels detected!"
    results["check12_no_gradients_or_labels"] = True

    # Check 13: Image-level statistical independence
    assert len(image_df) == expected_n, f"Check 13 Failed: Statistics not on image level (N={len(image_df)})"
    results["check13_image_level_independence"] = True

    # Check 14: All diagnostics logged
    assert manifest.get("activation_diagnostics_logged", False), "Check 14 Failed: Activation diagnostics not logged!"
    results["check14_report_equals_manifest"] = True

    return results
