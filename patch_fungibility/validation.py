from typing import Dict, Any, List
import pandas as pd
import numpy as np


def run_fungibility_v0_validations(
    image_df: pd.DataFrame,
    manifest: Dict[str, Any],
    expected_n: int = 1000,
    expected_depths: List[int] = [2, 4, 6, 8, 10]
) -> Dict[str, bool]:
    """
    Executes the 14 pre-registered programmatic assertions for Patch Content Fungibility V0.
    Raises AssertionError if any check fails.
    """
    results = {}

    # Check 1: Backbone weights verified unchanged
    assert manifest.get("backbone_weights_verified", False), "Check 1 Failed: Backbone weights were modified!"
    results["check1_backbone_unmodified"] = True

    # Check 2: Same N=1000 image IDs used everywhere
    assert len(image_df) == expected_n, f"Check 2 Failed: Image count is {len(image_df)}, expected {expected_n}"
    assert image_df["sample_id"].nunique() == expected_n, "Check 2 Failed: Sample IDs are not unique!"
    results["check2_same_image_ids"] = True

    # Check 3: Exactly 49 patches modified per image
    assert manifest.get("mask_count") == 49, f"Check 3 Failed: Modified patches = {manifest.get('mask_count')}, expected 49"
    results["check3_exactly_49_patches"] = True

    # Check 4: Same patch mask across interventions
    assert manifest.get("same_mask_verified", False), "Check 4 Failed: Different patch masks used across conditions"
    results["check4_same_patch_mask"] = True

    # Check 5: CLS token untouched at injection
    assert manifest.get("cls_untouched_verified", False), "Check 5 Failed: CLS token was modified!"
    results["check5_cls_untouched"] = True

    # Check 6: Donor image j != i for every image
    assert "donor_sample_id" in image_df.columns, "Check 6 Failed: Missing donor_sample_id in image records"
    assert (image_df["sample_id"] != image_df["donor_sample_id"]).all(), "Check 6 Failed: Donor image equals target image!"
    results["check6_donor_image_not_self"] = True

    # Check 7: Same-position donor preserves patch coordinate
    assert manifest.get("same_position_preserves_coord", False), "Check 7 Failed: Same-position condition altered coordinates"
    results["check7_same_position_donor_coordinate"] = True

    # Check 8: Random-position donor does not preserve coordinate
    assert manifest.get("rand_pos_perm_derangement", False), "Check 8 Failed: Random-position permutation has fixed points"
    results["check8_random_position_donor_coordinate"] = True

    # Check 9: Within-image shuffle is a true permutation
    assert manifest.get("within_image_shuffle_derangement", False), "Check 9 Failed: Within-image shuffle is not a derangement"
    results["check9_within_image_shuffle_bijective"] = True

    # Check 10: Downstream blocks identical across conditions
    assert manifest.get("downstream_blocks_identical", False), "Check 10 Failed: Downstream blocks altered"
    results["check10_downstream_blocks_identical"] = True

    # Check 11: Frozen random seeds
    assert manifest.get("mask_seed") == 2501, "Check 11 Failed: Mask seed mismatch"
    assert manifest.get("donor_seed") == 3501, "Check 11 Failed: Donor seed mismatch"
    assert manifest.get("rand_pos_seed") == 4501, "Check 11 Failed: Random pos seed mismatch"
    assert manifest.get("within_shuff_seed") == 5501, "Check 11 Failed: Within shuffle seed mismatch"
    results["check11_frozen_random_seeds"] = True

    # Check 12: Statistics use image-level observations
    assert len(image_df) == expected_n, f"Check 12 Failed: Statistics not on image level (N={len(image_df)})"
    results["check12_image_level_statistics"] = True

    # Check 13: Activation replacement norms are logged
    assert manifest.get("activation_norms_logged", False), "Check 13 Failed: Activation norms were not logged"
    results["check13_activation_norms_logged"] = True

    # Check 14: All 5 tested depths are evaluated
    evaluated_depths = manifest.get("evaluated_depths", [])
    assert sorted(evaluated_depths) == sorted(expected_depths), f"Check 14 Failed: Evaluated depths {evaluated_depths} != {expected_depths}"
    results["check14_all_depths_evaluated"] = True

    return results
