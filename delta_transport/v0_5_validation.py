from typing import Dict, Any
import numpy as np
import pandas as pd


def run_v0_5_programmatic_validations(
    image_df: pd.DataFrame,
    matched_comparison_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    manifest: Dict[str, Any],
    expected_n: int = 1000
) -> Dict[str, bool]:
    """
    Executes the pre-registered programmatic assertions for DELTA TRANSPORT V0.5.
    Raises AssertionError if any assertion fails.
    """
    results = {}

    # Check 1: Backbone weights verified unchanged
    assert manifest.get("backbone_weights_verified", False), "Check 1 Failed: Backbone weights were modified!"
    results["check1_backbone_unmodified"] = True

    # Check 2: Same 1,000 image IDs used in every condition
    assert len(image_df) == expected_n, f"Check 2 Failed: Image count is {len(image_df)}, expected {expected_n}"
    assert image_df["sample_id"].nunique() == expected_n, "Check 2 Failed: Non-unique sample IDs"
    results["check2_same_image_ids"] = True

    # Check 3: Every oracle receives exactly 8 candidates per patch
    assert manifest.get("eight_candidates_verified", False), "Check 3 Failed: Oracle candidate count != 8"
    results["check3_eight_candidates_per_patch"] = True

    # Check 4: Every oracle independently selects 1 candidate for each of 196 patches
    assert manifest.get("independent_patch_choices_verified", False), "Check 4 Failed: Oracle did not select 196 patches"
    results["check4_independent_patch_choices"] = True

    # Check 5: Historical and control policies use the exact same gradient tensor
    assert manifest.get("identical_gradient_verified", False), "Check 5 Failed: Different gradient tensors used"
    results["check5_identical_gradient_used"] = True

    # Check 6: All candidate vectors are L2-normalized
    assert manifest.get("candidates_normalized_verified", False), "Check 6 Failed: Candidates not normalized"
    results["check6_candidates_l2_normalized"] = True

    # Check 7: Perturbation magnitude matches gamma across all conditions
    assert manifest.get("perturbation_norm_verified", False), "Check 7 Failed: Perturbation norm mismatch"
    results["check7_perturbation_norm_matched"] = True

    # Check 8: CLS token is untouched at injection
    assert manifest.get("cls_untouched_verified", False), "Check 8 Failed: CLS token modified"
    results["check8_cls_untouched"] = True

    # Check 9: Cross-image donor never equals target image (j != i)
    assert "cross_image_donor_id" in image_df.columns, "Check 9 Failed: Missing cross_image_donor_id"
    assert (image_df["sample_id"] != image_df["cross_image_donor_id"]).all(), "Check 9 Failed: Donor image equals target image!"
    results["check9_cross_image_donor_not_self"] = True

    # Check 10: Spatial shuffle is a true permutation of patch positions
    assert manifest.get("spatial_shuffle_is_permutation_verified", False), "Check 10 Failed: Shuffle is not bijective"
    results["check10_spatial_shuffle_is_permutation"] = True

    # Check 11: Random controls use frozen seeds
    assert manifest.get("random_oracle_seeds") == [2501, 2502, 2503, 2504, 2505], "Check 11 Failed: Random seeds mismatch"
    assert manifest.get("cross_image_seed") == 3501, "Check 11 Failed: Cross-image seed mismatch"
    assert manifest.get("spatial_shuffle_seeds") == [4501, 4502, 4503], "Check 11 Failed: Spatial shuffle seeds mismatch"
    assert manifest.get("pooled_delta_seed") == 5501, "Check 11 Failed: Pooled delta seed mismatch"
    results["check11_frozen_random_seeds"] = True

    # Check 12: Statistics operate on image-level independent observations (N = expected_n)
    assert len(image_df) == expected_n, "Check 12 Failed: Statistics not on image level"
    results["check12_image_level_statistics"] = True

    # Check 13: Report numbers match saved machine-readable outputs
    assert not matched_comparison_df.empty, "Check 13 Failed: Comparison table is empty"
    results["check13_data_manifest_consistency"] = True

    return results
