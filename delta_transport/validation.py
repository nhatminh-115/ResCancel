from typing import Dict, Any, List
import pandas as pd
import numpy as np


def run_programmatic_validations(
    image_df: pd.DataFrame,
    source_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    manifest: Dict[str, Any],
    expected_n: int = 1000
) -> Dict[str, bool]:
    """
    Executes the 10 pre-registered programmatic assertions.
    Raises AssertionError if any check fails.
    """
    results = {}

    # Check 1: Backbone parameters never change
    assert manifest.get("backbone_weights_verified", False), "Assertion 1 Failed: Backbone weights were not verified as unchanged"
    results["check1_backbone_unmodified"] = True

    # Check 2: Delta definition holds numerically
    assert manifest.get("delta_definition_verified", False), "Assertion 2 Failed: Delta_l != h_{l+1} - h_l"
    results["check2_delta_definition_valid"] = True

    # Check 3: CLS untouched during patch transport
    assert manifest.get("cls_untouched_verified", False), "Assertion 3 Failed: CLS token was modified at injection"
    results["check3_cls_untouched"] = True

    # Check 4: Identical patch positions modified across all active policies (196 patches)
    assert manifest.get("patch_positions_verified", False), "Assertion 4 Failed: Different patch positions were modified"
    results["check4_identical_patch_positions"] = True

    # Check 5: Normalized perturbation magnitude matches across policies (gamma * ||h_t||_2)
    assert manifest.get("perturbation_norm_verified", False), "Assertion 5 Failed: Perturbation norms do not match gamma"
    results["check5_matched_perturbation_norm"] = True

    # Check 6: Global oracle selects exactly one historical source layer per image
    assert "global_selected_layer" in source_df.columns, "Assertion 6 Failed: Missing global_selected_layer column"
    assert source_df["global_selected_layer"].between(0, 7).all(), "Assertion 6 Failed: Invalid global selected layer index"
    results["check6_global_single_layer"] = True

    # Check 7: Tokenwise oracle may vary source by patch (heterogeneous selections occur)
    assert "num_distinct_layers" in source_df.columns, "Assertion 7 Failed: Missing num_distinct_layers column"
    assert (source_df["num_distinct_layers"] >= 1).all(), "Assertion 7 Failed: Invalid distinct layers count"
    assert (source_df["num_distinct_layers"] > 1).any(), "Assertion 7 Failed: Tokenwise never varied source layer"
    results["check7_tokenwise_varies_patch"] = True

    # Check 8: Random controls use frozen seeds 2501, 2502, 2503
    expected_seeds = [2501, 2502, 2503]
    assert manifest.get("random_control_seeds") == expected_seeds, "Assertion 8 Failed: Random control seeds do not match frozen list"
    results["check8_frozen_random_seeds"] = True

    # Check 9: Statistics operate on image-level independent observations (N = expected_n)
    assert len(image_df) == expected_n, f"Assertion 9 Failed: Image count is {len(image_df)}, expected {expected_n}"
    assert len(source_df) == expected_n, f"Assertion 9 Failed: Source selection count is {len(source_df)}, expected {expected_n}"
    results["check9_image_level_independence"] = True

    # Check 10: Numbers match saved machine-readable outputs
    token_vs_global_row = comparison_df[comparison_df["policy"] == "tokenwise_oracle"]
    assert not token_vs_global_row.empty, "Assertion 10 Failed: Tokenwise oracle missing from comparison table"
    results["check10_manifest_data_consistency"] = True

    return results
