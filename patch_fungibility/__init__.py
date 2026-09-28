"""
Patch Content Fungibility V0: Mechanistic Falsification of Depth-Wise Token Content Dependence in Vision Transformers.
"""

from patch_fungibility.masks import (
    get_patch_mask_indices,
    get_donor_derangement,
    get_shuffled_donor_positions,
    get_within_image_shuffled_positions
)
from patch_fungibility.interventions import (
    apply_zero,
    apply_layer_mean,
    apply_cross_image_same_pos,
    apply_cross_image_rand_pos,
    apply_within_image_shuffle,
    compute_intervention_norms
)
from patch_fungibility.metrics import (
    evaluate_paired_margin_diff,
    evaluate_paired_accuracy,
    benjamini_hochberg_fdr
)
from patch_fungibility.pipeline import PatchFungibilityPipeline
from patch_fungibility.decision import evaluate_fungibility_decision
from patch_fungibility.validation import run_fungibility_v0_validations

__version__ = "0.1.0"

__all__ = [
    "get_patch_mask_indices",
    "get_donor_derangement",
    "get_shuffled_donor_positions",
    "get_within_image_shuffled_positions",
    "apply_zero",
    "apply_layer_mean",
    "apply_cross_image_same_pos",
    "apply_cross_image_rand_pos",
    "apply_within_image_shuffle",
    "compute_intervention_norms",
    "evaluate_paired_margin_diff",
    "evaluate_paired_accuracy",
    "benjamini_hochberg_fdr",
    "PatchFungibilityPipeline",
    "evaluate_fungibility_decision",
    "run_fungibility_v0_validations"
]
