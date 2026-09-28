from typing import Dict, List, Tuple, Any
import numpy as np


def get_nested_patch_masks(
    total_patches: int = 196,
    counts: Dict[str, int] = {"10%": 20, "25%": 49, "50%": 98},
    seed: int = 9301
) -> Dict[str, Dict[str, Any]]:
    """
    Generates nested patch masks across 10%, 25%, and 50% fractions using frozen seed 9301.
    Guarantees:
      mask_20 is subset of mask_49
      mask_49 is subset of mask_98
      CLS token (sequence index 0) is never masked (patch indices mapped to idx + 1).
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_patches).tolist()

    masks = {}
    prev_patch_set = set()

    for frac_key, count in [("10%", 20), ("25%", 49), ("50%", 98)]:
        patch_indices = sorted(perm[:count])
        seq_mask = [p + 1 for p in patch_indices]
        unmasked_seq = [s for s in range(1, total_patches + 1) if s not in seq_mask]

        current_patch_set = set(patch_indices)
        assert len(patch_indices) == count
        assert len(seq_mask) == count
        assert 0 not in seq_mask, "CLS token (index 0) must not be masked!"
        assert prev_patch_set.issubset(current_patch_set), f"Nested subset invariant violated for {frac_key}!"
        prev_patch_set = current_patch_set

        masks[frac_key] = {
            "fraction_key": frac_key,
            "count": count,
            "patch_indices": patch_indices,
            "seq_mask": seq_mask,
            "unmasked_seq": unmasked_seq
        }

    return masks
