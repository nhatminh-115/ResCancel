from typing import Dict, List, Tuple, Any
import numpy as np


def get_nested_patch_masks_v0_7(
    total_patches: int = 196,
    counts: Dict[str, int] = {"25%": 49, "50%": 98, "75%": 147, "100%": 196},
    seed: int = 9601
) -> Dict[str, Dict[str, Any]]:
    """
    Generates nested patch masks across 25%, 50%, 75%, and 100% fractions using frozen seed 9601.
    Guarantees:
      mask_49 is subset of mask_98
      mask_98 is subset of mask_147
      mask_147 is subset of mask_196
      CLS token (sequence index 0) is never masked (patch indices mapped to idx + 1).
      At 100%, exactly all 196 patch tokens are masked, while CLS remains index 0 untouched.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_patches).tolist()

    masks = {}
    prev_patch_set = set()

    for frac_key, count in [("25%", 49), ("50%", 98), ("75%", 147), ("100%", 196)]:
        patch_indices = sorted(perm[:count])
        seq_mask = [p + 1 for p in patch_indices]
        unmasked_seq = [s for s in range(1, total_patches + 1) if s not in seq_mask]

        current_patch_set = set(patch_indices)
        assert len(patch_indices) == count, f"Expected {count} patch indices, got {len(patch_indices)}"
        assert len(seq_mask) == count
        assert 0 not in seq_mask, "CLS token (index 0) must never be in patch seq_mask!"
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
