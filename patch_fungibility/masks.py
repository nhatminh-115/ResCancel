from typing import List, Tuple
import numpy as np


def get_patch_mask_indices(
    total_patches: int = 196,
    mask_count: int = 49,
    seed: int = 2501
) -> Tuple[List[int], List[int], List[int]]:
    """
    Generates a deterministic random subset of 49 spatial patch indices.
    Returns:
      patch_mask_indices: indices in 0..195
      seq_mask_indices: indices in 1..196 (with CLS at 0 untouched)
      unmasked_seq_indices: non-masked patch indices in 1..196
    """
    rng = np.random.RandomState(seed)
    patch_indices = sorted(rng.choice(total_patches, size=mask_count, replace=False))
    seq_mask = [idx + 1 for idx in patch_indices]
    unmasked_seq = [idx for idx in range(1, total_patches + 1) if idx not in seq_mask]

    assert len(patch_indices) == mask_count
    assert len(seq_mask) == mask_count
    assert len(unmasked_seq) == total_patches - mask_count
    assert 0 not in seq_mask, "CLS token (index 0) must not be masked!"

    return patch_indices, seq_mask, unmasked_seq


def get_donor_derangement(n_samples: int = 1000, seed: int = 3501) -> np.ndarray:
    """
    Generates a deterministic derangement of range(n_samples) where perm[i] != i for all i.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    for i in range(n_samples):
        if perm[i] == i:
            swap_idx = (i + 1) % n_samples
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]
    assert (perm != np.arange(n_samples)).all(), "Derangement failed: self-donor detected!"
    return perm


def get_shuffled_donor_positions(seq_mask: List[int], seed: int = 4501) -> List[int]:
    """
    Permutes the 49 masked slots with no fixed points for cross-image random position assignment.
    """
    k = len(seq_mask)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(k)
    for i in range(k):
        if perm[i] == i:
            swap_idx = (i + 1) % k
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]
    assert (perm != np.arange(k)).all(), "Random position permutation has fixed point!"
    return [seq_mask[p] for p in perm]


def get_within_image_shuffled_positions(seq_mask: List[int], seed: int = 5501) -> List[int]:
    """
    Permutes the 49 masked slots with no fixed points for within-image spatial shuffle.
    """
    k = len(seq_mask)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(k)
    for i in range(k):
        if perm[i] == i:
            swap_idx = (i + 1) % k
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]
    assert (perm != np.arange(k)).all(), "Within-image shuffle permutation has fixed point!"
    return [seq_mask[p] for p in perm]
