"""
patch_fungibility/dense_fraction_masks.py

Mask permutation generation, dense fraction grid construction, and intervention hooks
for the Dense Fraction & Spatial-Mask Robustness Sweep.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch


def build_dense_fraction_grid(
    n_patches: int,
    step_percent: int = 1
) -> List[Dict[str, Any]]:
    """
    Constructs a deduplicated dense fraction grid from 0% to 100%.
    step_percent: 1 (default), 2 (fallback 1), or 5 (fallback 2).
    Always includes exact landmarks: 0%, 25%, 50%, 75%, 100%.
    """
    requested_percentages = list(range(0, 101, step_percent))
    for landmark in [0, 25, 50, 75, 100]:
        if landmark not in requested_percentages:
            requested_percentages.append(landmark)
    requested_percentages = sorted(set(requested_percentages))

    grid = []
    seen_k = set()

    for p in requested_percentages:
        f_req = p / 100.0
        k = int(round(f_req * n_patches))
        if k not in seen_k:
            seen_k.add(k)
            grid.append({
                "requested_fraction": f_req,
                "requested_percent": p,
                "actual_k": k,
                "actual_fraction": k / n_patches,
                "actual_percent": (k / n_patches) * 100.0
            })

    # Sort by actual_k
    grid = sorted(grid, key=lambda x: x["actual_k"])
    # Ensure exact 0 and exact n_patches exist
    assert grid[0]["actual_k"] == 0, "Grid must start at k=0"
    assert grid[-1]["actual_k"] == n_patches, f"Grid must end at k={n_patches}"

    return grid


def generate_spatial_permutations(
    n_patches: int,
    seeds: List[int] = [31001, 31002, 31003, 31004, 31005]
) -> Dict[int, np.ndarray]:
    """
    Generates 5 independent deterministic spatial permutations of indices 0..(n_patches-1).
    """
    perms = {}
    for s in seeds:
        rng = np.random.RandomState(s)
        p = rng.permutation(n_patches)
        perms[s] = p
    return perms


def build_nested_prefix_masks(
    permutation: np.ndarray,
    grid: List[Dict[str, Any]]
) -> Dict[int, np.ndarray]:
    """
    Builds nested prefix masks M(k) for each k in grid.
    Returns dict mapping k -> boolean array of shape (n_patches,) where True = replaced.
    Verifies M(k) subset of M(k_next).
    """
    n_patches = len(permutation)
    masks = {}

    prev_mask = None
    for entry in grid:
        k = entry["actual_k"]
        mask = np.zeros(n_patches, dtype=bool)
        if k > 0:
            selected_indices = permutation[:k]
            mask[selected_indices] = True

        if prev_mask is not None:
            # Nestedness check
            assert np.all(mask[prev_mask]), f"Prefix mask for k={k} is not a superset of previous mask!"

        masks[k] = mask
        prev_mask = mask

    return masks


class DensePatchInterventionHook:
    """
    Applies intervention to hidden states at target_depth.
    h: (B, 1 + n_patches, D)
    mask: boolean tensor of shape (n_patches,) where True indicates replaced.
    """
    def __init__(
        self,
        target_depth: int,
        mask: np.ndarray,
        intervention_type: str,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu")
    ):
        self.target_depth = target_depth
        self.mask_np = mask
        self.mask = torch.tensor(mask, dtype=torch.bool, device=device)
        self.intervention_type = intervention_type
        self.device = device
        self.seed = seed
        self.gen = None
        if self.seed is not None:
            self.gen = torch.Generator(device=self.device)
            self.gen.manual_seed(self.seed)

        self.mu = torch.tensor(mu, dtype=torch.float32, device=device) if mu is not None else None
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device) if sigma is not None else None

    def __call__(self, depth: int, h: torch.Tensor) -> torch.Tensor:
        if depth != self.target_depth:
            return h

        B, seq_len, D = h.shape
        cls_token = h[:, :1, :]
        patches = h[:, 1:, :].clone()

        n_masked = int(self.mask.sum().item())
        if n_masked == 0:
            return h

        if self.intervention_type == "ZERO":
            patches[:, self.mask, :] = 0.0

        elif self.intervention_type == "CENTROID":
            patches[:, self.mask, :] = self.mu.unsqueeze(0).unsqueeze(0)

        elif self.intervention_type == "DIAGONAL_GAUSSIAN":
            noise = torch.randn((B, n_masked, D), generator=self.gen, device=self.device, dtype=torch.float32)
            replacements = self.mu.unsqueeze(0).unsqueeze(0) + noise * self.sigma.unsqueeze(0).unsqueeze(0)
            patches[:, self.mask, :] = replacements

        else:
            raise ValueError(f"Unknown intervention type: {self.intervention_type}")

        h_out = torch.cat([cls_token, patches], dim=1)
        return h_out
