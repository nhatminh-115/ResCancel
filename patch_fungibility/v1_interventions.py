"""
patch_fungibility/v1_interventions.py

Intervention generators, nested mask builders, and calibration statistics calculation
for Patch Fungibility V1:
- Nested spatial patch masks (25%, 50%, 75%, 100%) with seed 21001.
- Calibration statistics (mean mu_l, std sigma_l) and Depth-8 PCA.
- Zero, Centroid, Diagonal Gaussian, Coordinate Permutation, Sign Flip.
- 100% replacement: Shared Noise vs. Independent Noise vs. Isotropic Noise.
- 1D Subspace: Natural PC1, Natural PC2, Random 1D.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn


def build_nested_masks(n_patches: int, seed: int = 21001) -> Dict[float, np.ndarray]:
    """
    Constructs deterministic nested spatial patch masks for fractions [0.25, 0.50, 0.75, 1.00].
    Uses a single deterministic permutation of patch indices 0..(n_patches-1).
    Returns dict mapping fraction -> boolean array of shape (n_patches,) where True = replaced.
    Ensures M_0.25 subset of M_0.50 subset of M_0.75 subset of M_1.00.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_patches)

    fractions = [0.25, 0.50, 0.75, 1.00]
    masks = {}
    for frac in fractions:
        k = int(round(n_patches * frac))
        selected_indices = perm[:k]
        mask = np.zeros(n_patches, dtype=bool)
        mask[selected_indices] = True
        masks[frac] = mask

    # Verify nestedness
    assert np.all(masks[0.50][masks[0.25]]), "Mask 0.25 is not a subset of 0.50"
    assert np.all(masks[0.75][masks[0.50]]), "Mask 0.50 is not a subset of 0.75"
    assert np.all(masks[1.00][masks[0.75]]), "Mask 0.75 is not a subset of 1.00"

    return masks


def compute_calibration_statistics(
    calib_activations: Dict[int, torch.Tensor]
) -> Dict[str, Any]:
    """
    Given cached calibration patch activations per depth:
      calib_activations[depth] is a tensor of shape (N_calib, N_patches, D).
    Computes:
      - mean_depth_{l}: (D,)
      - std_depth_{l}: (D,)
      - pca_v_depth_8: (D, D) principal direction matrix (columns or rows)
      - pca_lambdas_depth_8: (D,) eigenvalues
      - pca_total_variance_depth_8: float
    """
    stats = {}
    for depth, acts in calib_activations.items():
        # acts: (N_calib, N_patches, D)
        # Flatten over images and patches: (N_calib * N_patches, D)
        flat = acts.reshape(-1, acts.size(-1)).to(torch.float32)
        mu = flat.mean(dim=0)
        var = flat.var(dim=0, unbiased=True)
        sigma = torch.sqrt(torch.clamp(var, min=1e-8))

        stats[f"mu_{depth}"] = mu.cpu().numpy()
        stats[f"sigma_{depth}"] = sigma.cpu().numpy()

        if depth == 8:
            # Center activations for PCA
            centered = flat - mu.unsqueeze(0)
            n_samples = centered.size(0)
            cov = torch.matmul(centered.T, centered) / (n_samples - 1)
            # Eigen-decomposition of covariance matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Sort in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors v_k

            stats["pca_lambdas_8"] = eigenvalues.cpu().numpy()
            stats["pca_vectors_8"] = eigenvectors.cpu().numpy()  # column k is v_{k+1}
            stats["total_var_8"] = float(eigenvalues.sum().item())

    return stats


def get_coordinate_permutations(
    dim: int,
    seeds: List[int] = [23001, 23002, 23003]
) -> Dict[int, np.ndarray]:
    """
    Generates deterministic coordinate permutations for given dimension and seeds.
    Each permutation is a bijection pi: {0..D-1} -> {0..D-1}.
    """
    perms = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        p = rng.permutation(dim)
        perms[seed] = p
    return perms


def get_random_1d_directions(
    dim: int,
    seeds: List[int] = [25001, 25002, 25003]
) -> Dict[int, np.ndarray]:
    """
    Generates random unit directions on the sphere S^{D-1} for given dimension and seeds.
    u = z / ||z||_2 where z ~ N(0, I).
    """
    directions = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        z = rng.randn(dim).astype(np.float32)
        u = z / np.linalg.norm(z)
        directions[seed] = u
    return directions


class PatchInterventionHook:
    """
    Encapsulates intervention logic applied to block hidden states.
    h: tensor of shape (B, 1 + N_patches, D), where token 0 is CLS.
    mask: boolean tensor of shape (N_patches,) where True means replace that spatial patch.
    """
    def __init__(
        self,
        target_depth: int,
        mask: np.ndarray,
        intervention_type: str,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        device: torch.device = torch.device("cpu")
    ):
        self.target_depth = target_depth
        self.mask_np = mask
        self.mask = torch.tensor(mask, dtype=torch.bool, device=device)
        self.intervention_type = intervention_type
        self.device = device
        self.seed = seed
        self.extra_args = extra_args or {}

        self.mu = torch.tensor(mu, dtype=torch.float32, device=device) if mu is not None else None
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device) if sigma is not None else None

        # Precompute specific vectors if static
        if intervention_type == "COORDINATE_PERMUTED_CENTROID":
            perm = self.extra_args["perm"]
            perm_mu = mu[perm]
            self.replacement_vector = torch.tensor(perm_mu, dtype=torch.float32, device=device)
        elif intervention_type == "SIGN_FLIPPED_CENTROID":
            self.replacement_vector = -self.mu
        elif intervention_type in ("CENTROID", "STATIC_CENTROID"):
            self.replacement_vector = self.mu
        else:
            self.replacement_vector = None

    def __call__(self, depth: int, h: torch.Tensor) -> torch.Tensor:
        if depth != self.target_depth:
            return h

        # h shape: (B, 1 + N_patches, D)
        B, seq_len, D = h.shape
        cls_token = h[:, :1, :]
        patches = h[:, 1:, :].clone()  # (B, N_patches, D)

        n_masked = int(self.mask.sum().item())
        if n_masked == 0:
            return h

        # Generator for reproducible stochastic conditions
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.seed)

        if self.intervention_type == "ZERO":
            # Set masked patches to zero
            patches[:, self.mask, :] = 0.0

        elif self.intervention_type in ("CENTROID", "STATIC_CENTROID", "COORDINATE_PERMUTED_CENTROID", "SIGN_FLIPPED_CENTROID"):
            # Set masked patches to precomputed vector
            patches[:, self.mask, :] = self.replacement_vector.unsqueeze(0).unsqueeze(0)

        elif self.intervention_type == "DIAGONAL_GAUSSIAN":
            # Sample independent x_{t,d} ~ N(mu_d, sigma_d^2) for each image and masked patch
            noise = torch.randn((B, n_masked, D), generator=gen, device=self.device, dtype=torch.float32)
            replacements = self.mu.unsqueeze(0).unsqueeze(0) + noise * self.sigma.unsqueeze(0).unsqueeze(0)
            patches[:, self.mask, :] = replacements

        elif self.intervention_type == "SHARED_GAUSSIAN_NOISE":
            # For each image sample ONE epsilon ~ N(0, diag(sigma^2)), broadcast to all patches
            noise_img = torch.randn((B, 1, D), generator=gen, device=self.device, dtype=torch.float32)
            replacements = self.mu.unsqueeze(0).unsqueeze(0) + noise_img * self.sigma.unsqueeze(0).unsqueeze(0)
            # Broadcast across all masked patches
            patches[:, self.mask, :] = replacements.expand(B, n_masked, D)

        elif self.intervention_type == "INDEPENDENT_GAUSSIAN_NOISE":
            # Each patch receives independent epsilon_t ~ N(0, diag(sigma^2))
            noise_patches = torch.randn((B, n_masked, D), generator=gen, device=self.device, dtype=torch.float32)
            replacements = self.mu.unsqueeze(0).unsqueeze(0) + noise_patches * self.sigma.unsqueeze(0).unsqueeze(0)
            patches[:, self.mask, :] = replacements

        elif self.intervention_type == "ISOTROPIC_INDEPENDENT_NOISE":
            # Match total calibration variance energy: E_full = sum sigma_d^2
            # epsilon_t ~ N(0, (E_full / D) * I)
            e_full = float(torch.sum(self.sigma ** 2).item())
            iso_sigma = np.sqrt(e_full / D)
            noise_iso = torch.randn((B, n_masked, D), generator=gen, device=self.device, dtype=torch.float32) * iso_sigma
            replacements = self.mu.unsqueeze(0).unsqueeze(0) + noise_iso
            patches[:, self.mask, :] = replacements

        elif self.intervention_type == "NATURAL_PC1":
            # h_t = mu + z_t * sqrt(lambda_1) * v_1
            v_1 = torch.tensor(self.extra_args["v_1"], dtype=torch.float32, device=self.device)  # (D,)
            lambda_1 = float(self.extra_args["lambda_1"])
            scale = np.sqrt(lambda_1)
            z = torch.randn((B, n_masked, 1), generator=gen, device=self.device, dtype=torch.float32)
            replacements = self.mu.unsqueeze(0).unsqueeze(0) + z * (scale * v_1.unsqueeze(0).unsqueeze(0))
            patches[:, self.mask, :] = replacements

        elif self.intervention_type == "NATURAL_PC2":
            # h_t = mu + z_t * sqrt(lambda_2) * v_2
            v_2 = torch.tensor(self.extra_args["v_2"], dtype=torch.float32, device=self.device)  # (D,)
            lambda_2 = float(self.extra_args["lambda_2"])
            scale = np.sqrt(lambda_2)
            z = torch.randn((B, n_masked, 1), generator=gen, device=self.device, dtype=torch.float32)
            replacements = self.mu.unsqueeze(0).unsqueeze(0) + z * (scale * v_2.unsqueeze(0).unsqueeze(0))
            patches[:, self.mask, :] = replacements

        elif self.intervention_type == "RANDOM_1D":
            # h_t = mu + z_t * sqrt(lambda_1) * u
            u = torch.tensor(self.extra_args["u"], dtype=torch.float32, device=self.device)  # (D,)
            lambda_1 = float(self.extra_args["lambda_1"])
            scale = np.sqrt(lambda_1)
            z = torch.randn((B, n_masked, 1), generator=gen, device=self.device, dtype=torch.float32)
            replacements = self.mu.unsqueeze(0).unsqueeze(0) + z * (scale * u.unsqueeze(0).unsqueeze(0))
            patches[:, self.mask, :] = replacements

        else:
            raise ValueError(f"Unknown intervention type: {self.intervention_type}")

        # Assemble CLS and patches back
        h_out = torch.cat([cls_token, patches], dim=1)
        return h_out
