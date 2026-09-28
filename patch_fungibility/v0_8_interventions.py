import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


def apply_zero_to_mask(h: torch.Tensor, seq_mask: List[int]) -> torch.Tensor:
    h_mod = h.clone()
    h_mod[:, seq_mask, :] = 0.0
    return h_mod


def apply_centroid_to_mask(h: torch.Tensor, seq_mask: List[int], mu_8: torch.Tensor) -> torch.Tensor:
    h_mod = h.clone()
    h_mod[:, seq_mask, :] = mu_8.to(h.device)
    return h_mod


def generate_diagonal_gaussian_eps(
    B: int,
    M: int,
    sigma_8: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Independent diagonal Gaussian perturbation: eps_t ~ N(0, diag(sigma_8^2))
    Shape: (B, M, D)
    """
    D = sigma_8.shape[0]
    z = torch.randn((B, M, D), generator=generator, dtype=dtype).to(device)
    sigma_bc = sigma_8.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    eps = z * sigma_bc
    return eps


def generate_shared_noise_eps(
    B: int,
    M: int,
    sigma_8: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Shared Gaussian perturbation: eps_shared ~ N(0, diag(sigma_8^2)) broadcast to all M tokens.
    All M tokens receive the identical perturbation vector.
    Shape: (B, M, D)
    """
    D = sigma_8.shape[0]
    z = torch.randn((B, 1, D), generator=generator, dtype=dtype).to(device)
    sigma_bc = sigma_8.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    eps_shared = z * sigma_bc  # (B, 1, D)
    eps = eps_shared.expand(B, M, D).clone()
    return eps


def generate_isotropic_full_eps(
    B: int,
    M: int,
    D: int,
    E_full: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Isotropic Gaussian perturbation: eps_t ~ N(0, (E_full / D) * I_D)
    Shape: (B, M, D)
    """
    sigma_iso = math.sqrt(E_full / float(D))
    z = torch.randn((B, M, D), generator=generator, dtype=dtype).to(device)
    eps = z * sigma_iso
    return eps


def generate_pca_rank_eps(
    B: int,
    M: int,
    V_R: torch.Tensor,
    eigenvalues_R: torch.Tensor,
    E_full: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    PCA Rank-R perturbation:
      eps_t = sum_{r=1}^R a_{t,r} v_r = V_R a_t
    where a_{t,r} ~ N(0, s_R * lambda_r) with s_R = E_full / sum(lambda_r)
    Matches E[||eps_t||_2^2] = E_full exactly.
    Shape: (B, M, D)
    """
    R = V_R.shape[1]
    S_R = float(eigenvalues_R.sum().item())
    scale_sq = E_full / max(S_R, 1e-12)
    coeff_std = torch.sqrt(eigenvalues_R * scale_sq).to(device=device, dtype=dtype)  # (R,)

    # Sample coefficients: (B, M, R) on device
    z = torch.randn((B, M, R), generator=generator, dtype=dtype).to(device)
    a = z * coeff_std.unsqueeze(0).unsqueeze(0)

    # Project to D-dimensional space: eps = a @ V_R^T
    # V_R is (D, R), so a @ V_R^T is (B, M, D)
    V_R_dev = V_R.to(device=device, dtype=dtype)
    eps = torch.matmul(a, V_R_dev.T)
    return eps


def generate_random_rank_eps(
    B: int,
    M: int,
    U_R: torch.Tensor,
    E_full: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Random Rank-R perturbation:
      eps_t = U_R a_t
    where U_R is an orthonormal basis (D, R), and a_t ~ N(0, (E_full / R) * I_R).
    Matches E[||eps_t||_2^2] = E_full exactly.
    Shape: (B, M, D)
    """
    R = U_R.shape[1]
    std_r = math.sqrt(E_full / float(R))

    # Sample coefficients: (B, M, R) on device
    z = torch.randn((B, M, R), generator=generator, dtype=dtype).to(device)
    a = z * std_r

    # Project: eps = a @ U_R^T -> (B, M, D)
    U_R_dev = U_R.to(device=device, dtype=dtype)
    eps = torch.matmul(a, U_R_dev.T)
    return eps


def generate_grouped_diversity_eps(
    B: int,
    M: int,
    sigma_8: torch.Tensor,
    K: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Grouped diversity perturbation:
    For each sample in batch, generate K independent Gaussian vectors g_1..g_K ~ N(0, diag(sigma_8^2)).
    Assign the M spatial positions approximately evenly among them using deterministic indexing (t mod K).
    Matches E[||eps_t||_2^2] = E_full exactly.
    Shape: (B, M, D)
    """
    D = sigma_8.shape[0]
    # Sample K vectors: (B, K, D) on device
    z = torch.randn((B, K, D), generator=generator, dtype=dtype).to(device)
    sigma_bc = sigma_8.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    k_vecs = z * sigma_bc  # (B, K, D)

    # Assign positions 0..M-1 to indices 0..K-1
    group_indices = torch.tensor([t % K for t in range(M)], dtype=torch.long, device=device)  # (M,)
    # Gather across K: (B, M, D)
    eps = k_vecs[:, group_indices, :]
    return eps


def apply_perturbation_to_mask(
    h: torch.Tensor,
    seq_mask: List[int],
    mu_8: torch.Tensor,
    eps: torch.Tensor
) -> torch.Tensor:
    """
    Applies perturbation h'_t = mu_8 + eps_t to spatial patch tokens at seq_mask.
    Leaves CLS token (at index 0) strictly untouched.
    h: (B, 197, D)
    eps: (B, len(seq_mask), D)
    mu_8: (D,)
    """
    h_mod = h.clone()
    mu_bc = mu_8.to(h.device).unsqueeze(0).unsqueeze(0)
    h_mod[:, seq_mask, :] = mu_bc + eps.to(h.device)
    return h_mod
