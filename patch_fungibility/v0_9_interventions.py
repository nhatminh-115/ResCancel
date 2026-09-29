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


def generate_natural_pca_rank_eps(
    B: int,
    M: int,
    V_R: torch.Tensor,
    eigenvalues_R: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Natural PCA Rank-R perturbation:
      eps_t = sum_{r=1}^R a_{t,r} v_r = V_R a_t
    where a_{t,r} ~ N(0, lambda_r) independently per token WITHOUT ANY E_full rescaling.
    Expected perturbation energy: E_R = sum_{r=1}^R lambda_r.
    Shape: (B, M, D)
    """
    R = V_R.shape[1]
    coeff_std = torch.sqrt(eigenvalues_R).to(device=device, dtype=dtype)  # (R,)
    z = torch.randn((B, M, R), generator=generator, dtype=dtype).to(device)
    a = z * coeff_std.unsqueeze(0).unsqueeze(0)
    V_R_dev = V_R.to(device=device, dtype=dtype)
    eps = torch.matmul(a, V_R_dev.T)
    return eps


def generate_energy_matched_pca_rank_eps(
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
    Energy-matched PCA Rank-R perturbation (reproduces V0.8):
      eps_t = V_R a_t
    where a_{t,r} ~ N(0, (E_full / sum(lambda_R)) * lambda_r).
    Expected perturbation energy: E_full.
    Shape: (B, M, D)
    """
    R = V_R.shape[1]
    S_R = float(eigenvalues_R.sum().item())
    scale_sq = E_full / max(S_R, 1e-12)
    coeff_std = torch.sqrt(eigenvalues_R * scale_sq).to(device=device, dtype=dtype)  # (R,)
    z = torch.randn((B, M, R), generator=generator, dtype=dtype).to(device)
    a = z * coeff_std.unsqueeze(0).unsqueeze(0)
    V_R_dev = V_R.to(device=device, dtype=dtype)
    eps = torch.matmul(a, V_R_dev.T)
    return eps


def generate_pc1_scale_eps(
    B: int,
    M: int,
    v_1: torch.Tensor,
    lambda_1: float,
    scale: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PC1 perturbation scaled by factor s:
      eps_t = s * z_t * sqrt(lambda_1) * v_1
    where z_t ~ N(0, 1).
    Returns (eps, z_raw) where z_raw is (B, M) for geometric expansion tracking.
    """
    z = torch.randn((B, M), generator=generator, dtype=dtype).to(device)
    coeff = scale * math.sqrt(lambda_1)
    a = (z * coeff).unsqueeze(-1)  # (B, M, 1)
    v_dev = v_1.to(device=device, dtype=dtype).unsqueeze(0)  # (1, D)
    eps = a * v_dev  # (B, M, D)
    return eps, z


def generate_single_pc_eps(
    B: int,
    M: int,
    v_k: torch.Tensor,
    var_k: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Single PC perturbation along v_k with specified variance var_k:
      eps_t = z_t * sqrt(var_k) * v_k
    Shape: (B, M, D)
    """
    z = torch.randn((B, M), generator=generator, dtype=dtype).to(device)
    coeff = math.sqrt(var_k)
    a = (z * coeff).unsqueeze(-1)
    v_dev = v_k.to(device=device, dtype=dtype).unsqueeze(0)
    eps = a * v_dev
    return eps


def generate_random_1d_eps(
    B: int,
    M: int,
    u: torch.Tensor,
    target_var: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Random 1D perturbation along unit vector u with specified variance:
      eps_t = z_t * sqrt(target_var) * u
    u must have ||u||_2 = 1.0.
    Shape: (B, M, D)
    """
    u_norm = float(torch.norm(u, p=2).item())
    assert abs(u_norm - 1.0) < 1e-4, f"Random vector u not normalized! Norm = {u_norm}"
    z = torch.randn((B, M), generator=generator, dtype=dtype).to(device)
    coeff = math.sqrt(target_var)
    a = (z * coeff).unsqueeze(-1)
    u_dev = u.to(device=device, dtype=dtype).unsqueeze(0)
    eps = a * u_dev
    return eps


def generate_diagonal_gaussian_eps(
    B: int,
    M: int,
    sigma_8: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    D = sigma_8.shape[0]
    z = torch.randn((B, M, D), generator=generator, dtype=dtype).to(device)
    sigma_bc = sigma_8.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    eps = z * sigma_bc
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
    sigma_iso = math.sqrt(E_full / float(D))
    z = torch.randn((B, M, D), generator=generator, dtype=dtype).to(device)
    eps = z * sigma_iso
    return eps
