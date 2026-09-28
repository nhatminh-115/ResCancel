import os
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_calibration_pca(
    model: nn.Module,
    model_name: str,
    calib_loader: DataLoader,
    target_depth: int = 8,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Dict[str, Any]:
    """
    Computes PCA on Block-8 calibration patch representations (N = 1,000 images x 196 tokens = 196,000 tokens).
    Zero labels or gradients used.
    Returns:
      - mu_8: (D,) mean vector
      - sigma_8: (D,) coordinate-wise standard deviation
      - E_full: scalar total expected variance energy = sum(sigma_8^2)
      - eigenvalues: (D,) sorted descending
      - eigenvectors: (D, D) orthonormal principal directions V where V[:, r] is r-th component
      - explained_variance_ratio: (D,)
    """
    model.eval()
    embed_dim = model.embed_dim
    cached_tokens = []

    with torch.no_grad():
        for images, _ in calib_loader:
            images = images.to(device)
            feat = model.patch_embed(images)
            feat = model._pos_embed(feat)
            feat = model.patch_drop(feat)
            feat = model.norm_pre(feat)

            for b in range(target_depth + 1):
                feat = model.blocks[b](feat)

            # Spatial patch tokens only (tokens 1..196; CLS at index 0 excluded)
            spatial_tokens = feat[:, 1:, :].detach().cpu()
            cached_tokens.append(spatial_tokens)

    # Reshape: (N, 196, D) -> (N * 196, D)
    all_tokens = torch.cat(cached_tokens, dim=0).reshape(-1, embed_dim)
    n_tokens = all_tokens.shape[0]

    mu_8 = all_tokens.mean(dim=0)
    sigma_8 = all_tokens.std(dim=0, unbiased=True)
    E_full = float((sigma_8 ** 2).sum().item())

    # Center tokens
    centered_tokens = all_tokens - mu_8.unsqueeze(0)

    # Empirical covariance matrix: Sigma = (X_c^T X_c) / (n_tokens - 1)
    cov_matrix = torch.cov(centered_tokens.T)

    # Eigendecomposition: torch.linalg.eigh returns ascending eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Flip to descending order
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])

    # Enforce strict positive eigenvalue clipping to avoid numerical zero negatives
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)

    # Check orthonormality: V^T V == I_D
    v_gram = eigenvectors.T @ eigenvectors
    ident = torch.eye(embed_dim)
    ortho_err = float((v_gram - ident).abs().max().item())
    assert ortho_err < 1e-4, f"PCA eigenvectors not orthonormal! Max error = {ortho_err}"

    explained_var = (eigenvalues / eigenvalues.sum()).numpy().tolist()

    return {
        "model_name": model_name,
        "n_tokens": n_tokens,
        "embed_dim": embed_dim,
        "mu_8": mu_8.numpy().astype(np.float32),
        "sigma_8": sigma_8.numpy().astype(np.float32),
        "E_full": E_full,
        "eigenvalues": eigenvalues.numpy().astype(np.float32),
        "eigenvectors": eigenvectors.numpy().astype(np.float32),
        "explained_variance_ratio": explained_var,
        "ortho_error": ortho_err
    }


def get_random_orthonormal_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    """
    Constructs an orthonormal basis U_R in R^(dim x rank) using QR decomposition.
    Guarantees U_R^T U_R = I_R.
    """
    rng = np.random.RandomState(seed)
    mat = rng.randn(dim, rank).astype(np.float32)
    q, _ = np.linalg.qr(mat)
    basis = q[:, :rank].astype(np.float32)
    gram_err = float(np.max(np.abs(basis.T @ basis - np.eye(rank))))
    assert gram_err < 1e-4, f"Random basis not orthonormal! Max error = {gram_err}"
    return basis
