"""
patch_fungibility/amortized_operator.py

Core engine for Amortized Operator-Aware Token Compression:
1. Dataset split generator (disjoint Train, Val, Calib, Canonical Eval)
2. Fast Gram SVD oracle target extraction
3. Invariant Subspace metrics (Grassmannian overlap, principal angles)
4. Predictor architectures:
   - StaticSubspaceModel (global calibration average)
   - MLPSubspacePredictor (pooled statistics baseline)
   - FactorizedModePredictor (token x feature low-rank factorization)
   - DirectCarrierPredictor (direct carrier update baseline control)
5. Fast amortized carrier solver (r x r linear system)
"""

import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import io

from patch_fungibility.v0_6_dataset import ParquetImageSubset


def get_amortized_imagenet_splits(
    calib_seed: int = 9101,
    eval_seed: int = 9201,
    train_seed: int = 7101,
    val_seed: int = 7201,
    n_calib: int = 1000,
    n_eval: int = 1000,
    n_train: int = 2000,
    n_val: int = 500
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Constructs 4 mutually disjoint, stratified splits from ImageNet-1k validation set:
    - Calib: 1,000 images (seed 9101, 1 per class)
    - Eval: 1,000 images (seed 9201, 1 per class - canonical benchmark)
    - Train: 2,000 images (seed 7101, 2 per class)
    - Val: 500 images (seed 7201, 1 per 2 classes)
    Guarantees zero overlap in global image indices.
    """
    cache_pattern = os.path.expanduser(r"~/.cache/huggingface/hub/**/val-*.parquet")
    parquet_files = sorted(glob.glob(cache_pattern, recursive=True))
    if not parquet_files:
        raise FileNotFoundError("ImageNet-1k validation parquet files not found in cache.")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["global_index"] = np.arange(len(full_df))

    grouped = full_df.groupby("label")
    classes = sorted(grouped.groups.keys())

    rng_calib = np.random.RandomState(calib_seed)
    rng_eval = np.random.RandomState(eval_seed)
    rng_train = np.random.RandomState(train_seed)
    rng_val = np.random.RandomState(val_seed)

    calib_idx, eval_idx, train_idx, val_idx = [], [], [], []

    for c in classes:
        pool = list(grouped.groups[c])
        # 1. Calib
        c_choice = int(rng_calib.choice(pool))
        calib_idx.append(c_choice)
        pool.remove(c_choice)

        # 2. Canonical Eval
        e_choice = int(rng_eval.choice(pool))
        eval_idx.append(e_choice)
        pool.remove(e_choice)

        # 3. Train (2 per class for 1,000 classes = 2,000)
        t_choices = rng_train.choice(pool, size=2, replace=False).tolist()
        train_idx.extend(t_choices)
        for t in t_choices:
            pool.remove(t)

        # 4. Val (1 per 2 classes = 500 total)
        if len(val_idx) < n_val:
            v_choice = int(rng_val.choice(pool))
            val_idx.append(v_choice)
            pool.remove(v_choice)

    # Assert strict zero overlap
    assert len(set(calib_idx).intersection(set(eval_idx))) == 0
    assert len(set(train_idx).intersection(set(eval_idx))) == 0
    assert len(set(val_idx).intersection(set(eval_idx))) == 0
    assert len(set(train_idx).intersection(set(calib_idx))) == 0
    assert len(set(train_idx).intersection(set(val_idx))) == 0

    def extract_samples(indices):
        sub_df = full_df.loc[indices].reset_index(drop=True)
        samples = []
        for _, row in sub_df.iterrows():
            img_val = row["image"]
            raw_bytes = img_val["bytes"] if isinstance(img_val, dict) else img_val
            samples.append((raw_bytes, int(row["label"])))
        return samples

    calib_set = ParquetImageSubset(extract_samples(calib_idx))
    eval_set = ParquetImageSubset(extract_samples(eval_idx))
    train_set = ParquetImageSubset(extract_samples(train_idx))
    val_set = ParquetImageSubset(extract_samples(val_idx))

    return calib_set, eval_set, train_set, val_set


def extract_oracle_subspace(J: torch.Tensor, r: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes exact top-r right singular vectors V_r in R^{ND x r} and singular values sigma in R^r
    using fast Gram eigendecomposition: G = J J^T in R^{D_readout x D_readout}.
    """
    device = J.device
    D_readout, ND = J.shape

    Gram = J @ J.T
    evals, U = torch.linalg.eigh(Gram)
    evals = evals.flip(0)
    U = U.flip(1)
    sigmas = torch.sqrt(torch.clamp(evals, min=1e-12))

    actual_r = min(r, D_readout)
    U_r = U[:, :actual_r] # (D_readout, r)
    sigmas_r = sigmas[:actual_r] # (r,)
    # Right singular vectors: V_r = J^T (U_r Sigma_r^{-1})
    V_r = J.T @ (U_r / sigmas_r.unsqueeze(0)) # (ND, r)

    return V_r, sigmas_r


def subspace_overlap(V_true: torch.Tensor, V_pred: torch.Tensor) -> float:
    """
    Invariant Grassmannian overlap metric: ||V_true^T V_pred||_F^2 / r in [0, 1].
    Invariant to sign flips and arbitrary internal orthonormal basis rotations.
    """
    r = V_true.shape[1]
    M = V_true.T @ V_pred
    overlap = (torch.norm(M) ** 2 / float(r)).item()
    return float(overlap)


def subspace_principal_angles(V_true: torch.Tensor, V_pred: torch.Tensor) -> np.ndarray:
    """
    Computes principal angles between subspaces spanned by V_true and V_pred in degrees.
    """
    M = (V_true.T @ V_pred).detach().cpu()
    s = torch.linalg.svdvals(M).numpy()
    s = np.clip(s, 0.0, 1.0)
    angles_rad = np.arccos(s)
    angles_deg = np.degrees(angles_rad)
    return np.sort(angles_deg)


# ====================================================================
# PREDICTOR ARCHITECTURES
# ====================================================================

class StaticSubspaceModel(nn.Module):
    """
    Global / Static Subspace Baseline:
    Stores precomputed dataset-level average visible subspace V_static in R^{ND x r}.
    """
    def __init__(self, V_static: torch.Tensor):
        super().__init__()
        self.register_buffer("V_static", V_static)

    def forward(self, h_act: torch.Tensor) -> torch.Tensor:
        B_batch = h_act.size(0)
        return self.V_static.unsqueeze(0).repeat(B_batch, 1, 1)


class MLPSubspacePredictor(nn.Module):
    """
    Family B: Linear / MLP Activation-to-Subspace Predictor.
    Pools [CLS, mean(P), std(P)] (dim 3D) and projects through a 2-layer MLP.
    Output is reshaped to (ND, r) and orthonormalized via differentiable QR.
    """
    def __init__(self, N: int = 196, D: int = 384, r: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.N = N
        self.D = D
        self.r = r
        self.ND = N * D

        self.net = nn.Sequential(
            nn.Linear(3 * D, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.ND * r)
        )

    def forward(self, h_act: torch.Tensor) -> torch.Tensor:
        # h_act: (B_batch, 1+N, D)
        B_batch = h_act.size(0)
        cls_tok = h_act[:, 0, :]
        P = h_act[:, 1:, :]
        p_mean = P.mean(dim=1)
        p_std = P.std(dim=1)

        pooled = torch.cat([cls_tok, p_mean, p_std], dim=-1)
        V_raw = self.net(pooled).view(B_batch, self.ND, self.r)
        V_ortho, _ = torch.linalg.qr(V_raw)
        return V_ortho


class FactorizedModePredictor(nn.Module):
    """
    Family C: Factorized Mode Predictor.
    Exploits mode separability: each mode Q_k in R^{N x D} is rank-R separable:
        Q_k = sum_{s=0}^{R-1} a_{k, s} b_{k, s}^T
    Predicts token factors A in R^{N x (r*R)} and feature factors B in R^{D x (r*R)}.
    Output is reshaped and orthonormalized via differentiable QR.
    """
    def __init__(
        self,
        N: int = 196,
        D: int = 384,
        r: int = 32,
        R: int = 2,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.N = N
        self.D = D
        self.r = r
        self.R = R
        self.tot_comp = r * R

        # Token factor network: operates token-wise
        self.token_net = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.tot_comp)
        )

        # Feature factor network: operates on pooled CLS + mean(P)
        self.feat_net = nn.Sequential(
            nn.Linear(2 * D, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.tot_comp * D)
        )

    def forward(self, h_act: torch.Tensor) -> torch.Tensor:
        # h_act: (B_batch, 1+N, D)
        B_batch = h_act.size(0)
        cls_tok = h_act[:, 0, :]
        P = h_act[:, 1:, :]

        # Token factors: (B_batch, N, r, R)
        A = self.token_net(P).view(B_batch, self.N, self.r, self.R)

        # Feature factors: (B_batch, r, R, D)
        pooled = torch.cat([cls_tok, P.mean(dim=1)], dim=-1)
        B_feat = self.feat_net(pooled).view(B_batch, self.r, self.R, self.D)

        # Reconstruct modes: Q_{b, k, n, d} = sum_s A_{b, n, k, s} * B_{b, k, s, d}
        Q = torch.einsum('bnks,bksd->bknd', A, B_feat) # (B_batch, r, N, D)
        V_raw = Q.reshape(B_batch, self.r, self.N * self.D).transpose(1, 2) # (B_batch, ND, r)

        # Differentiable QR orthonormalization
        V_ortho, _ = torch.linalg.qr(V_raw)
        return V_ortho


class DirectCarrierPredictor(nn.Module):
    """
    Family D: Direct Carrier Predictor Control.
    Directly predicts carrier displacement Delta C in R^{B x D} from P and S
    without explicit operator subspace prediction.
    """
    def __init__(self, D: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.D = D
        self.net = nn.Sequential(
            nn.Linear(2 * D, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, D)
        )

    def forward(self, P: torch.Tensor, S: torch.Tensor, C_mean: torch.Tensor) -> torch.Tensor:
        # P: (N, D), S: (N, B), C_mean: (B, D)
        # For each group j, pool group features and concatenate with C_mean[j]
        B_tokens = S.shape[1]
        device = P.device
        delta_C = torch.zeros_like(C_mean)
        for j in range(B_tokens):
            members = (S[:, j] > 0)
            p_group = P[members]
            group_feat = torch.cat([C_mean[j], p_group.std(dim=0)], dim=-1)
            delta_C[j] = self.net(group_feat)
        return C_mean + delta_C


def solve_amortized_carrier(
    P: torch.Tensor,
    S: torch.Tensor,
    multiplicities: torch.Tensor,
    V_pred: torch.Tensor,
    lam_factor: float = 10.0
) -> Dict[str, Any]:
    """
    Fast closed-form Tikhonov carrier solve using predicted top-r subspace V_pred in R^{ND x r}:
        J_proj = V_pred^T in R^{r x ND}
    Since r is tiny (e.g. 16 or 32), the Gram matrix is r x r and solve is ultra-fast (<0.5 ms).
    """
    N, D = P.shape
    B_tokens = S.shape[1]
    r = V_pred.shape[1]
    device = P.device

    # 1. Group means
    C_mean = torch.zeros(B_tokens, D, device=device)
    for j in range(B_tokens):
        members = (S[:, j] > 0)
        C_mean[j] = P[members].mean(dim=0)

    E_mean = P - S @ C_mean
    J_proj = V_pred.T # (r, ND)
    r_mean = J_proj @ E_mean.reshape(-1) # (r,)
    norm_r_mean = torch.norm(r_mean).item()

    # 2. Block column sums K_j = sum_{i in G_j} V_{pred, i}^T in R^{r x D}
    V_blocks = J_proj.view(r, N, D)
    K = torch.zeros(r, B_tokens, D, device=device)
    for j in range(B_tokens):
        members = (S[:, j] > 0)
        K[:, j, :] = V_blocks[:, members, :].sum(dim=1)

    # 3. Gram matrix Sigma = sum_j (1 / m_j) K_j K_j^T in R^{r x r}
    Sigma = torch.zeros(r, r, device=device)
    for j in range(B_tokens):
        K_j = K[:, j, :]
        m_j = multiplicities[j].item()
        Sigma += (1.0 / m_j) * (K_j @ K_j.T)

    # 4. Tikhonov regularizer
    sig_trace = torch.trace(Sigma).item() / max(1, r)
    lam = lam_factor * sig_trace
    M = Sigma + lam * torch.eye(r, device=device)

    # 5. Exact r x r solve
    alpha = torch.linalg.solve(M, r_mean)

    # 6. Carrier adjustments
    C_opt = C_mean.clone()
    for j in range(B_tokens):
        K_j = K[:, j, :]
        m_j = multiplicities[j].item()
        delta_C_j = (1.0 / m_j) * (K_j.T @ alpha)
        C_opt[j] += delta_C_j

    E_opt = P - S @ C_opt
    r_opt = J_proj @ E_opt.reshape(-1)
    norm_r_opt = torch.norm(r_opt).item()
    norm_delta_C = torch.norm(C_opt - C_mean).item()

    return {
        "C_mean": C_mean,
        "C_opt": C_opt,
        "norm_r_mean": norm_r_mean,
        "norm_r_opt": norm_r_opt,
        "norm_delta_C": norm_delta_C,
        "E_opt": E_opt
    }
