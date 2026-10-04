"""
patch_fungibility/operator_compression_confirmatory.py

Strict Confirmatory Benchmark Engine for Operator-Aware Token Compression:
1. Fast batched-VJP downstream Jacobian construction
2. Frozen Primary & Baseline token reduction methods:
   - Random Pruning (multiple seeds)
   - Norm-based Pruning
   - Attention-based Pruning (CLS attention)
   - Group-Mean Merging (feature-similarity grouping)
   - Medoid Merging
   - Unweighted Centroid Carrier
   - ToMe-style Bipartite Soft Matching (BSM)
   - Operator-Aware Oracle Compression (full per-image J)
   - Low-Rank J-Aware Compression (r in {16, 32, 64})
   - Cumulative Joint Value-Key Compression (C_cum)
3. Controlled Grouping Ablations (Spatial, Random)
4. Same-Group Causal Controls
5. Paired Statistical Tests (McNemar, Paired t / Wilcoxon, Bootstrap 95% CI, Cohen's d_z)
6. AUC-Token Frontier and Retention Metrics
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional

from patch_fungibility.dense_fraction_models import forward_block_by_block
from patch_fungibility.compression_models import (
    forward_downstream_compressed,
    forward_downstream_reference
)


def compute_fast_downstream_jacobian(
    model: nn.Module,
    model_key: str,
    start_depth: int,
    clean_acts_img: torch.Tensor,
    chunk_size: int = 128
) -> torch.Tensor:
    """
    Computes exact downstream Jacobian J = d(z_readout) / d(vec(P^T)) using batched VJP.
    clean_acts_img: (1, 1+N, D) activation at start_depth
    Returns:
        J: (D_readout, N*D)
    """
    device = clean_acts_img.device
    B, total_tokens, D = clean_acts_img.shape
    N = total_tokens - 1

    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        D_readout = D
        def forward_downstream(h_in):
            h = h_in
            for b_idx in range(start_depth, len(model.blocks)):
                h = model.blocks[b_idx](h)
            h_norm = model.norm(h)
            return h_norm[:, 0, :] # CLS token readout: (curr_b, D)
    elif model_key == "dinov2":
        D_readout = 2 * D
        backbone = model.backbone
        def forward_downstream(h_in):
            h = h_in
            for b_idx in range(start_depth, len(backbone.blocks)):
                h = backbone.blocks[b_idx](h)
            h_norm = backbone.norm(h)
            cls_norm = h_norm[:, 0]
            patch_mean = h_norm[:, 1:].mean(dim=1)
            return torch.cat([cls_norm, patch_mean], dim=-1) # (curr_b, 2D)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    rows = []
    for d in range(0, D_readout, chunk_size):
        end_d = min(d + chunk_size, D_readout)
        curr_b = end_d - d
        h_batch = clean_acts_img.repeat(curr_b, 1, 1).requires_grad_(True)
        z_out = forward_downstream(h_batch)

        grad_outputs = torch.eye(D_readout, device=device)[d:end_d]
        g = torch.autograd.grad(z_out, h_batch, grad_outputs=grad_outputs)[0]
        # Only patch tokens (exclude CLS token at index 0)
        g_patch = g[:, 1:, :].reshape(curr_b, -1) # (curr_b, N*D)
        rows.append(g_patch)

    J = torch.cat(rows, dim=0) # (D_readout, N*D)
    return J


def extract_cls_attention_weights(
    model: nn.Module,
    model_key: str,
    depth: int,
    acts_img: torch.Tensor
) -> torch.Tensor:
    """
    Extracts attention weights from [CLS] to all patch tokens at block `depth`.
    Returns: (N,) tensor of weights summing to 1.
    """
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        block = model.blocks[depth]
        h = block.norm1(acts_img)
        B, N_tot, C = h.shape
        qkv = block.attn.qkv(h).reshape(B, N_tot, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = block.attn.q_norm(q), block.attn.k_norm(k)
        head_dim = block.attn.head_dim
    elif model_key == "dinov2":
        block = model.backbone.blocks[depth]
        h = block.norm1(acts_img)
        B, N_tot, C = h.shape
        num_heads = block.attn.num_heads
        head_dim = C // num_heads
        qkv = block.attn.qkv(h).reshape(B, N_tot, 3, num_heads, head_dim)
        q, k, v = torch.unbind(qkv, 2)
        q, k = [t.transpose(1, 2) for t in [q, k]]
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    q_cls = q[:, :, 0:1, :] # (B, H, 1, d)
    k_patch = k[:, :, 1:, :] # (B, H, N, d)
    scores = (q_cls @ k_patch.transpose(-2, -1)) / (head_dim ** 0.5)
    attn = F.softmax(scores, dim=-1).mean(dim=1).squeeze(0).squeeze(0) # (N,)
    return attn


def tome_bipartite_merge(
    P: torch.Tensor,
    B_target: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Faithful training-free ToMe-style Bipartite Soft Matching (BSM) token merging.
    Merges patch tokens until exactly B_target remain.
    Returns:
        cur_tokens: (B_target, D)
        cur_mults: (B_target,) multiplicities summing to N
    """
    N, D = P.shape
    device = P.device
    r_to_merge = N - B_target
    if r_to_merge <= 0:
        return P.clone(), torch.ones(N, device=device)

    cur_tokens = P.clone()
    cur_mults = torch.ones(N, device=device)

    while cur_tokens.size(0) > B_target:
        cur_N = cur_tokens.size(0)
        n_merge = min(cur_N // 2, cur_N - B_target)

        # Alternating split into A and B
        idx_A = torch.arange(0, cur_N, 2, device=device)
        idx_B = torch.arange(1, cur_N, 2, device=device)

        A = cur_tokens[idx_A]
        B = cur_tokens[idx_B]
        mult_A = cur_mults[idx_A]
        mult_B = cur_mults[idx_B]

        A_norm = F.normalize(A, p=2, dim=-1)
        B_norm = F.normalize(B, p=2, dim=-1)
        sim = A_norm @ B_norm.T # (|A|, |B|)

        max_sim, best_b = sim.max(dim=1)
        top_edges = torch.topk(max_sim, n_merge).indices
        merge_mask = torch.zeros(A.size(0), dtype=torch.bool, device=device)
        merge_mask[top_edges] = True

        for a_idx in top_edges:
            b_idx = best_b[a_idx]
            w_a = mult_A[a_idx]
            w_b = mult_B[b_idx]
            B[b_idx] = (w_b * B[b_idx] + w_a * A[a_idx]) / (w_b + w_a)
            mult_B[b_idx] += w_a

        keep_A = A[~merge_mask]
        keep_mult_A = mult_A[~merge_mask]

        cur_tokens = torch.cat([keep_A, B], dim=0)
        cur_mults = torch.cat([keep_mult_A, mult_B], dim=0)

    return cur_tokens, cur_mults


def create_groupings_confirmatory(
    P: torch.Tensor,
    B_tokens: int,
    strategy: str = "feature_similarity",
    seed: int = 42
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    """
    Partitions N patch tokens into B_tokens groups:
    - feature_similarity: cosine similarity clustering with deterministic medoids
    - spatial: 2D regular grid partition
    - random: uniform random partition with seed
    """
    N, D = P.shape
    device = P.device

    if strategy == "feature_similarity":
        P_norm = F.normalize(P, p=2, dim=-1)
        torch.manual_seed(seed)
        medoid_indices = [0]
        for _ in range(1, B_tokens):
            sims = P_norm @ P_norm[medoid_indices].T
            max_sims, _ = sims.max(dim=1)
            next_idx = int(torch.argmin(max_sims).item())
            medoid_indices.append(next_idx)

        sim_to_medoids = P_norm @ P_norm[medoid_indices].T
        assignments = sim_to_medoids.argmax(dim=1).cpu().numpy()

        groups = [[] for _ in range(B_tokens)]
        for i, a in enumerate(assignments):
            groups[a].append(i)

        for j in range(B_tokens):
            if len(groups[j]) == 0:
                lens = [len(g) for g in groups]
                max_idx = np.argmax(lens)
                groups[j].append(groups[max_idx].pop())

    elif strategy == "spatial":
        grid_h = int(math.isqrt(N))
        grid_w = N // grid_h
        coords = np.array([(i // grid_w, i % grid_w) for i in range(N)])
        aspect = grid_h / grid_w
        num_clusters_h = max(1, int(round(math.sqrt(B_tokens * aspect))))
        num_clusters_w = max(1, int(round(B_tokens / num_clusters_h)))

        cluster_h = np.clip((coords[:, 0] * num_clusters_h) // grid_h, 0, num_clusters_h - 1)
        cluster_w = np.clip((coords[:, 1] * num_clusters_w) // grid_w, 0, num_clusters_w - 1)
        raw_cluster_ids = cluster_h * num_clusters_w + cluster_w
        unique_ids = np.unique(raw_cluster_ids)
        groups = [np.where(raw_cluster_ids == uid)[0].tolist() for uid in unique_ids]

        while len(groups) > B_tokens:
            lens = [len(g) for g in groups]
            idx1, idx2 = np.argsort(lens)[:2]
            groups[idx1].extend(groups[idx2])
            groups.pop(idx2)

        while len(groups) < B_tokens:
            lens = [len(g) for g in groups]
            max_idx = np.argmax(lens)
            half = len(groups[max_idx]) // 2
            if half == 0:
                break
            new_g = groups[max_idx][half:]
            groups[max_idx] = groups[max_idx][:half]
            groups.append(new_g)

    elif strategy == "random":
        np.random.seed(seed)
        perm = np.random.permutation(N)
        chunks = np.array_split(perm, B_tokens)
        groups = [c.tolist() for c in chunks]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    B_actual = len(groups)
    S = torch.zeros(N, B_actual, device=device)
    multiplicities = torch.zeros(B_actual, device=device)
    for j, g in enumerate(groups):
        S[g, j] = 1.0
        multiplicities[j] = len(g)

    return groups, S, multiplicities


def solve_operator_aware_carriers_fast(
    P: torch.Tensor,
    S: torch.Tensor,
    multiplicities: torch.Tensor,
    J: torch.Tensor,
    lam_factor: float = 10.0
) -> Dict[str, Any]:
    """
    Exact closed-form Tikhonov-regularized operator-aware carrier optimization:
        min_C ||J vec((P - S C)^T)||^2 + lambda ||P - S C||_F^2
    """
    N, D = P.shape
    B_tokens = S.shape[1]
    D_readout = J.shape[0]
    device = P.device

    # 1. Group Means
    C_mean = torch.zeros(B_tokens, D, device=device)
    for j in range(B_tokens):
        members = (S[:, j] > 0)
        C_mean[j] = P[members].mean(dim=0)

    E_mean = P - S @ C_mean
    r_mean = J @ E_mean.reshape(-1)
    norm_r_mean = torch.norm(r_mean).item()

    # 2. Block column sums K_j = sum_{i in G_j} J_i in R^{D_readout x D}
    J_blocks = J.view(D_readout, N, D)
    K = torch.zeros(D_readout, B_tokens, D, device=device)
    for j in range(B_tokens):
        members = (S[:, j] > 0)
        K[:, j, :] = J_blocks[:, members, :].sum(dim=1)

    # 3. Gram matrix Sigma = sum_j (1 / m_j) K_j K_j^T
    Sigma = torch.zeros(D_readout, D_readout, device=device)
    for j in range(B_tokens):
        K_j = K[:, j, :]
        m_j = multiplicities[j].item()
        Sigma += (1.0 / m_j) * (K_j @ K_j.T)

    # 4. Tikhonov regularizer
    sig_trace = torch.trace(Sigma).item() / max(1, D_readout)
    lam = lam_factor * sig_trace
    M = Sigma + lam * torch.eye(D_readout, device=device)

    # 5. Exact linear solve
    alpha = torch.linalg.solve(M, r_mean)

    # 6. Carrier adjustments
    C_opt = C_mean.clone()
    for j in range(B_tokens):
        K_j = K[:, j, :]
        m_j = multiplicities[j].item()
        delta_C_j = (1.0 / m_j) * (K_j.T @ alpha)
        C_opt[j] += delta_C_j

    E_opt = P - S @ C_opt
    r_opt = J @ E_opt.reshape(-1)
    norm_r_opt = torch.norm(r_opt).item()
    norm_delta_C = torch.norm(C_opt - C_mean).item()

    return {
        "C_mean": C_mean,
        "C_opt": C_opt,
        "norm_r_mean": norm_r_mean,
        "norm_r_opt": norm_r_opt,
        "norm_delta_C": norm_delta_C,
        "E_mean": E_mean,
        "E_opt": E_opt
    }


def solve_low_rank_carriers_fast(
    P: torch.Tensor,
    S: torch.Tensor,
    multiplicities: torch.Tensor,
    J: torch.Tensor,
    rank_r: int,
    lam_factor: float = 10.0
) -> Dict[str, Any]:
    """
    Low-rank spectral projection before carrier solve.
    """
    D_readout = J.shape[0]
    device = P.device

    Gram_J = J @ J.T
    evals, U = torch.linalg.eigh(Gram_J)
    evals = evals.flip(0)
    U = U.flip(1)

    actual_r = min(rank_r, D_readout)
    U_r = U[:, :actual_r]
    J_proj = U_r.T @ J

    sol = solve_operator_aware_carriers_fast(P, S, multiplicities, J_proj, lam_factor=lam_factor)
    # Re-evaluate residual against full J
    E_opt = sol["E_opt"]
    sol["norm_r_opt"] = torch.norm(J @ E_opt.reshape(-1)).item()
    return sol


def compute_mcnemar_pvalue(b: int, c: int) -> float:
    """
    Computes exact two-sided McNemar p-value for discordant counts b and c.
    """
    n = b + c
    if n == 0:
        return 1.0
    import scipy.stats as stats
    # Exact binomial test with p = 0.5
    k = min(b, c)
    p_val = stats.binomtest(k, n, 0.5, alternative='two-sided').pvalue
    return float(p_val)


def bootstrap_mean_diff_ci(
    diffs: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Computes 95% bootstrap confidence interval for mean difference.
    """
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return ci_low, ci_high
