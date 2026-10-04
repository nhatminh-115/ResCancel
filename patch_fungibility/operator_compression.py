"""
Fungibility Operator Compression Module

Implements:
1. Exact closed-form linear least-squares operator-aware carrier optimization:
   minimize_C ||J_{l->L} vec((P - S C)^T)||^2 + lambda ||P - S C||_F^2
2. Multiplicity-aware sequence shortening via exact carrier collapse
3. Grouping strategies: Spatial, Feature-Similarity, Random
4. Low-rank spectral truncation (rank r in {8, 16, 32, 64, 128})
5. Calibration-averaged operator (J_bar) for zero-backprop inference
6. Full matched-budget benchmark suite
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

from patch_fungibility.dense_fraction_models import forward_block_by_block
from patch_fungibility.multiblock_operator import construct_downstream_jacobian
from patch_fungibility.joint_value_key import construct_cumulative_multiblock_joint_operator
from patch_fungibility.compression_models import (
    forward_downstream_compressed,
    forward_downstream_reference
)


def create_groupings(P, B_tokens, strategy="spatial", seed=42):
    """
    Partitions N patch tokens into B_tokens groups:
    - spatial: 2D grid partitioning of 14x14 patches into contiguous regions
    - feature_similarity: clustering based on cosine similarity of token representations
    - random: uniform random partitioning of tokens
    
    Returns:
        groups: list of lists containing token indices
        S: (N, B_tokens) assignment matrix
        multiplicities: (B_tokens,) token counts per group
    """
    N, D = P.shape
    device = P.device
    
    if strategy == "spatial":
        # Standard ViT 224x224 has 14x14 = 196 patches
        grid_h = int(math.isqrt(N))
        grid_w = N // grid_h
        
        # Partition grid into B_tokens blocks
        coords = np.array([(i // grid_w, i % grid_w) for i in range(N)])
        
        # Determine number of clusters along h and w
        aspect = grid_h / grid_w
        num_clusters_h = max(1, int(round(math.sqrt(B_tokens * aspect))))
        num_clusters_w = max(1, int(round(B_tokens / num_clusters_h)))
        
        # Quantize coordinates to clusters
        cluster_h = np.clip((coords[:, 0] * num_clusters_h) // grid_h, 0, num_clusters_h - 1)
        cluster_w = np.clip((coords[:, 1] * num_clusters_w) // grid_w, 0, num_clusters_w - 1)
        raw_cluster_ids = cluster_h * num_clusters_w + cluster_w
        
        # Remap unique clusters to range [0, B_tokens - 1]
        unique_ids = np.unique(raw_cluster_ids)
        groups = [np.where(raw_cluster_ids == uid)[0].tolist() for uid in unique_ids]
        
        # Adjust group count if rounding produced != B_tokens
        while len(groups) > B_tokens:
            # Merge smallest two groups
            lens = [len(g) for g in groups]
            idx1, idx2 = np.argsort(lens)[:2]
            groups[idx1].extend(groups[idx2])
            groups.pop(idx2)
            
        while len(groups) < B_tokens:
            # Split largest group
            lens = [len(g) for g in groups]
            max_idx = np.argmax(lens)
            half = len(groups[max_idx]) // 2
            if half == 0: break
            new_g = groups[max_idx][half:]
            groups[max_idx] = groups[max_idx][:half]
            groups.append(new_g)

    elif strategy == "feature_similarity":
        # Deterministic clustering based on representation projection
        P_norm = F.normalize(P, p=2, dim=-1)
        # Greedy farthest-point initialization
        torch.manual_seed(seed)
        medoid_indices = [0]
        for _ in range(1, B_tokens):
            sims = P_norm @ P_norm[medoid_indices].T # (N, num_selected)
            max_sims, _ = sims.max(dim=1)
            next_idx = int(torch.argmin(max_sims).item())
            medoid_indices.append(next_idx)
            
        # Assign all tokens to nearest medoid
        sim_to_medoids = P_norm @ P_norm[medoid_indices].T # (N, B_tokens)
        assignments = sim_to_medoids.argmax(dim=1).cpu().numpy()
        
        groups = [[] for _ in range(B_tokens)]
        for i, a in enumerate(assignments):
            groups[a].append(i)
            
        # Handle empty groups
        for j in range(B_tokens):
            if len(groups[j]) == 0:
                # Steal from largest group
                lens = [len(g) for g in groups]
                max_idx = np.argmax(lens)
                groups[j].append(groups[max_idx].pop())

    elif strategy == "random":
        np.random.seed(seed)
        perm = np.random.permutation(N)
        chunks = np.array_split(perm, B_tokens)
        groups = [c.tolist() for c in chunks]
    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")

    # Build S and multiplicities
    B_actual = len(groups)
    S = torch.zeros(N, B_actual, device=device)
    multiplicities = torch.zeros(B_actual, device=device)
    for j, g in enumerate(groups):
        S[g, j] = 1.0
        multiplicities[j] = len(g)
        
    return groups, S, multiplicities


def solve_operator_aware_carriers(P, S, multiplicities, J, lam_factor=10.0):
    """
    Solves the exact closed-form Tikhonov-regularized operator-aware carrier problem:
        min_C ||J vec((P - S C)^T)||^2 + lambda ||P - S C||_F^2
        
    Returns:
        C_mean: (B, D) group mean carriers
        C_opt: (B, D) operator-aware carriers
        norm_r_mean: ||J vec(E_mean^T)||
        norm_r_opt: ||J vec(E_opt^T)||
        norm_delta_C: ||C_opt - C_mean||_F
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
        
    E_mean = P - S @ C_mean # (N, D)
    vec_E_mean = E_mean.reshape(-1) # (ND,)
    r_mean = J @ vec_E_mean # (D_readout,)
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
        K_j = K[:, j, :] # (D_readout, D)
        m_j = multiplicities[j].item()
        Sigma += (1.0 / m_j) * (K_j @ K_j.T)
        
    # 4. Tikhonov regularizer
    sig_trace = torch.trace(Sigma).item() / max(1, D_readout)
    lam = lam_factor * sig_trace
    M = Sigma + lam * torch.eye(D_readout, device=device)
    
    # 5. Exact linear solve
    alpha = torch.linalg.solve(M, r_mean) # (D_readout,)
    
    # 6. Carrier adjustments
    C_opt = C_mean.clone()
    for j in range(B_tokens):
        K_j = K[:, j, :]
        m_j = multiplicities[j].item()
        delta_C_j = (1.0 / m_j) * (K_j.T @ alpha) # (D,)
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


def solve_low_rank_carriers(P, S, multiplicities, J, rank_r, lam_factor=10.0):
    """
    Truncates J to its top r singular modes before solving the closed-form carrier problem.
    """
    D_readout = J.shape[0]
    device = P.device
    
    Gram_J = J @ J.T
    evals, U = torch.linalg.eigh(Gram_J)
    evals = evals.flip(0)
    U = U.flip(1)
    
    actual_r = min(rank_r, D_readout)
    U_r = U[:, :actual_r] # (D_readout, r)
    
    # Projected operator: J_proj = U_r^T J in R^{r x ND}
    J_proj = U_r.T @ J
    
    sol = solve_operator_aware_carriers(P, S, multiplicities, J_proj, lam_factor=lam_factor)
    return sol


def evaluate_matched_budget_compression(
    model, arch_type, depth, clean_acts_img, clean_logits_img, targets_img, B_tokens,
    grouping_strategy="spatial", op_multi=None, op_cum=None, J_bar=None, device=None
):
    """
    Evaluates all matched-budget compression methods at budget B_tokens on a single image activation.
    """
    D_readout = op_multi["D_readout"]
    J_i = op_multi["J"]
    
    P = clean_acts_img[0, 1:, :] # (N, D)
    N, D = P.shape
    cls_token = clean_acts_img[:, 0:1, :]
    
    # 1. Grouping
    groups, S, multiplicities = create_groupings(P, B_tokens, strategy=grouping_strategy)
    mults = torch.cat([torch.tensor([1.0], device=device), multiplicities], dim=0)
    
    results = {}
    
    # Helper to evaluate compressed tokens
    def eval_compressed(C_tokens, mult_tensor=None):
        if mult_tensor is None:
            mult_tensor = mults
        h_comp = torch.cat([cls_token, C_tokens.unsqueeze(0)], dim=1)
        logits_comp = forward_downstream_compressed(model, arch_type, depth, h_comp, mult_tensor, N)
        l2 = torch.norm(logits_comp - clean_logits_img).item()
        
        p_c = F.softmax(clean_logits_img, dim=-1)
        log_p = F.log_softmax(logits_comp, dim=-1)
        kl = F.kl_div(log_p, p_c, reduction='batchmean').item()
        
        c_top1 = clean_logits_img.argmax(dim=-1).item()
        p_top1 = logits_comp.argmax(dim=-1).item()
        correct = (p_top1 == targets_img.item())
        flip = (c_top1 != p_top1)
        
        c_tgt = clean_logits_img[0, targets_img.item()].item()
        p_tgt = logits_comp[0, targets_img.item()].item()
        margin_dam = c_tgt - p_tgt
        
        return {
            "logit_l2": l2,
            "kl_div": kl,
            "top1_acc": 1.0 if correct else 0.0,
            "top1_flip": 1.0 if flip else 0.0,
            "margin_damage": margin_dam,
            "logits": logits_comp
        }
        
    # Helper to evaluate full surrogate (for collapse parity)
    def eval_full(C_tokens):
        h_full = clean_acts_img.clone()
        h_full[0, 1:, :] = S @ C_tokens
        logits_full = forward_downstream_reference(model, arch_type, depth, h_full)
        return logits_full

    # Method 1: Random Pruning
    torch.manual_seed(42)
    keep_indices = torch.randperm(N, device=device)[:B_tokens].sort()[0]
    C_rand_prune = P[keep_indices] # (B_tokens, D)
    unweighted_mults = torch.ones(1 + B_tokens, device=device)
    res_rand_prune = eval_compressed(C_rand_prune, unweighted_mults)
    results["random_pruning"] = res_rand_prune
    
    # Method 2: Norm-based Pruning
    norms = torch.norm(P, dim=-1)
    keep_norm_idx = torch.topk(norms, B_tokens).indices.sort()[0]
    C_norm_prune = P[keep_norm_idx]
    res_norm_prune = eval_compressed(C_norm_prune, unweighted_mults)
    results["norm_pruning"] = res_norm_prune
    
    # Method 3: Group Mean Merging
    sol_J = solve_operator_aware_carriers(P, S, multiplicities, J_i, lam_factor=10.0)
    C_mean = sol_J["C_mean"]
    res_mean = eval_compressed(C_mean, mults)
    logits_mean_full = eval_full(C_mean)
    parity_mean = torch.norm(res_mean["logits"] - logits_mean_full, p=float('inf')).item()
    res_mean["collapse_parity"] = parity_mean
    res_mean["operator_residual"] = sol_J["norm_r_mean"]
    results["group_mean"] = res_mean
    
    # Method 4: Medoid Merging
    C_medoid = torch.zeros(B_tokens, D, device=device)
    for j, g in enumerate(groups):
        p_g = P[g]
        diffs = torch.norm(p_g - C_mean[j:j+1], dim=-1)
        best_i = int(torch.argmin(diffs).item())
        C_medoid[j] = p_g[best_i]
    res_medoid = eval_compressed(C_medoid, mults)
    E_medoid = P - S @ C_medoid
    res_medoid["operator_residual"] = torch.norm(J_i @ E_medoid.reshape(-1)).item()
    results["medoid"] = res_medoid
    
    # Method 5: Unweighted Centroid Carrier (mult=1)
    res_unweighted = eval_compressed(C_mean, unweighted_mults)
    results["unweighted_centroid"] = res_unweighted
    
    # Method 6: Operator-Aware Carrier (J_{l->L})
    C_opt_J = sol_J["C_opt"]
    res_opt_J = eval_compressed(C_opt_J, mults)
    logits_opt_full = eval_full(C_opt_J)
    parity_opt = torch.norm(res_opt_J["logits"] - logits_opt_full, p=float('inf')).item()
    res_opt_J["collapse_parity"] = parity_opt
    res_opt_J["operator_residual"] = sol_J["norm_r_opt"]
    res_opt_J["delta_C_norm"] = sol_J["norm_delta_C"]
    results["operator_aware_J"] = res_opt_J
    
    # Method 7: Operator-Aware Carrier (Cumulative C_{l->L})
    if op_cum is not None:
        C_cum_op = op_cum["C_cum"]
        sol_cum = solve_operator_aware_carriers(P, S, multiplicities, C_cum_op, lam_factor=10.0)
        C_opt_cum = sol_cum["C_opt"]
        res_opt_cum = eval_compressed(C_opt_cum, mults)
        res_opt_cum["operator_residual"] = sol_cum["norm_r_opt"]
        results["operator_aware_C_cum"] = res_opt_cum
        
    # Method 8: Calibration-Averaged Operator (J_bar)
    if J_bar is not None:
        sol_bar = solve_operator_aware_carriers(P, S, multiplicities, J_bar, lam_factor=10.0)
        C_opt_bar = sol_bar["C_opt"]
        res_bar = eval_compressed(C_opt_bar, mults)
        E_bar = P - S @ C_opt_bar
        res_bar["operator_residual"] = torch.norm(J_i @ E_bar.reshape(-1)).item()
        results["calibration_J_bar"] = res_bar
        
    return results
