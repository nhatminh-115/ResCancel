"""
Full Fungibility Operator Module: Unified Token x Feature Transmission Spectrum

Constructs, diagonalizes, and validates the single mathematical operator A_l acting
directly on the entire patch-stream perturbation Delta P in R^{N x D}:
    vec(Delta z_readout) = A_l vec(Delta P^T).

Supports:
- Standard CLS-readout models (DeiT-Small, DeiT-Tiny, ViT-B/16)
- Dual-channel readout models (DINOv2 ViT-S/14: CLS + mean patch pooling)
"""

import os
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from torch.utils.data import DataLoader, Subset


def construct_full_operator(model, arch_type, depth, clean_acts, device):
    """
    Constructs the exact linear transmission operator A_l from clean activations.
    
    Returns:
        dict containing:
            'A_l': (D_out, N * D) tensor
            'Gram': (D_out, D_out) tensor
            'evals': (D_out,) eigenvalues of Gram sorted descending
            'U': (D_out, D_out) orthonormal eigenvectors
            'S': (D_out,) singular values = sqrt(evals)
            'N': int, number of patch tokens
            'D': int, embedding dimension
            'D_out': int, readout dimension (D for CLS, 2D for DINOv2)
            'sigma_P': float, natural patch activation scale
            'mean_w': attention weights
    """
    B, total_tokens, D = clean_acts.shape
    N = total_tokens - 1
    
    # Calculate natural patch activation scale
    patches = clean_acts[:, 1:, :].reshape(-1, D)
    sigma_P = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()
    
    # Extract weights based on architecture
    if arch_type == "dinov2":
        blk = model.backbone.blocks[depth]
        norm_c = blk.norm1(clean_acts)
        num_heads = blk.attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5
        
        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim)
        qc, kc, _ = torch.unbind(qkv_c, 2)
        qc = qc.transpose(1, 2)
        kc = kc.transpose(1, 2)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
        
        w_cls = A_clean[:, :, 0, 1:].mean(dim=0) # (H, N)
        w_patch = A_clean[:, :, 1:, 1:].mean(dim=2).mean(dim=0) # (H, N)
        
        W_V = blk.attn.qkv.weight[2*D : 3*D, :]
        W_O = blk.attn.proj.weight
        
        M_h_list = [W_O[:, h*head_dim : (h+1)*head_dim] @ W_V[h*head_dim : (h+1)*head_dim, :] for h in range(num_heads)]
        M_stack = torch.stack(M_h_list, dim=0) # (H, D, D)
        
        K_cls = torch.einsum('hn,hij->nij', w_cls, M_stack) # (N, D, D)
        K_patch = torch.einsum('hn,hij->nij', w_patch, M_stack) # (N, D, D)
        K_all = torch.cat([K_cls, K_patch], dim=1) # (N, 2D, D)
        
        D_out = 2 * D
        A_l = K_all.permute(1, 0, 2).reshape(D_out, N * D)
        mean_w = torch.stack([w_cls, w_patch], dim=0)
    else:
        # timm models (DeiT-Small, DeiT-Tiny, ViT-B/16)
        blk = model.blocks[depth]
        norm_c = blk.norm1(clean_acts)
        num_heads = blk.attn.num_heads
        head_dim = blk.attn.head_dim
        scale = blk.attn.scale
        
        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qc, kc, _ = qkv_c.unbind(0)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
        mean_w = A_clean[:, :, 0, 1:].mean(dim=0) # (H, N)
        
        W_V = blk.attn.qkv.weight[2*D : 3*D, :]
        W_O = blk.attn.proj.weight
        
        M_h_list = [W_O[:, h*head_dim : (h+1)*head_dim] @ W_V[h*head_dim : (h+1)*head_dim, :] for h in range(num_heads)]
        M_stack = torch.stack(M_h_list, dim=0) # (H, D, D)
        
        K_all = torch.einsum('hn,hij->nij', mean_w, M_stack) # (N, D, D)
        D_out = D
        A_l = K_all.permute(1, 0, 2).reshape(D_out, N * D)
        
    # Fast exact SVD via Gram matrix
    Gram = A_l @ A_l.T # (D_out, D_out)
    evals, U = torch.linalg.eigh(Gram)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    U = U[:, idx]
    S = torch.sqrt(torch.clamp(evals, min=0.0))
    
    return {
        "A_l": A_l,
        "Gram": Gram,
        "evals": evals,
        "U": U,
        "S": S,
        "N": N,
        "D": D,
        "D_out": D_out,
        "sigma_P": sigma_P,
        "mean_w": mean_w,
        "arch_type": arch_type,
        "depth": depth
    }


def extract_operator_modes(op_dict, seed=42):
    """
    Extracts representative singular modes from A_l:
    - Q_0 (top mode)
    - Q_1 (second top mode)
    - Q_mid (median transmission mode)
    - Q_bot (lowest non-zero transmission mode)
    - Q_null (exact mathematical null mode)
    - Q_random (unstructured Gaussian control)
    - Q_checkerboard (spatial checkerboard pattern)
    - Q_randsign (random-sign pattern)
    """
    A_l = op_dict["A_l"]
    U = op_dict["U"]
    S = op_dict["S"]
    N = op_dict["N"]
    D = op_dict["D"]
    D_out = op_dict["D_out"]
    device = A_l.device
    
    modes = {}
    
    # Helper to construct right singular vector from left eigenvector
    def get_right_mode(k):
        sigma = S[k].item()
        if sigma > 1e-10:
            v_flat = (1.0 / sigma) * (A_l.T @ U[:, k])
        else:
            v_flat = torch.randn(N * D, device=device)
        v_flat = v_flat / torch.norm(v_flat)
        return v_flat.view(N, D)
        
    modes["top_0"] = get_right_mode(0)
    modes["top_1"] = get_right_mode(1)
    
    mid_idx = D_out // 2
    modes["mid"] = get_right_mode(mid_idx)
    
    # Lowest non-zero index where S > 1e-5
    nz_indices = torch.where(S > 1e-5)[0]
    bot_idx = nz_indices[-1].item() if len(nz_indices) > 0 else D_out - 1
    modes["bot_nonzero"] = get_right_mode(bot_idx)
    
    # Exact null mode via fast pseudo-inverse projection
    torch.manual_seed(seed)
    r = torch.randn(N * D, device=device)
    inv_evals = torch.where(op_dict["evals"] > 1e-10, 1.0 / op_dict["evals"], torch.zeros_like(op_dict["evals"]))
    Ar = A_l @ r
    inv_G_Ar = U @ (inv_evals * (U.T @ Ar))
    r_row = A_l.T @ inv_G_Ar
    r_null = r - r_row
    r_null = r_null / torch.norm(r_null)
    modes["exact_null"] = r_null.view(N, D)
    
    # Random Gaussian control
    r_iso = torch.randn(N, D, device=device)
    modes["random_gaussian"] = r_iso / torch.norm(r_iso)
    
    # Spatial checkerboard mode on top feature direction
    # SVD of Q_0 gives top feature direction
    _, _, Vh_top = torch.linalg.svd(modes["top_0"])
    v_feat = Vh_top[0, :]
    v_feat = v_feat / torch.norm(v_feat)
    
    # Checkerboard spatial pattern
    grid_size = int(math.isqrt(N))
    check_grid = torch.ones(grid_size, grid_size, device=device)
    check_grid[::2, 1::2] = -1
    check_grid[1::2, ::2] = -1
    a_check = check_grid.reshape(-1)
    if len(a_check) < N:
        pad = torch.ones(N - len(a_check), device=device)
        a_check = torch.cat([a_check, pad])
    elif len(a_check) > N:
        a_check = a_check[:N]
    a_check = a_check / torch.norm(a_check)
    modes["checkerboard"] = torch.outer(a_check, v_feat)
    
    # Random-sign pattern on top feature direction
    a_rand = torch.randint(0, 2, (N,), device=device).float() * 2 - 1
    a_rand = a_rand / torch.norm(a_rand)
    modes["random_sign"] = torch.outer(a_rand, v_feat)
    
    # Verify all norms are 1.0
    for name, m in modes.items():
        assert abs(torch.norm(m).item() - 1.0) < 1e-4, f"Mode {name} norm is {torch.norm(m).item()}"
        
    return modes


def evaluate_mode_factorization(op_dict, num_modes=100):
    """
    Computes rank-1 and rank-2 energy fractions of right singular modes Q_k in R^{N x D}.
    """
    A_l = op_dict["A_l"]
    U = op_dict["U"]
    S = op_dict["S"]
    N = op_dict["N"]
    D = op_dict["D"]
    D_out = op_dict["D_out"]
    
    max_k = min(num_modes, D_out)
    results = []
    
    for k in range(max_k):
        sigma = S[k].item()
        if sigma < 1e-10:
            continue
        v_flat = (1.0 / sigma) * (A_l.T @ U[:, k])
        Q_k = v_flat.view(N, D)
        
        _, S_q, _ = torch.linalg.svd(Q_k)
        total_sq = torch.sum(S_q**2).item()
        r1 = (S_q[0]**2).item() / total_sq
        r2 = ((S_q[0]**2 + S_q[1]**2).item()) / total_sq
        r3 = ((S_q[0]**2 + S_q[1]**2 + S_q[2]**2).item()) / total_sq
        
        results.append({
            "mode_index": k,
            "singular_value": sigma,
            "rank1_energy_fraction": r1,
            "rank2_energy_fraction": r2,
            "rank3_energy_fraction": r3
        })
        
    return pd.DataFrame(results)


def run_full_model_eval(model, arch_type, depth, clean_logits, clean_acts, delta_P, eval_targets, device):
    """
    Runs the full unconstrained model with patch perturbation delta_P.
    """
    B, total_tokens, D = clean_acts.shape
    h_pert = clean_acts.clone()
    h_pert[:, 1:, :] += delta_P.unsqueeze(0)
    
    pert_logits, pert_collected = forward_block_by_block(
        model, arch_type, h_start=h_pert, start_depth=depth, collect_depths=(depth+1,)
    )
    
    diff = pert_logits - clean_logits
    logit_l2 = torch.norm(diff, dim=-1).mean().item()
    
    # KL divergence
    p_clean = F.softmax(clean_logits, dim=-1)
    log_p_pert = F.log_softmax(pert_logits, dim=-1)
    kl_div = F.kl_div(log_p_pert, p_clean, reduction='batchmean').item()
    
    # Margin damage
    clean_top1 = clean_logits.argmax(dim=-1)
    pert_top1 = pert_logits.argmax(dim=-1)
    top1_flip_rate = (clean_top1 != pert_top1).float().mean().item()
    top1_acc = (pert_top1 == eval_targets).float().mean().item()
    
    clean_target_logit = clean_logits.gather(1, eval_targets.unsqueeze(1)).squeeze(1)
    pert_target_logit = pert_logits.gather(1, eval_targets.unsqueeze(1)).squeeze(1)
    margin_damage = (clean_target_logit - pert_target_logit).mean().item()
    
    # Readout disturbance at next block
    h_next_clean = pert_collected[depth+1] # Note: pert_collected collected from h_pert
    # To get clean next block, compare token 0
    # Or measure norm of readout update directly
    readout_dist = torch.norm(h_pert[:, 0, :] - clean_acts[:, 0, :], dim=-1).mean().item()
    
    return {
        "logit_l2": logit_l2,
        "kl_div": kl_div,
        "margin_damage": margin_damage,
        "top1_acc": top1_acc,
        "top1_flip_rate": top1_flip_rate,
        "readout_disturbance": readout_dist
    }


def evaluate_finite_radius_curves(model, arch_type, depth, clean_logits, clean_acts, modes_dict, eval_targets, sigma_P, device):
    """
    Evaluates downstream model damage across finite perturbation scale sweep:
    s in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0]
    """
    scale_sweep = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0]
    rows = []
    
    for mode_name, Q_k in modes_dict.items():
        for s in scale_sweep:
            alpha = s * sigma_P
            delta_P = alpha * Q_k
            
            if s == 0.0:
                metrics = {
                    "logit_l2": 0.0,
                    "kl_div": 0.0,
                    "margin_damage": 0.0,
                    "top1_acc": (clean_logits.argmax(dim=-1) == eval_targets).float().mean().item(),
                    "top1_flip_rate": 0.0,
                    "readout_disturbance": 0.0
                }
            else:
                metrics = run_full_model_eval(
                    model, arch_type, depth, clean_logits, clean_acts, delta_P, eval_targets, device
                )
                
            row = {
                "arch": arch_type,
                "depth": depth,
                "mode_name": mode_name,
                "scale_factor": s,
                "alpha": alpha,
                **metrics
            }
            rows.append(row)
            
    return pd.DataFrame(rows)


def evaluate_random_held_out(model, arch_type, depth, clean_logits, clean_acts, op_dict, eval_targets, device, M=100, seed=42):
    """
    Evaluates M held-out perturbations:
    1) Pure isotropic Gaussian (M=100)
    2) Diverse geometry spanning null to top (M=100)
    Computes Pearson r and Spearman rho between predicted linear transmission tau and observed full-model damage.
    """
    A_l = op_dict["A_l"]
    U = op_dict["U"]
    S = op_dict["S"]
    evals = op_dict["evals"]
    N = op_dict["N"]
    D = op_dict["D"]
    sigma_P = op_dict["sigma_P"]
    
    scale_s = 0.4
    alpha = scale_s * sigma_P
    inv_evals = torch.where(evals > 1e-10, 1.0 / evals, torch.zeros_like(evals))
    
    rows = []
    
    # Batch 1: Pure isotropic Gaussian perturbations
    torch.manual_seed(seed)
    for j in range(M):
        R = torch.randn(N, D, device=device)
        R = R / torch.norm(R)
        tau = torch.norm(A_l @ R.reshape(-1)).item()
        
        delta_P = alpha * R
        metrics = run_full_model_eval(
            model, arch_type, depth, clean_logits, clean_acts, delta_P, eval_targets, device
        )
        rows.append({
            "set_type": "pure_isotropic_gaussian",
            "pert_id": j,
            "predicted_tau": tau,
            "scale_factor": scale_s,
            "alpha": alpha,
            **metrics
        })
        
    # Batch 2: Diverse geometric perturbations (spanning top, mid, null, and isotropic)
    torch.manual_seed(seed + 1000)
    diverse_list = []
    # 25 top linear combinations
    for _ in range(25):
        coeffs = torch.randn(min(50, len(S)), device=device)
        coeffs = coeffs / torch.norm(coeffs)
        u_comb = U[:, :len(coeffs)] @ (coeffs / S[:len(coeffs)])
        v_comb = A_l.T @ u_comb
        diverse_list.append(v_comb / torch.norm(v_comb))
        
    # 25 mid linear combinations
    mid_start = len(S) // 4
    mid_end = 3 * len(S) // 4
    mid_len = mid_end - mid_start
    for _ in range(25):
        coeffs = torch.randn(mid_len, device=device)
        coeffs = coeffs / torch.norm(coeffs)
        u_comb = U[:, mid_start:mid_end] @ (coeffs / S[mid_start:mid_end])
        v_comb = A_l.T @ u_comb
        diverse_list.append(v_comb / torch.norm(v_comb))
        
    # 25 null space perturbations
    for _ in range(25):
        r = torch.randn(N * D, device=device)
        Ar = A_l @ r
        inv_G_Ar = U @ (inv_evals * (U.T @ Ar))
        r_null = r - (A_l.T @ inv_G_Ar)
        diverse_list.append(r_null / torch.norm(r_null))
        
    # 25 isotropic Gaussian
    for _ in range(25):
        r = torch.randn(N * D, device=device)
        diverse_list.append(r / torch.norm(r))
        
    for j, v_flat in enumerate(diverse_list):
        R = v_flat.view(N, D)
        tau = torch.norm(A_l @ v_flat).item()
        delta_P = alpha * R
        metrics = run_full_model_eval(
            model, arch_type, depth, clean_logits, clean_acts, delta_P, eval_targets, device
        )
        rows.append({
            "set_type": "diverse_geometry",
            "pert_id": j,
            "predicted_tau": tau,
            "scale_factor": scale_s,
            "alpha": alpha,
            **metrics
        })
        
    return pd.DataFrame(rows)


def evaluate_depth_evolution(model, arch_type, depths, eval_images, device):
    """
    Computes the full operator A_l across fragile, peak, and terminal depths.
    Tracks singular values, effective rank, null dimension, and condition number.
    """
    rows = []
    spectrum_rows = []
    
    clean_logits, clean_collected = forward_block_by_block(
        model, arch_type, x=eval_images, start_depth=0, collect_depths=tuple(depths)
    )
    
    for d in depths:
        h_clean = clean_collected[d]
        op = construct_full_operator(model, arch_type, d, h_clean, device)
        S = op["S"]
        D_out = op["D_out"]
        total_dim = op["N"] * op["D"]
        
        # Non-zero singular values
        sigma_1 = S[0].item()
        sigma_last = S[-1].item()
        
        # Effective rank: exp(entropy(p))
        s_prob = S / (S.sum() + 1e-12)
        s_prob_nz = s_prob[s_prob > 1e-12]
        entropy = -(s_prob_nz * torch.log(s_prob_nz)).sum().item()
        effective_rank = math.exp(entropy)
        
        # Null dimensions
        null_dim_exact = total_dim - D_out
        null_dim_1e2 = total_dim - (S / sigma_1 >= 1e-2).sum().item()
        null_dim_1e3 = total_dim - (S / sigma_1 >= 1e-3).sum().item()
        null_dim_1e4 = total_dim - (S / sigma_1 >= 1e-4).sum().item()
        
        anisotropy_ratio = sigma_1 / (S.mean().item() + 1e-12)
        
        rows.append({
            "arch": arch_type,
            "depth": d,
            "total_dim": total_dim,
            "D_out": D_out,
            "sigma_1": sigma_1,
            "sigma_mean": S.mean().item(),
            "sigma_min": sigma_last,
            "effective_rank": effective_rank,
            "anisotropy_ratio": anisotropy_ratio,
            "exact_null_dim": null_dim_exact,
            "null_dim_1e2": null_dim_1e2,
            "null_dim_1e3": null_dim_1e3,
            "null_dim_1e4": null_dim_1e4,
            "null_dim_pct": (null_dim_exact / total_dim) * 100.0
        })
        
        for k in range(len(S)):
            spectrum_rows.append({
                "arch": arch_type,
                "depth": d,
                "mode_index": k,
                "singular_value": S[k].item(),
                "normalized_singular_value": (S[k] / sigma_1).item()
            })
            
    return pd.DataFrame(rows), pd.DataFrame(spectrum_rows)


def evaluate_intervention_projections(op_dict, clean_acts, device):
    """
    Projects canonical historical interventions onto the full operator spectrum:
    - Centroid displacement
    - Diagonal/Independent Gaussian
    - Shared Gaussian
    - Coherent PC1
    - Random-Sign PC1
    - Checkerboard PC1
    - Coordinate permutation
    - Sign inversion
    """
    A_l = op_dict["A_l"]
    U = op_dict["U"]
    S = op_dict["S"]
    N = op_dict["N"]
    D = op_dict["D"]
    D_out = op_dict["D_out"]
    
    patches = clean_acts[:, 1:, :] # (B, N, D)
    patch_flat = patches.reshape(-1, D)
    mu = patch_flat.mean(dim=0)
    diff = patch_flat - mu.unsqueeze(0)
    cov = (diff.T @ diff) / (patch_flat.shape[0] - 1)
    evals_cov, evecs_cov = torch.linalg.eigh(cov)
    pc1 = evecs_cov[:, -1]
    
    # Subspace thresholds: high (top 20%), mid (next 40%), low/null (remaining)
    k_high = int(0.20 * D_out)
    k_mid = int(0.60 * D_out)
    
    def project(delta_P):
        x = delta_P.reshape(-1)
        norm_x = torch.norm(x)
        if norm_x < 1e-12:
            return 0.0, 0.0, 1.0, 0.0
        x = x / norm_x
        Ax = A_l @ x
        coords = (1.0 / S) * (U.T @ Ax)
        e_high = torch.sum(coords[:k_high]**2).item()
        e_mid = torch.sum(coords[k_high:k_mid]**2).item()
        e_null = max(0.0, 1.0 - (e_high + e_mid))
        tau = torch.norm(Ax).item()
        return e_high, e_mid, e_null, tau

    interventions = {}
    
    # 1. Global Coherent PC1
    a_ones = torch.ones(N, device=device) / math.sqrt(N)
    interventions["Global Coherent (PC1)"] = torch.outer(a_ones, pc1)
    
    # 2. Random-Sign PC1
    torch.manual_seed(42)
    a_rand = torch.randint(0, 2, (N,), device=device).float() * 2 - 1
    a_rand = a_rand / math.sqrt(N)
    interventions["Random-Sign (PC1)"] = torch.outer(a_rand, pc1)
    
    # 3. Checkerboard PC1
    grid_size = int(math.isqrt(N))
    check_grid = torch.ones(grid_size, grid_size, device=device)
    check_grid[::2, 1::2] = -1
    check_grid[1::2, ::2] = -1
    a_check = check_grid.reshape(-1)
    if len(a_check) < N:
        a_check = torch.cat([a_check, torch.ones(N - len(a_check), device=device)])
    elif len(a_check) > N:
        a_check = a_check[:N]
    a_check = a_check / torch.norm(a_check)
    interventions["Checkerboard (PC1)"] = torch.outer(a_check, pc1)
    
    # 4. Centroid displacement
    interventions["Centroid Displacement"] = mu.unsqueeze(0) - patches[0]
    
    # 5. Shared Gaussian
    g_shared = torch.randn(D, device=device)
    g_shared = g_shared / torch.norm(g_shared)
    interventions["Shared Gaussian"] = g_shared.unsqueeze(0).expand(N, D) / math.sqrt(N)
    
    # 6. Independent Gaussian
    r_ind = torch.randn(N, D, device=device)
    interventions["Independent Gaussian"] = r_ind / torch.norm(r_ind)
    
    # 7. Coordinate Permutation
    perm = torch.randperm(D)
    diff_perm = patches[0][:, perm] - patches[0]
    interventions["Coordinate Permutation"] = diff_perm / torch.norm(diff_perm)
    
    # 8. Sign Inversion
    diff_sign = -2.0 * patches[0]
    interventions["Sign Inversion"] = diff_sign / torch.norm(diff_sign)
    
    rows = []
    for name, dP in interventions.items():
        eh, em, en, tau = project(dP)
        rows.append({
            "intervention": name,
            "high_trans_energy_pct": eh * 100.0,
            "mid_trans_energy_pct": em * 100.0,
            "low_null_energy_pct": en * 100.0,
            "transmission_tau": tau
        })
        
    return pd.DataFrame(rows)


def evaluate_residual_and_scaling(model, arch_type, depth, clean_logits, clean_acts, modes_dict, device):
    """
    1) Decomposes where residual damage arises for exact linear null modes:
       - Block l CLS diff vs patch diff
       - Block l+1 CLS diff vs patch diff
       - Later blocks & terminal output
    2) Measures scaling exponent p (damage proportional to alpha^p) across fine alpha grid.
    """
    sigma_P = (clean_acts[:, 1:, :].reshape(-1, clean_acts.shape[-1]).var(dim=0, unbiased=False).sum()).sqrt().item()
    
    # Scaling test over small alphas: [0.01, 0.02, 0.04, 0.08, 0.16, 0.25]
    small_alphas = [0.01, 0.02, 0.04, 0.08, 0.16, 0.25]
    scaling_rows = []
    residual_rows = []
    
    Q_top = modes_dict["top_0"]
    Q_null = modes_dict["exact_null"]
    
    for mode_name, Q in [("top_mode", Q_top), ("exact_null_mode", Q_null)]:
        for s in small_alphas:
            alpha = s * sigma_P
            delta_P = alpha * Q
            h_pert = clean_acts.clone()
            h_pert[:, 1:, :] += delta_P.unsqueeze(0)
            
            # Forward block l
            if arch_type == "dinov2":
                blk_l = model.backbone.blocks[depth]
                blk_next = model.backbone.blocks[depth+1] if depth + 1 < len(model.backbone.blocks) else None
            else:
                blk_l = model.blocks[depth]
                blk_next = model.blocks[depth+1] if depth + 1 < len(model.blocks) else None
                
            out_l_pert = blk_l(h_pert)
            out_l_clean = blk_l(clean_acts)
            
            diff_l_cls = torch.norm(out_l_pert[:, 0, :] - out_l_clean[:, 0, :], dim=-1).mean().item()
            diff_l_patch = torch.norm(out_l_pert[:, 1:, :] - out_l_clean[:, 1:, :], dim=-1).mean().item()
            
            if blk_next is not None:
                out_next_pert = blk_next(out_l_pert)
                out_next_clean = blk_next(out_l_clean)
                diff_next_cls = torch.norm(out_next_pert[:, 0, :] - out_next_clean[:, 0, :], dim=-1).mean().item()
                diff_next_patch = torch.norm(out_next_pert[:, 1:, :] - out_next_clean[:, 1:, :], dim=-1).mean().item()
            else:
                diff_next_cls = diff_l_cls
                diff_next_patch = diff_l_patch
                
            # Full model output
            pert_logits, _ = forward_block_by_block(model, arch_type, h_start=h_pert, start_depth=depth)
            diff_logits = pert_logits - clean_logits
            logit_l2 = torch.norm(diff_logits, dim=-1).mean().item()
            
            p_clean = F.softmax(clean_logits, dim=-1)
            log_p_pert = F.log_softmax(pert_logits, dim=-1)
            kl_div = F.kl_div(log_p_pert, p_clean, reduction='batchmean').item()
            
            scaling_rows.append({
                "mode_name": mode_name,
                "scale_factor": s,
                "alpha": alpha,
                "logit_l2": logit_l2,
                "kl_div": kl_div,
                "block_l_cls_diff": diff_l_cls,
                "block_l_patch_diff": diff_l_patch,
                "block_next_cls_diff": diff_next_cls
            })
            
            residual_rows.append({
                "mode_name": mode_name,
                "scale_factor": s,
                "block_l_cls_diff": diff_l_cls,
                "block_l_patch_diff": diff_l_patch,
                "block_next_cls_diff": diff_next_cls,
                "block_next_patch_diff": diff_next_patch,
                "terminal_logit_l2": logit_l2
            })
            
    df_scaling = pd.DataFrame(scaling_rows)
    df_residual = pd.DataFrame(residual_rows)
    
    # Compute empirical power-law exponent p = d(ln y) / d(ln x)
    exponent_summary = []
    for mode in ["top_mode", "exact_null_mode"]:
        sub = df_scaling[df_scaling["mode_name"] == mode]
        log_x = np.log(sub["scale_factor"].values)
        
        for metric in ["logit_l2", "kl_div", "block_l_cls_diff"]:
            log_y = np.log(np.maximum(sub[metric].values, 1e-12))
            # Linear regression: log_y = p * log_x + c
            poly = np.polyfit(log_x, log_y, 1)
            exponent_summary.append({
                "mode_name": mode,
                "metric": metric,
                "scaling_exponent_p": poly[0],
                "intercept_c": poly[1]
            })
            
    return df_residual, df_scaling, pd.DataFrame(exponent_summary)


def generate_full_operator_figures(output_dir, fig_dir):
    """
    Generates Figures A through H from committed CSV outputs.
    """
    os.makedirs(fig_dir, exist_ok=True)
    
    df_spec = pd.read_csv(os.path.join(output_dir, "operator_spectrum.csv"))
    df_curves = pd.read_csv(os.path.join(output_dir, "finite_radius_curves.csv"))
    df_heldout = pd.read_csv(os.path.join(output_dir, "random_perturbation_prediction.csv"))
    df_factor = pd.read_csv(os.path.join(output_dir, "mode_factorization.csv"))
    df_depth = pd.read_csv(os.path.join(output_dir, "depth_evolution.csv"))
    df_proj = pd.read_csv(os.path.join(output_dir, "intervention_projection.csv"))
    df_scaling = pd.read_csv(os.path.join(output_dir, "null_scaling.csv"))
    df_rep = pd.read_csv(os.path.join(output_dir, "replication_summary.csv"))
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------
    # Figure A: Full Operator Spectrum
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for arch in df_spec["arch"].unique():
        sub = df_spec[(df_spec["arch"] == arch) & (df_spec["depth"] == 8)]
        ax.plot(sub["mode_index"], sub["singular_value"], label=f"{arch} (depth 8)", lw=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Singular Mode Index k", fontsize=11, fontweight="bold")
    ax.set_ylabel("Singular Value \u03c3_k (log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Figure A: Full Token \u00d7 Feature Operator Spectrum (A_l)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_a_full_operator_spectrum.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure B: Top vs Null Modes Finite Radius Sweep
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_c = df_curves[df_curves["arch"] == "deit_small"]
    for mode in ["top_0", "mid", "bot_nonzero", "exact_null", "random_gaussian"]:
        sub_m = sub_c[sub_c["mode_name"] == mode]
        ax.plot(sub_m["scale_factor"], sub_m["logit_l2"], marker='o', label=mode, lw=1.8)
    ax.set_xlabel("Perturbation Scale (\u03b1 / \u03c3_P)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Full-Model Logit L2 Damage", fontsize=11, fontweight="bold")
    ax.set_title("Figure B: Top vs Null Modes across Finite Radius Sweep", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_b_top_vs_null_modes.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure C: Predicted Transmission vs Observed Damage
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_div = df_heldout[df_heldout["set_type"] == "diverse_geometry"]
    r_val, _ = pearsonr(sub_div["predicted_tau"], sub_div["logit_l2"])
    rho_val, _ = spearmanr(sub_div["predicted_tau"], sub_div["logit_l2"])
    
    ax.scatter(sub_div["predicted_tau"], sub_div["logit_l2"], alpha=0.75, c="#1f77b4", edgecolors='none', s=45)
    m, b = np.polyfit(sub_div["predicted_tau"], sub_div["logit_l2"], 1)
    x_grid = np.linspace(sub_div["predicted_tau"].min(), sub_div["predicted_tau"].max(), 50)
    ax.plot(x_grid, m * x_grid + b, color="red", lw=2, label=f"Fit (r = {r_val:.3f}, \u03c1 = {rho_val:.3f})")
    ax.set_xlabel("Predicted Linear Transmission \u03c4_j = ||A_l vec(\u0394P_j^T)||", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed Full-Model Logit L2 Damage", fontsize=11, fontweight="bold")
    ax.set_title("Figure C: Predicted vs Observed Damage on Held-Out Perturbations", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_c_predicted_vs_observed.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure D: Mode Factorization (Rank-1 vs Entangled)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(df_factor["mode_index"], df_factor["rank1_energy_fraction"], marker='o', color="#2ca02c", lw=2, label="Rank-1 Energy Fraction")
    ax.plot(df_factor["mode_index"], df_factor["rank2_energy_fraction"], marker='s', color="#1f77b4", lw=1.5, ls="--", label="Rank-2 Energy Fraction")
    ax.set_xlabel("Singular Mode Index k", fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy Fraction Explained", fontsize=11, fontweight="bold")
    ax.set_title("Figure D: Singular Mode Separability (Rank-1 Factorization)", fontsize=12, fontweight="bold")
    ax.set_ylim([0.3, 1.05])
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_d_mode_factorization.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure E: Depth Evolution of Spectrum
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_deit = df_depth[df_depth["arch"] == "deit_small"]
    ax.plot(sub_deit["depth"], sub_deit["effective_rank"], marker='o', lw=2, color="#d62728", label="Effective Rank")
    ax.set_xlabel("Model Depth l", fontsize=11, fontweight="bold")
    ax.set_ylabel("Effective Rank", fontsize=11, fontweight="bold", color="#d62728")
    ax.tick_params(axis='y', labelcolor="#d62728")
    
    ax2 = ax.twinx()
    ax2.plot(sub_deit["depth"], sub_deit["anisotropy_ratio"], marker='s', lw=2, color="#1f77b4", label="Anisotropy Ratio (\u03c3_1 / \u03c3_mean)")
    ax2.set_ylabel("Anisotropy Ratio", fontsize=11, fontweight="bold", color="#1f77b4")
    ax2.tick_params(axis='y', labelcolor="#1f77b4")
    
    ax.set_title("Figure E: Depth Evolution of Transmission Spectrum (DeiT-Small)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_e_depth_evolution.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure F: Historical Intervention Projection
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(df_proj))
    width = 0.55
    ax.bar(x, df_proj["high_trans_energy_pct"], width, label="High-Transmission Subspace", color="#d62728")
    ax.bar(x, df_proj["mid_trans_energy_pct"], width, bottom=df_proj["high_trans_energy_pct"], label="Mid-Transmission Subspace", color="#ff7f0e")
    ax.bar(x, df_proj["low_null_energy_pct"], width, bottom=df_proj["high_trans_energy_pct"] + df_proj["mid_trans_energy_pct"], label="Low/Null Subspace", color="#2ca02c")
    
    ax.set_ylabel("Spectral Energy Allocation (%)", fontsize=11, fontweight="bold")
    ax.set_title("Figure F: Projection of Canonical Interventions onto Operator Spectrum", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_proj["intervention"], rotation=30, ha="right", fontsize=9)
    ax.set_ylim([0, 105])
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_f_intervention_projection.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure G: Null Mode Scaling Exponent
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for mode in ["top_mode", "exact_null_mode"]:
        sub = df_scaling[df_scaling["mode_name"] == mode]
        ax.plot(sub["scale_factor"], sub["logit_l2"], marker='o', lw=2, label=f"{mode} (L2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Perturbation Scale (\u03b1 / \u03c3_P) [log scale]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Logit L2 Damage [log scale]", fontsize=11, fontweight="bold")
    ax.set_title("Figure G: Power-Law Scaling of Damage (Damage ~ alpha^p)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_g_null_mode_scaling.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure H: Cross-Architecture Replication
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    x = np.arange(len(df_rep))
    width = 0.35
    ax.bar(x - width/2, df_rep["top_damage_s04"], width, label="Top Mode Damage (s=0.4)", color="#d62728")
    ax.bar(x + width/2, df_rep["null_damage_s04"], width, label="Exact Null Mode Damage (s=0.4)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(df_rep["arch"], fontsize=10, fontweight="bold")
    ax.set_ylabel("Logit L2 Damage at s = 0.4", fontsize=11, fontweight="bold")
    ax.set_title("Figure H: Cross-Architecture Replication of Full Operator Hierarchy", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_h_cross_architecture_replication.png"))
    plt.close(fig)
    print("All 8 figures successfully generated!")
