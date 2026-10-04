"""
Multi-Block Operator Module: Downstream-Persistent Low-Transmission Geometry

Constructs and evaluates:
1. Exact clean-state downstream Jacobian J_{l->L} in R^{D_readout x (ND)}
2. Multi-block right singular modes (Q_multi_top, Q_multi_null)
3. Single-block vs multi-block nullity and survival analysis
4. Null-space rotation across downstream blocks (principal angles)
5. Block-by-block leakage tracing
6. Finite-radius sweeps and power-law scaling analysis
7. Out-of-sample prediction comparison (single-block tau vs multi-block tau)
"""

import os
import sys
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
from patch_fungibility.full_fungibility_operator import construct_full_operator


def construct_downstream_jacobian(model, arch_type, start_depth, clean_acts, device):
    """
    Constructs the exact clean-state downstream Jacobian J_{l->L}:
        J_{l->L} = d(z_final) / d(vec(P_l^T)) in R^{D_readout x (ND)}.
        
    Supports:
        - DeiT-Small, DeiT-Tiny, ViT-Base (CLS readout, D_readout = D)
        - DINOv2 (Dual CLS + mean patch pooling, D_readout = 2D)
    """
    B, total_tokens, D = clean_acts.shape
    N = total_tokens - 1
    total_dim = N * D
    
    patches = clean_acts[:, 1:, :].reshape(-1, D)
    sigma_P = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()
    
    # Define frozen downstream function mapping h_start to z_final
    if arch_type == "dinov2":
        backbone = model.backbone
        D_readout = 2 * D
        
        def forward_downstream(h_in):
            h = h_in
            for b_idx in range(start_depth, len(backbone.blocks)):
                h = backbone.blocks[b_idx](h)
            h_norm = backbone.norm(h)
            z_cls = h_norm[:, 0, :]
            z_patch_mean = h_norm[:, 1:, :].mean(dim=1)
            return torch.cat([z_cls, z_patch_mean], dim=-1) # (B, 2D)
    else:
        D_readout = D
        
        def forward_downstream(h_in):
            h = h_in
            for b_idx in range(start_depth, len(model.blocks)):
                h = model.blocks[b_idx](h)
            h_norm = model.norm(h)
            return h_norm[:, 0, :] # (B, D)

    # Use first image (or mean across batch) for clean linearization point
    h_var = clean_acts[:1].clone().detach().requires_grad_(True)
    z_clean = forward_downstream(h_var)
    
    # Compute Jacobian rows via VJPs with unit vectors
    rows = []
    chunk_size = 32
    for d in range(0, D_readout, chunk_size):
        end_d = min(d + chunk_size, D_readout)
        for k in range(d, end_d):
            grad_out = torch.zeros_like(z_clean)
            grad_out[0, k] = 1.0
            is_last = (k == D_readout - 1)
            g = torch.autograd.grad(z_clean, h_var, grad_outputs=grad_out, retain_graph=not is_last)[0]
            # Extract patch gradient: shape (1, N, D) -> (ND,)
            rows.append(g[0, 1:, :].reshape(-1))
            
    J = torch.stack(rows, dim=0) # (D_readout, ND)
    
    # Exact SVD of J via Gram matrix
    Gram_J = J @ J.T # (D_readout, D_readout)
    evals_J, U_J = torch.linalg.eigh(Gram_J)
    idx_J = torch.argsort(evals_J, descending=True)
    evals_J = evals_J[idx_J]
    U_J = U_J[:, idx_J]
    S_J = torch.sqrt(torch.clamp(evals_J, min=0.0))
    
    return {
        "J": J,
        "Gram_J": Gram_J,
        "evals_J": evals_J,
        "U_J": U_J,
        "S_J": S_J,
        "N": N,
        "D": D,
        "D_readout": D_readout,
        "total_dim": total_dim,
        "sigma_P": sigma_P,
        "arch_type": arch_type,
        "start_depth": start_depth
    }


def extract_multiblock_modes(op_multi, op_single, seed=42):
    """
    Extracts representative modes from multi-block Jacobian J and single-block operator A_l:
    - multi_top: top singular mode of J
    - multi_null: exact linear null mode of J
    - single_top: top singular mode of A_l
    - single_null: exact linear null mode of A_l
    - random_gaussian: isotropic random control
    """
    J = op_multi["J"]
    U_J = op_multi["U_J"]
    S_J = op_multi["S_J"]
    N = op_multi["N"]
    D = op_multi["D"]
    device = J.device
    
    modes = {}
    
    # 1. Multi-block top mode
    v_J_top = (1.0 / S_J[0].item()) * (J.T @ U_J[:, 0])
    v_J_top = v_J_top / torch.norm(v_J_top)
    modes["multi_top"] = v_J_top.view(N, D)
    
    # 2. Multi-block exact null mode
    torch.manual_seed(seed)
    r = torch.randn(N * D, device=device)
    inv_evals_J = torch.where(op_multi["evals_J"] > 1e-10, 1.0 / op_multi["evals_J"], torch.zeros_like(op_multi["evals_J"]))
    Ar_J = J @ r
    inv_G_Ar_J = U_J @ (inv_evals_J * (U_J.T @ Ar_J))
    r_null_J = r - (J.T @ inv_G_Ar_J)
    r_null_J = r_null_J / torch.norm(r_null_J)
    modes["multi_null"] = r_null_J.view(N, D)
    
    # 3. Single-block top mode
    A_l = op_single["A_l"]
    U_A = op_single["U"]
    S_A = op_single["S"]
    v_A_top = (1.0 / S_A[0].item()) * (A_l.T @ U_A[:, 0])
    v_A_top = v_A_top / torch.norm(v_A_top)
    modes["single_top"] = v_A_top.view(N, D)
    
    # 4. Single-block exact null mode
    torch.manual_seed(seed)
    r_A = torch.randn(N * D, device=device)
    inv_evals_A = torch.where(op_single["evals"] > 1e-10, 1.0 / op_single["evals"], torch.zeros_like(op_single["evals"]))
    Ar_A = A_l @ r_A
    inv_G_Ar_A = U_A @ (inv_evals_A * (U_A.T @ Ar_A))
    r_null_A = r_A - (A_l.T @ inv_G_Ar_A)
    r_null_A = r_null_A / torch.norm(r_null_A)
    modes["single_null"] = r_null_A.view(N, D)
    
    # 5. Isotropic random control
    r_iso = torch.randn(N, D, device=device)
    modes["random_gaussian"] = r_iso / torch.norm(r_iso)
    
    # Verify unit Frobenius norms
    for name, m in modes.items():
        assert abs(torch.norm(m).item() - 1.0) < 1e-4
        
    return modes


def evaluate_nullspace_rotation(model, arch_type, start_depth, clean_collected, device):
    """
    Measures the rotation of row spaces / null spaces between adjacent downstream blocks:
    For blocks b in [start_depth, L-1]:
        Computes principal cosines, principal angles, mean overlap, and min overlap between
        RowSpace(A_b) and RowSpace(A_{b+1}).
    """
    if arch_type == "dinov2":
        num_blocks = len(model.backbone.blocks)
    else:
        num_blocks = len(model.blocks)
        
    A_dict = {}
    U_dict = {}
    S_dict = {}
    
    for b in range(start_depth, num_blocks):
        op_b = construct_full_operator(model, arch_type, b, clean_collected[b], device)
        A_dict[b] = op_b["A_l"]
        U_dict[b] = op_b["U"]
        S_dict[b] = op_b["S"]
        
    rows = []
    for b in range(start_depth, num_blocks - 1):
        mask_b = S_dict[b] > 1e-4
        mask_next = S_dict[b+1] > 1e-4
        
        V_b = (A_dict[b].T @ U_dict[b][:, mask_b]) / S_dict[b][mask_b] # (ND, K_b)
        V_next = (A_dict[b+1].T @ U_dict[b+1][:, mask_next]) / S_dict[b+1][mask_next] # (ND, K_next)
        
        # Overlap matrix: M = V_b.T @ V_next of shape (K_b, K_next)
        M = V_b.T @ V_next
        s_overlap = torch.linalg.svdvals(M).cpu().numpy()
        
        top_cos = float(s_overlap[0])
        mean_cos = float(np.mean(s_overlap))
        min_cos = float(s_overlap[-1])
        top_angle_deg = float(np.arccos(np.clip(top_cos, 0.0, 1.0)) * 180.0 / np.pi)
        mean_angle_deg = float(np.arccos(np.clip(mean_cos, 0.0, 1.0)) * 180.0 / np.pi)
        
        rows.append({
            "arch": arch_type,
            "block_b": b,
            "block_b_plus_1": b + 1,
            "top_principal_cosine": top_cos,
            "mean_principal_cosine": mean_cos,
            "min_principal_cosine": min_cos,
            "top_principal_angle_deg": top_angle_deg,
            "mean_principal_angle_deg": mean_angle_deg
        })
        
    return pd.DataFrame(rows)


def evaluate_downstream_leakage_trace(model, arch_type, start_depth, clean_collected, modes_dict, device):
    """
    Traces perturbations block-by-block from start_depth to terminal layer:
    Records at each block:
    - Patch perturbation Frobenius norm: ||Delta P_b||_F
    - Readout transmission: ||A_b vec(Delta P_b^T)||
    - Readout token disturbance: ||Delta z_b||
    - Cumulative visibility score
    """
    if arch_type == "dinov2":
        blocks = model.backbone.blocks
    else:
        blocks = model.blocks
        
    num_blocks = len(blocks)
    alpha = 0.4 * (clean_collected[start_depth][:, 1:, :].reshape(-1, clean_collected[start_depth].shape[-1]).var(dim=0, unbiased=False).sum()).sqrt().item()
    
    rows = []
    
    for mode_name, Q in modes_dict.items():
        delta_P = alpha * Q
        h_pert = clean_collected[start_depth].clone()
        h_pert[:, 1:, :] += delta_P.unsqueeze(0)
        
        cum_vis = 0.0
        
        for b in range(start_depth, num_blocks):
            h_clean_b = clean_collected[b]
            op_b = construct_full_operator(model, arch_type, b, h_clean_b, device)
            A_b = op_b["A_l"]
            
            # Current patch perturbation at block b
            delta_P_b = (h_pert[:, 1:, :] - h_clean_b[:, 1:, :]).mean(dim=0) # (N, D)
            norm_P_b = torch.norm(delta_P_b).item()
            
            # Linear readout transmission through A_b
            tau_b = torch.norm(A_b @ delta_P_b.reshape(-1)).item()
            cum_vis += (tau_b ** 2)
            
            # Forward block b
            h_pert = blocks[b](h_pert)
            h_clean_next = blocks[b](h_clean_b)
            
            # Readout disturbance at output of block b
            cls_dist = torch.norm(h_pert[:, 0, :] - h_clean_next[:, 0, :], dim=-1).mean().item()
            patch_dist = torch.norm(h_pert[:, 1:, :] - h_clean_next[:, 1:, :], dim=-1).mean().item()
            
            rows.append({
                "mode_name": mode_name,
                "block": b,
                "patch_norm": norm_P_b,
                "transmission_tau_b": tau_b,
                "cls_disturbance": cls_dist,
                "patch_disturbance": patch_dist,
                "cumulative_visibility": cum_vis
            })
            
    return pd.DataFrame(rows)


def run_full_model_eval(model, arch_type, depth, clean_logits, clean_acts, delta_P, eval_targets, device):
    """
    Evaluates downstream model damage with patch perturbation delta_P.
    """
    B, total_tokens, D = clean_acts.shape
    h_pert = clean_acts.clone()
    h_pert[:, 1:, :] += delta_P.unsqueeze(0)
    
    pert_logits, _ = forward_block_by_block(
        model, arch_type, h_start=h_pert, start_depth=depth
    )
    
    diff = pert_logits - clean_logits
    logit_l2 = torch.norm(diff, dim=-1).mean().item()
    
    p_clean = F.softmax(clean_logits, dim=-1)
    log_p_pert = F.log_softmax(pert_logits, dim=-1)
    kl_div = F.kl_div(log_p_pert, p_clean, reduction='batchmean').item()
    
    clean_top1 = clean_logits.argmax(dim=-1)
    pert_top1 = pert_logits.argmax(dim=-1)
    top1_flip_rate = (clean_top1 != pert_top1).float().mean().item()
    top1_acc = (pert_top1 == eval_targets).float().mean().item()
    
    clean_target_logit = clean_logits.gather(1, eval_targets.unsqueeze(1)).squeeze(1)
    pert_target_logit = pert_logits.gather(1, eval_targets.unsqueeze(1)).squeeze(1)
    margin_damage = (clean_target_logit - pert_target_logit).mean().item()
    
    return {
        "logit_l2": logit_l2,
        "kl_div": kl_div,
        "margin_damage": margin_damage,
        "top1_acc": top1_acc,
        "top1_flip_rate": top1_flip_rate
    }


def evaluate_finite_radius_curves(model, arch_type, start_depth, clean_logits, clean_acts, modes_dict, eval_targets, sigma_P, device):
    """
    Evaluates downstream unconstrained model damage across scale sweep s in [0.0, 4.0].
    """
    scale_sweep = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0]
    rows = []
    
    for mode_name, Q in modes_dict.items():
        for s in scale_sweep:
            alpha = s * sigma_P
            delta_P = alpha * Q
            
            if s == 0.0:
                metrics = {
                    "logit_l2": 0.0,
                    "kl_div": 0.0,
                    "margin_damage": 0.0,
                    "top1_acc": (clean_logits.argmax(dim=-1) == eval_targets).float().mean().item(),
                    "top1_flip_rate": 0.0
                }
            else:
                metrics = run_full_model_eval(
                    model, arch_type, start_depth, clean_logits, clean_acts, delta_P, eval_targets, device
                )
                
            rows.append({
                "arch": arch_type,
                "depth": start_depth,
                "mode_name": mode_name,
                "scale_factor": s,
                "alpha": alpha,
                **metrics
            })
            
    return pd.DataFrame(rows)


def evaluate_heldout_prediction(model, arch_type, start_depth, clean_logits, clean_acts, op_multi, op_single, eval_targets, device, M=100, seed=42):
    """
    Evaluates M held-out perturbations:
    Compares:
        tau_single = ||A_l vec(Delta P^T)||
        tau_multi = ||J_{l->L} vec(Delta P^T)||
    against observed full-model damage.
    """
    J = op_multi["J"]
    A_l = op_single["A_l"]
    U_J = op_multi["U_J"]
    S_J = op_multi["S_J"]
    evals_J = op_multi["evals_J"]
    N = op_multi["N"]
    D = op_multi["D"]
    sigma_P = op_multi["sigma_P"]
    
    scale_s = 0.4
    alpha = scale_s * sigma_P
    inv_evals_J = torch.where(evals_J > 1e-10, 1.0 / evals_J, torch.zeros_like(evals_J))
    
    rows = []
    
    # Sample diverse set: 25 top linear combs, 25 mid combs, 25 null, 25 isotropic
    torch.manual_seed(seed)
    pert_list = []
    for _ in range(25):
        coeffs = torch.randn(min(50, len(S_J)), device=device)
        coeffs = coeffs / torch.norm(coeffs)
        u_comb = U_J[:, :len(coeffs)] @ (coeffs / S_J[:len(coeffs)])
        v_comb = J.T @ u_comb
        pert_list.append(v_comb / torch.norm(v_comb))
        
    mid_start = len(S_J) // 4
    mid_end = 3 * len(S_J) // 4
    mid_len = mid_end - mid_start
    for _ in range(25):
        coeffs = torch.randn(mid_len, device=device)
        coeffs = coeffs / torch.norm(coeffs)
        u_comb = U_J[:, mid_start:mid_end] @ (coeffs / S_J[mid_start:mid_end])
        v_comb = J.T @ u_comb
        pert_list.append(v_comb / torch.norm(v_comb))
        
    for _ in range(25):
        r = torch.randn(N * D, device=device)
        Ar_J = J @ r
        inv_G_Ar_J = U_J @ (inv_evals_J * (U_J.T @ Ar_J))
        r_null_J = r - (J.T @ inv_G_Ar_J)
        pert_list.append(r_null_J / torch.norm(r_null_J))
        
    for _ in range(25):
        r = torch.randn(N * D, device=device)
        pert_list.append(r / torch.norm(r))
        
    for j, v_flat in enumerate(pert_list):
        R = v_flat.view(N, D)
        tau_multi = torch.norm(J @ v_flat).item()
        tau_single = torch.norm(A_l @ v_flat).item()
        
        delta_P = alpha * R
        metrics = run_full_model_eval(
            model, arch_type, start_depth, clean_logits, clean_acts, delta_P, eval_targets, device
        )
        
        rows.append({
            "pert_id": j,
            "predicted_tau_multi": tau_multi,
            "predicted_tau_single": tau_single,
            "scale_factor": scale_s,
            "alpha": alpha,
            **metrics
        })
        
    return pd.DataFrame(rows)


def evaluate_scaling_exponents(model, arch_type, start_depth, clean_logits, clean_acts, modes_dict, device):
    """
    Measures scaling exponent p (Damage proportional to alpha^p) for single-block null vs multi-block null modes.
    """
    sigma_P = (clean_acts[:, 1:, :].reshape(-1, clean_acts.shape[-1]).var(dim=0, unbiased=False).sum()).sqrt().item()
    small_alphas = [0.01, 0.02, 0.04, 0.08, 0.16, 0.25]
    rows = []
    
    for mode_name in ["multi_null", "single_null", "multi_top", "single_top"]:
        Q = modes_dict[mode_name]
        for s in small_alphas:
            alpha = s * sigma_P
            delta_P = alpha * Q
            h_pert = clean_acts.clone()
            h_pert[:, 1:, :] += delta_P.unsqueeze(0)
            
            pert_logits, _ = forward_block_by_block(model, arch_type, h_start=h_pert, start_depth=start_depth)
            diff = pert_logits - clean_logits
            logit_l2 = torch.norm(diff, dim=-1).mean().item()
            
            p_clean = F.softmax(clean_logits, dim=-1)
            log_p_pert = F.log_softmax(pert_logits, dim=-1)
            kl_div = F.kl_div(log_p_pert, p_clean, reduction='batchmean').item()
            
            rows.append({
                "mode_name": mode_name,
                "scale_factor": s,
                "alpha": alpha,
                "logit_l2": logit_l2,
                "kl_div": kl_div
            })
            
    df_raw = pd.DataFrame(rows)
    
    # Fit power laws
    summary = []
    for mode in df_raw["mode_name"].unique():
        sub = df_raw[df_raw["mode_name"] == mode]
        log_x = np.log(sub["scale_factor"].values)
        
        for metric in ["logit_l2", "kl_div"]:
            log_y = np.log(np.maximum(sub[metric].values, 1e-12))
            poly = np.polyfit(log_x, log_y, 1)
            summary.append({
                "mode_name": mode,
                "metric": metric,
                "scaling_exponent_p": float(poly[0]),
                "intercept_c": float(poly[1])
            })
            
    return pd.DataFrame(summary), df_raw


def evaluate_historical_interventions(op_multi, op_single, clean_acts, device):
    """
    Projects canonical interventions onto multi-block Jacobian spectrum vs single-block operator spectrum.
    """
    J = op_multi["J"]
    U_J = op_multi["U_J"]
    S_J = op_multi["S_J"]
    A_l = op_single["A_l"]
    U_A = op_single["U"]
    S_A = op_single["S"]
    N = op_multi["N"]
    D = op_multi["D"]
    D_readout = op_multi["D_readout"]
    
    patches = clean_acts[:, 1:, :]
    patch_flat = patches.reshape(-1, D)
    mu = patch_flat.mean(dim=0)
    diff = patch_flat - mu.unsqueeze(0)
    cov = (diff.T @ diff) / (patch_flat.shape[0] - 1)
    evals_cov, evecs_cov = torch.linalg.eigh(cov)
    pc1 = evecs_cov[:, -1]
    
    interventions = {}
    a_ones = torch.ones(N, device=device) / math.sqrt(N)
    interventions["Global Coherent (PC1)"] = torch.outer(a_ones, pc1)
    
    torch.manual_seed(42)
    a_rand = torch.randint(0, 2, (N,), device=device).float() * 2 - 1
    a_rand = a_rand / math.sqrt(N)
    interventions["Random-Sign (PC1)"] = torch.outer(a_rand, pc1)
    
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
    
    interventions["Centroid Displacement"] = mu.unsqueeze(0) - patches[0]
    
    g_shared = torch.randn(D, device=device)
    interventions["Shared Gaussian"] = g_shared.unsqueeze(0).expand(N, D) / math.sqrt(N)
    interventions["Independent Gaussian"] = torch.randn(N, D, device=device)
    
    perm = torch.randperm(D)
    diff_perm = patches[0][:, perm] - patches[0]
    interventions["Coordinate Permutation"] = diff_perm / torch.norm(diff_perm)
    interventions["Sign Inversion"] = -2.0 * patches[0]
    
    rows = []
    k_high_J = int(0.20 * D_readout)
    k_high_A = int(0.20 * op_single["D_out"])
    
    for name, dP in interventions.items():
        x = dP.reshape(-1)
        norm_x = torch.norm(x)
        if norm_x < 1e-12:
            continue
        x = x / norm_x
        
        # Multi-block J
        Jx = J @ x
        tau_multi = torch.norm(Jx).item()
        coords_J = (1.0 / S_J) * (U_J.T @ Jx)
        e_high_multi = torch.sum(coords_J[:k_high_J]**2).item() * 100.0
        e_null_multi = max(0.0, 1.0 - torch.sum(coords_J**2).item()) * 100.0
        
        # Single-block A_l
        Ax = A_l @ x
        tau_single = torch.norm(Ax).item()
        coords_A = (1.0 / S_A) * (U_A.T @ Ax)
        e_high_single = torch.sum(coords_A[:k_high_A]**2).item() * 100.0
        e_null_single = max(0.0, 1.0 - torch.sum(coords_A**2).item()) * 100.0
        
        rows.append({
            "intervention": name,
            "tau_multi": tau_multi,
            "tau_single": tau_single,
            "multi_high_energy_pct": e_high_multi,
            "multi_null_energy_pct": e_null_multi,
            "single_high_energy_pct": e_high_single,
            "single_null_energy_pct": e_null_single
        })
        
    return pd.DataFrame(rows)


def evaluate_depth_comparison(model, arch_type, depths, eval_images, device):
    """
    Compares multi-block Jacobian properties across fragile (2), peak (8), and terminal (11) depths.
    """
    rows = []
    spectrum_rows = []
    
    clean_logits, clean_collected = forward_block_by_block(
        model, arch_type, x=eval_images, start_depth=0, collect_depths=tuple(depths)
    )
    
    for d in depths:
        h_clean = clean_collected[d]
        op_m = construct_downstream_jacobian(model, arch_type, d, h_clean, device)
        S_J = op_m["S_J"]
        total_dim = op_m["total_dim"]
        D_readout = op_m["D_readout"]
        
        sigma_1 = S_J[0].item()
        sigma_mean = S_J.mean().item()
        
        s_prob = S_J / (S_J.sum() + 1e-12)
        s_prob_nz = s_prob[s_prob > 1e-12]
        effective_rank = math.exp(-(s_prob_nz * torch.log(s_prob_nz)).sum().item())
        
        exact_null_dim = total_dim - D_readout
        null_dim_1e2 = total_dim - (S_J / sigma_1 >= 1e-2).sum().item()
        null_dim_1e3 = total_dim - (S_J / sigma_1 >= 1e-3).sum().item()
        
        rows.append({
            "arch": arch_type,
            "depth": d,
            "remaining_blocks": len(model.blocks if arch_type != "dinov2" else model.backbone.blocks) - d,
            "sigma_1": sigma_1,
            "sigma_mean": sigma_mean,
            "effective_rank": effective_rank,
            "anisotropy_ratio": sigma_1 / (sigma_mean + 1e-12),
            "exact_null_dim": exact_null_dim,
            "null_dim_1e2": null_dim_1e2,
            "null_dim_1e3": null_dim_1e3,
            "null_dim_pct": (exact_null_dim / total_dim) * 100.0
        })
        
        for k in range(len(S_J)):
            spectrum_rows.append({
                "arch": arch_type,
                "depth": d,
                "mode_index": k,
                "singular_value": S_J[k].item(),
                "normalized_singular_value": (S_J[k] / sigma_1).item()
            })
            
    return pd.DataFrame(rows), pd.DataFrame(spectrum_rows)


def generate_multiblock_figures(output_dir, fig_dir):
    """
    Generates Figures A through H cleanly from committed CSV files.
    """
    os.makedirs(fig_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    df_spec = pd.read_csv(os.path.join(output_dir, "multiblock_spectrum.csv"))
    df_curves = pd.read_csv(os.path.join(output_dir, "finite_radius_curves.csv"))
    df_pred = pd.read_csv(os.path.join(output_dir, "random_prediction.csv"))
    df_rot = pd.read_csv(os.path.join(output_dir, "nullspace_rotation.csv"))
    df_leak = pd.read_csv(os.path.join(output_dir, "leakage_trace.csv"))
    df_depth = pd.read_csv(os.path.join(output_dir, "depth_comparison.csv"))
    df_rep = pd.read_csv(os.path.join(output_dir, "replication_summary.csv"))
    
    # -------------------------------------------------------------
    # Figure A: Single-Block vs Multi-Block Spectrum
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_spec = df_spec[(df_spec["arch"] == "deit_small") & (df_spec["depth"] == 8)]
    ax.plot(sub_spec["mode_index"], sub_spec["singular_value"], label="Multi-Block Jacobian J_{8->12}", color="#1f77b4", lw=2.2)
    ax.set_yscale("log")
    ax.set_xlabel("Singular Mode Index k", fontsize=11, fontweight="bold")
    ax.set_ylabel("Singular Value sigma_k (log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Figure A: Multi-Block Downstream Spectrum (DeiT-Small, Depth 8)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_a_single_vs_multiblock_spectrum.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure B: Single-Block Null vs Multi-Block Null Finite Radius
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_c = df_curves[df_curves["arch"] == "deit_small"]
    for mode, col, ls in [("multi_top", "#d62728", "-"),
                          ("single_top", "#ff7f0e", "--"),
                          ("single_null", "#9467bd", "-."),
                          ("multi_null", "#2ca02c", "-"),
                          ("random_gaussian", "#7f7f7f", ":")]:
        sub_m = sub_c[sub_c["mode_name"] == mode]
        ax.plot(sub_m["scale_factor"], sub_m["logit_l2"], marker='o', label=mode, color=col, ls=ls, lw=1.8)
    ax.set_xlabel("Perturbation Scale (alpha / sigma_P)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Full-Model Logit L2 Damage", fontsize=11, fontweight="bold")
    ax.set_title("Figure B: Single-Block Null vs Multi-Block Null across Finite Radius", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_b_single_null_vs_multi_null.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure C: Predicted vs Observed Damage (Multi vs Single)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    r_multi, _ = pearsonr(df_pred["predicted_tau_multi"], df_pred["logit_l2"])
    r_single, _ = pearsonr(df_pred["predicted_tau_single"], df_pred["logit_l2"])
    
    ax1.scatter(df_pred["predicted_tau_single"], df_pred["logit_l2"], alpha=0.7, c="#ff7f0e", s=40)
    ax1.set_xlabel("Single-Block Transmission tau_single", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Observed Logit L2 Damage", fontsize=11, fontweight="bold")
    ax1.set_title(f"Single-Block Predictor (r = {r_single:.3f})", fontsize=11, fontweight="bold")
    
    ax2.scatter(df_pred["predicted_tau_multi"], df_pred["logit_l2"], alpha=0.7, c="#1f77b4", s=40)
    ax2.set_xlabel("Multi-Block Transmission tau_multi", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Observed Logit L2 Damage", fontsize=11, fontweight="bold")
    ax2.set_title(f"Multi-Block Predictor (r = {r_multi:.3f})", fontsize=11, fontweight="bold")
    
    plt.suptitle("Figure C: Out-of-Sample Damage Prediction (Single vs Multi-Block)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_c_predicted_vs_observed.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure D: Null-Space Rotation across Blocks
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    pairs = [f"B{r['block_b']}->B{r['block_b_plus_1']}" for _, r in df_rot.iterrows()]
    x = np.arange(len(pairs))
    width = 0.35
    ax.bar(x - width/2, df_rot["mean_principal_cosine"], width, label="Mean Principal Cosine", color="#1f77b4")
    ax.bar(x + width/2, df_rot["top_principal_cosine"], width, label="Top Principal Cosine", color="#aec7e8")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=10, fontweight="bold")
    ax.set_ylabel("Subspace Cosine Overlap", fontsize=11, fontweight="bold")
    ax.set_title("Figure D: Null-Space Rotation between Adjacent Downstream Blocks", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_d_nullspace_rotation.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure E: Downstream Leakage Trace
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for mode, col in [("single_null", "#9467bd"), ("multi_null", "#2ca02c"), 
                      ("single_top", "#ff7f0e"), ("multi_top", "#d62728")]:
        sub_l = df_leak[df_leak["mode_name"] == mode]
        ax.plot(sub_l["block"], sub_l["transmission_tau_b"], marker='o', label=mode, color=col, lw=2.0)
    ax.set_xlabel("Downstream Block Index b", fontsize=11, fontweight="bold")
    ax.set_ylabel("Readout Transmission ||A_b Delta P_b||", fontsize=11, fontweight="bold")
    ax.set_title("Figure E: Block-by-Block Leakage Trace of Perturbations", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_e_leakage_trace.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure F: Finite Radius Response Curves (Damage vs Radius)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_c = df_curves[df_curves["arch"] == "deit_small"]
    for mode, col in [("multi_top", "#d62728"), ("single_top", "#ff7f0e"),
                      ("single_null", "#9467bd"), ("multi_null", "#2ca02c")]:
        sub_m = sub_c[sub_c["mode_name"] == mode]
        ax.plot(sub_m["scale_factor"], sub_m["top1_flip_rate"] * 100.0, marker='s', label=mode, color=col, lw=1.8)
    ax.set_xlabel("Perturbation Scale (alpha / sigma_P)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Top-1 Flip Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Figure F: Classifier Robustness (Top-1 Flip Rate) vs Perturbation Scale", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_f_finite_radius_curves.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure G: Depth Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_d = df_depth[df_depth["arch"] == "deit_small"]
    ax.plot(sub_d["depth"], sub_d["sigma_1"], marker='o', lw=2.2, color="#d62728", label="Top Singular Value sigma_1")
    ax.plot(sub_d["depth"], sub_d["sigma_mean"] * 5.0, marker='s', lw=1.8, ls="--", color="#1f77b4", label="Mean Singular Value (x5)")
    ax.set_xlabel("Intervention Depth l", fontsize=11, fontweight="bold")
    ax.set_ylabel("Jacobian Singular Value", fontsize=11, fontweight="bold")
    ax.set_title("Figure G: Multi-Block Jacobian Spectrum vs Depth (DeiT-Small)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_g_depth_comparison.png"))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Figure H: Cross-Architecture Replication
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    x = np.arange(len(df_rep))
    width = 0.25
    ax.bar(x - width, df_rep["multi_top_damage_s04"], width, label="Multi-Block Top Mode", color="#d62728")
    ax.bar(x, df_rep["single_null_damage_s04"], width, label="Single-Block Null Mode", color="#9467bd")
    ax.bar(x + width, df_rep["multi_null_damage_s04"], width, label="Multi-Block Null Mode", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(df_rep["arch"], fontsize=10, fontweight="bold")
    ax.set_ylabel("Logit L2 Damage at s = 0.4", fontsize=11, fontweight="bold")
    ax.set_title("Figure H: Cross-Architecture Multi-Block Null Advantage", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure_h_cross_architecture_replication.png"))
    plt.close(fig)
    print("All 8 multi-block figures successfully generated!")
