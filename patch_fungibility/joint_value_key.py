"""
Joint Value-Key Downstream Low-Transmission Module

Decomposes first-order downstream transmission into:
1. VALUE pathway: V_l in R^{D_readout x (ND)}
2. KEY / attention-rerouting pathway: R_l in R^{D_readout x (ND)}
3. Stacked Joint Operator: C_l = [lambda_V V_l; lambda_K R_l]
4. Cumulative Multi-Block Joint Operator: C_{l->L} = [lambda_J J_{l->L}; lambda_R R_{l->L}]
5. Subspace overlap, principal angles, finite-radius sweeps, and power-law scaling analysis
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
from patch_fungibility.multiblock_operator import construct_downstream_jacobian


def construct_value_and_key_operators(model, arch_type, depth, clean_acts, device):
    """
    Constructs exact Taylor linear operators V_l and R_l:
        delta z_value = V_l vec(Delta P^T)
        delta z_reroute = R_l vec(Delta P^T)
        delta z_local = (V_l + R_l) vec(Delta P^T)
    """
    B, total_tokens, D = clean_acts.shape
    N = total_tokens - 1
    total_dim = N * D
    
    patches = clean_acts[:, 1:, :].reshape(-1, D)
    sigma_P = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()
    
    if arch_type == "dinov2":
        blk = model.backbone.blocks[depth]
        norm1 = blk.norm1
        attn = blk.attn
        num_heads = attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5
        D_readout = 2 * D
        
        # Clean activations
        h_clean = clean_acts[:1].clone().detach()
        hn = norm1(h_clean)
        qkv = attn.qkv(hn).reshape(1, total_tokens, 3, num_heads, head_dim)
        qc, kc, vc = torch.unbind(qkv, 2)
        qc = qc.transpose(1, 2)
        kc = kc.transpose(1, 2)
        vc = vc.transpose(1, 2)
        Ac = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
        
        def get_out_val(dP):
            hp = h_clean.clone(); hp[:, 1:, :] += dP
            qkv_p = attn.qkv(norm1(hp)).reshape(1, total_tokens, 3, num_heads, head_dim)
            _, _, vp = torch.unbind(qkv_p, 2)
            vp = vp.transpose(1, 2)
            out_heads = Ac @ vp
            out = attn.proj(out_heads.transpose(1, 2).reshape(1, total_tokens, D))
            z_cls = out[:, 0, :]
            z_patch = out[:, 1:, :].mean(dim=1)
            return torch.cat([z_cls, z_patch], dim=-1)
            
        def get_out_key(dP):
            hp = h_clean.clone(); hp[:, 1:, :] += dP
            qkv_p = attn.qkv(norm1(hp)).reshape(1, total_tokens, 3, num_heads, head_dim)
            _, kp, _ = torch.unbind(qkv_p, 2)
            kp = kp.transpose(1, 2)
            Ap = ((qc * scale) @ kp.transpose(-2, -1)).softmax(dim=-1)
            out_heads = Ap @ vc
            out = attn.proj(out_heads.transpose(1, 2).reshape(1, total_tokens, D))
            z_cls = out[:, 0, :]
            z_patch = out[:, 1:, :].mean(dim=1)
            return torch.cat([z_cls, z_patch], dim=-1)
            
    else:
        blk = model.blocks[depth]
        norm1 = blk.norm1
        attn = blk.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        scale = attn.scale
        D_readout = D
        
        # Clean activations
        h_clean = clean_acts[:1].clone().detach()
        hn = norm1(h_clean)
        qkv = attn.qkv(hn).reshape(1, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qc, kc, vc = qkv.unbind(0)
        Ac = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
        Ac_cls = Ac[:, :, 0:1, :]
        
        def get_out_val(dP):
            hp = h_clean.clone(); hp[:, 1:, :] += dP
            qkv_p = attn.qkv(norm1(hp)).reshape(1, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            _, _, vp = qkv_p.unbind(0)
            out_heads = (Ac_cls @ vp).transpose(1, 2).reshape(1, 1, D)
            return attn.proj(out_heads).squeeze(1)
            
        def get_out_key(dP):
            hp = h_clean.clone(); hp[:, 1:, :] += dP
            qkv_p = attn.qkv(norm1(hp)).reshape(1, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            _, kp, _ = qkv_p.unbind(0)
            Ap = ((qc * scale) @ kp.transpose(-2, -1)).softmax(dim=-1)
            out_heads = (Ap[:, :, 0:1, :] @ vc).transpose(1, 2).reshape(1, 1, D)
            return attn.proj(out_heads).squeeze(1)

    # Compute Jacobians via batched VJPs
    dP_zero = torch.zeros(1, N, D, device=device, requires_grad=True)
    val_zero = get_out_val(dP_zero)
    key_zero = get_out_key(dP_zero)
    
    V_rows, R_rows = [], []
    chunk_size = 32
    for d in range(0, D_readout, chunk_size):
        end_d = min(d + chunk_size, D_readout)
        for k in range(d, end_d):
            gv = torch.zeros_like(val_zero); gv[0, k] = 1.0
            V_rows.append(torch.autograd.grad(val_zero, dP_zero, grad_outputs=gv, retain_graph=True)[0].reshape(-1))
            
            gk = torch.zeros_like(key_zero); gk[0, k] = 1.0
            is_last = (k == D_readout - 1)
            R_rows.append(torch.autograd.grad(key_zero, dP_zero, grad_outputs=gk, retain_graph=not is_last)[0].reshape(-1))
            
    V_l = torch.stack(V_rows, dim=0) # (D_readout, ND)
    R_l = torch.stack(R_rows, dim=0) # (D_readout, ND)
    
    # Gram SVD of V_l
    Gram_V = V_l @ V_l.T
    evals_V, U_V = torch.linalg.eigh(Gram_V)
    evals_V = evals_V.flip(0)
    U_V = U_V.flip(1)
    S_V = torch.sqrt(torch.clamp(evals_V, min=0.0))
    
    # Gram SVD of R_l
    Gram_R = R_l @ R_l.T
    evals_R, U_R = torch.linalg.eigh(Gram_R)
    evals_R = evals_R.flip(0)
    U_R = U_R.flip(1)
    S_R = torch.sqrt(torch.clamp(evals_R, min=0.0))
    
    # Normalized Joint Operator C_l
    norm_V = torch.norm(V_l).item()
    norm_R = torch.norm(R_l).item()
    lam_V = 1.0 / (norm_V + 1e-12)
    lam_K = 1.0 / (norm_R + 1e-12)
    
    C_l = torch.cat([lam_V * V_l, lam_K * R_l], dim=0) # (2*D_readout, ND)
    Gram_C = C_l @ C_l.T
    evals_C, U_C = torch.linalg.eigh(Gram_C)
    evals_C = evals_C.flip(0)
    U_C = U_C.flip(1)
    S_C = torch.sqrt(torch.clamp(evals_C, min=0.0))
    
    # Subspace Overlap & Principal Angles
    # Basis for RowSpace(V_l) and RowSpace(R_l)
    mask_V = S_V > 1e-5
    mask_R = S_R > 1e-5
    Basis_V = (V_l.T @ U_V[:, mask_V]) / S_V[mask_V] # (ND, K_V)
    Basis_R = (R_l.T @ U_R[:, mask_R]) / S_R[mask_R] # (ND, K_R)
    
    M_overlap = Basis_V.T @ Basis_R # (K_V, K_R)
    s_overlap = torch.linalg.svdvals(M_overlap).cpu().numpy()
    
    top_cos = float(s_overlap[0])
    mean_cos = float(np.mean(s_overlap))
    min_cos = float(s_overlap[-1])
    top_angle = float(np.arccos(np.clip(top_cos, 0.0, 1.0)) * 180.0 / np.pi)
    mean_angle = float(np.arccos(np.clip(mean_cos, 0.0, 1.0)) * 180.0 / np.pi)
    
    # Effective ranks
    eff_rank_V = float(torch.sum(S_V) ** 2 / torch.sum(S_V ** 2))
    eff_rank_R = float(torch.sum(S_R) ** 2 / torch.sum(S_R ** 2))
    eff_rank_C = float(torch.sum(S_C) ** 2 / torch.sum(S_C ** 2))
    
    return {
        "V_l": V_l,
        "R_l": R_l,
        "C_l": C_l,
        "Gram_V": Gram_V,
        "Gram_R": Gram_R,
        "Gram_C": Gram_C,
        "evals_V": evals_V,
        "evals_R": evals_R,
        "evals_C": evals_C,
        "U_V": U_V,
        "U_R": U_R,
        "U_C": U_C,
        "S_V": S_V,
        "S_R": S_R,
        "S_C": S_C,
        "norm_V": norm_V,
        "norm_R": norm_R,
        "lam_V": lam_V,
        "lam_K": lam_K,
        "N": N,
        "D": D,
        "D_readout": D_readout,
        "total_dim": total_dim,
        "sigma_P": sigma_P,
        "s_overlap": s_overlap,
        "top_cos": top_cos,
        "mean_cos": mean_cos,
        "min_cos": min_cos,
        "top_angle": top_angle,
        "mean_angle": mean_angle,
        "eff_rank_V": eff_rank_V,
        "eff_rank_R": eff_rank_R,
        "eff_rank_C": eff_rank_C,
        "depth": depth,
        "arch_type": arch_type
    }


def construct_cumulative_multiblock_joint_operator(model, arch_type, start_depth, clean_acts_list, op_multi, device):
    """
    Constructs the cumulative multi-block joint operator:
        C_{l->L} = [lambda_J * J_{l->L}; lambda_R * R_{l->L}]
    stacking the downstream Jacobian with key rerouting sensitivities across all downstream blocks.
    """
    if arch_type == "dinov2":
        blocks = model.backbone.blocks
        num_blocks = len(blocks)
    else:
        blocks = model.blocks
        num_blocks = len(blocks)
        
    B, total_tokens, D = clean_acts_list[start_depth].shape
    N = total_tokens - 1
    total_dim = N * D
    D_readout = op_multi["D_readout"]
    J_multi = op_multi["J"]
    
    R_b_list = []
    
    for b in range(start_depth, num_blocks):
        blk = blocks[b]
        norm1 = blk.norm1
        attn = blk.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dim if hasattr(attn, "head_dim") else (D // num_heads)
        scale = attn.scale if hasattr(attn, "scale") else (head_dim ** -0.5)
        
        hc_b = clean_acts_list[b][:1]
        hn_b = norm1(hc_b)
        
        if arch_type == "dinov2":
            qkv_b = attn.qkv(hn_b).reshape(1, total_tokens, 3, num_heads, head_dim)
            qc_b, _, vc_b = torch.unbind(qkv_b, 2)
            qc_b = qc_b.transpose(1, 2)
            vc_b = vc_b.transpose(1, 2)
            
            def get_block_b_key_out(dP):
                h = clean_acts_list[start_depth][:1].clone()
                h[:, 1:, :] += dP
                for i in range(start_depth, b):
                    h = blocks[i](h)
                qkv_p = attn.qkv(norm1(h)).reshape(1, total_tokens, 3, num_heads, head_dim)
                _, kp, _ = torch.unbind(qkv_p, 2)
                kp = kp.transpose(1, 2)
                Ap = ((qc_b * scale) @ kp.transpose(-2, -1)).softmax(dim=-1)
                out_heads = Ap @ vc_b
                out = attn.proj(out_heads.transpose(1, 2).reshape(1, total_tokens, D))
                return torch.cat([out[:, 0, :], out[:, 1:, :].mean(dim=1)], dim=-1)
        else:
            qkv_b = attn.qkv(hn_b).reshape(1, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            qc_b, _, vc_b = qkv_b.unbind(0)
            
            def get_block_b_key_out(dP):
                h = clean_acts_list[start_depth][:1].clone()
                h[:, 1:, :] += dP
                for i in range(start_depth, b):
                    h = blocks[i](h)
                qkv_p = attn.qkv(norm1(h)).reshape(1, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                _, kp, _ = qkv_p.unbind(0)
                Ap = ((qc_b * scale) @ kp.transpose(-2, -1)).softmax(dim=-1)
                out_heads = (Ap[:, :, 0:1, :] @ vc_b).transpose(1, 2).reshape(1, 1, D)
                return attn.proj(out_heads).squeeze(1)

        dP_zero = torch.zeros(1, N, D, device=device, requires_grad=True)
        out_zero = get_block_b_key_out(dP_zero)
        rows = []
        chunk = 32
        for d in range(0, D_readout, chunk):
            end_d = min(d + chunk, D_readout)
            for k in range(d, end_d):
                g = torch.zeros_like(out_zero); g[0, k] = 1.0
                is_last = (k == D_readout - 1 and b == num_blocks - 1)
                rows.append(torch.autograd.grad(out_zero, dP_zero, grad_outputs=g, retain_graph=not is_last)[0].reshape(-1))
        R_b = torch.stack(rows, dim=0)
        R_b_list.append(R_b)

    R_cum = torch.cat(R_b_list, dim=0) # (num_blocks * D_readout, ND)
    norm_J = torch.norm(J_multi).item()
    norm_R_cum = torch.norm(R_cum).item()
    lam_J = 1.0 / (norm_J + 1e-12)
    lam_R_cum = 1.0 / (norm_R_cum + 1e-12)
    
    C_cum = torch.cat([lam_J * J_multi, lam_R_cum * R_cum], dim=0)
    Gram_cum = C_cum @ C_cum.T
    evals_cum, U_cum = torch.linalg.eigh(Gram_cum)
    evals_cum = evals_cum.flip(0)
    U_cum = U_cum.flip(1)
    S_cum = torch.sqrt(torch.clamp(evals_cum, min=0.0))
    
    return {
        "C_cum": C_cum,
        "R_cum": R_cum,
        "Gram_cum": Gram_cum,
        "evals_cum": evals_cum,
        "U_cum": U_cum,
        "S_cum": S_cum,
        "norm_J": norm_J,
        "norm_R_cum": norm_R_cum,
        "lam_J": lam_J,
        "lam_R_cum": lam_R_cum
    }


def extract_all_modes(op_vk, op_multi, op_cum=None, seed=42):
    """
    Extracts matched Frobenius-norm unit modes:
    - top_high_transmission: top singular mode of C_l
    - value_only_null: exact null mode of V_l
    - key_only_null: exact null mode of R_l
    - multiblock_J_null: exact null mode of downstream Jacobian J_{l->L}
    - joint_vk_null: exact null mode of C_l (N_V \cap N_K)
    - joint_cum_null: exact null mode of C_{l->L}
    - random_matched_norm: isotropic Gaussian control
    """
    N = op_vk["N"]
    D = op_vk["D"]
    total_dim = op_vk["total_dim"]
    device = op_vk["V_l"].device
    
    torch.manual_seed(seed)
    r = torch.randn(total_dim, device=device)
    
    def project_null(Op, Gram_evals, U_mat):
        inv_evals = torch.where(Gram_evals > 1e-10, 1.0 / Gram_evals, torch.zeros_like(Gram_evals))
        Ar = Op @ r
        inv_G_Ar = U_mat @ (inv_evals * (U_mat.T @ Ar))
        q = r - (Op.T @ inv_G_Ar)
        return (q / torch.norm(q)).view(N, D)
        
    modes = {}
    
    # 1. Top mode of C_l
    v_top = (1.0 / op_vk["S_C"][0].item()) * (op_vk["C_l"].T @ op_vk["U_C"][:, 0])
    modes["top_high_transmission"] = (v_top / torch.norm(v_top)).view(N, D)
    
    # 2. Value-only null mode
    modes["value_only_null"] = project_null(op_vk["V_l"], op_vk["evals_V"], op_vk["U_V"])
    
    # 3. Key-only null mode
    modes["key_only_null"] = project_null(op_vk["R_l"], op_vk["evals_R"], op_vk["U_R"])
    
    # 4. Joint local VK null mode
    modes["joint_vk_null"] = project_null(op_vk["C_l"], op_vk["evals_C"], op_vk["U_C"])
    
    # 5. Multi-block J null mode
    modes["multiblock_J_null"] = project_null(op_multi["J"], op_multi["evals_J"], op_multi["U_J"])
    
    # 6. Cumulative multi-block joint null mode (if provided)
    if op_cum is not None:
        modes["joint_cum_null"] = project_null(op_cum["C_cum"], op_cum["evals_cum"], op_cum["U_cum"])
        
    # 7. Random isotropic mode
    modes["random_matched_norm"] = (r / torch.norm(r)).view(N, D)
    
    return modes


def evaluate_finite_radius_and_attention(model, arch_type, depth, clean_acts_list, clean_logits, modes, eval_targets, device):
    """
    Evaluates full unconstrained model and attention shifts across radii:
    s in {0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0}.
    """
    B, total_tokens, D = clean_acts_list[depth].shape
    N = total_tokens - 1
    clean_acts = clean_acts_list[depth]
    
    patches = clean_acts[:, 1:, :].reshape(-1, D)
    sigma_P = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()
    
    if arch_type == "dinov2":
        blk = model.backbone.blocks[depth]
        norm1 = blk.norm1
        attn = blk.attn
        num_heads = attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5
        
        hn = norm1(clean_acts)
        qkv = attn.qkv(hn).reshape(B, total_tokens, 3, num_heads, head_dim)
        qc, kc, _ = torch.unbind(qkv, 2)
        Ac = ((qc.transpose(1, 2) * scale) @ kc.transpose(1, 2).transpose(-2, -1)).softmax(dim=-1)
    else:
        blk = model.blocks[depth]
        norm1 = blk.norm1
        attn = blk.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        scale = attn.scale
        
        hn = norm1(clean_acts)
        qkv = attn.qkv(hn).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qc, kc, _ = qkv.unbind(0)
        Ac = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)

    radii = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0]
    rows_eval = []
    rows_attn = []
    
    for mode_name, Q in modes.items():
        for s in radii:
            alpha = s * sigma_P
            dP = (alpha * Q).unsqueeze(0)
            hp = clean_acts.clone()
            hp[:, 1:, :] += dP
            
            # 1. Full model forward
            pert_logits, _ = forward_block_by_block(model, arch_type, h_start=hp, start_depth=depth)
            diff = pert_logits - clean_logits
            logit_l2 = torch.norm(diff, dim=-1).mean().item()
            
            p_clean = F.softmax(clean_logits, dim=-1)
            log_p_pert = F.log_softmax(pert_logits, dim=-1)
            kl_div = F.kl_div(log_p_pert, p_clean, reduction='batchmean').item()
            
            clean_top1 = clean_logits.argmax(dim=-1)
            pert_top1 = pert_logits.argmax(dim=-1)
            top1_flip = (clean_top1 != pert_top1).float().mean().item()
            top1_acc = (pert_top1 == eval_targets).float().mean().item()
            
            clean_target = clean_logits.gather(1, eval_targets.unsqueeze(1)).squeeze(1)
            pert_target = pert_logits.gather(1, eval_targets.unsqueeze(1)).squeeze(1)
            margin_damage = (clean_target - pert_target).mean().item()
            
            # Readout disturbance at depth
            if arch_type == "dinov2":
                out_p = blk(hp)
                out_c = blk(clean_acts)
                readout_dist = torch.norm(out_p[:, 0, :] - out_c[:, 0, :], dim=-1).mean().item()
            else:
                out_p = blk(hp)
                out_c = blk(clean_acts)
                readout_dist = torch.norm(out_p[:, 0, :] - out_c[:, 0, :], dim=-1).mean().item()
                
            # 2. Attention shift metrics
            hn_p = norm1(hp)
            if arch_type == "dinov2":
                qkv_p = attn.qkv(hn_p).reshape(B, total_tokens, 3, num_heads, head_dim)
                qp, kp, _ = torch.unbind(qkv_p, 2)
                Ap = ((qp.transpose(1, 2) * scale) @ kp.transpose(1, 2).transpose(-2, -1)).softmax(dim=-1)
            else:
                qkv_p = attn.qkv(hn_p).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                qp, kp, _ = qkv_p.unbind(0)
                Ap = ((qp * scale) @ kp.transpose(-2, -1)).softmax(dim=-1)
                
            attn_frob = torch.norm(Ap - Ac).item()
            attn_cls_diff = torch.norm(Ap[:, :, 0, 1:] - Ac[:, :, 0, 1:]).item()
            
            # Attention KL
            kl_attn = F.kl_div(torch.log(Ap.clamp(min=1e-12)), Ac, reduction='batchmean').item()
            # Entropy change
            H_clean = -(Ac * torch.log(Ac.clamp(min=1e-12))).sum(dim=-1).mean().item()
            H_pert = -(Ap * torch.log(Ap.clamp(min=1e-12))).sum(dim=-1).mean().item()
            entropy_diff = H_pert - H_clean
            
            rows_eval.append({
                "mode_name": mode_name,
                "radius_s": s,
                "alpha": alpha,
                "logit_l2": logit_l2,
                "kl_div": kl_div,
                "margin_damage": margin_damage,
                "top1_flip_rate": top1_flip,
                "top1_accuracy": top1_acc,
                "readout_disturbance": readout_dist,
                "attention_frob_shift": attn_frob
            })
            
            rows_attn.append({
                "mode_name": mode_name,
                "radius_s": s,
                "alpha": alpha,
                "attention_frob_shift": attn_frob,
                "readout_to_patch_shift": attn_cls_diff,
                "attention_kl": kl_attn,
                "entropy_change": entropy_diff
            })
            
    return pd.DataFrame(rows_eval), pd.DataFrame(rows_attn)


def fit_power_law_scaling(df_eval):
    """
    Fits power-law scaling exponents p: ln(damage) = p * ln(alpha) + c
    on small finite radii s in [0.05, 0.4].
    """
    sub = df_eval[(df_eval["radius_s"] >= 0.05) & (df_eval["radius_s"] <= 0.4)]
    modes = sub["mode_name"].unique()
    
    rows = []
    for mode in modes:
        m_sub = sub[sub["mode_name"] == mode]
        log_alpha = np.log(m_sub["alpha"].values)
        log_l2 = np.log(np.maximum(m_sub["logit_l2"].values, 1e-12))
        
        p, c = np.polyfit(log_alpha, log_l2, 1)
        # R^2
        pred_log = p * log_alpha + c
        ss_tot = np.sum((log_l2 - np.mean(log_l2)) ** 2)
        ss_res = np.sum((log_l2 - pred_log) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        
        rows.append({
            "mode_name": mode,
            "scaling_exponent_p": float(p),
            "intercept_c": float(c),
            "r_squared": float(r2)
        })
        
    return pd.DataFrame(rows)


def trace_blockwise_rerouting(model, arch_type, start_depth, clean_acts_list, modes, device):
    """
    Traces modes through each downstream block recording:
    - value transmission
    - key/logit disturbance
    - softmax attention shift
    - query shift ||delta Q_b|| / ||Q_b||
    - residual-stream norm
    - readout disturbance
    """
    if arch_type == "dinov2":
        blocks = model.backbone.blocks
    else:
        blocks = model.blocks
    num_blocks = len(blocks)
    
    B, total_tokens, D = clean_acts_list[start_depth].shape
    N = total_tokens - 1
    patches = clean_acts_list[start_depth][:, 1:, :].reshape(-1, D)
    sigma_P = (patches.var(dim=0, unbiased=False).sum()).sqrt().item()
    alpha = 0.4 * sigma_P
    
    rows = []
    
    for mode_name, Q in modes.items():
        dP = (alpha * Q).unsqueeze(0)
        h_pert = clean_acts_list[start_depth].clone()
        h_pert[:, 1:, :] += dP
        
        for b in range(start_depth, num_blocks):
            blk = blocks[b]
            norm1 = blk.norm1
            attn = blk.attn
            H = attn.num_heads
            d_h = attn.head_dim if hasattr(attn, "head_dim") else (D // H)
            scale = attn.scale if hasattr(attn, "scale") else (d_h ** -0.5)
            
            hc_b = clean_acts_list[b]
            
            # Norms and disturbances
            diff_patch = (h_pert[:, 1:, :] - hc_b[:, 1:, :]).mean(dim=0)
            norm_patch = torch.norm(diff_patch).item()
            
            # Clean and perturbed attention at block b
            if arch_type == "dinov2":
                qkv_c = attn.qkv(norm1(hc_b)).reshape(B, total_tokens, 3, H, d_h)
                qc, kc, vc = torch.unbind(qkv_c, 2)
                Ac = ((qc.transpose(1, 2) * scale) @ kc.transpose(1, 2).transpose(-2, -1)).softmax(dim=-1)
                
                qkv_p = attn.qkv(norm1(h_pert)).reshape(B, total_tokens, 3, H, d_h)
                qp, kp, vp = torch.unbind(qkv_p, 2)
                Ap = ((qp.transpose(1, 2) * scale) @ kp.transpose(1, 2).transpose(-2, -1)).softmax(dim=-1)
                
                # Query disturbance
                q_diff = torch.norm(qp - qc).item()
                q_norm = torch.norm(qc).item()
                query_shift = q_diff / (q_norm + 1e-12)
            else:
                qkv_c = attn.qkv(norm1(hc_b)).reshape(B, total_tokens, 3, H, d_h).permute(2, 0, 3, 1, 4)
                qc, kc, vc = qkv_c.unbind(0)
                Ac = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
                
                qkv_p = attn.qkv(norm1(h_pert)).reshape(B, total_tokens, 3, H, d_h).permute(2, 0, 3, 1, 4)
                qp, kp, vp = qkv_p.unbind(0)
                Ap = ((qp * scale) @ kp.transpose(-2, -1)).softmax(dim=-1)
                
                q_diff = torch.norm(qp[:, :, 0, :] - qc[:, :, 0, :]).item()
                q_norm = torch.norm(qc[:, :, 0, :]).item()
                query_shift = q_diff / (q_norm + 1e-12)
                
            attn_shift = torch.norm(Ap - Ac).item()
            cls_attn_shift = torch.norm(Ap[:, :, 0, :] - Ac[:, :, 0, :]).item()
            
            # Forward block b
            h_pert = blk(h_pert)
            hc_next = blk(hc_b)
            
            cls_dist = torch.norm(h_pert[:, 0, :] - hc_next[:, 0, :]).item()
            
            rows.append({
                "mode_name": mode_name,
                "block": b,
                "patch_norm": norm_patch,
                "attention_frob_shift": attn_shift,
                "cls_attention_shift": cls_attn_shift,
                "query_relative_shift": query_shift,
                "readout_disturbance": cls_dist
            })
            
    return pd.DataFrame(rows)


def evaluate_held_out_predictions(model, arch_type, depth, clean_acts, clean_logits, op_vk, op_multi, device, M=100, seed=123):
    """
    Evaluates out-of-sample prediction of damage for M=100 held-out random perturbations.
    Compares:
    - tau_value = ||V_l vec(Delta P^T)||
    - tau_key = ||R_l vec(Delta P^T)||
    - tau_joint = ||C_l vec(Delta P^T)||
    - tau_multi = ||J_{l->L} vec(Delta P^T)||
    """
    B, total_tokens, D = clean_acts.shape
    N = total_tokens - 1
    total_dim = N * D
    sigma_P = op_vk["sigma_P"]
    
    V_l = op_vk["V_l"]
    R_l = op_vk["R_l"]
    C_l = op_vk["C_l"]
    J_multi = op_multi["J"]
    
    torch.manual_seed(seed)
    scales = np.random.uniform(0.1, 1.0, size=M)
    
    rows = []
    
    for i in range(M):
        r = torch.randn(total_dim, device=device)
        r = r / torch.norm(r)
        alpha = float(scales[i] * sigma_P)
        q_vec = alpha * r
        dP = q_vec.view(1, N, D)
        
        # Linear transmission scores
        tau_val = torch.norm(V_l @ q_vec).item()
        tau_key = torch.norm(R_l @ q_vec).item()
        tau_joint = torch.norm(C_l @ q_vec).item()
        tau_multi = torch.norm(J_multi @ q_vec).item()
        
        # Real damage
        hp = clean_acts.clone()
        hp[:, 1:, :] += dP
        pert_logits, _ = forward_block_by_block(model, arch_type, h_start=hp, start_depth=depth)
        logit_l2 = torch.norm(pert_logits - clean_logits).item()
        
        p_clean = F.softmax(clean_logits, dim=-1)
        log_p_pert = F.log_softmax(pert_logits, dim=-1)
        kl_div = F.kl_div(log_p_pert, p_clean, reduction='batchmean').item()
        
        rows.append({
            "sample_idx": i,
            "scale_factor": float(scales[i]),
            "alpha": alpha,
            "tau_value": tau_val,
            "tau_key": tau_key,
            "tau_joint": tau_joint,
            "tau_multi": tau_multi,
            "logit_l2": logit_l2,
            "kl_div": kl_div
        })
        
    df_pred = pd.DataFrame(rows)
    
    # Compute correlation metrics
    r_val, _ = pearsonr(df_pred["tau_value"], df_pred["logit_l2"])
    rho_val, _ = spearmanr(df_pred["tau_value"], df_pred["logit_l2"])
    
    r_key, _ = pearsonr(df_pred["tau_key"], df_pred["logit_l2"])
    rho_key, _ = spearmanr(df_pred["tau_key"], df_pred["logit_l2"])
    
    r_joint, _ = pearsonr(df_pred["tau_joint"], df_pred["logit_l2"])
    rho_joint, _ = spearmanr(df_pred["tau_joint"], df_pred["logit_l2"])
    
    r_multi, _ = pearsonr(df_pred["tau_multi"], df_pred["logit_l2"])
    rho_multi, _ = spearmanr(df_pred["tau_multi"], df_pred["logit_l2"])
    
    corr_summary = {
        "r_value": float(r_val), "rho_value": float(rho_val), "r2_value": float(r_val**2),
        "r_key": float(r_key), "rho_key": float(rho_key), "r2_key": float(r_key**2),
        "r_joint": float(r_joint), "rho_joint": float(rho_joint), "r2_joint": float(r_joint**2),
        "r_multi": float(r_multi), "rho_multi": float(rho_multi), "r2_multi": float(r_multi**2)
    }
    
    return df_pred, corr_summary
