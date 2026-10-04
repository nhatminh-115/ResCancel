"""
patch_fungibility/attention_causal_audit.py

Causal path decomposition and mechanistic audit of the attention aggregation
cancellation hypothesis in Vision Transformers.

Decomposes the downstream response into:
1. Frozen-attention value aggregation pathway (A_clean @ V_pert)
2. Attention-rerouting pathway (A_pert @ V_clean)
3. Sub-projection pathways (Q, K, V combinations)
4. Head-wise cancellation scalar Gamma_h(a) = sum_i w_{h,readout,i} a_i
5. Head-level linear disturbance prediction Delta z_h^pred = alpha Gamma_h (v^T W_V^h)
6. Attention matrix shift (Frobenius norm, KL divergence, entropy)
7. Block-by-block disturbance propagation
8. Residual pathway vs Attention branch vs MLP branch
9. Spatial smoothness and 2D Fourier frequency response
"""

import os
import json
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns

from patch_fungibility.dense_fraction_models import (
    load_model_and_transform,
    forward_block_by_block
)
from patch_fungibility.joint_stream_geometry import (
    construct_token_patterns,
    construct_joint_feature_directions,
    estimate_functional_metric
)
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits


# ==============================================================================
# 1. Decomposed Block Execution Helpers
# ==============================================================================

def get_block_module(model: nn.Module, model_key: str, blk_idx: int) -> nn.Module:
    """Returns the block module at index blk_idx for timm models or DINOv2."""
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        return model.blocks[blk_idx]
    elif model_key == "dinov2":
        return model.backbone.blocks[blk_idx]
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def extract_readout_token(h: torch.Tensor, model_key: str) -> torch.Tensor:
    """Extracts the appropriate readout representation from hidden state h."""
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        return h[:, 0, :]
    elif model_key == "dinov2":
        cls_tok = h[:, 0, :]
        patch_mean = h[:, 1:, :].mean(dim=1)
        return torch.cat([cls_tok, patch_mean], dim=-1)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def extract_readout_attention(A: torch.Tensor, model_key: str) -> torch.Tensor:
    """
    Extracts attention weights from the readout query to spatial patches.
    A: (B, H, total_tokens, total_tokens)
    Returns: (B, H, N)
    """
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        return A[:, :, 0, 1:]
    elif model_key == "dinov2":
        w_cls = A[:, :, 0, 1:]
        w_patch = A[:, :, 1:, 1:].mean(dim=2)
        return 0.5 * (w_cls + w_patch)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def run_decomposed_block(
    model: nn.Module,
    model_key: str,
    blk_idx: int,
    h_in_clean: torch.Tensor,
    h_in_pert: torch.Tensor,
    q_src: str = 'pert',     # 'clean' or 'pert'
    k_src: str = 'pert',     # 'clean' or 'pert'
    v_src: str = 'pert',     # 'clean' or 'pert'
    res_src: str = 'pert',   # 'clean' or 'pert'
    frozen_A: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs a decomposed forward pass through block blk_idx.
    Returns: (h_out, A, attn_branch_out, mlp_branch_out, post_attn_res)
    """
    blk = get_block_module(model, model_key, blk_idx)
    B, total_tokens, C = h_in_clean.shape

    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        num_heads = blk.attn.num_heads
        head_dim = blk.attn.head_dim
        scale = blk.attn.scale

        norm_c = blk.norm1(h_in_clean)
        norm_p = blk.norm1(h_in_pert)

        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qkv_p = blk.attn.qkv(norm_p).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qc, kc, vc = qkv_c.unbind(0)
        qp, kp, vp = qkv_p.unbind(0)

        q = qp if q_src == 'pert' else qc
        k = kp if k_src == 'pert' else kc
        v = vp if v_src == 'pert' else vc
        res = h_in_pert if res_src == 'pert' else h_in_clean

        if frozen_A is not None:
            A = frozen_A
        else:
            A = ((q * scale) @ k.transpose(-2, -1)).softmax(dim=-1)

        ctx = (A @ v).transpose(1, 2).reshape(B, total_tokens, C)
        attn_branch = blk.drop_path1(blk.ls1(blk.attn.proj(ctx)))
        post_attn_res = res + attn_branch
        mlp_branch = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(post_attn_res))))
        h_out = post_attn_res + mlp_branch

        return h_out, A, attn_branch, mlp_branch, post_attn_res

    elif model_key == "dinov2":
        num_heads = blk.attn.num_heads
        head_dim = C // num_heads
        scale = head_dim ** -0.5

        norm_c = blk.norm1(h_in_clean)
        norm_p = blk.norm1(h_in_pert)

        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim)
        qkv_p = blk.attn.qkv(norm_p).reshape(B, total_tokens, 3, num_heads, head_dim)
        qc, kc, vc = torch.unbind(qkv_c, 2)
        qp, kp, vp = torch.unbind(qkv_p, 2)

        qc = qc.transpose(1, 2)
        kc = kc.transpose(1, 2)
        vc = vc.transpose(1, 2)
        qp = qp.transpose(1, 2)
        kp = kp.transpose(1, 2)
        vp = vp.transpose(1, 2)

        q = qp if q_src == 'pert' else qc
        k = kp if k_src == 'pert' else kc
        v = vp if v_src == 'pert' else vc
        res = h_in_pert if res_src == 'pert' else h_in_clean

        if frozen_A is not None:
            A = frozen_A
        else:
            A = ((q * scale) @ k.transpose(-2, -1)).softmax(dim=-1)

        ctx = (A @ v).transpose(1, 2).contiguous().view(B, total_tokens, C)
        attn_branch = blk.ls1(blk.attn.proj_drop(blk.attn.proj(ctx)))
        post_attn_res = res + attn_branch
        mlp_branch = blk.ls2(blk.mlp(blk.norm2(post_attn_res)))
        h_out = post_attn_res + mlp_branch

        return h_out, A, attn_branch, mlp_branch, post_attn_res

    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def forward_from_block_output(
    model: nn.Module,
    model_key: str,
    h_start: torch.Tensor,
    next_block_idx: int
) -> torch.Tensor:
    """Executes subsequent transformer blocks from h_start to final classification logits."""
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        h = h_start
        for b in range(next_block_idx, len(model.blocks)):
            h = model.blocks[b](h)
        logits = model.forward_head(model.norm(h))
        return logits

    elif model_key == "dinov2":
        h = h_start
        for b in range(next_block_idx, len(model.backbone.blocks)):
            h = model.backbone.blocks[b](h)
        h_norm = model.backbone.norm(h)
        cls_norm = h_norm[:, 0]
        patch_norm = h_norm[:, 1:]
        patch_mean = patch_norm.mean(dim=1)
        readout_in = torch.cat([cls_norm, patch_mean], dim=-1)
        logits = model.linear_head(readout_in)
        return logits

    else:
        raise ValueError(f"Unknown model_key: {model_key}")


# ==============================================================================
# 2. Main Causal Condition Evaluation
# ==============================================================================

def evaluate_causal_conditions(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor,
    h_clean_depth: torch.Tensor,
    feature_dirs: Dict[str, torch.Tensor],
    token_patterns: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_grid: List[float] = [0.5, 1.0, 2.0],
    device: torch.device = torch.device("cuda")
) -> List[Dict[str, Any]]:
    """
    Evaluates controlled causal conditions:
    - clean
    - full_perturbation
    - frozen_attn_v_only_pert_res
    - frozen_attn_v_only_clean_res
    - reroute_clean_res
    - reroute_pert_res
    """
    records = []
    blk_idx = depth  # 0-indexed: block depth takes h_depth to h_{depth+1}
    B, total_tokens, D = h_clean_depth.shape
    N = total_tokens - 1

    # Precompute clean block depth output
    with torch.no_grad():
        h_clean_next, A_clean, _, _, _ = run_decomposed_block(
            model, model_key, blk_idx, h_clean_depth, h_clean_depth,
            q_src='clean', k_src='clean', v_src='clean', res_src='clean'
        )
        z_clean_readout = extract_readout_token(h_clean_next, model_key)
        clean_target_logits = clean_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        clean_top1 = clean_logits.argmax(dim=-1)

    # 1. Clean condition record
    records.append({
        "model_key": model_key,
        "depth": depth,
        "token_pattern": "none",
        "feature_dir": "none",
        "scale_s": 0.0,
        "condition": "clean",
        "logit_margin_drop": 0.0,
        "top1_flip_rate": 0.0,
        "logit_l2": 0.0,
        "dz_readout_l1": 0.0,
        "dp_stream_l1": 0.0
    })

    target_fdirs = ["jac_top", "jac_null", "pc1", "rand_dir"]
    target_tmodes = ["global_coherent", "random_sign", "checkerboard"]

    conditions_to_test = [
        ("full_perturbation", 'pert', 'pert', 'pert', 'pert', None),
        ("frozen_attn_v_only_pert_res", 'clean', 'clean', 'pert', 'pert', A_clean),
        ("frozen_attn_v_only_clean_res", 'clean', 'clean', 'pert', 'clean', A_clean),
        ("reroute_clean_res", 'pert', 'pert', 'clean', 'clean', None),
        ("reroute_pert_res", 'pert', 'pert', 'clean', 'pert', None)
    ]

    for f_name in target_fdirs:
        v = feature_dirs[f_name].to(device)
        for t_name in target_tmodes:
            a = token_patterns[t_name].to(device)
            base_pert = torch.einsum('n,d->nd', a, v).unsqueeze(0).expand(B, -1, -1)

            for s in scale_grid:
                alpha = s * sigma_P
                delta_P = alpha * base_pert
                h_pert = h_clean_depth.clone()
                h_pert[:, 1:, :] += delta_P

                for cond_name, q_s, k_s, v_s, res_s, frz_A in conditions_to_test:
                    with torch.no_grad():
                        h_next, _, _, _, _ = run_decomposed_block(
                            model, model_key, blk_idx, h_clean_depth, h_pert,
                            q_src=q_s, k_src=k_s, v_src=v_s, res_src=res_s, frozen_A=frz_A
                        )
                        logits_pert = forward_from_block_output(model, model_key, h_next, blk_idx + 1)

                        # Metrics
                        pert_target_logits = logits_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
                        margin_drop = (clean_target_logits - pert_target_logits).mean().item()
                        top1_flips = (logits_pert.argmax(dim=-1) != clean_top1).float().mean().item()
                        logit_l2 = (logits_pert - clean_logits).norm(dim=-1).mean().item()

                        # Immediate block depth+1 changes
                        z_pert_readout = extract_readout_token(h_next, model_key)
                        dz_readout = (z_pert_readout - z_clean_readout).norm(dim=-1).mean().item()
                        dp_stream = (h_next[:, 1:, :] - h_clean_next[:, 1:, :]).norm(dim=(-2, -1)).mean().item()

                        records.append({
                            "model_key": model_key,
                            "depth": depth,
                            "token_pattern": t_name,
                            "feature_dir": f_name,
                            "scale_s": s,
                            "condition": cond_name,
                            "logit_margin_drop": margin_drop,
                            "top1_flip_rate": top1_flips,
                            "logit_l2": logit_l2,
                            "dz_readout_l1": dz_readout,
                            "dp_stream_l1": dp_stream
                        })

    return records


# ==============================================================================
# 3. Sub-Projection Decomposition (Q, K, V Combinations)
# ==============================================================================

def evaluate_qkv_decomposition(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor,
    h_clean_depth: torch.Tensor,
    feature_dirs: Dict[str, torch.Tensor],
    token_patterns: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_s: float = 1.0,
    device: torch.device = torch.device("cuda")
) -> List[Dict[str, Any]]:
    """
    Evaluates individual and joint projection combinations:
    - V_only: (Q_c, K_c, V_p)
    - K_only: (Q_c, K_p, V_c)
    - Q_only: (Q_p, K_c, V_c)
    - K_plus_V: (Q_c, K_p, V_p)
    - Q_plus_K: (Q_p, K_p, V_c)
    - Q_plus_K_plus_V: (Q_p, K_p, V_p)
    """
    records = []
    blk_idx = depth
    B, total_tokens, D = h_clean_depth.shape
    alpha = scale_s * sigma_P

    clean_target_logits = clean_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    clean_top1 = clean_logits.argmax(dim=-1)

    with torch.no_grad():
        h_clean_next, _, _, _, _ = run_decomposed_block(
            model, model_key, blk_idx, h_clean_depth, h_clean_depth,
            q_src='clean', k_src='clean', v_src='clean', res_src='clean'
        )
        z_clean_readout = extract_readout_token(h_clean_next, model_key)

    combos = [
        ("V_only", 'clean', 'clean', 'pert', 'clean'),
        ("K_only", 'clean', 'pert', 'clean', 'clean'),
        ("Q_only", 'pert', 'clean', 'clean', 'clean'),
        ("K_plus_V", 'clean', 'pert', 'pert', 'clean'),
        ("Q_plus_K", 'pert', 'pert', 'clean', 'clean'),
        ("Q_plus_K_plus_V", 'pert', 'pert', 'pert', 'clean')
    ]

    for f_name in ["jac_top", "jac_null"]:
        v = feature_dirs[f_name].to(device)
        for t_name in ["global_coherent", "random_sign", "checkerboard"]:
            a = token_patterns[t_name].to(device)
            delta_P = alpha * torch.einsum('n,d->nd', a, v).unsqueeze(0).expand(B, -1, -1)
            h_pert = h_clean_depth.clone()
            h_pert[:, 1:, :] += delta_P

            for pathway_name, q_s, k_s, v_s, res_s in combos:
                with torch.no_grad():
                    h_next, _, _, _, _ = run_decomposed_block(
                        model, model_key, blk_idx, h_clean_depth, h_pert,
                        q_src=q_s, k_src=k_s, v_src=v_s, res_src=res_s
                    )
                    logits_pert = forward_from_block_output(model, model_key, h_next, blk_idx + 1)

                    pert_target = logits_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
                    margin_drop = (clean_target_logits - pert_target).mean().item()
                    top1_flips = (logits_pert.argmax(dim=-1) != clean_top1).float().mean().item()

                    z_pert = extract_readout_token(h_next, model_key)
                    dz_readout = (z_pert - z_clean_readout).norm(dim=-1).mean().item()

                    records.append({
                        "model_key": model_key,
                        "depth": depth,
                        "token_pattern": t_name,
                        "feature_dir": f_name,
                        "scale_s": scale_s,
                        "pathway": pathway_name,
                        "logit_margin_drop": margin_drop,
                        "top1_flip_rate": top1_flips,
                        "dz_readout_l1": dz_readout
                    })

    return records


# ==============================================================================
# 4. Head-Wise Gamma Scalar & Linear Disturbance Prediction
# ==============================================================================

def compute_headwise_gamma_and_predictions(
    model: nn.Module,
    model_key: str,
    depth: int,
    h_clean_depth: torch.Tensor,
    feature_dirs: Dict[str, torch.Tensor],
    token_patterns: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_s: float = 1.0,
    device: torch.device = torch.device("cuda")
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes:
    1. Head-wise cancellation scalar Gamma_h(a) = sum_i w_{h,readout,i} a_i
    2. Linear value prediction Delta z_h^pred = alpha Gamma_h (v^T W_V^h)
       compared against measured head disturbance Delta z_h^obs
    """
    records_gamma = []
    records_pred = []
    blk_idx = depth
    B, total_tokens, D = h_clean_depth.shape
    N = total_tokens - 1
    alpha = scale_s * sigma_P
    blk = get_block_module(model, model_key, blk_idx)

    # 1. Clean attention matrix and projections
    norm_c = blk.norm1(h_clean_depth)
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        num_heads = blk.attn.num_heads
        head_dim = blk.attn.head_dim
        scale = blk.attn.scale
        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qc, kc, vc = qkv_c.unbind(0)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
    elif model_key == "dinov2":
        num_heads = blk.attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5
        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim)
        qc, kc, vc = torch.unbind(qkv_c, 2)
        qc = qc.transpose(1, 2)
        kc = kc.transpose(1, 2)
        vc = vc.transpose(1, 2)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
    else:
        raise ValueError(f"Unknown model: {model_key}")

    # Readout attention weights w: shape (B, H, N)
    w_readout = extract_readout_attention(A_clean, model_key)

    # Compute Gamma_h for each token pattern
    all_patterns = [
        "global_coherent", "random_sign", "checkerboard",
        "random_gaussian", "spatial_cluster_25%", "smooth_spatial"
    ]

    for t_name in all_patterns:
        if t_name not in token_patterns:
            continue
        a = token_patterns[t_name].to(device)
        # Gamma_h: shape (B, H)
        gamma = torch.einsum('bhn,n->bh', w_readout, a)

        for h in range(num_heads):
            w_h = w_readout[:, h, :]
            # Total variation on w_h (assuming 14x14 or 16x16 grid)
            grid_side = int(math.isqrt(N))
            if grid_side * grid_side == N:
                w_grid = w_h.view(B, grid_side, grid_side)
                tv = ((w_grid[:, 1:, :] - w_grid[:, :-1, :]).abs().mean() +
                      (w_grid[:, :, 1:] - w_grid[:, :, :-1]).abs().mean()).item()
            else:
                tv = 0.0

            # Attention entropy
            ent = -(w_h * (w_h + 1e-12).log()).sum(dim=-1).mean().item()

            records_gamma.append({
                "model_key": model_key,
                "depth": depth,
                "token_pattern": t_name,
                "head_idx": h,
                "gamma_h": gamma[:, h].mean().item(),
                "abs_gamma_h": gamma[:, h].abs().mean().item(),
                "w_mean": w_h.mean().item(),
                "w_entropy": ent,
                "w_tv": tv
            })

    # Linear Head Prediction: Compare Delta z_h^pred with Delta z_h^obs
    for f_name in ["jac_top", "jac_null"]:
        v = feature_dirs[f_name].to(device)
        for t_name in ["global_coherent", "random_sign", "checkerboard"]:
            a = token_patterns[t_name].to(device)
            delta_P = alpha * torch.einsum('n,d->nd', a, v).unsqueeze(0).expand(B, -1, -1)
            h_pert = h_clean_depth.clone()
            h_pert[:, 1:, :] += delta_P

            norm_p = blk.norm1(h_pert)
            if model_key in ("deit_tiny", "deit_small", "vit_base"):
                qkv_p = blk.attn.qkv(norm_p).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                _, _, vp = qkv_p.unbind(0)
            elif model_key == "dinov2":
                qkv_p = blk.attn.qkv(norm_p).reshape(B, total_tokens, 3, num_heads, head_dim)
                _, _, vp = torch.unbind(qkv_p, 2)
                vp = vp.transpose(1, 2)

            # Observed head disturbance:
            # (A_clean @ vp - A_clean @ vc)
            ctx_c = A_clean @ vc
            ctx_p = A_clean @ vp
            diff_ctx = ctx_p - ctx_c  # (B, H, total_tokens, head_dim)

            # Readout query index: for CLS readout, index 0; for DINOv2, average index 0 and patch mean
            if model_key in ("deit_tiny", "deit_small", "vit_base"):
                dz_obs = diff_ctx[:, :, 0, :]  # (B, H, head_dim)
                dz_pred = torch.einsum('bhn,bhnd->bhd', w_readout, (vp - vc)[:, :, 1:, :])
            elif model_key == "dinov2":
                dz_obs_cls = diff_ctx[:, :, 0, :]
                dz_obs_patch = diff_ctx[:, :, 1:, :].mean(dim=2)
                dz_obs = 0.5 * (dz_obs_cls + dz_obs_patch)
                dz_pred = torch.einsum('bhn,bhnd->bhd', w_readout, (vp - vc)[:, :, 1:, :])

            for h in range(num_heads):
                obs_h = dz_obs[:, h, :]   # (B, d)
                pred_h = dz_pred[:, h, :] # (B, d)

                cos_sim = torch.cosine_similarity(obs_h, pred_h, dim=-1).mean().item()
                norm_obs = obs_h.norm(dim=-1).mean().item()
                norm_pred = pred_h.norm(dim=-1).mean().item()
                ratio = norm_pred / (norm_obs + 1e-10)

                # Pearson correlation across flattened dimensions
                obs_flat = obs_h.flatten().cpu().numpy()
                pred_flat = pred_h.flatten().cpu().numpy()
                if np.std(obs_flat) > 1e-8 and np.std(pred_flat) > 1e-8:
                    pearson = float(np.corrcoef(obs_flat, pred_flat)[0, 1])
                else:
                    pearson = 1.0

                # R2 score
                ss_res = np.sum((obs_flat - pred_flat) ** 2)
                ss_tot = np.sum((obs_flat - np.mean(obs_flat)) ** 2) + 1e-10
                r2 = float(1.0 - (ss_res / ss_tot))

                records_pred.append({
                    "model_key": model_key,
                    "depth": depth,
                    "token_pattern": t_name,
                    "feature_dir": f_name,
                    "head_idx": h,
                    "cosine_sim": cos_sim,
                    "norm_ratio_pred_over_obs": ratio,
                    "pearson_r": pearson,
                    "r2_score": r2,
                    "pred_norm": norm_pred,
                    "obs_norm": norm_obs
                })

    return records_gamma, records_pred


# ==============================================================================
# 5. Attention Matrix Movement Metrics
# ==============================================================================

def measure_attention_movement(
    model: nn.Module,
    model_key: str,
    depth: int,
    h_clean_depth: torch.Tensor,
    feature_dirs: Dict[str, torch.Tensor],
    token_patterns: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_grid: List[float] = [0.5, 1.0, 2.0],
    device: torch.device = torch.device("cuda")
) -> List[Dict[str, Any]]:
    """
    Measures how much the softmax attention matrix moves:
    - Frobenius norm diff ||A_pert - A_clean||_F
    - KL divergence KL(A_clean || A_pert)
    - Attention entropy change Delta H
    - Readout-to-patch attention correlation
    - Max head Frobenius diff
    """
    records = []
    blk_idx = depth
    B, total_tokens, D = h_clean_depth.shape
    blk = get_block_module(model, model_key, blk_idx)

    # Compute clean attention
    norm_c = blk.norm1(h_clean_depth)
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        num_heads = blk.attn.num_heads
        head_dim = blk.attn.head_dim
        scale = blk.attn.scale
        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qc, kc, _ = qkv_c.unbind(0)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
    elif model_key == "dinov2":
        num_heads = blk.attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5
        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim)
        qc, kc, _ = torch.unbind(qkv_c, 2)
        qc = qc.transpose(1, 2)
        kc = kc.transpose(1, 2)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
    else:
        raise ValueError(f"Unknown model: {model_key}")

    ent_clean = -(A_clean * (A_clean + 1e-12).log()).sum(dim=-1).mean().item()
    w_c = extract_readout_attention(A_clean, model_key)

    for f_name in ["jac_top", "jac_null"]:
        v = feature_dirs[f_name].to(device)
        for t_name in ["global_coherent", "random_sign", "checkerboard"]:
            a = token_patterns[t_name].to(device)
            base_pert = torch.einsum('n,d->nd', a, v).unsqueeze(0).expand(B, -1, -1)

            for s in scale_grid:
                alpha = s * sigma_P
                delta_P = alpha * base_pert
                h_pert = h_clean_depth.clone()
                h_pert[:, 1:, :] += delta_P

                norm_p = blk.norm1(h_pert)
                if model_key in ("deit_tiny", "deit_small", "vit_base"):
                    qkv_p = blk.attn.qkv(norm_p).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    qp, kp, _ = qkv_p.unbind(0)
                    A_pert = ((qp * scale) @ kp.transpose(-2, -1)).softmax(dim=-1)
                elif model_key == "dinov2":
                    qkv_p = blk.attn.qkv(norm_p).reshape(B, total_tokens, 3, num_heads, head_dim)
                    qp, kp, _ = torch.unbind(qkv_p, 2)
                    qp = qp.transpose(1, 2)
                    kp = kp.transpose(1, 2)
                    A_pert = ((qp * scale) @ kp.transpose(-2, -1)).softmax(dim=-1)

                # 1. Frobenius norm diff
                diff_frob = (A_pert - A_clean).norm(dim=(-2, -1)).mean().item()
                head_diffs = (A_pert - A_clean).norm(dim=(-2, -1)).mean(dim=0) # (H,)
                max_head_diff = head_diffs.max().item()

                # 2. KL divergence: KL(A_clean || A_pert)
                kl = (A_clean * ((A_clean + 1e-12).log() - (A_pert + 1e-12).log())).sum(dim=-1).mean().item()

                # 3. Entropy change
                ent_pert = -(A_pert * (A_pert + 1e-12).log()).sum(dim=-1).mean().item()
                delta_ent = ent_pert - ent_clean

                # 4. Readout attention correlation
                w_p = extract_readout_attention(A_pert, model_key)
                readout_corr = torch.cosine_similarity(
                    w_c.flatten(start_dim=1), w_p.flatten(start_dim=1), dim=-1
                ).mean().item()

                records.append({
                    "model_key": model_key,
                    "depth": depth,
                    "token_pattern": t_name,
                    "feature_dir": f_name,
                    "scale_s": s,
                    "frobenius_norm_diff": diff_frob,
                    "kl_divergence": kl,
                    "entropy_clean": ent_clean,
                    "entropy_pert": ent_pert,
                    "delta_entropy": delta_ent,
                    "readout_corr": readout_corr,
                    "max_head_diff": max_head_diff
                })

    return records


# ==============================================================================
# 6. Block-by-Block Disturbance Propagation & Residual Decomposition
# ==============================================================================

def trace_blockwise_propagation_and_residuals(
    model: nn.Module,
    model_key: str,
    start_depth: int,
    images: torch.Tensor,
    clean_collected: Dict[int, torch.Tensor],
    feature_dirs: Dict[str, torch.Tensor],
    token_patterns: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_s: float = 1.0,
    device: torch.device = torch.device("cuda")
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Traces perturbation propagation through all downstream blocks:
    - blockwise_propagation: records stream norm, readout disturbance, cos sim, etc.
    - residual_path: decomposes block start_depth into attn branch, res in, post-attn, mlp branch.
    """
    records_prop = []
    records_res = []

    B = images.shape[0]
    total_blocks = len(model.blocks) if model_key in ("deit_tiny", "deit_small", "vit_base") else len(model.backbone.blocks)
    alpha = scale_s * sigma_P
    h_clean_start = clean_collected[start_depth]

    key_combos = [
        ("jac_top", "global_coherent"),
        ("jac_top", "random_sign"),
        ("jac_null", "global_coherent"),
        ("jac_null", "random_sign")
    ]

    for f_name, t_name in key_combos:
        v = feature_dirs[f_name].to(device)
        a = token_patterns[t_name].to(device)
        delta_P = alpha * torch.einsum('n,d->nd', a, v).unsqueeze(0).expand(B, -1, -1)
        h_pert = h_clean_start.clone()
        h_pert[:, 1:, :] += delta_P

        # Step 1: Trace residual breakdown in the first downstream block (blk_idx = start_depth)
        with torch.no_grad():
            h_next, A_p, attn_branch_p, mlp_branch_p, post_attn_p = run_decomposed_block(
                model, model_key, start_depth, h_clean_start, h_pert,
                q_src='pert', k_src='pert', v_src='pert', res_src='pert'
            )
            h_clean_next, A_c, attn_branch_c, mlp_branch_c, post_attn_c = run_decomposed_block(
                model, model_key, start_depth, h_clean_start, h_clean_start,
                q_src='clean', k_src='clean', v_src='clean', res_src='clean'
            )

            res_input_diff = (h_pert - h_clean_start).norm(dim=(-2, -1)).mean().item()
            attn_branch_diff = (attn_branch_p - attn_branch_c).norm(dim=(-2, -1)).mean().item()
            post_attn_diff = (post_attn_p - post_attn_c).norm(dim=(-2, -1)).mean().item()
            mlp_branch_diff = (mlp_branch_p - mlp_branch_c).norm(dim=(-2, -1)).mean().item()
            post_mlp_diff = (h_next - h_clean_next).norm(dim=(-2, -1)).mean().item()

            records_res.append({
                "model_key": model_key,
                "depth": start_depth,
                "token_pattern": t_name,
                "feature_dir": f_name,
                "scale_s": scale_s,
                "res_input_diff": res_input_diff,
                "attn_branch_diff": attn_branch_diff,
                "post_attn_diff": post_attn_diff,
                "mlp_branch_diff": mlp_branch_diff,
                "post_mlp_diff": post_mlp_diff
            })

        # Step 2: Trace downstream progression block by block
        curr_h_pert = h_next
        for b_idx in range(start_depth + 1, total_blocks + 1):
            depth_k = b_idx
            # If b_idx == start_depth + 1, curr_h_pert is already computed (output of block start_depth)
            if b_idx > start_depth + 1:
                with torch.no_grad():
                    curr_h_pert = get_block_module(model, model_key, b_idx - 1)(curr_h_pert)

            h_clean_k = clean_collected[depth_k]
            dp_stream = (curr_h_pert[:, 1:, :] - h_clean_k[:, 1:, :]).norm(dim=(-2, -1)).mean().item()
            z_clean = extract_readout_token(h_clean_k, model_key)
            z_pert = extract_readout_token(curr_h_pert, model_key)
            dz_readout = (z_pert - z_clean).norm(dim=-1).mean().item()
            cos_sim = torch.cosine_similarity(
                curr_h_pert.flatten(start_dim=1), h_clean_k.flatten(start_dim=1), dim=-1
            ).mean().item()

            records_prop.append({
                "model_key": model_key,
                "depth": start_depth,
                "block_idx": b_idx - 1,
                "downstream_step": b_idx - start_depth,
                "token_pattern": t_name,
                "feature_dir": f_name,
                "scale_s": scale_s,
                "dp_stream_norm": dp_stream,
                "dz_readout_norm": dz_readout,
                "cosine_sim": cos_sim
            })

    return records_prop, records_res


# ==============================================================================
# 7. Spatial Smoothness Analysis & 2D Fourier Response Map
# ==============================================================================

def analyze_spatial_smoothness_and_fourier_response(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor,
    h_clean_depth: torch.Tensor,
    feature_dirs: Dict[str, torch.Tensor],
    sigma_P: float,
    scale_s: float = 1.0,
    device: torch.device = torch.device("cuda")
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    1. Computes 2D Fourier power spectrum, low-frequency power ratio, total variation,
       and neighbor correlation of clean readout attention maps.
    2. Measures 2D Fourier spatial frequency response map H_l(k_x, k_y, v).
    """
    blk_idx = depth
    B, total_tokens, D = h_clean_depth.shape
    N = total_tokens - 1
    grid_side = int(math.isqrt(N))
    alpha = scale_s * sigma_P

    blk = get_block_module(model, model_key, blk_idx)
    norm_c = blk.norm1(h_clean_depth)

    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        num_heads = blk.attn.num_heads
        head_dim = blk.attn.head_dim
        scale = blk.attn.scale
        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        qc, kc, _ = qkv_c.unbind(0)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
    elif model_key == "dinov2":
        num_heads = blk.attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5
        qkv_c = blk.attn.qkv(norm_c).reshape(B, total_tokens, 3, num_heads, head_dim)
        qc, kc, _ = torch.unbind(qkv_c, 2)
        qc = qc.transpose(1, 2)
        kc = kc.transpose(1, 2)
        A_clean = ((qc * scale) @ kc.transpose(-2, -1)).softmax(dim=-1)
    else:
        raise ValueError(f"Unknown model: {model_key}")

    w_readout = extract_readout_attention(A_clean, model_key) # (B, H, N)
    w_grid = w_readout.view(B, num_heads, grid_side, grid_side)

    # 1. 2D FFT Power Spectrum
    fft_w = torch.fft.fft2(w_grid)
    power_spectrum = (fft_w.abs() ** 2) / (grid_side * grid_side) # (B, H, G, G)
    mean_power = power_spectrum.mean(dim=(0, 1))

    # Low-frequency quadrant: kx <= 2, ky <= 2
    low_freq_power = mean_power[:3, :3].sum().item()
    total_power = mean_power.sum().item() + 1e-10
    lfpr = low_freq_power / total_power

    # Total variation
    tv = ((w_grid[:, :, 1:, :] - w_grid[:, :, :-1, :]).abs().mean() +
          (w_grid[:, :, :, 1:] - w_grid[:, :, :, :-1]).abs().mean()).item()

    # Neighbor correlation
    flat_center_h = w_grid[:, :, :, :-1].flatten().cpu().numpy()
    flat_right_h = w_grid[:, :, :, 1:].flatten().cpu().numpy()
    neighbor_corr = float(np.corrcoef(flat_center_h, flat_right_h)[0, 1])

    smoothness_summary = {
        "model_key": model_key,
        "depth": depth,
        "low_frequency_power_ratio": lfpr,
        "total_variation": tv,
        "neighbor_correlation": neighbor_corr
    }

    # 2. 2D Cosine Spatial Frequency Response Map H_l(k_x, k_y, v)
    records_fourier = []
    clean_target_logits = clean_logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    u = torch.arange(grid_side, device=device).unsqueeze(1)
    v_idx = torch.arange(grid_side, device=device).unsqueeze(0)

    # Test spatial frequencies kx, ky in {0, 1, 2, 3, 4}
    for kx in range(5):
        for ky in range(5):
            mode_2d = torch.cos(math.pi * (u + 0.5) * kx / grid_side) * torch.cos(math.pi * (v_idx + 0.5) * ky / grid_side)
            a_mode = mode_2d.flatten()
            a_mode = a_mode / a_mode.norm()

            for f_name in ["jac_top", "pc1", "jac_null"]:
                v_feat = feature_dirs[f_name].to(device)
                delta_P = alpha * torch.einsum('n,d->nd', a_mode, v_feat).unsqueeze(0).expand(B, -1, -1)
                h_pert = h_clean_depth.clone()
                h_pert[:, 1:, :] += delta_P

                with torch.no_grad():
                    h_next, _, _, _, _ = run_decomposed_block(
                        model, model_key, blk_idx, h_clean_depth, h_pert,
                        q_src='pert', k_src='pert', v_src='pert', res_src='pert'
                    )
                    logits_pert = forward_from_block_output(model, model_key, h_next, blk_idx + 1)
                    pert_target = logits_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
                    damage = (clean_target_logits - pert_target).mean().item()

                    records_fourier.append({
                        "model_key": model_key,
                        "depth": depth,
                        "kx": kx,
                        "ky": ky,
                        "spatial_freq": math.sqrt(kx**2 + ky**2),
                        "feature_dir": f_name,
                        "scale_s": scale_s,
                        "damage": damage
                    })

    return smoothness_summary, records_fourier


# ==============================================================================
# 8. Publication Figures Generation
# ==============================================================================

def generate_causal_audit_figures(
    df_causal: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_prop: pd.DataFrame,
    df_shift: pd.DataFrame,
    df_rep: pd.DataFrame,
    df_fourier: Optional[pd.DataFrame],
    output_dir: str
):
    """Generates publication figures A through F."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ----------------------------------------------------
    # FIGURE A: Causal Path Decomposition
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))
    df_a = df_causal[
        (df_causal["feature_dir"] == "jac_top") &
        (df_causal["scale_s"] == 1.0) &
        (df_causal["condition"].isin([
            "full_perturbation",
            "frozen_attn_v_only_pert_res",
            "frozen_attn_v_only_clean_res",
            "reroute_clean_res"
        ]))
    ]
    if not df_a.empty:
        sns.barplot(
            data=df_a,
            x="token_pattern",
            y="logit_margin_drop",
            hue="condition",
            palette="Set2"
        )
        plt.title("Figure A: Causal Path Decomposition (True-Class Logit Drop along jac_top at s=1.0)")
        plt.xlabel("Token Pattern")
        plt.ylabel("Logit Margin Drop")
        plt.legend(title="Causal Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_a_causal_path_decomposition.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE B: Head-Wise Gamma Prediction
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    if not df_pred.empty:
        sns.scatterplot(
            data=df_pred,
            x="pred_norm",
            y="obs_norm",
            hue="token_pattern",
            style="feature_dir",
            s=80,
            alpha=0.9
        )
        max_val = max(df_pred["pred_norm"].max(), df_pred["obs_norm"].max()) * 1.1
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, label="Ideal 1:1 Parity")
        plt.title("Figure B: Linear Head Prediction vs Observed Head Disturbance")
        plt.xlabel(r"Predicted Value Disturbance $\|\Delta z_h^{pred}\|_2 = \alpha |\Gamma_h| \|v^T W_V^h\|$")
        plt.ylabel(r"Observed Disturbance $\|\Delta z_h^{obs}\|_2$ (Frozen Attention)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_b_headwise_gamma_prediction.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE C: Block-by-Block Disturbance Propagation
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    if not df_prop.empty:
        df_prop["condition_label"] = df_prop["feature_dir"] + " × " + df_prop["token_pattern"]
        sns.lineplot(
            data=df_prop,
            x="downstream_step",
            y="dz_readout_norm",
            hue="condition_label",
            marker="o",
            linewidth=2.5
        )
        plt.title("Figure C: Downstream Disturbance Propagation Across Blocks")
        plt.xlabel("Downstream Block Step After Intervention")
        plt.ylabel(r"Readout Disturbance Norm $\|\Delta z_b\|_2$")
        plt.legend(title="Intervention", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_c_blockwise_propagation.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE D: Attention Matrix Shift
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    if not df_shift.empty:
        df_shift_s1 = df_shift[df_shift["scale_s"] == 1.0]
        sns.barplot(
            data=df_shift_s1,
            x="token_pattern",
            y="frobenius_norm_diff",
            hue="feature_dir",
            palette="muted"
        )
        plt.title("Figure D: Attention Matrix Shift ||A_pert - A_clean||_F at s=1.0")
        plt.xlabel("Token Pattern")
        plt.ylabel("Frobenius Difference")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_d_attention_shift.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE E: Cross-Architecture Replication Summary
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))
    if not df_rep.empty:
        sns.barplot(
            data=df_rep,
            x="token_pattern",
            y="dz_readout_l1",
            hue="condition",
            palette="coolwarm"
        )
        plt.title("Figure E: Confirmatory Replication (Readout Disturbance at Block l+1)")
        plt.xlabel("Token Pattern")
        plt.ylabel("Readout Disturbance Norm")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_e_replication.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # FIGURE F: 2D Spatial Frequency Transfer Function (Optional)
    # ----------------------------------------------------
    if df_fourier is not None and not df_fourier.empty:
        plt.figure(figsize=(12, 4))
        fdirs = ["jac_top", "pc1", "jac_null"]
        for idx, fd in enumerate(fdirs, 1):
            sub = df_fourier[df_fourier["feature_dir"] == fd]
            if not sub.empty:
                pivot = sub.pivot(index="ky", columns="kx", values="damage")
                plt.subplot(1, 3, idx)
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar=(idx == 3))
                plt.title(f"H(kx, ky) for {fd}")
                plt.xlabel("kx (Horizontal freq)")
                plt.ylabel("ky (Vertical freq)")
        plt.suptitle("Figure F: 2D Spatial Frequency Response of the Downstream Transformer", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure_f_spatial_frequency_response.png"), dpi=300)
        plt.close()

    print(f"Generated all figures in {output_dir}")
