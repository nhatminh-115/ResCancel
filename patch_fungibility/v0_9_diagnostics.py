from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_per_image_representation_ranks(h_patch: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Computes representation rank metrics strictly per image for spatial patch representations H_patch (B, 196, D).
    Returns dictionary of arrays of shape (B,):
      - numerical_rank: count of singular values with s_i / s_1 > 1e-3
      - stable_rank: ||H_c||_F^2 / ||H_c||_2^2
      - effective_rank: (sum s_i^2)^2 / sum s_i^4 (participation ratio)
      - patch_variance: (1/196) * sum(s_i^2)
      - mean_pairwise_cosine: average off-diagonal cosine similarity across 196 patches
    """
    B, N, D = h_patch.shape
    # Center across spatial tokens per image: H_c = H - mean_token(H)
    h_mean = h_patch.mean(dim=1, keepdim=True)
    h_c = h_patch - h_mean

    # Compute singular values per image: shape (B, min(N, D))
    s = torch.linalg.svdvals(h_c)  # sorted descending
    s1 = s[:, 0].clamp(min=1e-12)

    # 1. Numerical rank: s_i / s_1 > 1e-3
    rel_s = s / s1.unsqueeze(-1)
    num_rank = (rel_s > 1e-3).sum(dim=-1).float()  # (B,)

    # 2. Stable rank: sum(s_i^2) / s_1^2
    fro_sq = (s ** 2).sum(dim=-1)  # (B,)
    stable_rank = fro_sq / (s1 ** 2)

    # 3. Participation ratio (effective rank): (sum s_i^2)^2 / sum(s_i^4)
    sum_s4 = (s ** 4).sum(dim=-1).clamp(min=1e-12)
    eff_rank = (fro_sq ** 2) / sum_s4

    # 4. Patch variance per token: fro_sq / N
    patch_var = fro_sq / float(N)

    # 5. Mean pairwise cosine similarity
    norm_h = F.normalize(h_patch, p=2, dim=-1)
    # Cosine Gram matrix: (B, N, N)
    gram = torch.bmm(norm_h, norm_h.transpose(1, 2))
    # Mask diagonal
    diag_mask = torch.eye(N, device=h_patch.device, dtype=torch.bool).unsqueeze(0)
    off_diag = gram.masked_select(~diag_mask).view(B, N * (N - 1))
    mean_cosine = off_diag.mean(dim=-1)  # (B,)

    return {
        "numerical_rank": num_rank.cpu().numpy(),
        "stable_rank": stable_rank.cpu().numpy(),
        "effective_rank": eff_rank.cpu().numpy(),
        "patch_variance": patch_var.cpu().numpy(),
        "mean_pairwise_cosine": mean_cosine.cpu().numpy()
    }


def forward_block_with_v0_9_diagnostics(
    block: nn.Module,
    x: torch.Tensor,
    compute_attn_svd: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes a standard ViT block forward pass while capturing exact attention diagnostics:
      - spatial patch Key variance across 196 tokens
      - spatial patch Value variance across 196 tokens
      - CLS-to-patch attention entropy
      - attention matrix stable rank (optional for efficiency)
    """
    B, N, C = x.shape
    x_norm = block.norm1(x)
    attn_mod = block.attn

    num_heads = attn_mod.num_heads
    head_dim = attn_mod.head_dim
    scale = attn_mod.scale

    # Compute Q, K, V
    qkv = attn_mod.qkv(x_norm).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # Shape: (B, num_heads, N, head_dim)
    q = attn_mod.q_norm(q)
    k = attn_mod.k_norm(k)

    # Spatial patch tokens only (index 1..196)
    k_patch = k[:, :, 1:, :]  # (B, H, 196, head_dim)
    v_patch = v[:, :, 1:, :]

    # Key and Value variance across 196 patch tokens
    key_var = k_patch.var(dim=2, unbiased=True).mean().item()
    val_var = v_patch.var(dim=2, unbiased=True).mean().item()

    # Softmax Attention weights: A = softmax(scale * Q @ K^T)
    # Shape: (B, H, N, N)
    attn_scores = (q * scale) @ k.transpose(-2, -1)
    attn_weights = attn_scores.softmax(dim=-1)

    # CLS-to-patch attention distribution: row 0, columns 1..196
    cls_patch_attn = attn_weights[:, :, 0, 1:]  # (B, H, 196)
    eps = 1e-9
    cls_entropy = -(cls_patch_attn * torch.log(cls_patch_attn + eps)).sum(dim=-1).mean().item()

    if compute_attn_svd:
        # Singular values of attention matrix
        attn_bh = attn_weights.reshape(-1, N, N)
        s_attn = torch.linalg.svdvals(attn_bh)
        s1_attn = s_attn[:, 0].clamp(min=1e-12)
        fro_sq_attn = (s_attn ** 2).sum(dim=-1)
        attn_stable_rank = (fro_sq_attn / (s1_attn ** 2)).mean().item()
    else:
        attn_stable_rank = 1.0

    # Complete attention computation
    out_attn = (attn_weights @ v).transpose(1, 2).reshape(B, N, attn_mod.attn_dim)
    out_attn = attn_mod.norm(out_attn)
    out_attn = attn_mod.proj(out_attn)
    out_attn = attn_mod.proj_drop(out_attn)

    # Residual 1 + MLP
    x = x + out_attn
    x = x + block.mlp(block.norm2(x))

    diag = {
        "key_variance": float(key_var),
        "value_variance": float(val_var),
        "cls_attention_entropy": float(cls_entropy),
        "attention_stable_rank": float(attn_stable_rank)
    }
    return x, diag


def compute_geometric_expansion_diagnostics(
    h_mod_8: torch.Tensor,
    h_11: torch.Tensor,
    v_1: torch.Tensor,
    z_raw: torch.Tensor
) -> Dict[str, float]:
    """
    For Natural PCA Rank 1:
    - h_mod_8: (B, 197, D) injection
    - h_11: (B, 197, D) output of Block 11
    - v_1: (D,) original leading PC
    - z_raw: (B, 196) scalar input coefficient for spatial tokens
    Measures:
      - pc1_correlation: correlation between z_raw and projection of h_11 onto v_1
      - pc1_variance_fraction: fraction of spatial patch variance at Block 11 aligned with v_1
      - orthogonal_variance_fraction: 1.0 - pc1_variance_fraction
    """
    # Spatial tokens at Block 11: (B, 196, D)
    h_patch = h_11[:, 1:, :]
    h_mean = h_patch.mean(dim=1, keepdim=True)
    h_c = h_patch - h_mean  # Centered

    v_1_dev = v_1.to(h_patch.device, dtype=h_patch.dtype)

    # Projection onto original PC1: (B, 196)
    proj_pc1 = torch.matmul(h_c, v_1_dev)

    # Compute correlation per sample between z_raw and proj_pc1
    z_mean = z_raw.mean(dim=-1, keepdim=True)
    z_c = z_raw - z_mean
    p_mean = proj_pc1.mean(dim=-1, keepdim=True)
    p_c = proj_pc1 - p_mean

    cov_zp = (z_c * p_c).mean(dim=-1)
    std_z = z_c.std(dim=-1, unbiased=True).clamp(min=1e-8)
    std_p = p_c.std(dim=-1, unbiased=True).clamp(min=1e-8)
    corr = (cov_zp / (std_z * std_p)).abs().mean().item()

    # Total spatial variance at Block 11
    total_var = (h_c ** 2).sum(dim=-1).mean().item()  # average sum of variances
    var_pc1 = (p_c ** 2).mean().item()

    pc1_var_frac = float(var_pc1 / max(total_var, 1e-12))
    ortho_var_frac = float(max(0.0, 1.0 - pc1_var_frac))

    return {
        "pc1_correlation": float(corr),
        "pc1_variance_fraction": pc1_var_frac,
        "orthogonal_variance_fraction": ortho_var_frac
    }
