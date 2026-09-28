from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_representation_ranks(h_patch: torch.Tensor) -> Dict[str, float]:
    """
    Computes representation rank metrics for spatial patch representations H_patch (B, 196, D).
    Returns batch-averaged metrics:
      - numerical_rank: count of singular values with s_i / s_1 > 1e-3
      - stable_rank: ||H_c||_F^2 / ||H_c||_2^2
      - effective_rank: (sum s_i^2)^2 / sum s_i^4 (participation ratio)
      - mean_pairwise_cosine: average off-diagonal cosine similarity across 196 patches
    """
    B, N, D = h_patch.shape
    # Center across spatial tokens: H_c = H - mean_token(H)
    h_mean = h_patch.mean(dim=1, keepdim=True)
    h_c = h_patch - h_mean

    # Compute singular values per sample: shape (B, min(N, D))
    s = torch.linalg.svdvals(h_c)  # s is sorted descending
    s1 = s[:, 0].clamp(min=1e-12)

    # 1. Numerical rank: s_i / s_1 > 1e-3
    rel_s = s / s1.unsqueeze(-1)
    num_rank = (rel_s > 1e-3).sum(dim=-1).float().mean().item()

    # 2. Stable rank: sum(s_i^2) / s_1^2
    fro_sq = (s ** 2).sum(dim=-1)
    stable_rank = (fro_sq / (s1 ** 2)).mean().item()

    # 3. Participation ratio (effective rank): (sum s_i^2)^2 / sum(s_i^4)
    eff_rank = ((fro_sq ** 2) / (s ** 4).sum(dim=-1).clamp(min=1e-12)).mean().item()

    # 4. Mean pairwise cosine similarity
    norm_h = F.normalize(h_patch, p=2, dim=-1)
    # Cosine gram matrix: (B, N, N)
    gram = torch.bmm(norm_h, norm_h.transpose(1, 2))
    # Mask diagonal
    diag_mask = torch.eye(N, device=h_patch.device, dtype=torch.bool).unsqueeze(0)
    off_diag = gram.masked_select(~diag_mask).view(B, N * (N - 1))
    mean_cosine = off_diag.mean().item()

    return {
        "numerical_rank": float(num_rank),
        "stable_rank": float(stable_rank),
        "effective_rank": float(eff_rank),
        "mean_pairwise_cosine": float(mean_cosine)
    }


def forward_block_with_diagnostics(
    block: nn.Module,
    x: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes a standard ViT block forward pass while capturing exact attention diagnostics:
      - spatial patch Key variance across 196 tokens
      - spatial patch Value variance across 196 tokens
      - CLS-to-patch attention entropy
      - attention matrix stable rank
    Returns:
      (output_tensor, diagnostics_dict)
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

    # Attention matrix stable rank per head
    # Flatten B and H: (B*H, N, N)
    attn_bh = attn_weights.reshape(-1, N, N)
    # Singular values of attention matrix
    s_attn = torch.linalg.svdvals(attn_bh)
    s1_attn = s_attn[:, 0].clamp(min=1e-12)
    fro_sq_attn = (s_attn ** 2).sum(dim=-1)
    attn_stable_rank = (fro_sq_attn / (s1_attn ** 2)).mean().item()

    # Complete attention computation
    out_attn = (attn_weights @ v).transpose(1, 2).reshape(B, N, attn_mod.attn_dim)
    out_attn = attn_mod.norm(out_attn)
    out_attn = attn_mod.proj(out_attn)
    out_attn = attn_mod.proj_drop(out_attn)

    # Residual 1
    x = x + out_attn
    # MLP
    x = x + block.mlp(block.norm2(x))

    diag = {
        "key_variance": float(key_var),
        "value_variance": float(val_var),
        "cls_attention_entropy": float(cls_entropy),
        "attention_stable_rank": float(attn_stable_rank)
    }
    return x, diag
