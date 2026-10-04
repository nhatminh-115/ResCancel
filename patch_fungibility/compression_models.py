"""
patch_fungibility/compression_models.py

Multiplicity-aware attention blocks, compressed downstream paths,
and full end-to-end forward functions for:
- DeiT-Tiny
- DeiT-Small
- ViT-B/16 AugReg
- DINOv2 ViT-S/14
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_compressed_block_timm(
    block: nn.Module,
    x: torch.Tensor,
    attn_bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Executes a single timm Vision Transformer block with multiplicity-aware attention bias.
    attn_bias: (1, 1, 1, T_comp) or None
    """
    h = block.norm1(x)
    B, N, C = h.shape
    qkv = block.attn.qkv(h).reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = block.attn.q_norm(q), block.attn.k_norm(k)

    # Scaled dot-product attention with log-multiplicity bias on keys
    out_attn = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_bias,
        dropout_p=block.attn.attn_drop.p if block.attn.training else 0.0
    )
    out_attn = out_attn.transpose(1, 2).reshape(B, N, C)
    out_attn = block.attn.proj(out_attn)

    # Residual 1 + LayerScale
    x = x + block.drop_path1(block.ls1(out_attn))

    # Norm 2 + MLP + Residual 2
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    return x


def forward_compressed_block_dinov2(
    block: nn.Module,
    x: torch.Tensor,
    attn_bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Executes a single DINOv2 block with multiplicity-aware attention bias.
    """
    h = block.norm1(x)
    B, N, C = h.shape
    qkv = block.attn.qkv(h).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

    out_attn = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_bias,
        dropout_p=block.attn.attn_drop if block.attn.training else 0.0
    )
    out_attn = out_attn.transpose(1, 2).contiguous().view(B, N, C)
    out_attn = block.attn.proj_drop(block.attn.proj(out_attn))

    # Residual 1 + LayerScale
    x = x + block.ls1(out_attn)

    # Norm 2 + MLP (SwiGLU) + Residual 2
    x = x + block.ls2(block.mlp(block.norm2(x)))
    return x


def forward_downstream_compressed(
    model: nn.Module,
    model_key: str,
    start_depth: int,
    h_comp: torch.Tensor,
    multiplicities: torch.Tensor,
    original_n_patches: int
) -> torch.Tensor:
    """
    Executes downstream blocks (from start_depth to final block) on compressed sequence.
    """
    device = h_comp.device
    T_comp = h_comp.size(1)

    # Build attention bias from multiplicities: (1, 1, 1, T_comp)
    # log(s_j) added to attention logits
    attn_bias = torch.log(multiplicities).view(1, 1, 1, T_comp).to(device)

    cur = h_comp
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        for b_idx in range(start_depth, len(model.blocks)):
            cur = forward_compressed_block_timm(model.blocks[b_idx], cur, attn_bias)

        h_norm = model.norm(cur)
        logits = model.forward_head(h_norm)
        return logits

    elif model_key == "dinov2":
        backbone = model.backbone
        for b_idx in range(start_depth, len(backbone.blocks)):
            cur = forward_compressed_block_dinov2(backbone.blocks[b_idx], cur, attn_bias)

        h_norm = backbone.norm(cur)
        cls_norm = h_norm[:, 0]

        patch_mults = multiplicities[1:].view(1, -1, 1)
        total_m = float(multiplicities[1:].sum().item())
        if abs(total_m - original_n_patches) < 1e-2:
            patch_mean = torch.sum(h_norm[:, 1:] * patch_mults, dim=1) / float(original_n_patches)
        else:
            patch_mean = h_norm[:, 1:].mean(dim=1)

        readout = torch.cat([cls_norm, patch_mean], dim=-1)
        logits = model.linear_head(readout)
        return logits
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def forward_downstream_reference(
    model: nn.Module,
    model_key: str,
    start_depth: int,
    h_ref: torch.Tensor
) -> torch.Tensor:
    """
    Executes downstream blocks on full uncompressed sequence.
    """
    cur = h_ref
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        for b_idx in range(start_depth, len(model.blocks)):
            cur = model.blocks[b_idx](cur)

        h_norm = model.norm(cur)
        logits = model.forward_head(h_norm)
        return logits

    elif model_key == "dinov2":
        backbone = model.backbone
        for b_idx in range(start_depth, len(backbone.blocks)):
            cur = backbone.blocks[b_idx](cur)

        h_norm = backbone.norm(cur)
        cls_norm = h_norm[:, 0]
        patch_mean = h_norm[:, 1:].mean(dim=1)
        readout = torch.cat([cls_norm, patch_mean], dim=-1)
        logits = model.linear_head(readout)
        return logits
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def forward_end_to_end_clean(
    model: nn.Module,
    model_key: str,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Executes standard uncompressed forward pass from image to logits.
    """
    return model(x)


def forward_end_to_end_compressed(
    model: nn.Module,
    model_key: str,
    x: torch.Tensor,
    start_depth: int,
    mask: torch.Tensor,
    condition: str,
    mu: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Executes complete end-to-end forward pass from raw image to logits,
    transitioning at block `start_depth` into a compressed sequence:
    - Blocks 0..start_depth-1 run on full sequence.
    - At start_depth, patches are partitioned into M real anchors and m discarded.
    - Discarded patches are represented via:
      - 'WEIGHTED_CENTROID_CARRIER': 1 carrier with mu, size m
      - 'UNWEIGHTED_CENTROID': 1 carrier with mu, size 1
      - 'IMAGE_MEAN_CARRIER': 1 carrier with mean(discarded), size m
      - 'RANDOM_PRUNING': 0 carriers, keep B = M + 1 real patches
    """
    device = x.device
    B_batch = x.size(0)

    # 1. Early blocks (0..start_depth-1)
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        h = model.patch_embed(x)
        h = model._pos_embed(h)
        h = model.patch_drop(h)
        h = model.norm_pre(h)
        for b_idx in range(start_depth):
            h = model.blocks[b_idx](h)
    elif model_key == "dinov2":
        h = model.backbone.prepare_tokens_with_masks(x)
        for b_idx in range(start_depth):
            h = model.backbone.blocks[b_idx](h)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    # 2. Sequence compression at output of block start_depth
    cls_token = h[:, :1, :]
    patches = h[:, 1:, :]
    n_patches = patches.size(1)

    m_discarded = int(mask.sum().item())
    M_real = n_patches - m_discarded

    if m_discarded == 0:
        # 0% replacement: no compression needed
        multiplicities = torch.ones(1 + n_patches, device=device)
        h_comp = h
        return forward_downstream_compressed(model, model_key, start_depth, h_comp, multiplicities, n_patches)

    # Extract real anchors
    real_patches = patches[:, ~mask, :]  # (B, M_real, D)

    if condition == "WEIGHTED_CENTROID_CARRIER":
        carrier = mu.unsqueeze(0).unsqueeze(0).expand(B_batch, 1, -1)  # (B, 1, D)
        h_comp = torch.cat([cls_token, real_patches, carrier], dim=1)
        multiplicities = torch.cat([
            torch.ones(1 + M_real, device=device),
            torch.tensor([float(m_discarded)], device=device)
        ])

    elif condition == "UNWEIGHTED_CENTROID":
        carrier = mu.unsqueeze(0).unsqueeze(0).expand(B_batch, 1, -1)
        h_comp = torch.cat([cls_token, real_patches, carrier], dim=1)
        multiplicities = torch.ones(2 + M_real, device=device)

    elif condition == "IMAGE_MEAN_CARRIER":
        discarded_patches = patches[:, mask, :]
        mean_carrier = discarded_patches.mean(dim=1, keepdim=True)  # (B, 1, D)
        h_comp = torch.cat([cls_token, real_patches, mean_carrier], dim=1)
        multiplicities = torch.cat([
            torch.ones(1 + M_real, device=device),
            torch.tensor([float(m_discarded)], device=device)
        ])

    elif condition == "RANDOM_PRUNING":
        # Keep B = M_real + 1 patches (or M_real if m_discarded == n_patches)
        B_target = min(M_real + 1, n_patches)
        if B_target > M_real:
            # Pick 1 patch from discarded to match token budget B
            extra_idx = torch.where(mask)[0][0]
            extra_patch = patches[:, extra_idx : extra_idx + 1, :]
            pruned_patches = torch.cat([real_patches, extra_patch], dim=1)
        else:
            pruned_patches = real_patches

        h_comp = torch.cat([cls_token, pruned_patches], dim=1)
        multiplicities = torch.ones(h_comp.size(1), device=device)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    # 3. Downstream blocks (start_depth..end)
    return forward_downstream_compressed(model, model_key, start_depth, h_comp, multiplicities, n_patches)
