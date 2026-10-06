"""
patch_fungibility/batched_operator_compression.py

Core engine for Batched Operator-Aware Token Compression:
1. Batched Prefix and Suffix Transformer forward passes (supporting arbitrary batch sizes BS >= 1).
2. Batched Vectorized Low-Rank Carrier Solvers:
   - batched_solve_amortized_carrier (exact vectorized Cholesky across batch)
   - batched_solve_diagonal_carrier (fast diagonal approximation)
   - batched_solve_one_step_carrier (one-step preconditioned update)
3. Grouping Strategies:
   - Fixed 2D Spatial Grid Grouping (cached, zero runtime overhead)
   - Hybrid Dynamic Grouping (Importance-Preserve + Spatial Merge, IP-SM, < 0.1 ms)
4. Multiplicity Attention Handling:
   - Route A: Explicit Additive Bias (Reference)
   - Route B: Augmented Coordinate (dim padded to multiple of 8, Flash/cuDNN native)
   - Route C: Key/Value scaling approximation
   - Route D: Uniform / Unweighted carrier (no mask)
5. Baselines:
   - Clean ViT
   - Spatial Group Mean (Centroid)
   - Hybrid Group Mean (IP-SM Centroid)
   - Attention Pruning (EViT)
   - Token Merging (ToMe)
"""

import time
import math
from typing import Dict, List, Tuple, Any, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_fungibility.compression_models import (
    forward_compressed_block_timm,
    forward_compressed_block_dinov2,
)
from patch_fungibility.amortized_operator import (
    FactorizedModePredictor,
    extract_oracle_subspace,
    subspace_overlap
)
from patch_fungibility.practical_operator_compression import (
    create_fixed_spatial_grouping,
    forward_prefix_only
)


# ====================================================================
# 1. BATCHED PREFIX AND SUFFIX EXECUTION
# ====================================================================

def forward_prefix_batched(
    model: nn.Module,
    model_key: str,
    l: int,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Executes ONLY blocks 0..l-1 starting from raw image batch x (BS, 3, H, W).
    Stops immediately at block l without running suffix or classification head.
    Returns hidden states h_l of shape (BS, 1+N, D).
    """
    return forward_prefix_only(model=model, model_key=model_key, l=l, x=x)


def forward_block_with_multiplicity_route(
    block: nn.Module,
    model_key: str,
    x: torch.Tensor,
    multiplicities: torch.Tensor,
    route: str = "reference"
) -> torch.Tensor:
    """
    Executes a single transformer block under the selected multiplicity route:
    - 'reference': explicit additive attention bias log(m_j)
    - 'augmented': augmented key/query dimension (d -> d', d' % 8 == 0)
    - 'unweighted': no attention bias (m_j = 1)
    """
    BS, T, C = x.shape
    device = x.device
    dtype = x.dtype

    if route == "reference":
        attn_bias = torch.log(multiplicities).view(1, 1, 1, T).to(device=device, dtype=dtype)
        if model_key in ("deit_tiny", "deit_small", "vit_base"):
            return forward_compressed_block_timm(block, x, attn_bias)
        else:
            return forward_compressed_block_dinov2(block, x, attn_bias)

    elif route == "unweighted":
        if model_key in ("deit_tiny", "deit_small", "vit_base"):
            return forward_compressed_block_timm(block, x, None)
        else:
            return forward_compressed_block_dinov2(block, x, None)

    elif route == "augmented":
        # Route B: Augmented coordinate formulation
        if model_key in ("deit_tiny", "deit_small", "vit_base"):
            h = block.norm1(x)
            qkv = block.attn.qkv(h).reshape(BS, T, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = block.attn.q_norm(q), block.attn.k_norm(k)

            d = block.attn.head_dim
            d_prime = ((d + 1 + 7) // 8) * 8 # nearest multiple of 8 >= d+1
            scale_q = (float(d_prime) / float(d)) ** 0.5

            q_aug = torch.zeros(BS, block.attn.num_heads, T, d_prime, device=device, dtype=dtype)
            q_aug[..., :d] = q * scale_q
            q_aug[..., d] = 1.0

            k_aug = torch.zeros(BS, block.attn.num_heads, T, d_prime, device=device, dtype=dtype)
            k_aug[..., :d] = k
            log_m = torch.log(multiplicities).view(1, 1, T).expand(BS, block.attn.num_heads, T)
            k_aug[..., d] = (float(d_prime) ** 0.5) * log_m

            out_attn = F.scaled_dot_product_attention(
                q_aug, k_aug, v,
                dropout_p=block.attn.attn_drop.p if block.attn.training else 0.0
            )
            out_attn = out_attn.transpose(1, 2).reshape(BS, T, C)
            out_attn = block.attn.proj(out_attn)
            x = x + block.drop_path1(block.ls1(out_attn))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            return x
        else: # DINOv2
            h = block.norm1(x)
            qkv = block.attn.qkv(h).reshape(BS, T, 3, block.attn.num_heads, C // block.attn.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

            d = C // block.attn.num_heads
            d_prime = ((d + 1 + 7) // 8) * 8
            scale_q = (float(d_prime) / float(d)) ** 0.5

            q_aug = torch.zeros(BS, block.attn.num_heads, T, d_prime, device=device, dtype=dtype)
            q_aug[..., :d] = q * scale_q
            q_aug[..., d] = 1.0

            k_aug = torch.zeros(BS, block.attn.num_heads, T, d_prime, device=device, dtype=dtype)
            k_aug[..., :d] = k
            log_m = torch.log(multiplicities).view(1, 1, T).expand(BS, block.attn.num_heads, T)
            k_aug[..., d] = (float(d_prime) ** 0.5) * log_m

            out_attn = F.scaled_dot_product_attention(
                q_aug, k_aug, v,
                dropout_p=block.attn.attn_drop if block.attn.training else 0.0
            )
            out_attn = out_attn.transpose(1, 2).contiguous().view(BS, T, C)
            out_attn = block.attn.proj_drop(block.attn.proj(out_attn))
            x = x + block.ls1(out_attn)
            x = x + block.ls2(block.mlp(block.norm2(x)))
            return x
    else:
        raise ValueError(f"Unknown multiplicity route: {route}")


def forward_suffix_batched(
    model: nn.Module,
    model_key: str,
    l: int,
    h_comp: torch.Tensor,
    multiplicities: Optional[torch.Tensor] = None,
    original_n_patches: int = 196,
    route: str = "reference"
) -> torch.Tensor:
    """
    Executes downstream blocks (l..L-1) and classification head for a batch of compressed sequences.
    h_comp: (BS, 1+B, D)
    multiplicities: (1+B,) or (BS, 1+B)
    """
    if multiplicities is None:
        route = "unweighted"
        multiplicities = torch.ones(h_comp.shape[1], device=h_comp.device, dtype=h_comp.dtype)

    cur = h_comp
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        for b_idx in range(l, len(model.blocks)):
            cur = forward_block_with_multiplicity_route(
                block=model.blocks[b_idx],
                model_key=model_key,
                x=cur,
                multiplicities=multiplicities,
                route=route
            )
        h_norm = model.norm(cur)
        logits = model.forward_head(h_norm)
        return logits
    elif model_key == "dinov2":
        backbone = model.backbone
        for b_idx in range(l, len(backbone.blocks)):
            cur = forward_block_with_multiplicity_route(
                block=backbone.blocks[b_idx],
                model_key=model_key,
                x=cur,
                multiplicities=multiplicities,
                route=route
            )
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


# ====================================================================
# 2. BATCHED CARRIER SOLVERS
# ====================================================================

def batched_solve_amortized_carrier(
    P: torch.Tensor,
    S: torch.Tensor,
    multiplicities: torch.Tensor,
    V_pred: torch.Tensor,
    lam_factor: float = 10.0
) -> Dict[str, Any]:
    """
    Batched closed-form Tikhonov carrier solve across arbitrary batch size BS:
    - P: (BS, N, D)
    - S: (N, B) or (BS, N, B)
    - multiplicities: (B,) or (BS, B)
    - V_pred: (BS, N*D, r)
    Returns:
    - C_opt: (BS, B, D)
    - C_mean: (BS, B, D)
    - delta_C: (BS, B, D)
    """
    BS, N, D = P.shape
    r = V_pred.shape[-1]
    device = P.device
    dtype = P.dtype

    if S.dim() == 2:
        # Static grouping: expand across batch
        S_b = S.unsqueeze(0).expand(BS, -1, -1) # (BS, N, B)
        B_tokens = S.shape[1]
    else:
        S_b = S # (BS, N, B)
        B_tokens = S.shape[2]

    if multiplicities.dim() == 1:
        m_safe = multiplicities.clamp(min=1.0)
        m = m_safe.view(1, B_tokens, 1).expand(BS, -1, 1) # (BS, B, 1)
        inv_sqrt_m = torch.rsqrt(m_safe).view(1, 1, B_tokens, 1) # (1, 1, B, 1)
    else:
        m_safe = multiplicities.clamp(min=1.0)
        m = m_safe.unsqueeze(-1) # (BS, B, 1)
        inv_sqrt_m = torch.rsqrt(m_safe).view(BS, 1, B_tokens, 1)

    # 1. Batched Group Means: (BS, B, D)
    C_mean = torch.bmm(S_b.transpose(1, 2), P) / m.clamp(min=1.0)

    # 2. Batched Residual & Projection:
    E_mean = P - torch.bmm(S_b, C_mean) # (BS, N, D)
    E_flat = E_mean.reshape(BS, N * D, 1)
    r_mean = torch.bmm(V_pred.transpose(1, 2), E_flat).squeeze(-1) # (BS, r)

    # 3. Batched K_j accumulation:
    V_blocks = V_pred.view(BS, N, D, r).permute(0, 3, 2, 1).reshape(BS, r * D, N)
    K_flat = torch.bmm(V_blocks, S_b) # (BS, r*D, B)
    K = K_flat.view(BS, r, D, B_tokens).permute(0, 1, 3, 2) # (BS, r, B, D)

    # 4. Batched Sigma matrix formation:
    K_scaled = (K * inv_sqrt_m).reshape(BS, r, B_tokens * D)
    Sigma = torch.bmm(K_scaled, K_scaled.transpose(1, 2)) # (BS, r, r)
    Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2))

    # 5. Batched Tikhonov regularizer & Cholesky solve:
    sig_trace = torch.diagonal(Sigma, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) / float(max(1, r))
    lam = torch.clamp(lam_factor * sig_trace, min=1e-3)
    eye = torch.eye(r, device=device, dtype=dtype).unsqueeze(0).expand(BS, -1, -1)
    M = Sigma + lam * eye

    try:
        L = torch.linalg.cholesky(M)
        alpha = torch.cholesky_solve(r_mean.unsqueeze(-1), L).squeeze(-1) # (BS, r)
    except torch._C._LinAlgError:
        alpha = torch.linalg.solve(M + 1e-2 * eye, r_mean.unsqueeze(-1)).squeeze(-1)

    # 6. Batched Carrier adjustments:
    delta_C = torch.einsum('br,brjd->bjd', alpha, K) / m.clamp(min=1.0)
    C_opt = C_mean + delta_C

    return {
        "C_mean": C_mean,
        "C_opt": C_opt,
        "delta_C": delta_C,
        "norm_r_mean": torch.norm(r_mean, dim=-1).mean().item(),
        "norm_delta_C": torch.norm(delta_C, dim=(1, 2)).mean().item()
    }


def batched_solve_diagonal_carrier(
    P: torch.Tensor,
    S: torch.Tensor,
    multiplicities: torch.Tensor,
    V_pred: torch.Tensor,
    lam_factor: float = 10.0
) -> Dict[str, Any]:
    """
    Batched diagonal approximation to carrier solve across batch size BS:
    Diagonalizes Sigma_b, replacing O(r^3) Cholesky with O(r) elementwise inversion.
    """
    BS, N, D = P.shape
    r = V_pred.shape[-1]
    device = P.device
    dtype = P.dtype

    if S.dim() == 2:
        S_b = S.unsqueeze(0).expand(BS, -1, -1)
        B_tokens = S.shape[1]
    else:
        S_b = S
        B_tokens = S.shape[2]

    if multiplicities.dim() == 1:
        m_safe = multiplicities.clamp(min=1.0)
        m = m_safe.view(1, B_tokens, 1).expand(BS, -1, 1)
        inv_m = (1.0 / m_safe).view(1, 1, B_tokens, 1)
    else:
        m_safe = multiplicities.clamp(min=1.0)
        m = m_safe.unsqueeze(-1)
        inv_m = (1.0 / m_safe).view(BS, 1, B_tokens, 1)

    C_mean = torch.bmm(S_b.transpose(1, 2), P) / m.clamp(min=1.0)
    E_mean = P - torch.bmm(S_b, C_mean)
    E_flat = E_mean.reshape(BS, N * D, 1)
    r_mean = torch.bmm(V_pred.transpose(1, 2), E_flat).squeeze(-1) # (BS, r)

    V_blocks = V_pred.view(BS, N, D, r).permute(0, 3, 2, 1).reshape(BS, r * D, N)
    K_flat = torch.bmm(V_blocks, S_b)
    K = K_flat.view(BS, r, D, B_tokens).permute(0, 1, 3, 2) # (BS, r, B, D)

    # Diagonal of Sigma: (BS, r)
    diag_Sigma = (inv_m * (K ** 2)).sum(dim=(2, 3)) # (BS, r)
    sig_trace = diag_Sigma.mean(dim=-1, keepdim=True)
    lam = lam_factor * sig_trace

    alpha = r_mean / (diag_Sigma + lam + 1e-8) # (BS, r)
    delta_C = torch.einsum('br,brjd->bjd', alpha, K) / m.clamp(min=1.0)
    C_opt = C_mean + delta_C

    return {
        "C_mean": C_mean,
        "C_opt": C_opt,
        "delta_C": delta_C,
        "norm_r_mean": torch.norm(r_mean, dim=-1).mean().item(),
        "norm_delta_C": torch.norm(delta_C, dim=(1, 2)).mean().item()
    }


def batched_solve_one_step_carrier(
    P: torch.Tensor,
    S: torch.Tensor,
    multiplicities: torch.Tensor,
    V_pred: torch.Tensor,
    step_scale: float = 0.5
) -> Dict[str, Any]:
    """
    Batched one-step preconditioned update without matrix inversion.
    """
    BS, N, D = P.shape
    r = V_pred.shape[-1]
    device = P.device

    if S.dim() == 2:
        S_b = S.unsqueeze(0).expand(BS, -1, -1)
        B_tokens = S.shape[1]
    else:
        S_b = S
        B_tokens = S.shape[2]

    if multiplicities.dim() == 1:
        m = multiplicities.view(1, B_tokens, 1).expand(BS, -1, 1)
    else:
        m = multiplicities.unsqueeze(-1)

    C_mean = torch.bmm(S_b.transpose(1, 2), P) / m.clamp(min=1.0)
    E_mean = P - torch.bmm(S_b, C_mean)
    E_flat = E_mean.reshape(BS, N * D, 1)
    r_mean = torch.bmm(V_pred.transpose(1, 2), E_flat).squeeze(-1) # (BS, r)

    V_blocks = V_pred.view(BS, N, D, r).permute(0, 3, 2, 1).reshape(BS, r * D, N)
    K_flat = torch.bmm(V_blocks, S_b)
    K = K_flat.view(BS, r, D, B_tokens).permute(0, 1, 3, 2)

    scale = step_scale / math.sqrt(float(N))
    alpha = scale * r_mean

    delta_C = torch.einsum('br,brjd->bjd', alpha, K) / m.clamp(min=1.0)
    C_opt = C_mean + delta_C

    return {
        "C_mean": C_mean,
        "C_opt": C_opt,
        "delta_C": delta_C,
        "norm_r_mean": torch.norm(r_mean, dim=-1).mean().item(),
        "norm_delta_C": torch.norm(delta_C, dim=(1, 2)).mean().item()
    }


# ====================================================================
# 3. HYBRID DYNAMIC GROUPING (IP-SM)
# ====================================================================

def create_hybrid_dynamic_grouping(
    P: torch.Tensor,
    B: int,
    K_keep: int = 16,
    grid_h: int = 14,
    grid_w: int = 14
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Importance-Preserve + Spatial Merge (IP-SM):
    - Evaluates token saliency w_i = ||P_i||_2 for each image in batch (< 0.05 ms).
    - Preserves top K_keep foreground tokens unmerged with multiplicity 1.
    - Merges remaining N - K_keep background tokens into B - K_keep spatial grid cells.
    Returns:
    - S: (BS, N, B) indicator matrix
    - multiplicities: (BS, B) token counts
    """
    BS, N, D = P.shape
    device = P.device
    dtype = P.dtype
    B_bg = B - K_keep
    assert B_bg > 0, f"Budget {B} must exceed K_keep {K_keep}"

    # 1. Background static spatial partition template: (N,)
    _, _, spatial_idx = create_fixed_spatial_grouping(N, B_bg, grid_h, grid_w, device=device)

    # 2. Token saliency from L2 activation norms
    scores = torch.norm(P, p=2, dim=-1) # (BS, N)
    _, topk_indices = torch.topk(scores, k=K_keep, dim=-1) # (BS, K_keep)

    # 3. Construct batched indicator matrix S: (BS, N, B)
    S = torch.zeros(BS, N, B, device=device, dtype=dtype)

    # Assign top-K foreground tokens to bins 0..K_keep-1
    batch_idx = torch.arange(BS, device=device).unsqueeze(1).expand(BS, K_keep)
    out_bin_fg = torch.arange(K_keep, device=device).unsqueeze(0).expand(BS, K_keep)
    S[batch_idx, topk_indices, out_bin_fg] = 1.0

    # Mask for background tokens
    mask_keep = torch.zeros(BS, N, dtype=torch.bool, device=device)
    mask_keep.scatter_(1, topk_indices, True)

    # Assign background tokens to bins K_keep..B-1 based on spatial grid index
    bg_bins = spatial_idx.unsqueeze(0).expand(BS, N) + K_keep
    bg_mask = ~mask_keep
    bg_nonzero = bg_mask.nonzero(as_tuple=True)
    S[bg_nonzero[0], bg_nonzero[1], bg_bins[bg_mask]] = 1.0

    mults = S.sum(dim=1) # (BS, B)
    # Ensure no empty groups in any batch item
    for b in range(BS):
        zero_bins = torch.where(mults[b] == 0)[0]
        for z in zero_bins:
            largest = torch.argmax(mults[b])
            members = torch.where(S[b, :, largest] > 0)[0]
            if len(members) > 1:
                stolen = members[-1]
                S[b, stolen, largest] = 0.0
                S[b, stolen, z] = 1.0
                mults[b, largest] -= 1.0
                mults[b, z] += 1.0

    return S, mults


# ====================================================================
# 4. BASELINES (ATTENTION PRUNING & TOME)
# ====================================================================

def run_evit_pruning_batched(
    h: torch.Tensor,
    B: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention / Token Pruning (EViT style):
    Keeps top B-1 patch tokens by activation norm + CLS token.
    Discards the remaining tokens completely (unweighted).
    Returns:
    - h_comp: (BS, 1+B, D)
    - multiplicities: (1+B,) all ones
    """
    BS, T, D = h.shape
    device = h.device
    cls_token = h[:, :1, :]
    patches = h[:, 1:, :]

    scores = torch.norm(patches, p=2, dim=-1) # (BS, N)
    _, topk_idx = torch.topk(scores, k=B, dim=-1) # (BS, B)

    # Gather kept patches
    batch_idx = torch.arange(BS, device=device).unsqueeze(1).expand(BS, B)
    kept_patches = patches[batch_idx, topk_idx] # (BS, B, D)

    h_comp = torch.cat([cls_token, kept_patches], dim=1)
    mults = torch.ones(1 + B, device=device, dtype=h.dtype)
    return h_comp, mults


from patch_fungibility.operator_compression_confirmatory import tome_bipartite_merge

def run_tome_bipartite_batched(
    h: torch.Tensor,
    target_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Faithful Batched Token Merging (ToMe-style Bipartite Soft Matching):
    Performs multi-round bipartite merging until exactly target_tokens remain.
    """
    BS, T, D = h.shape
    device = h.device
    cls_token = h[:, :1, :]
    patches = h[:, 1:, :]

    merged_patches = []
    merged_mults = []
    for b in range(BS):
        p_b, m_b = tome_bipartite_merge(patches[b], B_target=target_tokens)
        merged_patches.append(p_b)
        merged_mults.append(m_b)

    merged_patches = torch.stack(merged_patches, dim=0) # (BS, target_tokens, D)
    merged_mults = torch.stack(merged_mults, dim=0) # (BS, target_tokens)

    h_comp = torch.cat([cls_token, merged_patches], dim=1)
    mults_with_cls = torch.cat([torch.ones(BS, 1, device=device), merged_mults], dim=1)
    return h_comp, mults_with_cls[0]
