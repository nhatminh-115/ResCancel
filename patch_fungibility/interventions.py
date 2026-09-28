from typing import List, Dict, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np


def apply_zero(
    h: torch.Tensor,
    seq_mask: List[int]
) -> torch.Tensor:
    """
    Zero ablation: replaces the 49 masked tokens with 0.
    h: (B, 197, D)
    """
    h_out = h.clone()
    h_out[:, seq_mask, :] = 0.0
    return h_out


def apply_layer_mean(
    h: torch.Tensor,
    seq_mask: List[int],
    unmasked_seq: List[int]
) -> torch.Tensor:
    """
    Layer mean replacement: replaces the 49 masked tokens with the mean of the
    147 unmasked spatial patch tokens from the SAME image.
    h: (B, 197, D)
    """
    h_out = h.clone()
    # Compute mean across unmasked spatial tokens for each image in batch: shape (B, 1, D)
    mu_l = h[:, unmasked_seq, :].mean(dim=1, keepdim=True)
    h_out[:, seq_mask, :] = mu_l
    return h_out


def apply_cross_image_same_pos(
    h: torch.Tensor,
    h_donor: torch.Tensor,
    seq_mask: List[int]
) -> torch.Tensor:
    """
    Cross-image same-position replacement:
    replaces h_{l,t}(i) with h_{l,t}(j) from donor image j != i.
    h: (B, 197, D)
    h_donor: (B, 197, D)
    """
    h_out = h.clone()
    h_out[:, seq_mask, :] = h_donor[:, seq_mask, :]
    return h_out


def apply_cross_image_rand_pos(
    h: torch.Tensor,
    h_donor: torch.Tensor,
    seq_mask: List[int],
    shuffled_donor_positions: List[int]
) -> torch.Tensor:
    """
    Cross-image random-position replacement:
    replaces h_{l,t}(i) with h_{l,t'}(j) from donor image j != i where t' is a permuted position.
    h: (B, 197, D)
    h_donor: (B, 197, D)
    """
    h_out = h.clone()
    h_out[:, seq_mask, :] = h_donor[:, shuffled_donor_positions, :]
    return h_out


def apply_within_image_shuffle(
    h: torch.Tensor,
    seq_mask: List[int],
    within_shuffled_positions: List[int]
) -> torch.Tensor:
    """
    Within-image spatial shuffle:
    permutes the 49 masked tokens within the SAME image.
    h: (B, 197, D)
    """
    h_out = h.clone()
    h_out[:, seq_mask, :] = h[:, within_shuffled_positions, :]
    return h_out


def compute_intervention_norms(
    h_orig: torch.Tensor,
    h_mod: torch.Tensor,
    seq_mask: List[int]
) -> Dict[str, float]:
    """
    Computes activation magnitude and distributional diagnostics for the intervention:
    - mean L2 norm of the replaced tokens
    - median L2 norm of the replaced tokens
    - std L2 norm of the replaced tokens
    - mean cosine similarity to original tokens at those slots
    - whole-sequence mean L2 norm per token
    """
    with torch.no_grad():
        orig_tokens = h_orig[:, seq_mask, :] # (B, 49, D)
        mod_tokens = h_mod[:, seq_mask, :]   # (B, 49, D)
        
        orig_norms = torch.norm(orig_tokens, p=2, dim=-1) # (B, 49)
        mod_norms = torch.norm(mod_tokens, p=2, dim=-1)   # (B, 49)
        
        # Cosine similarity per token
        # Clamp denominator to avoid division by zero
        dot_prod = (orig_tokens * mod_tokens).sum(dim=-1)
        denom = orig_norms * mod_norms
        cos_sim = torch.where(denom > 1e-8, dot_prod / denom, torch.zeros_like(dot_prod))
        
        # Whole sequence norm
        seq_norm_orig = torch.norm(h_orig, p=2, dim=-1).mean()
        seq_norm_mod = torch.norm(h_mod, p=2, dim=-1).mean()

        return {
            "mean_token_l2": float(mod_norms.mean().item()),
            "std_token_l2": float(mod_norms.std().item()),
            "median_token_l2": float(mod_norms.median().item()),
            "orig_mean_token_l2": float(orig_norms.mean().item()),
            "mean_cosine_to_orig": float(cos_sim.mean().item()),
            "whole_seq_norm": float(seq_norm_mod.item()),
            "orig_whole_seq_norm": float(seq_norm_orig.item())
        }
