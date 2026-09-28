from typing import List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F


def apply_norm_matched_random_sphere(
    h: torch.Tensor,
    seq_mask: List[int],
    seed: int
) -> torch.Tensor:
    """
    Norm-Matched Random Sphere:
    Sample r ~ N(0, I), normalize u = r / ||r||, set h'_t = ||h_t|| * u.
    Exactly preserves original token L2 norm with random direction.
    """
    B, T, D = h.shape
    K = len(seq_mask)
    
    # Deterministic PyTorch generator for reproducibility
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(B, K, D, generator=gen, dtype=torch.float32).to(h.device)
    u = r / torch.norm(r, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    
    orig_tokens = h[:, seq_mask, :]
    orig_norms = torch.norm(orig_tokens, p=2, dim=-1, keepdim=True)
    
    h_out = h.clone()
    h_out[:, seq_mask, :] = (orig_norms * u).to(h.dtype)
    
    # Verify norm matching within 1e-5 relative error
    new_norms = torch.norm(h_out[:, seq_mask, :], p=2, dim=-1, keepdim=True)
    rel_error = torch.abs(new_norms - orig_norms) / (orig_norms + 1e-8)
    assert rel_error.max().item() < 1e-5, f"Norm matching failed: max rel error = {rel_error.max().item()}"
    
    return h_out


def apply_layer_statistics_gaussian(
    h: torch.Tensor,
    seq_mask: List[int],
    mu_d: torch.Tensor,
    sigma_d: torch.Tensor,
    seed: int
) -> torch.Tensor:
    """
    Layer-Statistics Gaussian:
    Sample r_d ~ N(mu_d, sigma_d^2) for each feature dimension d.
    Matches empirical first- and second-order activation moments.
    """
    B, T, D = h.shape
    K = len(seq_mask)
    
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(B, K, D, generator=gen, dtype=torch.float32).to(h.device)
    
    mu_exp = mu_d.view(1, 1, D).to(h.device)
    sigma_exp = sigma_d.view(1, 1, D).to(h.device)
    r = mu_exp + sigma_exp * eps
    
    h_out = h.clone()
    h_out[:, seq_mask, :] = r.to(h.dtype)
    return h_out


def apply_norm_matched_layer_mean(
    h: torch.Tensor,
    seq_mask: List[int],
    unmasked_seq: List[int]
) -> torch.Tensor:
    """
    Norm-Matched Same-Image Layer Mean:
    mu_i = mean of unmasked spatial patch tokens from SAME image.
    h'_t = ||h_t|| * mu_i / ||mu_i||.
    Preserves direction of intra-image mean while scaling to exact token L2 norm.
    """
    orig_tokens = h[:, seq_mask, :]
    orig_norms = torch.norm(orig_tokens, p=2, dim=-1, keepdim=True) # (B, K, 1)
    
    mu_i = h[:, unmasked_seq, :].mean(dim=1, keepdim=True) # (B, 1, D)
    mu_norm = torch.norm(mu_i, p=2, dim=-1, keepdim=True).clamp(min=1e-8) # (B, 1, 1)
    u_mu = mu_i / mu_norm # (B, 1, D)
    
    h_out = h.clone()
    h_out[:, seq_mask, :] = orig_norms * u_mu
    
    new_norms = torch.norm(h_out[:, seq_mask, :], p=2, dim=-1, keepdim=True)
    rel_error = torch.abs(new_norms - orig_norms) / (orig_norms + 1e-8)
    assert rel_error.max().item() < 1e-5, f"Norm matching failed: max rel error = {rel_error.max().item()}"
    
    return h_out


def apply_feature_shuffled_original(
    h: torch.Tensor,
    seq_mask: List[int],
    perm_d: List[int]
) -> torch.Tensor:
    """
    Feature-Shuffled Original:
    h'_t[d] = h_t[pi(d)].
    Permutes embedding dimensions using a fixed random permutation pi.
    Preserves token norm, mean, variance, and scalar multiset while destroying learned feature geometry.
    """
    D = h.shape[-1]
    assert len(perm_d) == D, "Permutation length must match feature dimension D"
    assert sorted(perm_d) == list(range(D)), "Permutation must be a valid bijection of 0..D-1"
    
    perm_tensor = torch.tensor(perm_d, dtype=torch.long, device=h.device)
    
    h_out = h.clone()
    h_out[:, seq_mask, :] = torch.index_select(h[:, seq_mask, :], dim=-1, index=perm_tensor)
    
    # Assert exact norm preservation
    orig_norms = torch.norm(h[:, seq_mask, :], p=2, dim=-1)
    new_norms = torch.norm(h_out[:, seq_mask, :], p=2, dim=-1)
    assert torch.allclose(orig_norms, new_norms, atol=1e-5), "Feature shuffle must preserve L2 norm exactly!"
    
    return h_out


def apply_norm_matched_cross_image(
    h: torch.Tensor,
    h_donor: torch.Tensor,
    seq_mask: List[int]
) -> torch.Tensor:
    """
    Norm-Matched Cross-Image Token:
    h'_t = ||h_t|| * c_t / ||c_t||.
    Scales foreign donor token to exact target token L2 norm.
    """
    orig_tokens = h[:, seq_mask, :]
    orig_norms = torch.norm(orig_tokens, p=2, dim=-1, keepdim=True)
    
    donor_tokens = h_donor[:, seq_mask, :]
    donor_norms = torch.norm(donor_tokens, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    u_donor = donor_tokens / donor_norms
    
    h_out = h.clone()
    h_out[:, seq_mask, :] = orig_norms * u_donor
    
    new_norms = torch.norm(h_out[:, seq_mask, :], p=2, dim=-1, keepdim=True)
    rel_error = torch.abs(new_norms - orig_norms) / (orig_norms + 1e-8)
    assert rel_error.max().item() < 1e-5, f"Norm matching failed: max rel error = {rel_error.max().item()}"
    
    return h_out


def compute_distribution_diagnostics(
    h_orig: torch.Tensor,
    h_mod: torch.Tensor,
    seq_mask: List[int],
    unmasked_seq: List[int],
    mu_layer: torch.Tensor,
    sigma_layer: torch.Tensor
) -> Dict[str, float]:
    """
    Computes activation magnitude and distributional diagnostics for the replacement:
    - mean & std token L2 norm
    - cosine to original token
    - cosine to same-image layer mean
    - Euclidean distance to layer centroid mu_layer
    - Diagonal Mahalanobis distance D_Mahal
    """
    with torch.no_grad():
        orig_tokens = h_orig[:, seq_mask, :] # (B, K, D)
        mod_tokens = h_mod[:, seq_mask, :]   # (B, K, D)
        
        orig_norms = torch.norm(orig_tokens, p=2, dim=-1) # (B, K)
        mod_norms = torch.norm(mod_tokens, p=2, dim=-1)   # (B, K)
        
        # Cosine to original token
        dot_orig = (orig_tokens * mod_tokens).sum(dim=-1)
        denom_orig = orig_norms * mod_norms
        cos_to_orig = torch.where(denom_orig > 1e-8, dot_orig / denom_orig, torch.zeros_like(dot_orig))
        
        # Cosine to same-image layer mean
        mu_i = h_orig[:, unmasked_seq, :].mean(dim=1, keepdim=True) # (B, 1, D)
        mu_i_norm = torch.norm(mu_i, p=2, dim=-1) # (B, 1)
        dot_mu = (mod_tokens * mu_i).sum(dim=-1) # (B, K)
        denom_mu = mod_norms * mu_i_norm # (B, K)
        cos_to_mu = torch.where(denom_mu > 1e-8, dot_mu / denom_mu, torch.zeros_like(dot_mu))
        
        # Distance to layer centroid
        mu_exp = mu_layer.view(1, 1, -1).to(h_mod.device)
        dist_to_centroid = torch.norm(mod_tokens - mu_exp, p=2, dim=-1) # (B, K)
        
        # Diagonal Mahalanobis distance
        sigma_exp = sigma_layer.view(1, 1, -1).to(h_mod.device)
        z_scores = (mod_tokens - mu_exp) / (sigma_exp + 1e-6)
        mahal_dist = (z_scores ** 2).sum(dim=-1) # (B, K)

        return {
            "mean_token_l2": float(mod_norms.mean().item()),
            "std_token_l2": float(mod_norms.std().item()),
            "mean_cosine_to_orig": float(cos_to_orig.mean().item()),
            "mean_cosine_to_mean": float(cos_to_mu.mean().item()),
            "mean_dist_to_centroid": float(dist_to_centroid.mean().item()),
            "mean_mahalanobis_dist": float(mahal_dist.mean().item())
        }
