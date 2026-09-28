from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

from delta_transport.oracle import forward_from_injection, compute_true_class_margins


def generate_cross_image_derangement(n_samples: int = 1000, seed: int = 3501) -> np.ndarray:
    """
    Generates a deterministic derangement of range(n_samples) where perm[i] != i for all i.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    for i in range(n_samples):
        if perm[i] == i:
            swap_idx = (i + 1) % n_samples
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]
    assert (perm != np.arange(n_samples)).all(), "Derangement verification failed: donor image matches target image!"
    return perm


def generate_random_oracle_candidates(
    batch_size: int,
    num_candidates: int,
    num_patches: int,
    embed_dim: int,
    seed: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generates num_candidates L2-normalized random vectors in R^embed_dim for each patch independently.
    Shape: (B, num_candidates, num_patches, embed_dim)
    """
    # Deterministic generation using PyTorch Generator on CPU, then move to device
    rng = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(batch_size, num_candidates, num_patches, embed_dim, generator=rng)
    r = r.to(device)
    norm = r.norm(dim=-1, keepdim=True) + 1e-7
    u_rand = r / norm
    return u_rand


def generate_spatial_shuffle_candidates(
    u_hist: torch.Tensor,
    seed: int
) -> torch.Tensor:
    """
    Randomly permutes the 196 patch positions independently for each candidate layer.
    Shape: (B, num_candidates, num_patches, embed_dim)
    """
    batch_size, num_candidates, num_patches, embed_dim = u_hist.shape
    rng = np.random.RandomState(seed)
    u_shuff = torch.empty_like(u_hist)

    for l in range(num_candidates):
        perm = rng.permutation(num_patches)
        # Verify valid permutation
        assert len(np.unique(perm)) == num_patches, "Spatial permutation is not bijective!"
        u_shuff[:, l, :, :] = u_hist[:, l, perm, :]

    return u_shuff


def generate_pooled_delta_candidates(
    u_hist: torch.Tensor,
    seed: int
) -> torch.Tensor:
    """
    Constructs 8 candidates per patch by uniformly sampling real historical deltas from the pool
    of all (num_candidates * num_patches) deltas belonging to that image.
    Shape: (B, num_candidates, num_patches, embed_dim)
    """
    batch_size, num_candidates, num_patches, embed_dim = u_hist.shape
    rng = np.random.RandomState(seed)
    total_pool_size = num_candidates * num_patches

    # Flatten spatial and layer dimensions: (B, 8*196, d)
    flat_pool = u_hist.view(batch_size, total_pool_size, embed_dim)
    u_pooled = torch.empty(batch_size, num_candidates, num_patches, embed_dim, device=u_hist.device, dtype=u_hist.dtype)

    for b in range(batch_size):
        # Sample 8 candidates for each of the 196 patches: shape (196, 8)
        sampled_indices = [rng.choice(total_pool_size, size=num_candidates, replace=False) for _ in range(num_patches)]
        sampled_indices = np.array(sampled_indices).T # (8, 196)
        for k in range(num_candidates):
            u_pooled[b, k, :, :] = flat_pool[b, sampled_indices[k], :]

    return u_pooled


def evaluate_matched_oracle_condition(
    model: nn.Module,
    h_inj: torch.Tensor,
    patch_grads: torch.Tensor,
    patch_norm: torch.Tensor,
    cls_h: torch.Tensor,
    candidate_u: torch.Tensor,
    gamma: float,
    start_block: int,
    total_blocks: int,
    targets: torch.Tensor
) -> Dict[str, Any]:
    """
    Applies the exact same label-aware argmax oracle rule over candidate_u:
      s_k,t = g_t^T u_k,t
      k*_t = argmax_k s_k,t
      h'_t = h_t + gamma * ||h_t|| * u_{k*_t, t}
    CLS token untouched.
    """
    batch_size, num_candidates, num_patches, embed_dim = candidate_u.shape

    # 1. Candidate vector normalization assertion
    cand_norms = candidate_u.norm(dim=-1)
    assert torch.allclose(cand_norms, torch.ones_like(cand_norms), atol=1e-3), "Candidate vectors are not L2-normalized!"

    # 2. First-order usefulness score: s_k,t = g_t^T u_k,t
    # patch_grads: (B, 196, d) -> unsqueeze(1): (B, 1, 196, d)
    s = (candidate_u * patch_grads.unsqueeze(1)).sum(dim=-1) # (B, num_candidates, 196)

    # 3. Independent per-patch selection: k*_t = argmax_k s_k,t
    selected_k = s.argmax(dim=1) # (B, 196)

    # 4. Gather selected candidate direction
    u_selected = torch.gather(
        candidate_u, 1, selected_k.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, num_patches, embed_dim)
    ).squeeze(1) # (B, 196, d)

    # 5. Injection with strictly matched perturbation budget
    patch_h = h_inj[:, 1:, :].detach()
    pert = gamma * patch_norm * u_selected
    injected_patch = patch_h + pert

    # Verify perturbation norm
    rel_norm = pert.norm(dim=-1) / (patch_norm.squeeze(-1) + 1e-12)
    assert torch.allclose(rel_norm, torch.full_like(rel_norm, gamma), atol=1e-4), "Perturbation norm mismatch!"

    # Construct injected state and verify CLS is untouched
    h_prime = torch.cat([cls_h, injected_patch], dim=1)
    assert torch.equal(h_prime[:, 0, :], cls_h[:, 0, :]), "CLS token was modified during injection!"

    # 6. Downstream forward pass through remaining blocks
    with torch.no_grad():
        logits_prime = forward_from_injection(model, h_prime, start_block, total_blocks)
    margins_prime = compute_true_class_margins(logits_prime, targets)
    preds_prime = logits_prime.argmax(dim=1)
    correct_prime = (preds_prime == targets)

    # 7. Spatial diversity metrics for this condition
    diversity_stats = []
    for b in range(batch_size):
        chosen_indices = selected_k[b].cpu().numpy()
        counts = np.bincount(chosen_indices, minlength=num_candidates)
        probs = counts / float(num_patches)
        entropy = -float(np.sum([p * np.log2(p + 1e-12) for p in probs if p > 0]))
        dominant_frac = float(np.max(probs))
        num_distinct = int(np.sum(counts > 0))
        diversity_stats.append({
            "candidate_counts": counts.tolist(),
            "entropy": entropy,
            "dominant_fraction": dominant_frac,
            "num_distinct_candidates": num_distinct
        })

    return {
        "margins": margins_prime.cpu().numpy(),
        "preds": preds_prime.cpu().numpy(),
        "correct": correct_prime.cpu().numpy(),
        "selected_indices": selected_k.cpu().numpy(),
        "diversity_stats": diversity_stats
    }
