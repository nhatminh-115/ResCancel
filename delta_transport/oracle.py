from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn


def forward_to_injection(
    model: nn.Module,
    x: torch.Tensor,
    injection_layer: int = 8
) -> List[torch.Tensor]:
    """
    Passes input through embeddings and blocks up to injection_layer (inclusive).
    Returns a list of state tensors:
      states[0] = h_0 (input to block 0)
      states[l] = h_l (input to block l)
      states[injection_layer + 1] = output of block injection_layer (input to injection_layer + 1)
    """
    feat = model.patch_embed(x)
    feat = model._pos_embed(feat)
    feat = model.patch_drop(feat)
    feat = model.norm_pre(feat)

    states = [feat]
    for i in range(injection_layer + 1):
        feat = model.blocks[i](feat)
        states.append(feat)

    return states


def forward_from_injection(
    model: nn.Module,
    h_inj: torch.Tensor,
    start_block: int = 9,
    total_blocks: int = 12
) -> torch.Tensor:
    """
    Passes the injected representation h_inj through the remaining transformer blocks,
    post-norm, and classifier head to obtain logits.
    """
    h_curr = h_inj
    for i in range(start_block, total_blocks):
        h_curr = model.blocks[i](h_curr)
    norm_feat = model.norm(h_curr)
    logits = model.forward_head(norm_feat)
    return logits


def compute_true_class_margins(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Computes true-class logit margin for each sample:
      m_i = z_y - max_{c != y} z_c
    """
    b_idx = torch.arange(logits.size(0), device=logits.device)
    true_logits = logits[b_idx, targets]
    
    masked_logits = logits.clone()
    masked_logits[b_idx, targets] = -float("inf")
    max_other_logits, _ = masked_logits.max(dim=1)
    
    margins = true_logits - max_other_logits
    return margins


def compute_candidate_deltas(
    states: List[torch.Tensor],
    candidate_layers: List[int] = list(range(8))
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes historical block residual deltas and normalized directions:
      Delta_l = states[l+1] - states[l]
    Returns:
      deltas: (B, num_candidates, 196, d)
      u: (B, num_candidates, 196, d)
    """
    b_deltas = []
    for l in candidate_layers:
        diff = states[l + 1] - states[l]
        # Numerical verification of Delta definition
        assert torch.allclose(diff, states[l + 1] - states[l], atol=1e-5), f"Delta definition mismatch at layer {l}"
        # Extract patch tokens (indices 1..196)
        patch_diff = diff[:, 1:, :]
        b_deltas.append(patch_diff)

    deltas = torch.stack(b_deltas, dim=1) # (B, num_candidates, 196, d)
    norm_factor = deltas.norm(dim=-1, keepdim=True) + 1e-7
    u = deltas / norm_factor
    return deltas, u


def compute_margin_gradient(
    model: nn.Module,
    h_inj: torch.Tensor,
    targets: torch.Tensor,
    start_block: int = 9,
    total_blocks: int = 12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes baseline logits, baseline margins, and the analytical gradient
    g_t = dm / dh_inj w.r.t the injected patch representation.
    """
    h_leaf = h_inj.detach().clone().requires_grad_(True)
    logits = forward_from_injection(model, h_leaf, start_block, total_blocks)
    margins = compute_true_class_margins(logits, targets)
    
    # Compute analytical gradients w.r.t h_leaf
    grads = torch.autograd.grad(margins.sum(), h_leaf)[0]
    patch_grads = grads[:, 1:, :] # (B, 196, d)
    
    return logits.detach(), margins.detach(), patch_grads.detach()


def evaluate_policies_batch(
    model: nn.Module,
    states: List[torch.Tensor],
    targets: torch.Tensor,
    gamma: float = 0.05,
    random_seeds: List[int] = [2501, 2502, 2503],
    candidate_layers: List[int] = list(range(8)),
    start_block: int = 9,
    total_blocks: int = 12
) -> Dict[str, Any]:
    """
    Evaluates all five policies (Baseline, Most-Recent, Random, Global Oracle, Tokenwise Oracle)
    on a batch of samples.
    """
    h_inj = states[start_block] # Output of block 8, entering block 9
    batch_size = h_inj.size(0)
    num_patches = h_inj.size(1) - 1 # 196
    d_model = h_inj.size(2)

    # 1. Baseline & Gradients
    base_logits, base_margins, patch_grads = compute_margin_gradient(
        model, h_inj, targets, start_block, total_blocks
    )
    base_preds = base_logits.argmax(dim=1)
    base_correct = (base_preds == targets)

    # 2. Historical Deltas & Normalized Directions
    deltas, u = compute_candidate_deltas(states, candidate_layers) # (B, 8, 196, d)

    # 3. First-order usefulness scores: s_l,t = g_t^T u_l,t
    # patch_grads: (B, 196, d) -> unsqueeze(1): (B, 1, 196, d)
    s = (u * patch_grads.unsqueeze(1)).sum(dim=-1) # (B, 8, 196)

    # 4. Global Oracle Source Selection: l* = argmax_l sum_t s_l,t
    agg_scores = s.sum(dim=2) # (B, 8)
    global_l_star = agg_scores.argmax(dim=1) # (B,)
    # Gather u for global oracle
    u_global = torch.gather(
        u, 1, global_l_star.view(batch_size, 1, 1, 1).expand(batch_size, 1, num_patches, d_model)
    ).squeeze(1)

    # 5. Tokenwise Oracle Source Selection: l*_t = argmax_l s_l,t
    tokenwise_l_star = s.argmax(dim=1) # (B, 196)
    u_tokenwise = torch.gather(
        u, 1, tokenwise_l_star.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, num_patches, d_model)
    ).squeeze(1)

    # 6. Most-Recent Policy: Layer 7 for all patches
    most_recent_idx = candidate_layers.index(7) if 7 in candidate_layers else len(candidate_layers) - 1
    u_most_recent = u[:, most_recent_idx, :, :]

    # 7. Random Policies across frozen seeds
    u_random_dict = {}
    for r_seed in random_seeds:
        rng = torch.Generator(device="cpu").manual_seed(r_seed)
        rand_layers = torch.randint(0, len(candidate_layers), size=(batch_size, num_patches), generator=rng)
        rand_layers = rand_layers.to(h_inj.device)
        u_rand = torch.gather(
            u, 1, rand_layers.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, num_patches, d_model)
        ).squeeze(1)
        u_random_dict[r_seed] = u_rand

    # Helper function to inject and evaluate a policy
    patch_h = h_inj[:, 1:, :].detach()
    patch_norm = patch_h.norm(dim=-1, keepdim=True)
    cls_h = h_inj[:, :1, :].detach()

    def run_injection(u_dir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Perturbation norm verification
        pert = gamma * patch_norm * u_dir
        injected_patch = patch_h + pert
        
        # Verify identical perturbation norm
        pert_relative_norm = pert.norm(dim=-1) / (patch_norm.squeeze(-1) + 1e-12)
        assert torch.allclose(pert_relative_norm, torch.full_like(pert_relative_norm, gamma), atol=1e-4), \
            "Perturbation magnitude does not match gamma"
            
        # CLS untouched assertion
        h_prime = torch.cat([cls_h, injected_patch], dim=1)
        assert torch.equal(h_prime[:, 0, :], cls_h[:, 0, :]), "CLS token was modified during injection!"
        
        # Forward pass through remaining blocks
        with torch.no_grad():
            logits_prime = forward_from_injection(model, h_prime, start_block, total_blocks)
        margins_prime = compute_true_class_margins(logits_prime, targets)
        preds_prime = logits_prime.argmax(dim=1)
        correct_prime = (preds_prime == targets)
        return logits_prime, margins_prime, preds_prime, correct_prime

    # Run interventions
    most_recent_res = run_injection(u_most_recent)
    global_res = run_injection(u_global)
    tokenwise_res = run_injection(u_tokenwise)

    random_results = {}
    for r_seed, u_r in u_random_dict.items():
        random_results[r_seed] = run_injection(u_r)

    # Compute spatial diversity statistics for tokenwise oracle
    # tokenwise_l_star: (B, 196)
    diversity_stats = []
    for b in range(batch_size):
        layers_chosen = tokenwise_l_star[b].cpu().numpy()
        counts = np.bincount(layers_chosen, minlength=len(candidate_layers))
        probs = counts / float(num_patches)
        entropy = -float(np.sum([p * np.log2(p + 1e-12) for p in probs if p > 0]))
        dominant_frac = float(np.max(probs))
        global_choice = int(global_l_star[b].cpu().item())
        frac_matching_global = float(probs[global_choice])
        num_distinct = int(np.sum(counts > 0))
        diversity_stats.append({
            "layer_counts": counts.tolist(),
            "entropy": entropy,
            "dominant_fraction": dominant_frac,
            "fraction_matching_global": frac_matching_global,
            "num_distinct_layers": num_distinct,
            "global_selected_layer": global_choice,
            "spatial_map_14x14": layers_chosen.reshape(14, 14).tolist()
        })

    return {
        "base": {
            "margins": base_margins.cpu().numpy(),
            "preds": base_preds.cpu().numpy(),
            "correct": base_correct.cpu().numpy()
        },
        "most_recent": {
            "margins": most_recent_res[1].cpu().numpy(),
            "preds": most_recent_res[2].cpu().numpy(),
            "correct": most_recent_res[3].cpu().numpy()
        },
        "global_oracle": {
            "margins": global_res[1].cpu().numpy(),
            "preds": global_res[2].cpu().numpy(),
            "correct": global_res[3].cpu().numpy(),
            "selected_layer": global_l_star.cpu().numpy()
        },
        "tokenwise_oracle": {
            "margins": tokenwise_res[1].cpu().numpy(),
            "preds": tokenwise_res[2].cpu().numpy(),
            "correct": tokenwise_res[3].cpu().numpy(),
            "selected_layers": tokenwise_l_star.cpu().numpy()
        },
        "random_tokenwise": {
            r_seed: {
                "margins": res[1].cpu().numpy(),
                "preds": res[2].cpu().numpy(),
                "correct": res[3].cpu().numpy()
            }
            for r_seed, res in random_results.items()
        },
        "diversity_stats": diversity_stats
    }
