import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, NamedTuple


class ResidualMetrics(NamedTuple):
    cos_sim: torch.Tensor       # Shape: [B, N]
    r_mag: torch.Tensor         # Shape: [B, N]
    q_contract: torch.Tensor    # Shape: [B, N]
    c_l1: torch.Tensor          # Shape: [B, N]
    x_norm: torch.Tensor        # Shape: [B, N]
    delta_norm: torch.Tensor    # Shape: [B, N]


def compute_residual_metrics(
    x: torch.Tensor,
    delta: torch.Tensor,
    eps: float = 1e-8
) -> ResidualMetrics:
    """
    Computes residual cancellation and geometry metrics for tensors x and delta.
    
    Args:
        x: Input tensor to residual addition, shape [B, N, D] or [N, D]
        delta: Residual update tensor, shape [B, N, D] or [N, D]
        eps: Small constant for numerical stability
        
    Returns:
        ResidualMetrics namedtuple containing:
            cos_sim: Geometric cosine similarity cos(x, delta)
            r_mag: Relative update magnitude ||delta||_2 / ||x||_2
            q_contract: Residual contraction ||x + delta||_2 / ||x||_2
            c_l1: Coordinate cancellation ratio (||x||_1 + ||delta||_1) / ||x + delta||_1
            x_norm: ||x||_2
            delta_norm: ||delta||_2
    """
    # Norms along the embedding dimension D (last dim)
    x_l2 = torch.linalg.norm(x, ord=2, dim=-1)
    delta_l2 = torch.linalg.norm(delta, ord=2, dim=-1)
    new_x = x + delta
    new_x_l2 = torch.linalg.norm(new_x, ord=2, dim=-1)

    # 1. Coordinate cancellation C_L1
    x_l1 = torch.linalg.norm(x, ord=1, dim=-1)
    delta_l1 = torch.linalg.norm(delta, ord=1, dim=-1)
    new_x_l1 = torch.linalg.norm(new_x, ord=1, dim=-1)
    c_l1 = (x_l1 + delta_l1) / (new_x_l1 + eps)

    # 2. Geometric anti-alignment cosine = cos(x, delta)
    dot_product = (x * delta).sum(dim=-1)
    cos_sim = dot_product / (x_l2 * delta_l2 + eps)
    # Clamp cosine to [-1, 1] for safety against floating point inaccuracies
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # 3. Relative update magnitude r
    r_mag = delta_l2 / (x_l2 + eps)

    # 4. Residual contraction q
    q_contract = new_x_l2 / (x_l2 + eps)

    return ResidualMetrics(
        cos_sim=cos_sim,
        r_mag=r_mag,
        q_contract=q_contract,
        c_l1=c_l1,
        x_norm=x_l2,
        delta_norm=delta_l2
    )


def compute_prediction_metrics(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    eps: float = 1e-10
) -> Dict[str, torch.Tensor]:
    """
    Computes classification metrics for model output logits.
    
    Args:
        logits: Logits tensor, shape [B, C]
        targets: Optional ground truth labels, shape [B]
        
    Returns:
        Dict with top1_pred, top1_prob, margin, entropy, and (if targets provided) is_correct.
    """
    probs = F.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(logits, k=2, dim=-1)
    top1_pred = top_indices[:, 0]
    top1_logit = top_values[:, 0]
    top2_logit = top_values[:, 1]
    
    # Margin = top-1 vs top-2 logit margin
    margin = top1_logit - top2_logit
    
    # Top-1 probability
    top1_prob = probs.gather(1, top1_pred.unsqueeze(-1)).squeeze(-1)
    
    # Predictive entropy: -sum(p * log(p))
    entropy = - (probs * torch.log(probs + eps)).sum(dim=-1)
    
    res = {
        "top1_pred": top1_pred,
        "top1_prob": top1_prob,
        "margin": margin,
        "entropy": entropy
    }
    
    if targets is not None:
        res["is_correct"] = (top1_pred == targets).to(torch.int32)
        
    return res


def is_extreme_cancellation(
    cos_sim: float,
    r_mag: float,
    q_contract: float,
    c_l1: float,
    cos_thresh: float = -0.60,
    r_thresh: float = 0.60,
    q_thresh: float = 0.85,
    c_l1_thresh: float = 2.00
) -> bool:
    """
    Pre-registered criterion for an extreme cancellation event.
    Must satisfy ALL 4 conditions simultaneously:
      1. Strong negative cosine: cos <= -0.60
      2. Comparable magnitude: r >= 0.60
      3. Substantial contraction: q <= 0.85
      4. High coordinate cancellation: C_L1 >= 2.00
    """
    return (
        cos_sim <= cos_thresh and
        r_mag >= r_thresh and
        q_contract <= q_thresh and
        c_l1 >= c_l1_thresh
    )
