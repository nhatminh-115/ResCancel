import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_calibration_statistics(
    model: nn.Module,
    model_name: str,
    calib_loader: DataLoader,
    tested_depths: List[int],
    device: torch.device,
    output_path: Optional[str] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Computes coordinate-wise activation statistics (mu, sigma, var) across ALL spatial patch tokens
    from the strictly disjoint Calibration set (N = 1,000 images).
    Labels are NEVER used.
    Returns:
      {depth: {"mu": torch.Tensor (D,), "sigma": torch.Tensor (D,), "var": torch.Tensor (D,)}}
    """
    model.eval()
    max_depth = max(tested_depths)
    cached_patch_tokens: Dict[int, List[torch.Tensor]] = {d: [] for d in tested_depths}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calib_loader):
            images = images.to(device)
            feat = model.patch_embed(images)
            feat = model._pos_embed(feat)
            feat = model.patch_drop(feat)
            feat = model.norm_pre(feat)

            for b in range(max_depth + 1):
                feat = model.blocks[b](feat)
                if b in tested_depths:
                    # Spatial patch tokens only (tokens 1..196; CLS at 0 excluded)
                    spatial_tokens = feat[:, 1:, :].detach().cpu()
                    cached_patch_tokens[b].append(spatial_tokens)

    stats_by_depth = {}
    for d in tested_depths:
        # Concatenate across batches: (N, 196, D) -> (N * 196, D)
        all_tokens = torch.cat(cached_patch_tokens[d], dim=0).reshape(-1, model.embed_dim)
        mu_d = all_tokens.mean(dim=0)
        sigma_d = all_tokens.std(dim=0, unbiased=True)
        var_d = sigma_d ** 2

        stats_by_depth[d] = {
            "mu": mu_d,
            "sigma": sigma_d,
            "var": var_d,
            "total_tokens": int(all_tokens.shape[0]),
            "embed_dim": int(model.embed_dim)
        }

    return stats_by_depth
