import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import types
from rescancel.metrics import compute_residual_metrics, is_extreme_cancellation


@dataclass
class InterventionConfig:
    """Configuration for causal intervention during forward pass."""
    active: bool = False
    target_layer: Optional[int] = None       # None means apply to all layers
    target_site: Optional[str] = None        # 'attn', 'mlp', or None (both)
    target_token: str = "cls"                # 'cls', 'patch', or 'all'
    mode: str = "weaken_opposing"            # 'weaken_opposing', 'random_direction', 'uniform_scaling'
    alpha: float = 1.0                       # 1.0 = untouched, 0.75, 0.50, 0.25
    seed: int = 42
    only_extreme: bool = False               # Only intervene if event satisfies extreme cancellation criteria


@dataclass
class ResidualRecord:
    """Stores recorded metrics for a single residual addition site."""
    layer: int
    site: str                                # 'attn' or 'mlp'
    # CLS token metrics [B]
    cls_cos: torch.Tensor
    cls_r: torch.Tensor
    cls_q: torch.Tensor
    cls_c_l1: torch.Tensor
    cls_x_norm: torch.Tensor
    cls_delta_norm: torch.Tensor
    # Patch token aggregated metrics [B]
    patch_cos_mean: torch.Tensor
    patch_cos_min: torch.Tensor
    patch_r_mean: torch.Tensor
    patch_q_mean: torch.Tensor
    patch_c_l1_mean: torch.Tensor
    patch_x_norm_mean: torch.Tensor
    patch_delta_norm_mean: torch.Tensor
    patch_extreme_frac: torch.Tensor         # Fraction of patches satisfying extreme criteria


class InstrumentedBlock(nn.Module):
    """
    Wraps a transformer block to instrument and/or intervene on residual additions
    around self-attention and MLP.
    """
    def __init__(self, original_block: nn.Module, layer_idx: int):
        super().__init__()
        self.block = original_block
        self.layer_idx = layer_idx
        self.logging_enabled: bool = True
        self.records: List[ResidualRecord] = []
        self.intervention: InterventionConfig = InterventionConfig(active=False)

    def set_intervention(self, config: InterventionConfig):
        self.intervention = config

    def clear_records(self):
        self.records.clear()

    def _apply_intervention(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        site: str
    ) -> torch.Tensor:
        """
        Applies causal intervention to delta according to self.intervention config.
        """
        if not self.intervention.active:
            return delta

        # Check target layer and site
        if self.intervention.target_layer is not None and self.intervention.target_layer != self.layer_idx:
            return delta
        if self.intervention.target_site is not None and self.intervention.target_site != site:
            return delta

        alpha = self.intervention.alpha
        if alpha == 1.0 and self.intervention.mode != "random_direction":
            return delta

        B, N, D = delta.shape
        eps = 1e-8

        # Determine which tokens to intervene on
        token_mask = torch.zeros(B, N, 1, dtype=torch.bool, device=delta.device)
        if self.intervention.target_token in ("cls", "all"):
            token_mask[:, 0:1, :] = True
        if self.intervention.target_token in ("patch", "all"):
            token_mask[:, 1:, :] = True

        # Decompose delta = delta_parallel + delta_perp relative to x
        dot = (delta * x).sum(dim=-1, keepdim=True)
        x_sq_norm = (x * x).sum(dim=-1, keepdim=True) + eps
        delta_parallel = (dot / x_sq_norm) * x
        delta_perp = delta - delta_parallel

        # Check if delta opposes x
        opposing = dot < 0.0

        if self.intervention.only_extreme:
            # Check extreme cancellation criteria per token
            metrics = compute_residual_metrics(x, delta, eps=eps)
            is_ext = (
                (metrics.cos_sim <= -0.60) &
                (metrics.r_mag >= 0.60) &
                (metrics.q_contract <= 0.85) &
                (metrics.c_l1 >= 2.00)
            ).unsqueeze(-1)
            target_mask = token_mask & opposing & is_ext
        else:
            target_mask = token_mask & opposing

        if not target_mask.any():
            return delta

        mode = self.intervention.mode
        delta_mod = delta.clone()

        if mode == "weaken_opposing":
            # delta' = alpha * delta_parallel + delta_perp
            weakened = alpha * delta_parallel + delta_perp
            delta_mod = torch.where(target_mask, weakened, delta_mod)

        elif mode == "random_direction":
            # Perturb in a random direction with norm equal to ||(1 - alpha) * delta_parallel||
            # using fixed generator for reproducibility
            gen = torch.Generator(device=delta.device)
            gen.manual_seed(self.intervention.seed + self.layer_idx * 10 + (0 if site == "attn" else 1))
            noise = torch.randn(delta.shape, generator=gen, device=delta.device, dtype=delta.dtype)
            noise_norm = torch.linalg.norm(noise, dim=-1, keepdim=True) + eps
            unit_noise = noise / noise_norm
            pert_magnitude = (1.0 - alpha) * torch.linalg.norm(delta_parallel, dim=-1, keepdim=True)
            delta_mod = torch.where(target_mask, delta + unit_noise * pert_magnitude, delta_mod)

        elif mode == "uniform_scaling":
            # Scale delta uniformly by (1 - beta) such that norm change matches (1 - alpha) * ||delta_parallel||
            delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True) + eps
            pert_magnitude = (1.0 - alpha) * torch.linalg.norm(delta_parallel, dim=-1, keepdim=True)
            beta = pert_magnitude / delta_norm
            scaled = (1.0 - beta) * delta
            delta_mod = torch.where(target_mask, scaled, delta_mod)

        return delta_mod

    def _record_metrics(self, x: torch.Tensor, delta: torch.Tensor, site: str):
        """Computes and detaches metrics for logging, transferring to CPU to save GPU VRAM."""
        with torch.no_grad():
            metrics = compute_residual_metrics(x, delta)
            
            # CLS token (index 0)
            cls_cos = metrics.cos_sim[:, 0].detach().cpu()
            cls_r = metrics.r_mag[:, 0].detach().cpu()
            cls_q = metrics.q_contract[:, 0].detach().cpu()
            cls_c_l1 = metrics.c_l1[:, 0].detach().cpu()
            cls_x_norm = metrics.x_norm[:, 0].detach().cpu()
            cls_delta_norm = metrics.delta_norm[:, 0].detach().cpu()

            # Patch tokens (indices 1..N-1)
            patch_cos = metrics.cos_sim[:, 1:].detach()
            patch_r = metrics.r_mag[:, 1:].detach()
            patch_q = metrics.q_contract[:, 1:].detach()
            patch_c_l1 = metrics.c_l1[:, 1:].detach()
            patch_x_norm = metrics.x_norm[:, 1:].detach()
            patch_delta_norm = metrics.delta_norm[:, 1:].detach()

            # Extreme patch condition
            extreme_mask = (
                (patch_cos <= -0.60) &
                (patch_r >= 0.60) &
                (patch_q <= 0.85) &
                (patch_c_l1 >= 2.00)
            ).float()
            patch_extreme_frac = extreme_mask.mean(dim=1).cpu()

            record = ResidualRecord(
                layer=self.layer_idx,
                site=site,
                cls_cos=cls_cos,
                cls_r=cls_r,
                cls_q=cls_q,
                cls_c_l1=cls_c_l1,
                cls_x_norm=cls_x_norm,
                cls_delta_norm=cls_delta_norm,
                patch_cos_mean=patch_cos.mean(dim=1).cpu(),
                patch_cos_min=patch_cos.amin(dim=1).cpu(),
                patch_r_mean=patch_r.mean(dim=1).cpu(),
                patch_q_mean=patch_q.mean(dim=1).cpu(),
                patch_c_l1_mean=patch_c_l1.mean(dim=1).cpu(),
                patch_x_norm_mean=patch_x_norm.mean(dim=1).cpu(),
                patch_delta_norm_mean=patch_delta_norm.mean(dim=1).cpu(),
                patch_extreme_frac=patch_extreme_frac
            )
            self.records.append(record)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        # --- 1. Attention residual addition ---
        x_in_attn = x
        norm_x1 = self.block.norm1(x_in_attn)
        # Call attention with appropriate signature
        try:
            attn_out = self.block.attn(norm_x1, attn_mask=attn_mask, is_causal=is_causal)
        except TypeError:
            attn_out = self.block.attn(norm_x1)

        delta_attn = self.block.drop_path1(self.block.ls1(attn_out))

        if self.logging_enabled:
            self._record_metrics(x_in_attn, delta_attn, site="attn")

        # Causal intervention at attention site
        delta_attn_eff = self._apply_intervention(x_in_attn, delta_attn, site="attn")
        x_post_attn = x_in_attn + delta_attn_eff

        # --- 2. MLP residual addition ---
        x_in_mlp = x_post_attn
        norm_x2 = self.block.norm2(x_in_mlp)
        mlp_out = self.block.mlp(norm_x2)
        delta_mlp = self.block.drop_path2(self.block.ls2(mlp_out))

        if self.logging_enabled:
            self._record_metrics(x_in_mlp, delta_mlp, site="mlp")

        # Causal intervention at MLP site
        delta_mlp_eff = self._apply_intervention(x_in_mlp, delta_mlp, site="mlp")
        x_post_mlp = x_in_mlp + delta_mlp_eff

        return x_post_mlp


class InstrumentedViT:
    """
    Manages instrumentation and intervention hooks across all layers of a timm VisionTransformer.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.instrumented_blocks: List[InstrumentedBlock] = []
        self._wrap_blocks()

    def _wrap_blocks(self):
        for i, block in enumerate(self.model.blocks):
            inst_block = InstrumentedBlock(block, layer_idx=i)
            self.instrumented_blocks.append(inst_block)
            self.model.blocks[i] = inst_block

    def unwrap(self):
        """Restores original uninstrumented blocks."""
        for i, inst_block in enumerate(self.instrumented_blocks):
            self.model.blocks[i] = inst_block.block
        self.instrumented_blocks.clear()

    def set_logging(self, enabled: bool):
        for b in self.instrumented_blocks:
            b.logging_enabled = enabled

    def clear_records(self):
        for b in self.instrumented_blocks:
            b.clear_records()

    def set_intervention(self, config: InterventionConfig):
        for b in self.instrumented_blocks:
            b.set_intervention(config)

    def disable_intervention(self):
        config = InterventionConfig(active=False)
        for b in self.instrumented_blocks:
            b.set_intervention(config)

    def collect_records(self) -> List[ResidualRecord]:
        """Collects all recorded metrics from all blocks in order."""
        records = []
        for b in self.instrumented_blocks:
            records.extend(b.records)
        return records
