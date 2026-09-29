"""
patch_fungibility/v1_models.py

Model definitions and manual block-level forward execution for:
- Model A: Supervised Vanilla ViT (vit_base_patch16_224.augreg_in1k via timm)
- Model B: Self-Supervised DINOv2 (dinov2_vits14_lc with layers=1, pretrained=True via torch.hub)

Provides verified manual forward execution that exactly reproduces official forward logits.
"""

from typing import Dict, Any, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm


def get_vitb_model(device: torch.device = torch.device("cpu")) -> Tuple[nn.Module, transforms.Compose, Dict[str, Any]]:
    """
    Loads pretrained vit_base_patch16_224.augreg_in1k from timm.
    Freezes all parameters and returns (model, transform, metadata).
    """
    model = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    metadata = {
        "model_id": "vit_base_patch16_224.augreg_in1k",
        "source": "timm",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
        "num_patches": 196,
        "regime": "supervised_augreg",
        "readout": "cls_token",
        "data_config": {k: str(v) for k, v in data_config.items()}
    }
    return model, transform, metadata


def get_dinov2_model(device: torch.device = torch.device("cpu")) -> Tuple[nn.Module, transforms.Compose, Dict[str, Any]]:
    """
    Loads official dinov2_vits14_lc (layers=1, pretrained=True) from torch.hub.
    Freezes all parameters and returns (model, transform, metadata).
    IMPORTANT: This is the official NO-REGISTER model with frozen ImageNet-1k linear head.
    Readout consumes [norm(cls), mean(norm(patch))] -> linear_head.
    """
    # Load via torch hub
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_lc", layers=1, pretrained=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    # Standard official DINOv2 validation transform
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    metadata = {
        "model_id": "dinov2_vits14_lc_layers1",
        "source": "facebookresearch/dinov2",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 14,
        "num_patches": 256,
        "regime": "self_supervised_dinov2",
        "readout": "cls_and_patch_mean",
        "head_in_features": 768,  # 384 * 2
        "num_classes": 1000
    }
    return model, transform, metadata


def forward_vitb_manual(
    model: nn.Module,
    x: Optional[torch.Tensor] = None,
    start_depth: int = 0,
    h_start: Optional[torch.Tensor] = None,
    intervention_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    collect_depths: Optional[Tuple[int, ...]] = None
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Executes block-by-block forward pass for ViT-B/16.
    Can start from raw image `x` (start_depth=0) or from hidden state `h_start` after block `start_depth`.
    Supports intervening at any block output via `intervention_fn(depth, h)`.
    collect_depths: tuple of block depths (1-indexed, 1..12) whose post-block activations should be returned.

    Hidden state shape: (B, 197, 768) where token 0 is CLS, tokens 1..196 are patches.
    """
    collected = {}

    if start_depth == 0:
        if x is None:
            raise ValueError("Must provide x when start_depth=0")
        # Embedding
        h = model.patch_embed(x)
        h = model._pos_embed(h)
        h = model.patch_drop(h)
        h = model.norm_pre(h)
        current_block = 0
    else:
        if h_start is None:
            raise ValueError("Must provide h_start when start_depth > 0")
        h = h_start
        current_block = start_depth

    # Iterate through blocks (0-indexed in model.blocks, 1-indexed depth: 1..12)
    for b_idx in range(current_block, len(model.blocks)):
        h = model.blocks[b_idx](h)
        depth = b_idx + 1

        if collect_depths is not None and depth in collect_depths:
            collected[depth] = h.detach().clone()

        if intervention_fn is not None:
            h = intervention_fn(depth, h)

    # Final norm & head
    h_norm = model.norm(h)
    logits = model.forward_head(h_norm)
    return logits, collected


def forward_dinov2_manual(
    model: nn.Module,
    x: Optional[torch.Tensor] = None,
    start_depth: int = 0,
    h_start: Optional[torch.Tensor] = None,
    intervention_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    collect_depths: Optional[Tuple[int, ...]] = None
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Executes block-by-block forward pass for DINOv2 ViT-S/14 with linear head.
    The model is an instance of `_LinearClassifierWrapper` containing `backbone` and `linear_head`.
    Can start from raw image `x` (start_depth=0) or from hidden state `h_start` after block `start_depth`.
    Supports intervening at any block output via `intervention_fn(depth, h)`.
    collect_depths: tuple of block depths (1-indexed, 1..12) whose post-block activations should be returned.

    Hidden state shape: (B, 257, 384) where token 0 is CLS, tokens 1..256 are patches.
    Readout:
      h_norm = backbone.norm(h)
      cls_norm = h_norm[:, 0]
      patch_norm = h_norm[:, 1:]
      patch_mean = patch_norm.mean(dim=1)
      head_in = concat([cls_norm, patch_mean], dim=1)
      logits = linear_head(head_in)
    """
    collected = {}
    backbone = model.backbone
    linear_head = model.linear_head

    if start_depth == 0:
        if x is None:
            raise ValueError("Must provide x when start_depth=0")
        # Embedding
        h = backbone.prepare_tokens_with_masks(x)
        current_block = 0
    else:
        if h_start is None:
            raise ValueError("Must provide h_start when start_depth > 0")
        h = h_start
        current_block = start_depth

    # Iterate through blocks (0-indexed in backbone.blocks, 1-indexed depth: 1..12)
    for b_idx in range(current_block, len(backbone.blocks)):
        h = backbone.blocks[b_idx](h)
        depth = b_idx + 1

        if collect_depths is not None and depth in collect_depths:
            collected[depth] = h.detach().clone()

        if intervention_fn is not None:
            h = intervention_fn(depth, h)

    # Final norm & readout
    h_norm = backbone.norm(h)
    cls_norm = h_norm[:, 0]
    patch_norm = h_norm[:, 1:]
    patch_mean = patch_norm.mean(dim=1)
    readout_in = torch.cat([cls_norm, patch_mean], dim=-1)
    logits = linear_head(readout_in)
    return logits, collected


def validate_manual_forwards(
    vitb_model: nn.Module,
    dinov2_model: nn.Module,
    eval_loader_vitb: torch.utils.data.DataLoader,
    eval_loader_dinov2: torch.utils.data.DataLoader,
    device: torch.device,
    n_images: int = 64
) -> Dict[str, Any]:
    """
    Validates manual block forward implementations against untouched official forward
    on at least n_images evaluation images across multiple batches.
    Requires:
      - max absolute logit difference < 1e-5
      - prediction agreement = 100%
    """
    validation_results = {}

    # 1. ViT-B validation
    vitb_max_diff = 0.0
    vitb_agree = 0
    vitb_total = 0

    with torch.no_grad():
        for images, labels in eval_loader_vitb:
            images = images.to(device)
            official_logits = vitb_model(images)
            manual_logits, _ = forward_vitb_manual(vitb_model, x=images)

            diff = torch.max(torch.abs(official_logits - manual_logits)).item()
            if diff > vitb_max_diff:
                vitb_max_diff = diff

            pred_off = official_logits.argmax(dim=-1)
            pred_man = manual_logits.argmax(dim=-1)
            vitb_agree += (pred_off == pred_man).sum().item()
            vitb_total += images.size(0)

            if vitb_total >= n_images:
                break

    vitb_pass = (vitb_max_diff < 1e-5) and (vitb_agree == vitb_total)
    validation_results["vit_base_patch16_224.augreg_in1k"] = {
        "n_evaluated": vitb_total,
        "max_abs_logit_diff": float(vitb_max_diff),
        "prediction_agreement_count": vitb_agree,
        "prediction_agreement_pct": float(100.0 * vitb_agree / vitb_total),
        "passed": bool(vitb_pass)
    }

    # 2. DINOv2 validation
    dinov2_max_diff = 0.0
    dinov2_agree = 0
    dinov2_total = 0

    with torch.no_grad():
        for images, labels in eval_loader_dinov2:
            images = images.to(device)
            official_logits = dinov2_model(images)
            manual_logits, _ = forward_dinov2_manual(dinov2_model, x=images)

            diff = torch.max(torch.abs(official_logits - manual_logits)).item()
            if diff > dinov2_max_diff:
                dinov2_max_diff = diff

            pred_off = official_logits.argmax(dim=-1)
            pred_man = manual_logits.argmax(dim=-1)
            dinov2_agree += (pred_off == pred_man).sum().item()
            dinov2_total += images.size(0)

            if dinov2_total >= n_images:
                break

    dinov2_pass = (dinov2_max_diff < 1e-5) and (dinov2_agree == dinov2_total)
    validation_results["dinov2_vits14_lc"] = {
        "n_evaluated": dinov2_total,
        "max_abs_logit_diff": float(dinov2_max_diff),
        "prediction_agreement_count": dinov2_agree,
        "prediction_agreement_pct": float(100.0 * dinov2_agree / dinov2_total),
        "passed": bool(dinov2_pass)
    }

    assert vitb_pass, f"ViT-B manual forward validation failed! Max diff: {vitb_max_diff}, Agree: {vitb_agree}/{vitb_total}"
    assert dinov2_pass, f"DINOv2 manual forward validation failed! Max diff: {dinov2_max_diff}, Agree: {dinov2_agree}/{dinov2_total}"

    return validation_results
