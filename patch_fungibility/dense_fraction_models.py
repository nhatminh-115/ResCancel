"""
patch_fungibility/dense_fraction_models.py

Unified model loading, data transforms, and verified block-by-block forward paths for:
- Model A: DeiT-Tiny (deit_tiny_patch16_224, D=192, N=196)
- Model B: DeiT-Small (deit_small_patch16_224, D=384, N=196)
- Model C: ViT-B/16 AugReg (vit_base_patch16_224.augreg_in1k, D=768, N=196)
- Model D: DINOv2 ViT-S/14 (dinov2_vits14_lc, layers=1, D=384, N=256)
"""

from typing import Dict, Any, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm


def load_model_and_transform(
    model_key: str,
    device: torch.device
) -> Tuple[nn.Module, transforms.Compose, Dict[str, Any]]:
    """
    Loads one of the four frozen models and returns (model, transform, metadata).
    """
    if model_key == "deit_tiny":
        model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        meta = {
            "model_key": "deit_tiny",
            "model_id": "deit_tiny_patch16_224",
            "embed_dim": 192,
            "num_patches": 196,
            "depth": 12,
            "primary_depth": 8,
            "readout": "cls_token"
        }

    elif model_key == "deit_small":
        model = timm.create_model("deit_small_patch16_224", pretrained=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        meta = {
            "model_key": "deit_small",
            "model_id": "deit_small_patch16_224",
            "embed_dim": 384,
            "num_patches": 196,
            "depth": 12,
            "primary_depth": 8,
            "readout": "cls_token"
        }

    elif model_key == "vit_base":
        model = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        meta = {
            "model_key": "vit_base",
            "model_id": "vit_base_patch16_224.augreg_in1k",
            "embed_dim": 768,
            "num_patches": 196,
            "depth": 12,
            "primary_depth": 7,
            "secondary_depth": 8,
            "readout": "cls_token"
        }

    elif model_key == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_lc", layers=1, pretrained=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        meta = {
            "model_key": "dinov2",
            "model_id": "dinov2_vits14_lc",
            "embed_dim": 384,
            "num_patches": 256,
            "depth": 12,
            "primary_depth": 9,
            "secondary_depth": 8,
            "readout": "cls_and_patch_mean"
        }
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    return model, transform, meta


def forward_block_by_block(
    model: nn.Module,
    model_key: str,
    x: Optional[torch.Tensor] = None,
    start_depth: int = 0,
    h_start: Optional[torch.Tensor] = None,
    collect_depths: Optional[Tuple[int, ...]] = None,
    intervention_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Executes block forward pass starting from image `x` (start_depth=0)
    or from hidden state `h_start` after block `start_depth`.
    Supports all four models.
    """
    collected = {}

    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        if start_depth == 0:
            if x is None:
                raise ValueError("Must provide x when start_depth=0")
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

        for b_idx in range(current_block, len(model.blocks)):
            h = model.blocks[b_idx](h)
            depth = b_idx + 1

            if collect_depths is not None and depth in collect_depths:
                collected[depth] = h.detach().clone()

            if intervention_fn is not None:
                h = intervention_fn(depth, h)

        h_norm = model.norm(h)
        logits = model.forward_head(h_norm)
        return logits, collected

    elif model_key == "dinov2":
        backbone = model.backbone
        linear_head = model.linear_head

        if start_depth == 0:
            if x is None:
                raise ValueError("Must provide x when start_depth=0")
            h = backbone.prepare_tokens_with_masks(x)
            current_block = 0
        else:
            if h_start is None:
                raise ValueError("Must provide h_start when start_depth > 0")
            h = h_start
            current_block = start_depth

        for b_idx in range(current_block, len(backbone.blocks)):
            h = backbone.blocks[b_idx](h)
            depth = b_idx + 1

            if collect_depths is not None and depth in collect_depths:
                collected[depth] = h.detach().clone()

            if intervention_fn is not None:
                h = intervention_fn(depth, h)

        h_norm = backbone.norm(h)
        cls_norm = h_norm[:, 0]
        patch_norm = h_norm[:, 1:]
        patch_mean = patch_norm.mean(dim=1)
        readout_in = torch.cat([cls_norm, patch_mean], dim=-1)
        logits = linear_head(readout_in)
        return logits, collected

    else:
        raise ValueError(f"Unknown model_key: {model_key}")
