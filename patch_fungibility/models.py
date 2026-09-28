import hashlib
from typing import Tuple
import torch
import torch.nn as nn
import timm


def get_model_parameter_hash(model: nn.Module) -> str:
    """Computes a SHA-256 hash of all model parameters to verify weight immutability."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def load_deit_model(model_name: str, device: torch.device) -> Tuple[nn.Module, str]:
    """
    Loads a pretrained DeiT model from timm, disables gradients on all parameters,
    and returns the model along with an initial SHA-256 parameter hash.
    """
    model = timm.create_model(model_name, pretrained=True)
    model.to(device)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    initial_hash = get_model_parameter_hash(model)
    return model, initial_hash


def verify_backbone_unchanged(model: nn.Module, initial_hash: str) -> bool:
    """Verifies that the current model weights match the initial hash exactly."""
    current_hash = get_model_parameter_hash(model)
    if current_hash != initial_hash:
        raise AssertionError("CRITICAL VIOLATION: Model backbone weights have been modified during execution!")
    return True
