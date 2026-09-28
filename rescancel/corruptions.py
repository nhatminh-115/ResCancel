import torch
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.functional as TF
from typing import Dict, Callable


def apply_gaussian_noise(x: torch.Tensor, sigma: float = 0.08, seed: int = 42) -> torch.Tensor:
    """
    Applies mild additive Gaussian noise to a normalized image tensor [B, C, H, W].
    """
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * sigma
    return x + noise


def apply_gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Applies mild Gaussian blur to an image tensor [B, C, H, W].
    """
    return TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def apply_mild_contrast(x: torch.Tensor, factor: float = 0.8) -> torch.Tensor:
    """
    Applies mild contrast reduction to an image tensor [B, C, H, W].
    """
    return TF.adjust_contrast(x, factor)


PERTURBATIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "clean": lambda x: x,
    "gaussian_noise": lambda x: apply_gaussian_noise(x, sigma=0.08),
    "gaussian_blur": lambda x: apply_gaussian_blur(x, kernel_size=5, sigma=1.0),
}
