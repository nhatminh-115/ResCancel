import io
import os
import glob
from typing import Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageNetValidationSubset(Dataset):
    """
    Reproducible ImageNet validation dataset loaded directly from cached Parquet files.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        seed: int = 42,
        stratified: bool = True,
        transform: Optional[transforms.Compose] = None
    ):
        self.num_samples = num_samples
        self.seed = seed
        self.stratified = stratified
        
        # Standard timm DeiT preprocessing: 256 resize -> 224 center crop
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self._samples: List[Tuple[bytes, int]] = []
        self._load_data()

    def _load_data(self):
        # Locate cached parquet files
        cache_pattern = os.path.expanduser(
            r"~/.cache/huggingface/hub/datasets--evanarlian--imagenet_1k_resized_256/snapshots/*/data/val-*.parquet"
        )
        parquet_files = sorted(glob.glob(cache_pattern))
        
        if not parquet_files:
            # If not in snapshot, try searching anywhere in cache
            alt_pattern = os.path.expanduser(
                r"~/.cache/huggingface/hub/**/val-*.parquet"
            )
            parquet_files = sorted(glob.glob(alt_pattern, recursive=True))

        if not parquet_files:
            raise FileNotFoundError(
                "ImageNet-1k validation parquet files not found in cache. "
                "Please run huggingface_hub download first."
            )

        # Load both files
        dfs = [pd.read_parquet(f) for f in parquet_files]
        full_df = pd.concat(dfs, ignore_index=True)
        
        if self.stratified and self.num_samples <= 1000:
            # Pick 1 sample per class deterministically using seed
            rng = np.random.RandomState(self.seed)
            grouped = full_df.groupby("label")
            selected_indices = []
            classes = sorted(grouped.groups.keys())[:self.num_samples]
            for c in classes:
                indices = grouped.groups[c]
                chosen_idx = rng.choice(indices)
                selected_indices.append(chosen_idx)
            selected_df = full_df.loc[selected_indices].reset_index(drop=True)
        else:
            # Deterministic random sample
            selected_df = full_df.sample(n=min(self.num_samples, len(full_df)), random_state=self.seed).reset_index(drop=True)

        for _, row in selected_df.iterrows():
            img_val = row["image"]
            if isinstance(img_val, dict) and "bytes" in img_val:
                raw_bytes = img_val["bytes"]
            elif isinstance(img_val, bytes):
                raw_bytes = img_val
            else:
                raise ValueError(f"Unknown image data format: {type(img_val)}")
            label = int(row["label"])
            self._samples.append((raw_bytes, label))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raw_bytes, label = self._samples[idx]
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def get_imagenet_val_loader(
    num_samples: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0
) -> DataLoader:
    """Returns a DataLoader for the reproducible ImageNet validation subset."""
    dataset = ImageNetValidationSubset(num_samples=num_samples, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return loader
