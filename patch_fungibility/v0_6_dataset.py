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


class ParquetImageSubset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[bytes, int]],
        transform: Optional[transforms.Compose] = None
    ):
        self._samples = samples
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raw_bytes, label = self._samples[idx]
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def get_disjoint_imagenet_splits(
    calib_seed: int = 9101,
    eval_seed: int = 9201,
    n_per_split: int = 1000,
    output_dir: Optional[str] = None
) -> Tuple[Dataset, Dataset, pd.DataFrame, pd.DataFrame]:
    """
    Constructs strictly disjoint Calibration and Evaluation subsets of ImageNet-1k validation set.
    Each split contains exactly n_per_split images (stratified 1 per class).
    Guarantees zero overlap in global image indices.
    """
    cache_pattern = os.path.expanduser(
        r"~/.cache/huggingface/hub/**/val-*.parquet"
    )
    parquet_files = sorted(glob.glob(cache_pattern, recursive=True))
    if not parquet_files:
        raise FileNotFoundError("ImageNet-1k validation parquet files not found in cache.")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["global_index"] = np.arange(len(full_df))

    grouped = full_df.groupby("label")
    classes = sorted(grouped.groups.keys())[:n_per_split]

    rng_calib = np.random.RandomState(calib_seed)
    rng_eval = np.random.RandomState(eval_seed)

    calib_indices = []
    eval_indices = []

    for c in classes:
        all_class_indices = list(grouped.groups[c])
        # Pick 1 sample for calibration
        calib_choice = int(rng_calib.choice(all_class_indices))
        calib_indices.append(calib_choice)

        # Remaining available indices for evaluation
        remaining_indices = [idx for idx in all_class_indices if idx != calib_choice]
        eval_choice = int(rng_eval.choice(remaining_indices))
        eval_indices.append(eval_choice)

    assert len(calib_indices) == n_per_split
    assert len(eval_indices) == n_per_split
    intersection = set(calib_indices).intersection(set(eval_indices))
    assert len(intersection) == 0, f"Disjoint split failed! Overlap: {intersection}"

    calib_df = full_df.loc[calib_indices].reset_index(drop=True)
    eval_df = full_df.loc[eval_indices].reset_index(drop=True)

    # Build raw samples
    def extract_samples(df):
        samples = []
        for _, row in df.iterrows():
            img_val = row["image"]
            if isinstance(img_val, dict) and "bytes" in img_val:
                raw_bytes = img_val["bytes"]
            elif isinstance(img_val, bytes):
                raw_bytes = img_val
            else:
                raise ValueError(f"Unknown image data format: {type(img_val)}")
            label = int(row["label"])
            samples.append((raw_bytes, label))
        return samples

    calib_samples = extract_samples(calib_df)
    eval_samples = extract_samples(eval_df)

    calib_dataset = ParquetImageSubset(calib_samples)
    eval_dataset = ParquetImageSubset(eval_samples)

    calib_manifest = pd.DataFrame({
        "sample_id": np.arange(len(calib_df)),
        "global_index": calib_df["global_index"].values,
        "label": calib_df["label"].values
    })
    eval_manifest = pd.DataFrame({
        "sample_id": np.arange(len(eval_df)),
        "global_index": eval_df["global_index"].values,
        "label": eval_df["label"].values
    })

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        calib_manifest.to_csv(os.path.join(output_dir, "calibration_split.csv"), index=False)
        eval_manifest.to_csv(os.path.join(output_dir, "evaluation_split.csv"), index=False)

    return calib_dataset, eval_dataset, calib_manifest, eval_manifest
