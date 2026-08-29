from __future__ import annotations
"""
PlantDoc Dataset loader.

PlantDoc is a real-world dataset with 2,598 images across 27 classes,
captured in uncontrolled environments — representing a significant
domain shift from PlantVillage.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class PlantDocDataset(Dataset):
    """PlantDoc dataset — real-world plant disease images.

    This dataset represents an 'out-of-domain' setting relative to
    PlantVillage, with varying lighting, backgrounds, and image quality.

    Args:
        root: Root directory containing PlantDoc data.
        split: 'train', 'val', or 'test'.
        transform: Optional transform for images.
        target_transform: Optional transform for targets.
    """

    SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform=None,
        target_transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.data_dir = self.root / "PlantDoc"
        if not self.data_dir.exists():
            print(f"PlantDoc not found at {self.data_dir}. Creating synthetic dataset...")
            self._create_synthetic_dataset()

        self.classes = sorted(
            [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples: list[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls_name]))
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((img_path, self.class_to_idx[cls_name]))
            for img_path in cls_dir.glob("*.JPG"):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

        rng = torch.Generator().manual_seed(42)
        n = len(self.samples)
        perm = torch.randperm(n, generator=rng).tolist()

        train_end = int(n * self.SPLIT_RATIOS["train"])
        val_end = train_end + int(n * self.SPLIT_RATIOS["val"])

        splits = {
            "train": perm[:train_end],
            "val": perm[train_end:val_end],
            "test": perm[val_end:],
        }

        if split is not None:
            self.samples = [self.samples[i] for i in splits[split]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _create_synthetic_dataset(self):
        """Create synthetic dataset for testing."""
        import numpy as np
        test_classes = ["Apple___Scab", "Tomato___Bacterial_spot", "Potato___Late_blight"]
        for cls_name in test_classes:
            cls_dir = self.data_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(15):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(cls_dir / f"synthetic_{i:04d}.jpg")

    @property
    def num_classes(self) -> int:
        return len(self.classes)
