"""
New Plant Diseases Dataset (Augmented).

Large-scale dataset with 87,848 images across 38 classes.
Source: https://www.kaggle.com/datasets/emmarex/plantdisease

This dataset is an augmented version of PlantVillage with
additional real-world images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class NewPlantDiseasesDataset(Dataset):
    """New Plant Diseases Dataset with 87K+ images.

    Args:
        root: Root directory containing the dataset.
        split: 'train', 'val', or 'test'.
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
    """

    SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

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

        self.data_dir = self.root / "plantvillage" / "plantvillage"

        # Try alternative paths
        if not self.data_dir.exists():
            self.data_dir = self.root / "New Plant Diseases Dataset(Augmented)" / "train"
        if not self.data_dir.exists():
            self.data_dir = self.root / "plant-disease"

        if not self.data_dir.exists():
            print(f"Dataset not found. Creating synthetic NewPlantDiseases...")
            self._create_synthetic()

        self.classes = sorted(
            [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        # Deterministic split
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

    def _create_synthetic(self):
        """Create synthetic dataset for testing."""
        classes = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight",
            "Tomato___Late_blight", "Tomato___healthy",
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Corn___Common_rust", "Corn___healthy",
            "Grape___Black_rot", "Grape___healthy",
            "Pepper___Bacterial_spot", "Pepper___healthy",
        ]
        for cls_name in classes:
            cls_dir = self.data_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(25):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(arr).save(cls_dir / f"synthetic_{i:04d}.jpg")

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

    @property
    def num_classes(self) -> int:
        return len(self.classes)
