from __future__ import annotations
"""
Coffee Leaf Disease Dataset loader.

Coffee leaf dataset with images of coffee rust and other diseases.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class CoffeeLeafDataset(Dataset):
    """Coffee leaf disease classification dataset.

    Args:
        root: Root directory containing coffee leaf data.
        split: 'train', 'val', or 'test'.
        transform: Optional transform for images.
        target_transform: Optional transform for targets.
    """

    CLASS_NAMES = [
        "healthy",
        "rust",
        "miner",
        "phoma",
        "cercospora",
    ]

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform=None,
        target_transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.target_transform = target_transform

        self.data_dir = self.root / "CoffeeLeaf"
        if not self.data_dir.exists():
            print(f"CoffeeLeaf not found at {self.data_dir}. Creating synthetic dataset...")
            self._create_synthetic_dataset()

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.CLASS_NAMES)
        }

        self.samples: list[Tuple[Path, int]] = []
        for cls_name in self.CLASS_NAMES:
            cls_dir = self.data_dir / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append(
                        (img_path, self.class_to_idx[cls_name])
                    )

        rng = torch.Generator().manual_seed(42)
        n = len(self.samples)
        perm = torch.randperm(n, generator=rng).tolist()

        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

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
        for cls_name in self.CLASS_NAMES:
            cls_dir = self.data_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(15):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(cls_dir / f"synthetic_{i:04d}.jpg")

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)
