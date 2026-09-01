"""
Cassava Leaf Disease Dataset.

Cassava is a major food crop in Africa. This dataset contains
21,397 images across 5 classes of cassava leaf diseases.

Source: https://www.kaggle.com/c/cassava-leaf-disease-classification
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class CassavaLeafDataset(Dataset):
    """Cassava Leaf Disease Dataset.

    Classes:
        0: Cassava Bacterial Blight (CBB)
        1: Cassava Brown Streak Disease (CBSD)
        2: Cassava Green Mottle (CGM)
        3: Cassava Mosaic Disease (CMD)
        4: Healthy

    Args:
        root: Root directory containing the dataset.
        split: 'train', 'val', or 'test'.
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
    """

    CLASS_NAMES = [
        "Cassava Bacterial Blight",
        "Cassava Brown Streak Disease",
        "Cassava Green Mottle",
        "Cassava Mosaic Disease",
        "Healthy",
    ]

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
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.target_transform = target_transform

        self.data_dir = self.root / "cassava-leaf-disease" / "train_images"

        if not self.data_dir.exists():
            # Try alternative paths
            alt_paths = [
                self.root / "cassava-leaf-disease-classification" / "train_images",
                self.root / "cassava" / "train_images",
            ]
            for alt in alt_paths:
                if alt.exists():
                    self.data_dir = alt
                    break

        if not self.data_dir.exists():
            print("CassavaLeaf not found. Attempting download...")
            self._download_huggingface()

        if not self.data_dir.exists():
            print("CassavaLeaf download failed. Creating synthetic dataset...")
            self._create_synthetic()

        # Load labels from CSV if available (check multiple paths)
        self.samples = []
        labels_file = None
        for csv_candidate in [
            self.root / "cassava-leaf-disease" / "train.csv",
            self.root / "cassava-leaf-disease-classification" / "train.csv",
            self.root / "cassava" / "train.csv",
        ]:
            if csv_candidate.exists():
                labels_file = csv_candidate
                break

        if labels_file is not None:
            import csv
            with open(labels_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = self.data_dir / row["image_id"]
                    label = int(row["label"])
                    if img_path.exists():
                        self.samples.append((img_path, label))
        else:
            # Fallback: scan directories
            for i, cls_name in enumerate(self.CLASS_NAMES):
                cls_dir = self.data_dir / str(i)
                if cls_dir.exists():
                    for ext in ("*.jpg", "*.png"):
                        for img_path in cls_dir.glob(ext):
                            self.samples.append((img_path, i))

        # Deterministic split
        if self.samples:
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
        """Create synthetic dataset."""
        for i in range(5):
            cls_dir = self.data_dir / str(i)
            cls_dir.mkdir(parents=True, exist_ok=True)
            for j in range(20):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(arr).save(cls_dir / f"synthetic_{j:04d}.jpg")

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

    def _download_huggingface(self) -> bool:
        """Download Cassava Leaf Disease from HuggingFace."""
        try:
            from datasets import load_dataset
            print("Downloading CassavaLeaf from HuggingFace...")
            hf_dataset = load_dataset(
                "pufanyi/cassava-leaf-disease-classification",
                trust_remote_code=True,
            )
            self.data_dir.mkdir(parents=True, exist_ok=True)
            for split_name in ["train", "validation", "test"]:
                if split_name not in hf_dataset:
                    continue
                for item in hf_dataset[split_name]:
                    label = int(item["label"])
                    cls_dir = self.data_dir / str(label)
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    img_id = item.get("image_id", f"{split_name}_{len(list(cls_dir.glob('*')))}.jpg")
                    img_path = cls_dir / f"{img_id}.jpg"
                    if not img_path.exists():
                        item["image"].save(str(img_path))
            print(f"CassavaLeaf download complete. Saved to {self.data_dir}")
            return True
        except Exception as e:
            print(f"CassavaLeaf HuggingFace download failed: {e}")
            return False

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)
