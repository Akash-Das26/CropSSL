from __future__ import annotations
"""
Plant Pathology 2020 (FGVC7) Dataset.

Apple foliar disease classification with severity estimation.
4 classes: healthy, multiple_diseases, rust, scab.
~1,821 training images from Apple orchards.

Source: https://www.kaggle.com/c/plant-pathology-2020-fgvc7
Paper: https://arxiv.org/abs/2006.13285

Also usable with Plant Pathology 2021 (FGVC8) extension for severity:
https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8
"""

import csv
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class PlantPathologyDataset(Dataset):
    """Plant Pathology 2020 dataset — apple leaf disease.

    Classes:
        0: healthy
        1: multiple_diseases
        2: rust
        3: scab

    The dataset includes severity labels for the 2021 extension:
    severity in {0=healthy, 1, 2, 3, 4} where higher = worse.

    Expected directory structure:
        root/
            PlantPathology/
                images/
                    apple_0.jpg
                    apple_1.jpg
                    ...
                train.csv  (optional, for label loading)
                test.csv   (optional)

    Args:
        root: Root directory containing PlantPathology data.
        split: 'train', 'val', or 'test'. If None, uses all data.
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
        include_severity: If True, also returns severity level (0-4).
    """

    CLASS_NAMES = ["healthy", "multiple_diseases", "rust", "scab"]
    SEVERITY_CLASSES = ["healthy", "minor", "moderate", "severe", "critical"]

    SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform=None,
        target_transform=None,
        include_severity: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.include_severity = include_severity

        self.data_dir = self.root / "PlantPathology"
        self.images_dir = self.data_dir / "images"

        if not self.data_dir.exists():
            print(
                f"PlantPathology not found at {self.data_dir}. "
                "Creating synthetic dataset..."
            )
            self._create_synthetic()

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.CLASS_NAMES)
        }

        self.samples: list[Tuple[Path, int, int]] = []

        # Try to load from CSV first
        csv_path = self.data_dir / "train.csv"
        if csv_path.exists():
            self._load_from_csv(csv_path)
        else:
            # Fallback: scan images directory
            self._load_from_directory()

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

    def _load_from_csv(self, csv_path: Path):
        """Load samples from train.csv with multi-label columns."""
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_id", row.get("image", ""))
                img_path = self.images_dir / f"{img_name}.jpg"
                if not img_path.exists():
                    # Try without extension
                    img_path = self.images_dir / img_name
                if not img_path.exists():
                    continue

                # Multi-label format: healthy,multiple_diseases,rust,scab
                if "healthy" in row:
                    label = -1
                    for cls_name in self.CLASS_NAMES:
                        if row.get(cls_name, "0") == "1":
                            label = self.class_to_idx[cls_name]
                            break
                    if label == -1:
                        label = self.class_to_idx.get("healthy", 0)
                else:
                    # Single label format
                    label = int(row.get("label", 0))

                # Severity (if available)
                severity = int(row.get("severity", 0))

                self.samples.append((img_path, label, severity))

    def _load_from_directory(self):
        """Scan images directory for files and infer labels from names."""
        if not self.images_dir.exists():
            return

        for img_path in sorted(self.images_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            name_lower = img_path.stem.lower()
            label = 0  # default healthy
            for cls_name in self.CLASS_NAMES:
                if cls_name in name_lower:
                    label = self.class_to_idx[cls_name]
                    break

            severity = 0
            for i, sev_name in enumerate(self.SEVERITY_CLASSES):
                if sev_name in name_lower:
                    severity = i
                    break

            self.samples.append((img_path, label, severity))

    def _create_synthetic(self):
        """Create synthetic dataset for testing."""
        import numpy as np

        self.images_dir.mkdir(parents=True, exist_ok=True)

        for i, cls_name in enumerate(self.CLASS_NAMES):
            for j in range(20):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(self.images_dir / f"{cls_name}_{j:04d}.jpg")

        # Create synthetic CSV
        csv_path = self.data_dir / "train.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "healthy", "multiple_diseases", "rust", "scab", "severity"])
            idx = 0
            for cls_idx, cls_name in enumerate(self.CLASS_NAMES):
                for j in range(20):
                    row = [f"{cls_name}_{j:04d}"]
                    for k in range(4):
                        row.append("1" if k == cls_idx else "0")
                    row.append(min(cls_idx, 4))
                    writer.writerow(row)
                    idx += 1

        print(f"Created synthetic PlantPathology at {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, severity = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        if self.include_severity:
            return image, label, severity
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)

    @property
    def num_severity_levels(self) -> int:
        return len(self.SEVERITY_CLASSES)

    def get_class_distribution(self) -> dict[str, int]:
        """Return sample count per class."""
        dist = {cls: 0 for cls in self.CLASS_NAMES}
        for _, label, _ in self.samples:
            dist[self.CLASS_NAMES[label]] += 1
        return dist
