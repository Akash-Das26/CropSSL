from __future__ import annotations
"""
iCassava 2019 Dataset (FGVC6).

Cassava disease classification — predecessor to the Cassava Leaf Disease
Kaggle competition. Same domain (Ugandan cassava fields) but different
geographic/domain-shift split. Useful for cross-dataset generalization.

5 classes: cmd, cbb, cbsd, cgm, healthy
~5,656 training images.

Source: https://www.kaggle.com/c/cassava-disease
Paper: https://arxiv.org/abs/1908.03309
"""

import csv
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ICassava2019Dataset(Dataset):
    """iCassava 2019 dataset — cassava disease from Ugandan fields.

    Classes:
        0: cassava bacterial blight (cbb)
        1: cassava brown streak disease (cbsd)
        2: cassava green mottle (cgm)
        3: cassava mosaic disease (cmd)
        4: healthy

    This dataset provides a geographic/domain-shift pair with the
    newer Cassava Leaf Disease dataset (2020) for cross-dataset
    evaluation.

    Expected directory structure:
        root/
            iCassava2019/
                train/
                    cbb/
                    cbsd/
                    cgm/
                    cmd/
                    healthy/
                train.csv  (optional)

    Args:
        root: Root directory containing iCassava2019 data.
        split: 'train', 'val', or 'test'. If None, uses all data.
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
    """

    CLASS_NAMES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]

    CLASS_DISPLAY_NAMES = [
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

        self.data_dir = self.root / "iCassava2019"
        self.train_dir = self.data_dir / "train"

        if not self.data_dir.exists():
            print(
                f"iCassava2019 not found at {self.data_dir}. "
                "Creating synthetic dataset..."
            )
            self._create_synthetic()

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.CLASS_NAMES)
        }

        self.samples: list[Tuple[Path, int]] = []

        # Try to load from CSV
        csv_path = self.data_dir / "train.csv"
        if csv_path.exists():
            self._load_from_csv(csv_path)
        else:
            # Scan class directories
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
        """Load samples from train.csv."""
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("image_id", row.get("filename", ""))
                label_str = row.get("label", row.get("category", ""))

                # Map string labels to indices
                if isinstance(label_str, str) and label_str in self.class_to_idx:
                    label = self.class_to_idx[label_str]
                else:
                    try:
                        label = int(label_str)
                    except (ValueError, TypeError):
                        continue

                img_path = self.train_dir / img_id
                if not img_path.exists():
                    img_path = self.train_dir / label_str / img_id
                if img_path.exists():
                    self.samples.append((img_path, label))

    def _load_from_directory(self):
        """Scan class subdirectories for images."""
        if not self.train_dir.exists():
            return

        for cls_name in self.CLASS_NAMES:
            cls_dir = self.train_dir / cls_name
            if not cls_dir.exists():
                # Try with full display name
                cls_dir = self.train_dir / cls_name.upper()

            if cls_dir.exists():
                label = self.class_to_idx[cls_name]
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
                    for img_path in cls_dir.glob(ext):
                        self.samples.append((img_path, label))

    def _create_synthetic(self):
        """Create synthetic dataset for testing."""
        import numpy as np

        for cls_name in self.CLASS_NAMES:
            cls_dir = self.train_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for j in range(20):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(cls_dir / f"synthetic_{j:04d}.jpg")

        # Create synthetic CSV
        csv_path = self.data_dir / "train.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "label"])
            for cls_name in self.CLASS_NAMES:
                for j in range(20):
                    writer.writerow([f"synthetic_{j:04d}.jpg", cls_name])

        print(f"Created synthetic iCassava2019 at {self.data_dir}")

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
        return len(self.CLASS_NAMES)

    def get_class_distribution(self) -> dict[str, int]:
        """Return sample count per class."""
        dist = {cls: 0 for cls in self.CLASS_NAMES}
        for _, label in self.samples:
            dist[self.CLASS_NAMES[label]] += 1
        return dist
