from __future__ import annotations
"""
BRACOL Dataset (2019).

Brazilian Arabica coffee leaf images captured on 5 different smartphone models
to maximize sensor-domain diversity. 1,747 images with severity labels for
4 biotic stresses.

Mendeley: https://data.mendeley.com/datasets/yy2k5y8mxg/1

Key value: Built explicitly around device heterogeneity (different phone cameras
= different noise/color profiles), making it ideal for testing robustness to
camera/sensor variation — a different axis of domain shift from background/lighting.
"""

import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class BRACOLDataset(Dataset):
    """BRACOL dataset — Brazilian Arabica coffee leaf disease.

    Features:
        - 1,747 images of Arabica coffee leaves
        - 5 different smartphone models (sensor diversity)
        - 4 biotic stress classes + healthy
        - Severity labels for each disease
        - Captured in Brazilian coffee plantations

    Smartphone Models (sensor diversity):
        - Model A: High-end (e.g., Samsung Galaxy S series)
        - Model B: Mid-range (e.g., Moto G series)
        - Model C: Low-end (e.g., Samsung J series)
        - Model D: Another brand (e.g., Xiaomi Redmi)
        - Model E: Budget (e.g., older model)

    Classes:
        0: healthy
        1: coffee_leaf_rust (CLR)
        2: coffee_leaf_miner (CLM)
        3: coffee_leaf_phoma
        4: coffee_cercosporiosis

    Severity: 0-3 scale (0=healthy, 1=mild, 2=moderate, 3=severe)

    Expected directory structure:
        root/
            BRACOL/
                images/
                    img_0001.jpg
                    img_0002.jpg
                    ...
                metadata.csv
                    columns: image_id, class, severity, phone_model, ...

    Args:
        root: Root directory containing BRACOL data.
        split: 'train', 'val', or 'test'. If None, uses all data.
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
        task: 'classification', 'severity', or 'both'.
        include_phone_model: If True, also returns phone model ID.
        download: If True, create synthetic fallback if missing.
    """

    CLASS_NAMES = [
        "healthy",
        "coffee_leaf_rust",
        "coffee_leaf_miner",
        "coffee_leaf_phoma",
        "coffee_cercosporiosis",
    ]

    CLASS_DISPLAY_NAMES = [
        "Healthy",
        "Coffee Leaf Rust (CLR)",
        "Coffee Leaf Miner (CLM)",
        "Coffee Leaf Phoma",
        "Coffee Cercosporiosis",
    ]

    SEVERITY_LEVELS = {
        0: "healthy",
        1: "mild",
        2: "moderate",
        3: "severe",
    }

    PHONE_MODELS = [
        "samsung_galaxy_s",     # High-end
        "motorola_moto_g",      # Mid-range
        "samsung_galaxy_j",     # Low-end
        "xiaomi_redmi",         # Another brand
        "budget_generic",       # Budget
    ]

    SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

    MENDELEY_URL = "https://data.mendeley.com/datasets/yy2k5y8mxg/1"

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform=None,
        target_transform=None,
        task: str = "classification",
        include_phone_model: bool = False,
        download: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.include_phone_model = include_phone_model

        self.data_dir = self.root / "BRACOL"
        self.images_dir = self.data_dir / "images"

        if not self.data_dir.exists():
            if download:
                print(
                    f"BRACOL not found at {self.data_dir}. "
                    "Creating synthetic dataset..."
                )
                self._create_synthetic()
            else:
                raise FileNotFoundError(
                    f"BRACOL not found at {self.data_dir}. "
                    f"Download from: {self.MENDELEY_URL}"
                )

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.CLASS_NAMES)
        }
        self.phone_to_idx = {
            name: idx for idx, name in enumerate(self.PHONE_MODELS)
        }

        # Samples: (img_path, class_idx, severity, phone_idx)
        self.samples: list[Tuple[Path, int, int, int]] = []
        self._load_samples()

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

    def _load_samples(self):
        """Load from metadata CSV or scan directory."""
        csv_path = self.data_dir / "metadata.csv"
        if csv_path.exists():
            self._load_from_csv(csv_path)
        else:
            self._load_from_directory()

    def _load_from_csv(self, csv_path: Path):
        """Load from metadata CSV."""
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_id", row.get("filename", ""))
                img_path = self.images_dir / img_name

                if not img_path.exists():
                    continue

                # Class
                class_name = row.get("class", "healthy").lower().strip()
                class_idx = self.class_to_idx.get(class_name, 0)

                # Severity
                try:
                    severity = int(row.get("severity", 0))
                    severity = max(0, min(3, severity))
                except (ValueError, TypeError):
                    severity = 0

                # Phone model
                phone = row.get("phone_model", self.PHONE_MODELS[0]).lower().strip()
                phone_idx = self.phone_to_idx.get(phone, 0)

                self.samples.append((img_path, class_idx, severity, phone_idx))

    def _load_from_directory(self):
        """Scan images directory and infer from filenames."""
        if not self.images_dir.exists():
            return

        for img_path in sorted(self.images_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            name_lower = img_path.stem.lower()

            # Infer class
            class_idx = 0
            for cls_name in self.CLASS_NAMES:
                keywords = cls_name.split("_")
                if any(kw in name_lower for kw in keywords if len(kw) > 2):
                    class_idx = self.class_to_idx[cls_name]
                    break

            # Infer severity
            severity = 0
            if "severe" in name_lower or "sev" in name_lower:
                severity = 3
            elif "moderate" in name_lower or "mod" in name_lower:
                severity = 2
            elif "mild" in name_lower or "light" in name_lower:
                severity = 1

            # Infer phone model
            phone_idx = 0
            for i, phone in enumerate(self.PHONE_MODELS):
                keywords = phone.split("_")
                if any(kw in name_lower for kw in keywords if len(kw) > 2):
                    phone_idx = i
                    break

            self.samples.append((img_path, class_idx, severity, phone_idx))

    def _create_synthetic(self):
        """Create synthetic dataset simulating 5 different phone sensors."""
        import numpy as np

        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Simulate different phone color profiles
        phone_tints = [
            [10, 0, -5],      # Slight warm
            [-5, 5, 10],      # Slight cool
            [0, 0, 0],        # Neutral
            [15, -10, 5],     # Warm + green
            [-10, -5, 15],    # Cool + blue
        ]

        for i, cls_name in enumerate(self.CLASS_NAMES):
            n_images = 15 + np.random.randint(0, 20)
            for j in range(n_images):
                phone_idx = np.random.randint(5)
                severity = 0 if cls_name == "healthy" else np.random.randint(1, 4)

                # Base image
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

                # Apply phone-specific color tint (simulating sensor differences)
                tint = phone_tints[phone_idx]
                for c in range(3):
                    arr[:, :, c] = np.clip(
                        arr[:, :, c].astype(int) + tint[c], 0, 255
                    ).astype(np.uint8)

                # Add slight green bias for coffee leaf
                arr[:, :, 1] = np.clip(
                    arr[:, :, 1].astype(int) + 20, 0, 255
                ).astype(np.uint8)

                img = Image.fromarray(arr)

                phone_name = self.PHONE_MODELS[phone_idx]
                img_name = f"{cls_name}_s{severity}_{phone_name}_{j:04d}.jpg"
                img.save(self.images_dir / img_name)

        # Create metadata CSV
        metadata_path = self.data_dir / "metadata.csv"
        with open(metadata_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "class", "severity", "phone_model"])
            for cls_name in self.CLASS_NAMES:
                n_images = 15 + np.random.randint(0, 20)
                for j in range(n_images):
                    phone_idx = np.random.randint(5)
                    severity = 0 if cls_name == "healthy" else np.random.randint(1, 4)
                    phone_name = self.PHONE_MODELS[phone_idx]
                    img_name = f"{cls_name}_s{severity}_{phone_name}_{j:04d}.jpg"
                    writer.writerow([img_name, cls_name, severity, phone_name])

        print(f"Created synthetic BRACOL at {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_idx, severity, phone_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        label = class_idx
        if self.target_transform:
            label = self.target_transform(label)

        if self.task == "severity":
            return image, torch.tensor(severity, dtype=torch.long)
        elif self.task == "both":
            return image, (
                torch.tensor(class_idx, dtype=torch.long),
                torch.tensor(severity, dtype=torch.long),
            )
        elif self.include_phone_model:
            return image, label, torch.tensor(phone_idx, dtype=torch.long)
        else:
            return image, label

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)

    @property
    def num_phone_models(self) -> int:
        return len(self.PHONE_MODELS)

    def get_class_distribution(self) -> dict[str, int]:
        """Return sample count per class."""
        dist = {cls: 0 for cls in self.CLASS_NAMES}
        for _, class_idx, _, _ in self.samples:
            dist[self.CLASS_NAMES[class_idx]] += 1
        return dist

    def get_phone_distribution(self) -> dict[str, int]:
        """Return sample count per phone model."""
        dist = {phone: 0 for phone in self.PHONE_MODELS}
        for _, _, _, phone_idx in self.samples:
            dist[self.PHONE_MODELS[phone_idx]] += 1
        return dist

    def get_severity_distribution(self) -> dict[str, int]:
        """Return sample count per severity level."""
        dist = {name: 0 for name in self.SEVERITY_LEVELS.values()}
        for _, _, severity, _ in self.samples:
            dist[self.SEVERITY_LEVELS[severity]] += 1
        return dist
