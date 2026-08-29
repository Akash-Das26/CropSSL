from __future__ import annotations
"""
DiaMOS Plant Dataset (2021).

Pear leaf disease and severity dataset from Italy.
3,505 field images with continuous severity levels (0–100% leaf area affected),
plus fruit growth-stage images collected across an entire growing season.

Zenodo: https://doi.org/10.5281/zenodo.5557313
Kaggle: https://www.kaggle.com/datasets/alexandraneagu101/diamos-plant-dataset

Key value: Provides continuous severity regression targets (not just binary
healthy/disease), enabling severity estimation as an additional task.
"""

import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class DiaMOSPlantDataset(Dataset):
    """DiaMOS Plant dataset — pear disease with severity regression.

    Features:
        - 3,505 field images of pear leaves/fruit
        - Continuous severity levels (0–100% leaf area affected)
        - Collected across an entire growing season in Italy
        - Fruit growth-stage annotations
        - Disease type classification + severity regression

    Tasks supported:
        1. Disease classification (binary: healthy vs diseased)
        2. Disease type classification (multi-class)
        3. Severity regression (continuous 0–100%)

    Expected directory structure:
        root/
            DiaMOSPlant/
                images/
                    img_0001.jpg
                    img_0002.jpg
                    ...
                annotations.csv
                    columns: image_id, disease, severity, growth_stage, ...

    Args:
        root: Root directory containing DiaMOSPlant data.
        split: 'train', 'val', or 'test'. If None, uses all data.
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
        task: 'classification', 'severity', or 'both'.
        download: If True, create synthetic fallback if missing.
    """

    # Disease classes
    DISEASE_CLASSES = [
        "healthy", "rust", "fire_blight", "pear_pear_psylla",
        "black_spot", "brown_spot", "leaf_spot", "scab",
        "pear_midge", "aphid_damage",
    ]

    # Growth stages across the season
    GROWTH_STAGES = [
        "bud_break", "early_leaf", "full_bloom",
        "fruit_set", "fruit_growth", "maturation", "harvest",
    ]

    SEVERITY_BINS = [0, 10, 25, 50, 75, 100]  # For severity classification

    SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform=None,
        target_transform=None,
        task: str = "classification",
        download: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.task = task

        self.data_dir = self.root / "DiaMOSPlant"
        self.images_dir = self.data_dir / "images"

        if not self.data_dir.exists():
            if download:
                print(
                    f"DiaMOSPlant not found at {self.data_dir}. "
                    "Creating synthetic dataset..."
                )
                self._create_synthetic()
            else:
                raise FileNotFoundError(
                    f"DiaMOSPlant not found at {self.data_dir}. "
                    "Download from: https://doi.org/10.5281/zenodo.5557313"
                )

        self.disease_to_idx = {
            name: idx for idx, name in enumerate(self.DISEASE_CLASSES)
        }
        self.stage_to_idx = {
            name: idx for idx, name in enumerate(self.GROWTH_STAGES)
        }

        # Samples: (img_path, disease_idx, severity_float, growth_stage_idx)
        self.samples: list[Tuple[Path, int, float, int]] = []
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
        """Load from annotations CSV or scan directory."""
        csv_path = self.data_dir / "annotations.csv"
        if csv_path.exists():
            self._load_from_csv(csv_path)
        else:
            self._load_from_directory()

    def _load_from_csv(self, csv_path: Path):
        """Load from annotations CSV."""
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_id", row.get("filename", ""))
                img_path = self.images_dir / img_name

                if not img_path.exists():
                    continue

                # Disease class
                disease = row.get("disease", "healthy").lower().strip()
                disease_idx = self.disease_to_idx.get(disease, 0)

                # Severity (0-100 float)
                try:
                    severity = float(row.get("severity", 0.0))
                    severity = max(0.0, min(100.0, severity))
                except (ValueError, TypeError):
                    severity = 0.0

                # Growth stage
                stage = row.get("growth_stage", "full_bloom").lower().strip()
                stage_idx = self.stage_to_idx.get(stage, 3)  # default full_bloom

                self.samples.append((img_path, disease_idx, severity, stage_idx))

    def _load_from_directory(self):
        """Scan images directory and infer labels from filenames."""
        if not self.images_dir.exists():
            return

        for img_path in sorted(self.images_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            name_lower = img_path.stem.lower()

            # Infer disease from name
            disease_idx = 0
            for cls_name in self.DISEASE_CLASSES:
                if cls_name.replace("_", "") in name_lower.replace("_", ""):
                    disease_idx = self.disease_to_idx[cls_name]
                    break

            # Infer severity from name (e.g., "severity_45")
            severity = 0.0
            if "severity" in name_lower:
                try:
                    parts = name_lower.split("severity")
                    severity = float(parts[-1].strip("_").split("_")[0])
                except (ValueError, IndexError):
                    pass

            # Infer growth stage
            stage_idx = 3
            for stage_name, idx in self.stage_to_idx.items():
                if stage_name in name_lower:
                    stage_idx = idx
                    break

            self.samples.append((img_path, disease_idx, severity, stage_idx))

    def _create_synthetic(self):
        """Create synthetic dataset for testing."""
        import numpy as np

        self.images_dir.mkdir(parents=True, exist_ok=True)

        for i, cls_name in enumerate(self.DISEASE_CLASSES):
            n_images = 15 + np.random.randint(0, 20)
            for j in range(n_images):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                # Add some green tint (pear leaf characteristics)
                arr[:, :, 1] = np.clip(
                    arr[:, :, 1].astype(int) + 30, 0, 255
                ).astype(np.uint8)
                img = Image.fromarray(arr)

                severity = np.random.uniform(0, 100) if cls_name != "healthy" else 0.0
                stage = self.GROWTH_STAGES[np.random.randint(len(self.GROWTH_STAGES))]

                img_name = f"{cls_name}_s{int(severity)}_{stage}_{j:04d}.jpg"
                img.save(self.images_dir / img_name)

        # Create annotations CSV
        annotations_path = self.data_dir / "annotations.csv"
        with open(annotations_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "disease", "severity", "growth_stage"])
            for cls_name in self.DISEASE_CLASSES:
                n_images = 15 + np.random.randint(0, 20)
                for j in range(n_images):
                    severity = np.random.uniform(0, 100) if cls_name != "healthy" else 0.0
                    stage = self.GROWTH_STAGES[np.random.randint(len(self.GROWTH_STAGES))]
                    img_name = f"{cls_name}_s{int(severity)}_{stage}_{j:04d}.jpg"
                    writer.writerow([img_name, cls_name, f"{severity:.1f}", stage])

        print(f"Created synthetic DiaMOSPlant at {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, disease_idx, severity, stage_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        label = disease_idx
        if self.target_transform:
            label = self.target_transform(label)

        if self.task == "severity":
            # Return severity as float for regression
            return image, torch.tensor(severity, dtype=torch.float32)
        elif self.task == "both":
            # Return (disease_class, severity)
            return image, (
                torch.tensor(disease_idx, dtype=torch.long),
                torch.tensor(severity, dtype=torch.float32),
            )
        else:
            return image, label

    @property
    def num_classes(self) -> int:
        return len(self.DISEASE_CLASSES)

    @property
    def has_severity(self) -> bool:
        return True

    def get_severity_stats(self) -> dict:
        """Return severity distribution statistics."""
        import numpy as np
        severities = [s[2] for s in self.samples]
        arr = np.array(severities)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
            "num_healthy": int((arr == 0).sum()),
            "num_diseased": int((arr > 0).sum()),
        }

    def get_class_distribution(self) -> dict[str, int]:
        """Return sample count per disease class."""
        dist = {cls: 0 for cls in self.DISEASE_CLASSES}
        for _, disease_idx, _, _ in self.samples:
            dist[self.DISEASE_CLASSES[disease_idx]] += 1
        return dist

    def get_growth_stage_distribution(self) -> dict[str, int]:
        """Return sample count per growth stage."""
        dist = {stage: 0 for stage in self.GROWTH_STAGES}
        for _, _, _, stage_idx in self.samples:
            dist[self.GROWTH_STAGES[stage_idx]] += 1
        return dist
