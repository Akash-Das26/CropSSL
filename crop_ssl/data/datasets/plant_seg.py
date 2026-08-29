from __future__ import annotations
"""
PlantSeg Dataset (2024).

Plant disease segmentation — the largest disease segmentation dataset.
11,400+ images with pixel-level annotations for 115 plant diseases.
Collected in-the-wild from diverse real environments.

Source: https://github.com/tqwei05/PlantSeg
Paper: https://www.nature.com/articles/s41597-025-06513-4
Data: Zenodo (linked in repo)

This dataset provides segmentation masks, enabling models to learn
spatial disease localization — a much stronger robustness signal than
classification alone.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class PlantSegDataset(Dataset):
    """PlantSeg dataset — pixel-level disease segmentation.

    Features:
        - 11,400+ images from diverse real environments
        - 115 plant disease classes
        - Segmentation masks (pixel-level annotations)
        - In-the-wild collected (not lab-cleaned)

    The dataset supports both:
        1. Segmentation mode (returns image + mask)
        2. Classification mode (returns image + class label, derived from mask)

    Expected directory structure:
        root/
            PlantSeg/
                images/
                    img_000001.jpg
                    img_000002.jpg
                    ...
                masks/
                    img_000001.png
                    img_000002.png
                    ...
                class_map.json  (optional: maps pixel values to disease names)
                split.json       (optional: train/val/test splits)

    Args:
        root: Root directory containing PlantSeg data.
        split: 'train', 'val', or 'test'. If None, uses all data.
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
        mode: 'segmentation' (returns mask) or 'classification' (returns class).
        max_classes: If set, limit to top N most frequent classes.
    """

    # Top common disease categories (subset of 115)
    DISEASE_CATEGORIES = [
        "healthy", "bacterial_spot", "early_blight", "late_blight",
        "leaf_mold", "septoria_leaf_spot", "spider_mites",
        "target_spot", "yellow_leaf_curl", "mosaic_virus",
        "rust", "powdery_mildew", "downy_mildew", "anthracnose",
        "cercospora_leaf_spot", "bacterial_wilt", "fusarium_wilt",
        "root_rot", "wilt", "blight", "leaf_spot", "fruit_rot",
        "flower_blight", "stem_canker", "canker", "scab",
        "fire_blight", "brown_rot", "black_rot", "leaf_blight",
        "gray_mold", "soft_rot", "bacterial_blight",
    ]

    NUM_CLASSES_FULL = 115  # Full dataset has 115 classes

    SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        transform=None,
        target_transform=None,
        mode: str = "classification",
        max_classes: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.max_classes = max_classes

        self.data_dir = self.root / "PlantSeg"
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"

        # Load class mapping first (needed by _create_synthetic)
        self.class_names = self._load_class_map()
        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        if not self.data_dir.exists():
            print(
                f"PlantSeg not found at {self.data_dir}. "
                "Creating synthetic dataset..."
            )
            self._create_synthetic()

        # Load samples
        self.samples: list[Tuple[Path, int]] = []
        self.mask_samples: list[Tuple[Path, Path, int]] = []
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
                indices = splits[split]
                self.samples = [self.samples[i] for i in indices]
                self.mask_samples = [self.mask_samples[i] for i in indices]

    def _load_class_map(self) -> list[str]:
        """Load class names from JSON or use defaults."""
        class_map_path = self.data_dir / "class_map.json"
        if class_map_path.exists():
            with open(class_map_path) as f:
                mapping = json.load(f)
            # Could be {name: pixel_val} or {pixel_val: name}
            if isinstance(mapping, dict):
                if all(isinstance(v, str) for v in list(mapping.values())[:5]):
                    return list(mapping.keys())
                return list(mapping.values())
        return self.DISEASE_CATEGORIES[:min(
            self.max_classes or len(self.DISEASE_CATEGORIES),
            len(self.DISEASE_CATEGORIES),
        )]

    def _load_samples(self):
        """Load image paths and labels."""
        if not self.images_dir.exists():
            return

        for img_path in sorted(self.images_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                continue

            # Try to determine class from filename or mask
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            label = self._infer_label_from_mask(mask_path) if mask_path.exists() else self._infer_label_from_name(img_path.name)

            if self.max_classes and label >= self.max_classes:
                continue

            self.samples.append((img_path, label))
            self.mask_samples.append((img_path, mask_path, label))

    def _infer_label_from_mask(self, mask_path: Path) -> int:
        """Infer class label from dominant class in segmentation mask."""
        try:
            mask = Image.open(mask_path)
            if mask.mode != "L":
                mask = mask.convert("L")
            import numpy as np
            mask_arr = np.array(mask)
            # Use most frequent non-zero value as class
            unique, counts = np.unique(mask_arr[mask_arr > 0], return_counts=True)
            if len(unique) > 0:
                dominant = unique[counts.argmax()]
                return int(dominant) % len(self.class_names)
        except Exception:
            pass
        return 0

    def _infer_label_from_name(self, name: str) -> int:
        """Infer class label from filename keywords."""
        name_lower = name.lower()
        for cls_idx, cls_name in enumerate(self.class_names):
            if cls_name.replace("_", " ") in name_lower or cls_name in name_lower:
                return cls_idx
        return 0

    def _create_synthetic(self):
        """Create synthetic dataset for testing."""
        import numpy as np

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        for i, cls_name in enumerate(self.class_names[:min(20, len(self.class_names))]):
            for j in range(20):
                # Image
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(self.images_dir / f"{cls_name}_{j:04d}.jpg")

                # Mask (sparse — only some pixels are labeled)
                mask = np.zeros((224, 224), dtype=np.uint8)
                # Create a small region for this class
                y, x = np.random.randint(50, 174, 2)
                mask[y:y+50, x:x+50] = (i + 1) % 256
                mask_img = Image.fromarray(mask)
                mask_img.save(self.masks_dir / f"{cls_name}_{j:04d}.png")

        # Save class map
        class_map = {name: idx for idx, name in enumerate(self.class_names[:20])}
        with open(self.data_dir / "class_map.json", "w") as f:
            json.dump(class_map, f, indent=2)

        print(f"Created synthetic PlantSeg at {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.mode == "segmentation" and idx < len(self.mask_samples):
            img_path, mask_path, label = self.mask_samples[idx]
            try:
                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path)
                if mask.mode != "L":
                    mask = mask.convert("L")
            except Exception:
                image = Image.new("RGB", (224, 224), (128, 128, 128))
                mask = Image.new("L", (224, 224), 0)

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            # Convert mask to tensor
            import torchvision.transforms.functional as TF
            mask_tensor = TF.to_tensor(mask).squeeze().long()

            return image, mask_tensor, label

        else:
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
        return min(len(self.class_names), self.max_classes or len(self.class_names))

    def get_class_distribution(self) -> dict[str, int]:
        """Return sample count per class."""
        dist = {cls: 0 for cls in self.class_names[:self.num_classes]}
        for _, label in self.samples:
            if label < len(self.class_names):
                dist[self.class_names[label]] += 1
        return dist

    def get_mask_stats(self) -> dict:
        """Return statistics about mask coverage."""
        total_pixels = 0
        labeled_pixels = 0
        for _, mask_path, _ in self.mask_samples:
            if not mask_path.exists():
                continue
            try:
                import numpy as np
                mask = np.array(Image.open(mask_path))
                total_pixels += mask.size
                labeled_pixels += (mask > 0).sum()
            except Exception:
                continue
        return {
            "total_pixels": total_pixels,
            "labeled_pixels": labeled_pixels,
            "coverage_ratio": labeled_pixels / max(total_pixels, 1),
        }
