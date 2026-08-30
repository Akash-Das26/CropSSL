from __future__ import annotations
"""
DomainNet-Plant: Custom domain-shifted plant disease dataset.

Simulates realistic domain shifts (weather conditions, camera types,
geographic variations) for robustness evaluation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class DomainNetPlant(Dataset):
    """Cross-domain plant disease dataset with multiple domains.

    Domains include:
        - studio: Controlled lab conditions (source domain)
        - greenhouse: Semi-controlled greenhouse conditions
        - field: Natural field conditions with varying weather
        - mobile: Mobile phone captures with varying quality
        - aerial: Drone/aerial imagery

    Args:
        root: Root directory containing domain data.
        domain: Which domain to use ('studio', 'greenhouse', etc.).
            If None, uses all domains.
        split: 'train', 'val', or 'test'.
        transform: Optional transform for images.
        target_transform: Optional transform for targets.
    """

    DOMAINS = ["studio", "greenhouse", "field", "mobile", "aerial"]

    CLASS_NAMES = [
        "apple_scab",
        "apple_rust",
        "corn_rust",
        "corn_blight",
        "grape_rot",
        "grape_black_rot",
        "tomato_blight",
        "tomato_bacterial_spot",
        "potato_early_blight",
        "potato_late_blight",
        "pepper_bacterial_spot",
        "healthy",
    ]

    def __init__(
        self,
        root: str,
        domain: Optional[str] = None,
        split: Optional[str] = "train",
        transform=None,
        target_transform=None,
    ):
        self.root = Path(root)
        self.domain = domain
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.CLASS_NAMES)
        }

        self.samples: list[Tuple[Path, int, str]] = []
        domains_to_load = [domain] if domain else self.DOMAINS

        for d in domains_to_load:
            domain_dir = self.root / "DomainNetPlant" / d
            if not domain_dir.exists():
                continue
            for cls_name in self.CLASS_NAMES:
                cls_dir = domain_dir / cls_name
                if not cls_dir.exists():
                    continue
                for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG"):
                    for img_path in cls_dir.glob(ext):
                        self.samples.append(
                            (img_path, self.class_to_idx[cls_name], d)
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
        img_path, label, _ = self.samples[idx]
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

    def get_domain_stats(self) -> Dict[str, int]:
        """Get sample count per domain."""
        stats: Dict[str, int] = {}
        for _, _, d in self.samples:
            stats[d] = stats.get(d, 0) + 1
        return stats
