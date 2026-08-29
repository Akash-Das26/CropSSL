from __future__ import annotations
"""
Cross-Domain Dataset wrapper.

Provides a unified interface for evaluating models across multiple
plant disease datasets with different domain characteristics.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, ConcatDataset

from crop_ssl.data.datasets.plantvillage import PlantVillageDataset
from crop_ssl.data.datasets.plantdoc import PlantDocDataset
from crop_ssl.data.datasets.rice_leaf import RiceLeafDataset
from crop_ssl.data.datasets.coffee_leaf import CoffeeLeafDataset
from crop_ssl.data.datasets.domainnet_plant import DomainNetPlant


DATASET_REGISTRY = {
    "plantvillage": PlantVillageDataset,
    "plantdoc": PlantDocDataset,
    "rice_leaf": RiceLeafDataset,
    "coffee_leaf": CoffeeLeafDataset,
    "domainnet_plant": DomainNetPlant,
}


class CrossDomainDataset(Dataset):
    """Unified cross-domain dataset for evaluation.

    Combines source and target domain datasets for domain adaptation
    experiments and cross-domain robustness evaluation.

    Args:
        source_dataset_name: Name of source domain dataset.
        target_dataset_name: Name of target domain dataset.
        source_root: Root path for source dataset.
        target_root: Root path for target dataset.
        source_split: Split for source dataset.
        target_split: Split for target dataset.
        transform: Shared transform pipeline.
        few_shot_k: If set, limit target to K shots per class.
    """

    def __init__(
        self,
        source_dataset_name: str,
        target_dataset_name: str,
        source_root: str,
        target_root: str,
        source_split: str = "train",
        target_split: str = "test",
        transform=None,
        few_shot_k: Optional[int] = None,
    ):
        source_cls = DATASET_REGISTRY.get(source_dataset_name)
        target_cls = DATASET_REGISTRY.get(target_dataset_name)

        if source_cls is None:
            raise ValueError(
                f"Unknown source dataset: {source_dataset_name}. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )
        if target_cls is None:
            raise ValueError(
                f"Unknown target dataset: {target_dataset_name}. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )

        self.source_dataset = source_cls(
            root=source_root, split=source_split, transform=transform
        )
        self.target_dataset = target_cls(
            root=target_root, split=target_split, transform=transform
        )

        if few_shot_k is not None:
            self.target_dataset = self._apply_few_shot(
                self.target_dataset, few_shot_k
            )

        self.source_dataset_name = source_dataset_name
        self.target_dataset_name = target_dataset_name
        self.few_shot_k = few_shot_k

    def _apply_few_shot(
        self, dataset: Dataset, k: int
    ) -> Dataset:
        """Sample K examples per class from the dataset."""
        class_indices: Dict[int, List[int]] = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        rng = torch.Generator().manual_seed(42)
        selected: List[int] = []
        for label, indices in class_indices.items():
            perm = torch.randperm(len(indices), generator=rng)
            n_select = min(k, len(indices))
            selected.extend([indices[perm[i]] for i in range(n_select)])

        return torch.utils.data.Subset(dataset, selected)

    def __len__(self) -> int:
        return len(self.source_dataset) + len(self.target_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < len(self.source_dataset):
            return self.source_dataset[idx]
        return self.target_dataset[idx - len(self.source_dataset)]

    @property
    def num_classes(self) -> int:
        return max(
            self.source_dataset.num_classes,
            self.target_dataset.num_classes,
        )

    def get_domain_info(self) -> Dict[str, str]:
        """Return metadata about source/target domains."""
        return {
            "source": self.source_dataset_name,
            "target": self.target_dataset_name,
            "source_size": len(self.source_dataset),
            "target_size": len(self.target_dataset),
            "few_shot_k": self.few_shot_k,
        }
