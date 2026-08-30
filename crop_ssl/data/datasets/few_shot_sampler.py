"""
Few-Shot Sampler for cross-domain evaluation.

Implements multiple sampling strategies for few-shot learning:
- Episodic sampling (N-way K-shot)
- Balanced class sampling
- Domain-aware stratified sampling
"""

import random
from typing import Dict, List

from torch.utils.data import Dataset, Sampler


class FewShotSampler(Sampler):
    """Episodic few-shot sampler.

    Samples episodes of N-way K-shot for meta-learning evaluation.

    Args:
        dataset: The dataset to sample from.
        n_way: Number of classes per episode.
        k_shot: Number of support samples per class.
        q_query: Number of query samples per class.
        num_episodes: Number of episodes to generate.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        num_episodes: int = 100,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes

        # Group indices by class
        self.class_indices: Dict[int, List[int]] = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.rng = random.Random(seed)

    def __iter__(self):
        """Generate episodic episodes."""
        all_indices: List[int] = []
        available_classes = [
            c for c, ids in self.class_indices.items()
            if len(ids) >= self.k_shot + self.q_query
        ]

        for _ in range(self.num_episodes):
            episode_classes = self.rng.sample(
                available_classes,
                min(self.n_way, len(available_classes)),
            )

            for cls in episode_classes:
                indices = self.rng.sample(
                    self.class_indices[cls],
                    self.k_shot + self.q_query,
                )
                all_indices.extend(indices)

        return iter(all_indices)

    def __len__(self) -> int:
        return self.num_episodes * self.n_way * (self.k_shot + self.q_query)

    def get_episode_info(self) -> List[Dict]:
        """Generate episode metadata for logging."""
        available_classes = [
            c for c, ids in self.class_indices.items()
            if len(ids) >= self.k_shot + self.q_query
        ]

        episodes = []
        for ep_idx in range(self.num_episodes):
            classes = self.rng.sample(
                available_classes,
                min(self.n_way, len(available_classes)),
            )
            episodes.append({
                "episode": ep_idx,
                "classes": classes,
                "k_shot": self.k_shot,
                "q_query": self.q_query,
                "total_samples": len(classes) * (self.k_shot + self.q_query),
            })
        return episodes


class BalancedClassSampler(Sampler):
    """Ensures balanced class representation during training.

    Oversamples minority classes and undersamples majority classes
    to achieve uniform class distribution.
    """

    def __init__(
        self,
        dataset: Dataset,
        samples_per_class: int = 50,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.samples_per_class = samples_per_class
        self.rng = random.Random(seed)

        self.class_indices: Dict[int, List[int]] = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            # Convert tensor label to int for dict key compatibility
            label = int(label) if hasattr(label, 'item') else label
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def __iter__(self):
        indices: List[int] = []
        for cls, cls_indices in self.class_indices.items():
            if len(cls_indices) >= self.samples_per_class:
                sampled = self.rng.sample(
                    cls_indices, self.samples_per_class
                )
            else:
                # Oversample with replacement
                sampled = [
                    cls_indices[i % len(cls_indices)]
                    for i in range(self.samples_per_class)
                ]
            indices.extend(sampled)

        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.class_indices) * self.samples_per_class


class DomainStratifiedSampler(Sampler):
    """Stratified sampler that maintains domain proportions.

    Ensures each batch has balanced representation from both
    source and target domains.
    """

    def __init__(
        self,
        source_dataset: Dataset,
        target_dataset: Dataset,
        batch_size: int = 32,
        source_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        self.source_ratio = source_ratio
        self.rng = random.Random(seed)

        self.source_indices = list(range(len(source_dataset)))
        self.target_indices = list(range(len(target_dataset)))

    def __iter__(self):
        source_per_batch = int(self.batch_size * self.source_ratio)
        target_per_batch = self.batch_size - source_per_batch

        source_idx = self.rng.sample(
            self.source_indices, len(self.source_indices)
        )
        target_idx = self.rng.sample(
            self.target_indices, len(self.target_indices)
        )

        all_indices: List[int] = []
        s_pos, t_pos = 0, 0

        total_batches = min(
            len(source_idx) // source_per_batch,
            len(target_idx) // target_per_batch,
        )

        for _ in range(total_batches):
            all_indices.extend(
                source_idx[s_pos : s_pos + source_per_batch]
            )
            all_indices.extend(
                target_idx[t_pos : t_pos + target_per_batch]
            )
            s_pos += source_per_batch
            t_pos += target_per_batch

        return iter(all_indices)

    def __len__(self) -> int:
        return min(
            len(self.source_indices),
            len(self.target_indices),
        )
