"""
Active Learning for Crop Disease Detection.

Selects the most informative unlabeled samples for annotation,
maximizing model performance with minimal labeling effort.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActiveLearner:
    """Active learning query strategies.

    Args:
        model: Trained model for uncertainty estimation.
        device: Device for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def uncertainty_sampling(
        self,
        unlabeled_loader,
        n_samples: int = 100,
    ) -> List[int]:
        """Select samples with highest prediction uncertainty.

        Uses entropy of predicted probabilities as uncertainty measure.

        Args:
            unlabeled_loader: DataLoader for unlabeled data.
            n_samples: Number of samples to select.

        Returns:
            List of selected sample indices.
        """
        uncertainties = []
        indices = []

        idx_offset = 0
        for images, _ in unlabeled_loader:
            images = images.to(self.device)
            logits = self.model(images)
            probs = F.softmax(logits, dim=-1)

            # Entropy: -sum(p * log(p))
            entropy = -(probs * probs.log()).sum(dim=-1)

            batch_size = images.shape[0]
            uncertainties.extend(entropy.cpu().tolist())
            indices.extend(range(idx_offset, idx_offset + batch_size))
            idx_offset += batch_size

        # Sort by uncertainty (descending)
        ranked = sorted(
            zip(uncertainties, indices), key=lambda x: -x[0]
        )
        return [idx for _, idx in ranked[:n_samples]]

    @torch.no_grad()
    def margin_sampling(
        self,
        unlabeled_loader,
        n_samples: int = 100,
    ) -> List[int]:
        """Select samples with smallest margin between top-2 predictions.

        Low margin = model is confused between two classes.

        Args:
            unlabeled_loader: DataLoader for unlabeled data.
            n_samples: Number of samples to select.

        Returns:
            List of selected sample indices.
        """
        margins = []
        indices = []

        idx_offset = 0
        for images, _ in unlabeled_loader:
            images = images.to(self.device)
            logits = self.model(images)
            probs = F.softmax(logits, dim=-1)

            # Sort probabilities
            sorted_probs, _ = probs.sort(dim=-1, descending=True)
            # Margin = p1 - p2
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]

            batch_size = images.shape[0]
            margins.extend(margin.cpu().tolist())
            indices.extend(range(idx_offset, idx_offset + batch_size))
            idx_offset += batch_size

        # Sort by margin (ascending = least certain)
        ranked = sorted(zip(margins, indices), key=lambda x: x[0])
        return [idx for _, idx in ranked[:n_samples]]

    @torch.no_grad()
    def query_by_committee(
        self,
        unlabeled_loader,
        committee: List[nn.Module],
        n_samples: int = 100,
    ) -> List[int]:
        """Select samples with highest disagreement among committee.

        Uses vote entropy to measure committee disagreement.

        Args:
            unlabeled_loader: DataLoader for unlabeled data.
            committee: List of models forming the committee.
            n_samples: Number of samples to select.

        Returns:
            List of selected sample indices.
        """
        all_votes = []
        indices = []

        idx_offset = 0
        for images, _ in unlabeled_loader:
            images = images.to(self.device)
            batch_votes = []

            for model in committee:
                model.eval()
                logits = model(images)
                preds = logits.argmax(dim=-1)
                batch_votes.append(preds.cpu())

            # Stack votes: (committee_size, batch_size)
            votes = torch.stack(batch_votes, dim=0)
            batch_size = images.shape[0]

            # Vote entropy for each sample
            for i in range(batch_size):
                sample_votes = votes[:, i]
                unique, counts = sample_votes.unique(return_counts=True)
                probs = counts.float() / counts.sum()
                entropy = -(probs * probs.log()).sum()
                all_votes.append(entropy.item())

            indices.extend(range(idx_offset, idx_offset + batch_size))
            idx_offset += batch_size

        # Sort by disagreement (descending)
        ranked = sorted(zip(all_votes, indices), key=lambda x: -x[0])
        return [idx for _, idx in ranked[:n_samples]]

    @torch.no_grad()
    def core_set(
        self,
        unlabeled_features: np.ndarray,
        labeled_features: np.ndarray,
        n_samples: int = 100,
    ) -> List[int]:
        """Select samples that maximize coverage of feature space.

        Greedy furthest-point sampling to diversify selections.

        Args:
            unlabeled_features: Features of unlabeled data (M, D).
            labeled_features: Features of labeled data (N, D).
            n_samples: Number of samples to select.

        Returns:
            List of selected sample indices.
        """
        from scipy.spatial.distance import cdist

        # Start with labeled set centroids
        center = labeled_features.mean(axis=0, keepdims=True)

        selected = []
        remaining = list(range(len(unlabeled_features)))

        for _ in range(n_samples):
            if not remaining:
                break

            # Compute min distance from each remaining sample to selected set
            if selected:
                selected_features = unlabeled_features[selected]
                dists = cdist(
                    unlabeled_features[remaining], selected_features
                ).min(axis=1)
            else:
                dists = cdist(
                    unlabeled_features[remaining], center
                ).flatten()

            # Select the furthest point
            best_idx = np.argmax(dists)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        return selected

    def compute_al_cycle(
        self,
        labeled_loader,
        unlabeled_loader,
        val_loader,
        n_samples: int = 100,
        strategy: str = "uncertainty",
    ) -> Dict:
        """Run one active learning cycle.

        Args:
            labeled_loader: Labeled data loader.
            unlabeled_loader: Unlabeled data loader.
            val_loader: Validation data loader.
            n_samples: Samples to query.
            strategy: 'uncertainty', 'margin', or 'committee'.

        Returns:
            Dict with query results and current performance.
        """
        # Evaluate current model
        current_acc = self._evaluate(val_loader)

        # Query new samples
        if strategy == "uncertainty":
            selected = self.uncertainty_sampling(unlabeled_loader, n_samples)
        elif strategy == "margin":
            selected = self.margin_sampling(unlabeled_loader, n_samples)
        else:
            selected = self.uncertainty_sampling(unlabeled_loader, n_samples)

        return {
            "selected_indices": selected,
            "n_selected": len(selected),
            "current_accuracy": current_acc,
            "strategy": strategy,
        }

    @torch.no_grad()
    @torch.no_grad()
    def _evaluate(self, dataloader) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / max(total, 1) * 100
