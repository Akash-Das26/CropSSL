"""
Model Ensembling for Cross-Domain Robustness.

Combines predictions from multiple SSL-pretrained models for
improved accuracy and uncertainty estimation.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelEnsemble(nn.Module):
    """Ensemble of multiple SSL models with weighted averaging.

    Args:
        models: List of (model, weight) tuples.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        models: List[tuple],
        num_classes: int,
    ):
        super().__init__()
        self.models = nn.ModuleList([m for m, _ in models])
        self.weights = torch.tensor([w for _, w in models])
        self.weights = self.weights / self.weights.sum()
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with weighted ensemble prediction.

        Args:
            x: Input tensor (B, C, H, W).
            return_individual: Whether to return individual model predictions.

        Returns:
            Dict with 'logits', 'probs', 'pred', 'confidence'.
            Optionally 'individual_logits' if return_individual=True.
        """
        all_logits = []
        for model in self.models:
            if hasattr(model, "encode"):
                features = model.encode(x)
                # Add a simple linear head if no classifier
                logits = features  # Caller should have a classifier
            else:
                logits = model(x)
            all_logits.append(logits)

        all_logits = torch.stack(all_logits)  # (M, B, C)
        weights = self.weights.to(all_logits.device).view(-1, 1, 1)

        # Weighted average of logits
        ensemble_logits = (all_logits * weights).sum(dim=0)
        probs = F.softmax(ensemble_logits, dim=-1)

        result = {
            "logits": ensemble_logits,
            "probs": probs,
            "pred": probs.argmax(dim=-1),
            "confidence": probs.max(dim=-1).values,
        }

        if return_individual:
            result["individual_logits"] = all_logits
            result["individual_preds"] = all_logits.argmax(dim=-1)

        return result


class SnapshotEnsemble:
    """Snapshot ensemble — combine checkpoints from different epochs.

    Loads multiple checkpoints and averages their predictions.
    """

    def __init__(
        self,
        model_class,
        checkpoint_paths: List[str],
        num_classes: int,
        device: str = "cpu",
    ):
        self.models = []
        self.device = device

        for path in checkpoint_paths:
            model = model_class()
            ckpt = torch.load(path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            model.to(device)
            self.models.append(model)

        self.num_classes = num_classes

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict with snapshot ensemble averaging."""
        all_probs = []
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs)  # (M, B, C)
        mean_probs = all_probs.mean(dim=0)

        return {
            "logits": torch.log(mean_probs + 1e-8),
            "probs": mean_probs,
            "pred": mean_probs.argmax(dim=-1),
            "confidence": mean_probs.max(dim=-1).values,
            "std": all_probs.std(dim=0).mean(dim=-1),
        }


class AdaptiveEnsemble:
    """Domain-aware adaptive ensemble.

    Dynamically adjusts model weights based on estimated domain shift.
    Models that perform better on the current domain get higher weights.
    """

    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int,
        temperature: float = 1.0,
    ):
        self.models = models
        self.num_classes = num_classes
        self.temperature = temperature
        self.domain_weights = None

    def estimate_domain_weights(
        self, calibration_data: torch.Tensor
    ) -> torch.Tensor:
        """Estimate per-model domain weights from calibration data.

        Args:
            calibration_data: Small calibration set (N, C, H, W).

        Returns:
            Model weights (M,).
        """
        all_entropies = []
        for model in self.models:
            with torch.no_grad():
                logits = model(calibration_data)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                all_entropies.append(entropy.item())

        # Lower entropy = more confident = higher weight
        entropies = torch.tensor(all_entropies)
        weights = F.softmax(-entropies / self.temperature, dim=0)
        self.domain_weights = weights
        return weights

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict with adaptive ensemble."""
        if weights is None:
            weights = self.domain_weights
        if weights is None:
            weights = torch.ones(len(self.models)) / len(self.models)

        all_logits = []
        for model in self.models:
            logits = model(x)
            all_logits.append(logits)

        all_logits = torch.stack(all_logits)
        w = weights.to(all_logits.device).view(-1, 1, 1)
        ensemble_logits = (all_logits * w).sum(dim=0)
        probs = F.softmax(ensemble_logits, dim=-1)

        return {
            "logits": ensemble_logits,
            "probs": probs,
            "pred": probs.argmax(dim=-1),
            "confidence": probs.max(dim=-1).values,
        }
