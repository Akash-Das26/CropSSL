"""
Confidence Calibration for Crop Disease Detection.

Post-hoc calibration methods to align predicted probabilities
with actual correctness likelihoods.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration.

    Learns a single temperature parameter T to rescale logits:
        p_i = softmax(z_i / T)

    Lower T = more confident, Higher T = less confident.

    Args:
        init_temperature: Initial temperature value.
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> Dict[str, float]:
        """Learn optimal temperature on validation set.

        Args:
            logits: Validation logits (N, C).
            labels: Validation labels (N,).
            lr: Learning rate for temperature optimization.
            max_iter: Maximum optimization iterations.

        Returns:
            Dict with 'temperature', 'ece_before', 'ece_after'.
        """
        # Compute ECE before calibration
        ece_before = self._compute_ece(logits, labels)

        # Optimize temperature
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Compute ECE after calibration
        with torch.no_grad():
            scaled = self.forward(logits)
        ece_after = self._compute_ece(scaled, labels)

        return {
            "temperature": self.temperature.item(),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "ece_improvement": ece_before - ece_after,
        }

    def _compute_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> float:
        """Compute Expected Calibration Error."""
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)
        correct = (preds == labels).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            in_bin = (max_probs > lower) & (max_probs <= upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                avg_confidence = max_probs[in_bin].mean()
                accuracy = correct[in_bin].mean()
                ece += (accuracy - avg_confidence).abs() * prop_in_bin

        return ece.item()

    @torch.no_grad()
    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return self.forward(logits)


class PlattScaling(nn.Module):
    """Platt scaling for binary/multiclass calibration.

    Learns a linear transformation: logits_new = A * logits + B
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.scale + self.bias

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        epochs: int = 50,
    ) -> Dict[str, float]:
        """Learn scaling parameters on validation set."""
        optimizer = torch.optim.LBFGS(
            [self.scale, self.bias], lr=lr, max_iter=epochs
        )

        ece_before = self._compute_ece(logits, labels)

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        with torch.no_grad():
            scaled = self.forward(logits)
        ece_after = self._compute_ece(scaled, labels)

        return {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "ece_improvement": ece_before - ece_after,
        }

    def _compute_ece(
        self, logits, labels, n_bins=15
    ) -> float:
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)
        correct = (preds == labels).float()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (max_probs > bin_boundaries[i]) & (
                max_probs <= bin_boundaries[i + 1]
            )
            prop = in_bin.float().mean()
            if prop > 0:
                ece += (correct[in_bin].mean() - max_probs[in_bin].mean()).abs() * prop
        return ece.item()


class CalibrationPipeline:
    """Full calibration pipeline: fit on val, apply on test.

    Args:
        method: 'temperature' or 'platt'.
        num_classes: Number of classes.
    """

    METHODS = ["temperature", "platt"]

    def __init__(
        self,
        method: str = "temperature",
        num_classes: int = 38,
    ):
        self.method = method
        self.num_classes = num_classes
        self.calibrator = None

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Fit calibrator on validation set.

        Args:
            val_logits: Validation logits (N, C).
            val_labels: Validation labels (N,).

        Returns:
            Calibration results dict.
        """
        if self.method == "temperature":
            self.calibrator = TemperatureScaling()
        elif self.method == "platt":
            self.calibrator = PlattScaling(self.num_classes)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        results = self.calibrator.calibrate(val_logits, val_labels)
        return results

    @torch.no_grad()
    def calibrate(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply calibration to logits."""
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self.calibrator(logits)

    def get_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute ECE for given logits."""
        if self.calibrator is None:
            # Compute raw ECE
            probs = F.softmax(logits, dim=1)
            max_probs, preds = probs.max(dim=1)
            correct = (preds == labels).float()
            bin_boundaries = torch.linspace(0, 1, 16)
            ece = 0.0
            for i in range(15):
                in_bin = (max_probs > bin_boundaries[i]) & (
                    max_probs <= bin_boundaries[i + 1]
                )
                prop = in_bin.float().mean()
                if prop > 0:
                    ece += (correct[in_bin].mean() - max_probs[in_bin].mean()).abs() * prop
            return ece.item()

        scaled = self.calibrator(logits)
        probs = F.softmax(scaled, dim=1)
        max_probs, preds = probs.max(dim=1)
        correct = (preds == labels).float()
        bin_boundaries = torch.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            in_bin = (max_probs > bin_boundaries[i]) & (
                max_probs <= bin_boundaries[i + 1]
            )
            prop = in_bin.float().mean()
            if prop > 0:
                ece += (correct[in_bin].mean() - max_probs[in_bin].mean()).abs() * prop
        return ece.item()
