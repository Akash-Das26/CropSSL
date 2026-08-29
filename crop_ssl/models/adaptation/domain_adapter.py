"""
Domain Adaptation Module.

Implements domain adaptation techniques for cross-domain
robustness in crop disease detection:
1. Domain-Adversarial Neural Network (DANN)
2. Maximum Mean Discrepancy (MMD) alignment
3. Feature alignment with CORAL loss
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalLayer(Function):
    """Gradient Reversal Layer for adversarial domain adaptation.

    Passes features forward unchanged but reverses gradients
    during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainAdversarialHead(nn.Module):
    """Domain discriminator for DANN.

    Predicts whether features come from source or target domain.

    Args:
        input_dim: Feature dimension.
        hidden_dim: Hidden layer dimension.
        num_domains: Number of domains (typically 2).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_domains: int = 2,
    ):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(
        self, features: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """Forward with gradient reversal.

        Args:
            features: Input features (B, D).
            alpha: Gradient reversal strength.

        Returns:
            Domain predictions (B, num_domains).
        """
        reversed_features = GradientReversalLayer.apply(features, alpha)
        return self.discriminator(reversed_features)


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss.

    Aligns source and target feature distributions using
    Gaussian kernel with multiple bandwidths.

    Args:
        kernel_bandwidths: List of kernel bandwidth values.
    """

    def __init__(
        self,
        kernel_bandwidths: Optional[list] = None,
    ):
        super().__init__()
        if kernel_bandwidths is None:
            kernel_bandwidths = [0.1, 1.0, 10.0]
        self.bandwidths = kernel_bandwidths

    def gaussian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bandwidth: float,
    ) -> torch.Tensor:
        """Compute Gaussian kernel matrix."""
        xx = x @ x.T
        yy = y @ y.T
        xy = x @ y.T

        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.T + rx - 2 * xx
        dyy = ry.T + ry - 2 * yy
        dxy = rx.T + ry - 2 * xy

        Kxx = torch.exp(-dxx / (2 * bandwidth**2))
        Kyy = torch.exp(-dyy / (2 * bandwidth**2))
        Kxy = torch.exp(-dxy / (2 * bandwidth**2))

        return Kxx, Kyy, Kxy

    def forward(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute MMD loss.

        Args:
            source_features: Source domain features (Ns, D).
            target_features: Target domain features (Nt, D).

        Returns:
            Scalar MMD loss.
        """
        loss = 0.0
        for bw in self.bandwidths:
            Kxx, Kyy, Kxy = self.gaussian_kernel(
                source_features, target_features, bw
            )
            n_s = source_features.shape[0]
            n_t = target_features.shape[0]

            loss += Kxx.sum() / (n_s * n_s)
            loss += Kyy.sum() / (n_t * n_t)
            loss -= 2 * Kxy.sum() / (n_s * n_t)

        return loss / len(self.bandwidths)


class CORALLoss(nn.Module):
    """CORrelation ALignment loss.

    Aligns second-order statistics (covariance) of source
    and target feature distributions.

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim

    def forward(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute CORAL loss.

        Args:
            source_features: Source features (Ns, D).
            target_features: Target features (Nt, D).

        Returns:
            Scalar CORAL loss.
        """
        d = source_features.shape[1]

        # Source covariance
        xm = source_features - source_features.mean(dim=0, keepdim=True)
        xc = (xm.T @ xm) / (source_features.shape[0] - 1)

        # Target covariance
        xmt = target_features - target_features.mean(dim=0, keepdim=True)
        xct = (xmt.T @ xmt) / (target_features.shape[0] - 1)

        # Frobenius norm of difference
        loss = (xc - xct).pow(2).sum() / (4 * d * d)

        return loss


class DomainAdaptationModule(nn.Module):
    """Unified domain adaptation module.

    Combines task classifier with domain adaptation losses.

    Args:
        backbone: Feature encoder backbone.
        num_classes: Number of task classes.
        adaptation_type: One of 'dann', 'mmd', 'coral', 'combined'.
        input_dim: Feature dimension from backbone.
    """

    ADAPTATION_TYPES = ["dann", "mmd", "coral", "combined"]

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        adaptation_type: str = "dann",
        input_dim: int = 768,
    ):
        super().__init__()
        self.backbone = backbone
        self.adaptation_type = adaptation_type

        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Domain adaptation components
        if adaptation_type in ("dann", "combined"):
            self.domain_head = DomainAdversarialHead(input_dim)

        if adaptation_type in ("mmd", "combined"):
            self.mmd_loss = MMDLoss()

        if adaptation_type in ("coral", "combined"):
            self.coral_loss = CORALLoss(input_dim)

    def forward(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
        alpha: float = 1.0,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for domain adaptation.

        Args:
            source_x: Source domain images (B, C, H, W).
            target_x: Target domain images (B, C, H, W).
            alpha: Domain adversarial strength (for DANN).
            return_features: Whether to return raw features.

        Returns:
            Dict with 'task_loss', 'domain_loss', 'total_loss',
            'source_logits', 'target_logits', and optionally 'features'.
        """
        # Extract features
        source_features = self.backbone.forward_features(source_x)
        target_features = self.backbone.forward_features(target_x)

        # Task predictions
        source_logits = self.task_classifier(source_features)
        target_logits = self.task_classifier(target_features)

        # Domain adaptation loss
        domain_loss = torch.tensor(0.0, device=source_x.device)

        if self.adaptation_type in ("dann", "combined"):
            # Source = label 0, Target = label 1
            source_domain = self.domain_head(source_features, alpha)
            target_domain = self.domain_head(target_features, alpha)

            B = source_x.shape[0]
            domain_labels = torch.cat([
                torch.zeros(B, dtype=torch.long, device=source_x.device),
                torch.ones(B, dtype=torch.long, device=source_x.device),
            ])
            domain_preds = torch.cat([source_domain, target_domain], dim=0)
            domain_loss += F.cross_entropy(domain_preds, domain_labels)

        if self.adaptation_type in ("mmd", "combined"):
            domain_loss += self.mmd_loss(source_features, target_features)

        if self.adaptation_type in ("coral", "combined"):
            domain_loss += self.coral_loss(source_features, target_features)

        result = {
            "source_logits": source_logits,
            "target_logits": target_logits,
            "domain_loss": domain_loss,
        }

        if return_features:
            result["source_features"] = source_features
            result["target_features"] = target_features

        return result

    def compute_task_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute task classification loss."""
        return F.cross_entropy(logits, labels)

    def compute_total_loss(
        self,
        task_loss: torch.Tensor,
        domain_loss: torch.Tensor,
        domain_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute total combined loss."""
        return task_loss + domain_weight * domain_loss
