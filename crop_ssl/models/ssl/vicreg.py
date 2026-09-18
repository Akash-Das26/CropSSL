"""
VICReg: Variance-Invariance-Covariance Regularization.

Implementation based on:
"VICReg: Variance-Invariance-Covariance Regularization
 for Self-Supervised Learning"
(Bardes, Ponce, & Lecun, ICLR 2022)

Key features:
- Non-contrastive: no negative pairs, no momentum encoder, no large batches
- Explicitly regularizes embedding variance (anti-collapse) and covariance
  (anti-redundancy) alongside the two-view invariance objective
- Hinge losses computed on flattened per-device batches
"""

from typing import Dict

import torch
import torch.nn as nn

from crop_ssl.models.backbones.vit import (
    vit_small_patch16,
    vit_base_patch16,
    vit_large_patch16,
)
from crop_ssl.models.heads.projection import SimCLRProjectionHead


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Flattened off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICReg(nn.Module):
    """VICReg self-supervised learning model.

    Args:
        backbone: ViT backbone architecture name.
        embed_dim: Encoder embedding dimension.
        proj_dim: Projection (expander) dimension.
        sim_weight: Weight of the mean-squared-error invariance term.
        var_weight: Weight of the variance (anti-collapse) hinge term.
        cov_weight: Weight of the covariance (anti-redundancy) hinge term.
    """

    BACKBONE_REGISTRY = {
        "vit_small": vit_small_patch16,
        "vit_base": vit_base_patch16,
        "vit_large": vit_large_patch16,
    }

    def __init__(
        self,
        backbone: str = "vit_base",
        embed_dim: int = 768,
        proj_dim: int = 128,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 25.0,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

        # Build backbone
        backbone_fn = self.BACKBONE_REGISTRY[backbone]
        self.encoder = backbone_fn(embed_dim=embed_dim)
        # Note: the paper uses a 3-layer expander head; the shared 2-layer
        # SimCLRProjectionHead keeps the factory pattern uniform.
        self.projector = SimCLRProjectionHead(
            in_dim=embed_dim, out_dim=proj_dim
        )

    def vicreg_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        eps: float = 1e-4,
    ) -> Dict[str, torch.Tensor]:
        """VICReg objective over the two view projections.

        Args:
            z_i: Projections from view 1 (B, proj_dim).
            z_j: Projections from view 2 (B, proj_dim).
            eps: Variance hinge gamma (std must stay above sqrt(eps)).

        Returns:
            Dict with 'loss' plus the individual 'invariance', 'variance',
            and 'covariance' terms (all scalar tensors).
        """
        B, D = z_i.shape
        x = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Invariance: MSE between aligned pairs
        invariance = torch.nn.functional.mse_loss(z_i, z_j)

        # Variance hinge: each dimension's std across the batch must exceed 1
        std = torch.sqrt(x.var(dim=0) + eps)
        variance = torch.mean(torch.relu(1.0 - std))

        # Covariance hinge: off-diagonal covariances must vanish
        xc = x - x.mean(dim=0)
        cov = (xc.T @ xc) / (x.shape[0] - 1)
        covariance = _off_diagonal(cov).pow_(2).sum() / D

        loss = (
            self.sim_weight * invariance
            + self.var_weight * variance
            + self.cov_weight * covariance
        )
        return {
            "loss": loss,
            "invariance": invariance,
            "variance": variance,
            "covariance": covariance,
        }

    def forward(
        self,
        view_1: torch.Tensor,
        view_2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with two augmented views.

        Args:
            view_1: First augmented view (B, C, H, W).
            view_2: Second augmented view (B, C, H, W).

        Returns:
            Dict with 'loss', 'invariance', 'variance', 'covariance',
            'z_i', 'z_j', 'features'.
        """
        # Encode
        feat_i = self.encoder.forward_features(view_1)
        feat_j = self.encoder.forward_features(view_2)

        # Project
        z_i = self.projector(feat_i)
        z_j = self.projector(feat_j)

        losses = self.vicreg_loss(z_i, z_j)

        return {
            **losses,
            "z_i": z_i,
            "z_j": z_j,
            "features": torch.cat([feat_i, feat_j], dim=0),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for downstream tasks.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Feature tensor (B, D).
        """
        return self.encoder.forward_features(x)

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "encoder" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder"])
        else:
            self.encoder.load_state_dict(state_dict)
