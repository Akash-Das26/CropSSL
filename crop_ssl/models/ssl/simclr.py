"""
SimCLR: Simple Framework for Contrastive Learning.

Implementation based on:
"A Simple Framework for Contrastive Learning of Visual Representations"
(Chen et al., 2020)

Key features:
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
- Two augmented views per image
- Symmetric contrastive learning
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from crop_ssl.models.backbones.vit import (
    VisionTransformer,
    vit_small_patch16,
    vit_base_patch16,
    vit_large_patch16,
)
from crop_ssl.models.heads.projection import SimCLRProjectionHead


class SimCLR(nn.Module):
    """SimCLR self-supervised learning model.

    Args:
        backbone: ViT backbone architecture name.
        embed_dim: Embedding dimension.
        proj_dim: Projection dimension.
        temperature: Temperature for NT-Xent loss.
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
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        # Build backbone
        backbone_fn = self.BACKBONE_REGISTRY[backbone]
        self.encoder = backbone_fn(embed_dim=embed_dim)
        self.projector = SimCLRProjectionHead(
            in_dim=embed_dim, out_dim=proj_dim
        )

    def nt_xent_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """NT-Xent contrastive loss.

        Args:
            z_i: Projections from view 1 (B, proj_dim).
            z_j: Projections from view 2 (B, proj_dim).

        Returns:
            Scalar loss.
        """
        B = z_i.shape[0]

        # Concatenate projections
        z = torch.cat([z_i, z_j], dim=0)  # (2B, proj_dim)
        z = F.normalize(z, dim=1)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarities
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=sim.device)
        sim = sim.masked_select(mask).view(2 * B, 2 * B - 1)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.zeros(2 * B, dtype=torch.long, device=sim.device)

        # Adjust labels for removed diagonal
        # After removing diagonal, positive index shifts
        # For row i (i < B): positive is at i+B-1 (shifted left by 1)
        # For row i (i >= B): positive is at i-1 (shifted left by 1)
        adjusted_labels = []
        for i in range(2 * B):
            pos = i + B if i < B else i - B
            if pos > i:
                pos -= 1
            adjusted_labels.append(pos)
        labels = torch.tensor(adjusted_labels, device=sim.device)

        loss = F.cross_entropy(sim, labels)
        return loss

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
            Dict with 'loss', 'z_i', 'z_j', 'features'.
        """
        # Encode
        feat_i = self.encoder.forward_features(view_1)
        feat_j = self.encoder.forward_features(view_2)

        # Project
        z_i = self.projector(feat_i)
        z_j = self.projector(feat_j)

        # Contrastive loss
        loss = self.nt_xent_loss(z_i, z_j)

        return {
            "loss": loss,
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
