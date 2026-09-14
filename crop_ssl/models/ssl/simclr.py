"""
SimCLR: Simple Framework for Contrastive Learning.

Implementation based on:
"A Simple Framework for Contrastive Learning of Visual Representations"
(Chen et al., 2020)

Key features:
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
- Two augmented views per image
- Symmetric contrastive learning
- Optional supervised contrastive loss ("SupCon", Khosla et al., 2020)
  selected with loss="supcon" when pair labels are available.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from crop_ssl.models.backbones.vit import (
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
        loss: str = "nt_xent",
    ):
        super().__init__()
        self.temperature = temperature
        if loss not in ("nt_xent", "supcon"):
            raise ValueError(
                f"Unknown loss: {loss}. Available: 'nt_xent', 'supcon'"
            )
        self.loss_type = loss

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
        # After removing diagonal, positive index shifts for i < pos
        # For i < B: positive was at i+B, shifts to i+B-1
        # For i >= B: positive was at i-B, stays at i-B
        labels = torch.arange(2 * B, device=sim.device)
        labels = torch.where(
            labels < B,
            labels + B - 1,   # i < B: positive at i+B-1
            labels - B,         # i >= B: positive at i-B
        )

        loss = F.cross_entropy(sim, labels)
        return loss

    def supcon_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised contrastive loss (SupCon, Khosla et al., 2020).

        Anchors are pulled toward all same-label projections (both views)
        and pushed away from everything else. Anchors with no positive
        (singleton classes) are excluded from the mean, so the loss stays
        finite for any label distribution.

        Args:
            z_i: Projections from view 1 (B, proj_dim).
            z_j: Projections from view 2 (B, proj_dim).
            labels: Class labels for the B samples (B,).

        Returns:
            Scalar loss.
        """
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)              # (2B, proj_dim)
        z = F.normalize(z, dim=1)
        y = torch.cat([labels, labels], dim=0)        # (2B,)

        sim = torch.mm(z, z.t()) / self.temperature   # (2B, 2B)
        # Numerical stability: shift by row max before exp
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
        positive_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
        anchor_mask = positive_mask.any(dim=1)        # anchors with >=1 positive
        if not anchor_mask.any():
            # No positives at all (all labels distinct): fall back to identity
            # pull — equivalent to NT-Xent restricted to the pair itself.
            positive_mask = torch.zeros_like(eye)
            for i in range(B):
                positive_mask[i, i + B] = True
                positive_mask[i + B, i] = True
            anchor_mask = torch.ones(2 * B, dtype=torch.bool, device=sim.device)

        exp_sim = torch.exp(sim.masked_fill(eye, float("-inf")))  # self excluded
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = positive_mask.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / pos_counts
        loss = -(mean_log_prob_pos[anchor_mask]).mean()
        return loss

    def forward(
        self,
        view_1: torch.Tensor,
        view_2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with two augmented views.

        Args:
            view_1: First augmented view (B, C, H, W).
            view_2: Second augmented view (B, C, H, W).
            labels: Class labels (B,), required when loss_type == 'supcon'.

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
        if self.loss_type == "supcon":
            if labels is None:
                raise ValueError(
                    "loss='supcon' requires pair labels; pass labels=<tensor> "
                    "or create the model with loss='nt_xent'"
                )
            loss = self.supcon_loss(z_i, z_j, labels)
        else:
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
