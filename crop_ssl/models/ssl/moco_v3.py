"""
MoCo v3: Momentum Contrast v3.

Implementation based on:
"An Empirical Study of Training Self-Supervised Vision Transformers"
(Wang et al., 2021)

Key features:
- Momentum-updated key encoder
- Learnable queue with temperature scaling
- Asymmetric contrastive learning
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crop_ssl.models.backbones.vit import (
    VisionTransformer,
    vit_small_patch16,
    vit_base_patch16,
    vit_large_patch16,
)
from crop_ssl.models.heads.projection import MoCoProjectionHead


class MoCoV3(nn.Module):
    """MoCo v3 self-supervised learning model.

    Args:
        backbone: ViT backbone architecture name.
        embed_dim: Embedding dimension.
        proj_dim: Projection dimension.
        queue_size: Size of the key queue.
        momentum: Momentum for EMA update.
        temperature: Temperature for InfoNCE loss.
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
        proj_dim: int = 256,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Build backbones
        backbone_fn = self.BACKBONE_REGISTRY[backbone]

        self.query_encoder = backbone_fn(embed_dim=embed_dim)
        self.key_encoder = backbone_fn(embed_dim=embed_dim)

        self.query_proj = MoCoProjectionHead(
            in_dim=embed_dim, out_dim=proj_dim
        )
        self.key_proj = MoCoProjectionHead(
            in_dim=embed_dim, out_dim=proj_dim
        )

        # Initialize key encoder from query encoder
        self.key_encoder.load_state_dict(
            self.query_encoder.state_dict()
        )
        self.key_proj.load_state_dict(self.query_proj.state_dict())

        # Freeze key encoder
        for p in self.key_encoder.parameters():
            p.requires_grad = False
        for p in self.key_proj.parameters():
            p.requires_grad = False

        # Queue
        self.register_buffer(
            "queue", F.normalize(torch.randn(proj_dim, queue_size), dim=0)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            # Handle overflow
            self.queue[:, ptr:] = keys[: self.queue_size - ptr].T
            self.queue[:, :batch_size - (self.queue_size - ptr)] = (
                keys[self.queue_size - ptr :].T
            )
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_key_encoder(self):
        """Momentum update of key encoder."""
        for param_q, param_k in zip(
            self.query_encoder.parameters(),
            self.key_encoder.parameters(),
        ):
            param_k.data = (
                self.momentum * param_k.data
                + (1.0 - self.momentum) * param_q.data
            )
        for param_q, param_k in zip(
            self.query_proj.parameters(),
            self.key_proj.parameters(),
        ):
            param_k.data = (
                self.momentum * param_k.data
                + (1.0 - self.momentum) * param_q.data
            )

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with query and key views.

        Args:
            x_q: Query view (B, C, H, W).
            x_k: Key view (B, C, H, W).

        Returns:
            Dict with 'loss', 'logits', 'labels'.
        """
        # Query features
        q_feat = self.query_encoder.forward_features(x_q)
        q = F.normalize(self.query_proj(q_feat), dim=1)

        # Key features (no grad)
        with torch.no_grad():
            self.update_key_encoder()
            k_feat = self.key_encoder.forward_features(x_k)
            k = F.normalize(self.key_proj(k_feat), dim=1)

        # Positive logits: q·k (B, 1)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: q·queue (B, queue_size)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Logits: (B, 1 + queue_size)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        labels = torch.zeros(
            logits.shape[0], dtype=torch.long, device=logits.device
        )

        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        return {
            "loss": loss,
            "logits": logits,
            "labels": labels,
        }

    def encode(
        self, x: torch.Tensor, use_key: bool = False
    ) -> torch.Tensor:
        """Extract features for downstream.

        Args:
            x: Input tensor (B, C, H, W).
            use_key: If True, use key encoder.

        Returns:
            Feature tensor (B, D).
        """
        encoder = self.key_encoder if use_key else self.query_encoder
        return encoder.forward_features(x)

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "query_encoder" in state_dict:
            self.query_encoder.load_state_dict(state_dict["query_encoder"])
            self.key_encoder.load_state_dict(state_dict["key_encoder"])
        else:
            self.query_encoder.load_state_dict(state_dict)
