"""
MAE: Masked Autoencoder for Vision Pre-training.

Implementation based on:
"Masked Autoencoders Are Scalable Vision Learners"
(He et al., 2022)

Key features:
- High masking ratio (75%) for efficient pre-training
- Asymmetric encoder-decoder architecture
- Reconstruction of raw pixel values
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from crop_ssl.models.backbones.vit import (
    vit_small_patch16,
    vit_base_patch16,
    vit_large_patch16,
)


class MAE(nn.Module):
    """Masked Autoencoder for vision pre-training.

    Args:
        backbone: ViT backbone architecture name.
        embed_dim: Encoder embedding dimension.
        patch_size: Patch size.
        img_size: Input image size.
        decoder_dim: Decoder embedding dimension.
        decoder_depth: Number of decoder layers.
        decoder_heads: Number of decoder attention heads.
        mask_ratio: Fraction of patches to mask.
        norm_pix_loss: Whether to use normalized pixel loss.
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
        patch_size: int = 16,
        img_size: int = 224,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_heads: int = 16,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = False,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss

        # Encoder (backbone without CLS token for MAE)
        backbone_fn = self.BACKBONE_REGISTRY[backbone]
        self.encoder = backbone_fn(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            global_pool=False,
        )
        # Remove CLS token for MAE: keep pos_embed WITHOUT the CLS position
        # pos_embed[:, 0, :] is the CLS position; we keep indices 1..N
        self.encoder.pos_embed = nn.Parameter(
            self.encoder.pos_embed[:, 1:, :].clone()
        )
        # Mark that this encoder has no CLS token
        self.encoder._no_cls = True

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(
            decoder_dim, patch_size**2 * 3
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random masking of patches.

        Args:
            x: Patch embeddings (B, N, D).
            mask_ratio: Fraction of patches to mask.

        Returns:
            x_masked: Visible patches only.
            mask: Binary mask (B, N) where 1 = masked.
            ids_restore: Indices to restore original order.
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        # Binary mask: 0 = keep, 1 = mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode with masking.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            latent: Encoder output for visible patches.
            mask: Binary mask.
            ids_restore: Restoration indices.
        """
        # Patch embed
        x = self.encoder.patch_embed(x)

        # Add positional embedding (CLS already removed in __init__)
        x = x + self.encoder.pos_embed

        # Random masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Transformer blocks
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Decode masked patches to pixel space.

        Args:
            x: Encoder output (B, N_vis, D).
            ids_restore: Restoration indices (B, N_total).

        Returns:
            Reconstructed patches (B, N_total, P^2 * 3).
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)

        # Append mask tokens
        B, N_vis, D = x.shape
        N_total = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(B, N_total - N_vis, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)

        # Unshuffle to original order
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )

        # Add positional embedding
        x_full = x_full + self.decoder_pos_embed[:, 1:, :]

        # Decoder blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)

        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)

        return pred

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches.

        Args:
            imgs: Input images (B, C, H, W).

        Returns:
            Patches (B, N, P^2 * C).
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images.

        Args:
            x: Patches (B, N, P^2 * C).

        Returns:
            Images (B, C, H, W).
        """
        p = self.patch_size
        h = w = int(math.sqrt(x.shape[1]))
        B, N, C = x.shape
        x = x.reshape(B, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, 3, h * p, w * p)
        return x

    def patch_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss on masked patches.

        Args:
            pred: Predicted patches (B, N, P^2 * 3).
            target: Target patches (B, N, P^2 * 3).
            mask: Binary mask (B, N).

        Returns:
            Scalar loss.
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) per-patch loss

        # Mean loss on masked patches only
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(
        self, imgs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            imgs: Input images (B, C, H, W).

        Returns:
            Dict with 'loss', 'pred', 'mask', 'target'.
        """
        # Encode
        latent, mask, ids_restore = self.forward_encoder(imgs)

        # Decode
        pred = self.forward_decoder(latent, ids_restore)

        # Target: original patches
        target = self.patchify(imgs)

        # Loss
        loss = self.patch_loss(pred, target, mask)

        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
            "target": target,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (full encoder without masking).

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Feature tensor (B, D).
        """
        # Encode without CLS token for MAE
        feat = self.encoder.patch_embed(x)
        feat = feat + self.encoder.pos_embed
        for block in self.encoder.blocks:
            feat = block(feat)
        feat = self.encoder.norm(feat)
        return feat.mean(dim=1)  # Global average pooling over patches

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "encoder" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder"], strict=False)
        else:
            self.encoder.load_state_dict(state_dict, strict=False)
