"""
ViT Backbone for CropSSL.

Implements a flexible Vision Transformer that can serve as the
backbone for DINOv2, MoCo v3, SimCLR, and other SSL methods.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings.

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, N, D) where N = num_patches.
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # Each: (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        if return_attention:
            return x, attn
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    Args:
        embed_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim ratio.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        if return_attention:
            attn_out, attn_weights = self.attn(
                self.norm1(x), return_attention=True
            )
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) backbone.

    Configurable ViT for SSL pre-training and downstream tasks.
    Supports different architectures (ViT-S/16, ViT-B/16, ViT-L/16).

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_channels: Input channels.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim ratio.
        dropout: Dropout rate.
        num_classes: Number of output classes (None for SSL backbone).
        global_pool: Whether to use CLS token for pooling.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
        global_pool: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        if num_classes is not None:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Extract features without classification head.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Feature tensor (B, D).
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.global_pool:
            return x[:, 0]  # CLS token
        return x.mean(dim=1)  # Global average pooling

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Classification logits or features.
        """
        features = self.forward_features(x)
        return self.head(features)

    def get_attention_maps(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        """Extract attention maps from all layers.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            List of attention maps, one per layer.
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        attn_maps = []
        for block in self.blocks:
            x, attn = block(x, return_attention=True)
            attn_maps.append(attn)

        return attn_maps


def vit_small_patch16(**kwargs) -> VisionTransformer:
    """ViT-S/16 configuration."""
    defaults = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return VisionTransformer(patch_size=16, **defaults)


def vit_base_patch16(**kwargs) -> VisionTransformer:
    """ViT-B/16 configuration."""
    defaults = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return VisionTransformer(patch_size=16, **defaults)


def vit_large_patch16(**kwargs) -> VisionTransformer:
    """ViT-L/16 configuration."""
    defaults = dict(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return VisionTransformer(patch_size=16, **defaults)
