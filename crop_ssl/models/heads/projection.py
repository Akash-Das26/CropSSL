"""
SSL Projection Heads.

Projection heads used by various SSL methods to map backbone
features to the contrastive/reconstruction space.
"""

import torch
import torch.nn as nn


class MLPProjectionHead(nn.Module):
    """MLP-based projection head used by DINO/DINOv2.

    Maps backbone features to projection space with
    optional stop-gradient for the student.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output projection dimension.
        num_layers: Number of hidden layers.
        batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        out_dim: int = 256,
        num_layers: int = 3,
        batch_norm: bool = True,
    ):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, in_dim).

        Returns:
            Projected features (B, out_dim).
        """
        return self.mlp(x)


class SimCLRProjectionHead(nn.Module):
    """Projection head for SimCLR.

    Two-layer MLP with ReLU, used in the contrastive loss computation.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output projection dimension.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 2048,
        out_dim: int = 128,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class MoCoProjectionHead(nn.Module):
    """Projection head for MoCo v3.

    Three-layer MLP with BN-ReLU, following the original paper.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output projection dimension.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        out_dim: int = 256,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class MAEReconstructionHead(nn.Module):
    """Decoder head for MAE reconstruction.

    Lightweight decoder that reconstructs pixel values from
    visible patch embeddings.

    Args:
        embed_dim: Encoder embedding dimension.
        decoder_dim: Decoder hidden dimension.
        decoder_depth: Number of decoder layers.
        decoder_heads: Number of decoder attention heads.
        patch_size: Patch size for output resolution.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_heads: int = 16,
        patch_size: int = 16,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 197, decoder_dim)
        )

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

    def forward(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct images from encoded patches.

        Args:
            x: Encoder output (B, N_visible, D).
            ids_restore: Indices to restore full sequence order (B, N_total).

        Returns:
            Reconstructed pixel patches (B, N_total, P^2 * C).
        """
        x = self.decoder_embed(x)

        # Append mask tokens
        B, N_vis, D = x.shape
        N_total = ids_restore.shape[1]
        N_mask = N_total - N_vis

        mask_tokens = x.new_zeros(B, N_mask, D)
        x = torch.cat([x, mask_tokens], dim=1)

        # Unshuffle
        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x
