"""
DINOv2: Self-Supervised Vision Transformer with Self-Distillation.

Implementation based on:
"DINOv2: Learning Robust Visual Features without Supervision"
(Oquab et al., 2023)

Key features:
- Student-teacher architecture with EMA updates
- Multi-crop strategy (global + local crops)
- Cross-entropy loss on sharpened softmax distributions
- Centering to prevent mode collapse
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crop_ssl.models.backbones.vit import (
    VisionTransformer,
    vit_small_patch16,
    vit_base_patch16,
    vit_large_patch16,
)
from crop_ssl.models.heads.projection import MLPProjectionHead


class DINOv2(nn.Module):
    """DINOv2 self-supervised learning model.

    Args:
        backbone: ViT backbone architecture name.
        embed_dim: Embedding dimension (auto-set from backbone).
        out_dim: Projection output dimension.
        patch_size: Patch size.
        momentum_teacher: EMA momentum for teacher.
        teacher_temp: Teacher temperature for sharpening.
        student_temp: Student temperature.
        center_momentum: Momentum for center update.
        local_crops_number: Number of local crops.
        global_crops_scale: Scale range for global crops.
        local_crops_scale: Scale range for local crops.
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
        out_dim: int = 65536,
        patch_size: int = 16,
        momentum_teacher: float = 0.996,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        local_crops_number: int = 8,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
    ):
        super().__init__()

        self.momentum_teacher = momentum_teacher
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.local_crops_number = local_crops_number

        # Build backbones
        backbone_fn = self.BACKBONE_REGISTRY.get(backbone)
        if backbone_fn is None:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Available: {list(self.BACKBONE_REGISTRY.keys())}"
            )

        self.student_backbone = backbone_fn(
            embed_dim=embed_dim, patch_size=patch_size
        )
        self.teacher_backbone = backbone_fn(
            embed_dim=embed_dim, patch_size=patch_size
        )

        # Freeze teacher backbone
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        # Projection heads
        self.student_head = MLPProjectionHead(
            in_dim=embed_dim,
            hidden_dim=embed_dim * 4,
            out_dim=out_dim,
        )
        self.teacher_head = MLPProjectionHead(
            in_dim=embed_dim,
            hidden_dim=embed_dim * 4,
            out_dim=out_dim,
        )

        # Initialize teacher from student
        self.teacher_backbone.load_state_dict(
            self.student_backbone.state_dict()
        )
        self.teacher_head.load_state_dict(
            self.student_head.state_dict()
        )

        # Freeze teacher head (after loading weights)
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Center for teacher outputs
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher from student."""
        m = self.momentum_teacher
        for param_s, param_t in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters(),
        ):
            param_t.data = m * param_t.data + (1 - m) * param_s.data

        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters(),
        ):
            param_t.data = m * param_t.data + (1 - m) * param_s.data

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Update center with exponential moving average."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

    def forward(self, crops: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through student and teacher.

        Args:
            crops: List of image crops from multi-crop transform.
                First N are global crops, rest are local crops.

        Returns:
            Dict with 'student_out', 'teacher_out', 'teacher_cls',
            'student_cls', 'logits', 'labels'.
        """
        n_global = 2
        n_local = len(crops) - n_global

        # Student forward
        student_cls = []
        student_out = []
        for crop in crops:
            # Handle both batched (B,C,H,W) and unbatched (C,H,W) inputs
            if crop.dim() == 3:
                crop = crop.unsqueeze(0)
            features = self.student_backbone.forward_features(crop)
            proj = self.student_head(features)
            student_cls.append(features)
            student_out.append(proj)

        student_cls = torch.cat(student_cls, dim=0)  # (N, D)
        student_out = torch.cat(student_out, dim=0)  # (N, out_dim)

        # Teacher forward (only global crops)
        with torch.no_grad():
            teacher_cls = []
            teacher_out = []
            for crop in crops[:n_global]:
                if crop.dim() == 3:
                    crop = crop.unsqueeze(0)
                features = self.teacher_backbone.forward_features(crop)
                proj = self.teacher_head(features)
                teacher_cls.append(features)
                teacher_out.append(proj)

            teacher_cls = torch.cat(teacher_cls, dim=0)
            teacher_out = torch.cat(teacher_out, dim=0)

        # Normalize
        student_out = F.log_softmax(
            student_out / self.student_temp, dim=-1
        )
        teacher_out = F.softmax(
            (teacher_out - self.center) / self.teacher_temp, dim=-1
        )

        # Build teacher output for all crops
        teacher_all = []
        for i in range(n_global):
            teacher_all.append(teacher_out[i])
        # Repeat teacher global outputs for local crops
        for _ in range(n_local):
            teacher_all.append(teacher_out.mean(dim=0))
        teacher_all = torch.stack(teacher_all, dim=0)

        # Cross-entropy loss: each student crop predicts teacher output
        # For global crops (0,1): student[i] predicts teacher[i]
        # For local crops (2..N): student[i] predicts mean of teacher globals
        n_crops = len(crops)
        total_loss = 0.0
        for s_idx in range(n_crops):
            loss = F.kl_div(
                student_out[s_idx].unsqueeze(0),
                teacher_all[s_idx].unsqueeze(0),
                reduction="batchmean",
            )
            total_loss += loss

        total_loss /= n_crops

        return {
            "loss": total_loss,
            "student_out": student_out,
            "teacher_out": teacher_all,
            "student_cls": student_cls,
            "teacher_cls": teacher_cls,
        }

    def encode(
        self, x: torch.Tensor, use_teacher: bool = True
    ) -> torch.Tensor:
        """Extract features for downstream tasks.

        Args:
            x: Input tensor (B, C, H, W).
            use_teacher: If True, use teacher backbone.

        Returns:
            Feature tensor (B, D).
        """
        backbone = (
            self.teacher_backbone if use_teacher
            else self.student_backbone
        )
        return backbone.forward_features(x)

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "student_backbone" in state_dict:
            self.student_backbone.load_state_dict(
                state_dict["student_backbone"]
            )
            self.teacher_backbone.load_state_dict(
                state_dict["teacher_backbone"]
            )
        else:
            self.student_backbone.load_state_dict(state_dict)
