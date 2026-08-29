"""
Grad-CAM Visualization for Crop Disease Localization.

Generates class activation maps showing which regions of a plant leaf
the model focuses on for disease classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Visualizes which image regions are most important for classification.

    Args:
        model: Trained model with forward_features method.
        target_layer: The layer to hook for activation/gradient extraction.
        device: Device for computation.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None

        # Find the last transformer block as default target
        if target_layer is None:
            target_layer = self._find_last_attention_layer()

        self.target_layer = target_layer
        self._register_hooks()

    def _find_last_attention_layer(self) -> nn.Module:
        """Find the last transformer block's attention layer."""
        last_attn = None
        last_linear = None

        # Try common ViT patterns
        for name, module in self.model.named_modules():
            if "blocks" in name and "attn" in name and "proj" in name:
                last_attn = module
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_attn is not None:
            return last_attn
        if last_linear is not None:
            return last_linear

        raise RuntimeError(
            "Could not find a target layer for GradCAM. "
            "Please provide target_layer explicitly."
        )

    def _register_hooks(self):
        """Register forward and backward hooks."""
        self._hooks = []

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image (1, C, H, W).
            target_class: Class index. If None, uses predicted class.

        Returns:
            Heatmap tensor (H, W) normalized to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        if output.dim() > 1:
            output = output.squeeze(0)

        if target_class is None:
            target_class = output.argmax().item()

        # Backward pass
        self.model.zero_grad()
        target = output[target_class]
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture activations/gradients")

        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, N)
        activations = self.activations[0]  # (C, N)

        # Handle ViT-style (B, N, D) or CNN-style (B, C, H, W)
        if gradients.dim() == 3 and activations.dim() == 3:
            # ViT: (B, N, D) — weighted sum over D dimension
            weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, D)
            cam = (weights * activations).sum(dim=-1).squeeze(0)  # (N,)
        elif gradients.dim() == 4 and activations.dim() == 4:
            # CNN: (B, C, H, W) — standard GradCAM
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = (weights * activations).sum(dim=1).squeeze(0)  # (H, W)
        else:
            # Fallback: flatten and take first spatial dim
            cam = activations.flatten(1).mean(dim=1).squeeze(0)

        # Reshape to spatial dimensions (exclude CLS token if present)
        # ViT outputs (B, N, D) where N = num_patches + 1 (CLS)
        num_patches = cam.shape[0]
        # Check if CLS token is included (non-square N)
        sqrt_n = int(num_patches ** 0.5)
        if sqrt_n * sqrt_n == num_patches:
            h = w = sqrt_n
        else:
            # Exclude first token (CLS) if N = h*w + 1
            h = w = int((num_patches - 1) ** 0.5)
            cam = cam[1:]  # Remove CLS token activation
        cam = cam.reshape(h, w)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def generate_batch(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate Grad-CAM for a batch.

        Args:
            input_tensor: Batch of images (B, C, H, W).
            target_class: Class index for all images.

        Returns:
            Batch of heatmaps (B, H, W).
        """
        heatmaps = []
        for i in range(input_tensor.shape[0]):
            hm = self.generate(input_tensor[i:i+1], target_class)
            heatmaps.append(hm)
        return torch.stack(heatmaps)

    def save_visualization(
        self,
        input_tensor: torch.Tensor,
        save_path: str,
        target_class: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        alpha: float = 0.5,
    ):
        """Save Grad-CAM overlay visualization.

        Args:
            input_tensor: Input image (1, C, H, W) or (C, H, W).
            save_path: Path to save the visualization.
            target_class: Class index.
            class_names: Optional class names for title.
            alpha: Overlay transparency.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        cam = self.generate(input_tensor, target_class)

        # Denormalize image
        img = input_tensor[0].cpu().detach()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        # Upsample heatmap to image size
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(img.shape[0], img.shape[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(cam_upsampled, cmap="jet")
        axes[1].set_title("Grad-CAM", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].imshow(cam_upsampled, cmap="jet", alpha=alpha)
        title = "Overlay"
        if class_names and target_class is not None:
            title += f" — {class_names[target_class]}"
        axes[2].set_title(title, fontsize=12)
        axes[2].axis("off")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Grad-CAM saved to {save_path}")
