"""
Test-Time Augmentation (TTA) for robust crop disease inference.

Applies multiple augmentations at test time and aggregates predictions
for improved accuracy and calibration.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class TestTimeAugmentation:
    """Test-Time Augmentation ensemble.

    Applies N augmentations to each test image and averages predictions.

    Args:
        model: Trained classification model.
        num_augmentations: Number of augmented views per image.
        scales: Multi-scale input sizes.
        flip: Whether to include horizontal flip.
        device: Device for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        num_augmentations: int = 10,
        scales: Optional[List[int]] = None,
        flip: bool = True,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.num_augmentations = num_augmentations
        self.flip = flip

        if scales is None:
            scales = [224, 256, 288]
        self.scales = scales

        # Build augmentation pipeline
        self.augmentations = self._build_augmentations()

    def _build_augmentations(self) -> List[T.Compose]:
        """Build diverse augmentation transforms."""
        augs = []

        for scale in self.scales:
            # Standard augmentations at different scales
            for _ in range(self.num_augmentations // len(self.scales)):
                transforms = [
                    T.Resize(scale + 32),
                    T.RandomCrop(scale),
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                    T.RandomGrayscale(p=0.1),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                ]
                augs.append(T.Compose(transforms))

                if self.flip:
                    flip_transforms = [
                        T.Resize(scale + 32),
                        T.RandomCrop(scale),
                        T.RandomHorizontalFlip(p=1.0),
                        T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
                    ]
                    augs.append(T.Compose(flip_transforms))

        return augs

    @torch.no_grad()
    def predict(
        self,
        image,
        return_std: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict with TTA.

        Args:
            image: PIL Image or tensor (C, H, W).
            return_std: Whether to return prediction standard deviation.

        Returns:
            Dict with 'logits', 'probs', 'pred', 'confidence'.
            Optionally 'std' for uncertainty estimation.
        """
        from PIL import Image as PILImage
        import numpy as np

        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_np = (image.cpu() * std + mean).permute(1, 2, 0).numpy()
                image = PILImage.fromarray((img_np * 255).clip(0, 255).astype("uint8"))

        all_logits = []
        for aug in self.augmentations:
            augmented = aug(image).unsqueeze(0).to(self.device)
            logits = self.model(augmented)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)  # (N, C)
        mean_logits = all_logits.mean(dim=0)  # (C,)
        probs = F.softmax(mean_logits, dim=-1)

        result = {
            "logits": mean_logits.unsqueeze(0),
            "probs": probs.unsqueeze(0),
            "pred": probs.argmax().item(),
            "confidence": probs.max().item(),
        }

        if return_std:
            # Prediction standard deviation as uncertainty measure
            all_probs = F.softmax(all_logits, dim=-1)
            result["std"] = all_probs.std(dim=0).mean().item()
            result["individual_preds"] = all_probs

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: List,
        batch_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """Predict a batch with TTA.

        Args:
            images: List of PIL Images.
            batch_size: Batch size for model inference.

        Returns:
            Dict with aggregated predictions.
        """
        all_results = []
        for img in images:
            result = self.predict(img, return_std=True)
            all_results.append(result)

        # Aggregate
        preds = [r["pred"] for r in all_results]
        confidences = [r["confidence"] for r in all_results]
        uncertainties = [r["std"] for r in all_results]

        return {
            "predictions": preds,
            "confidences": confidences,
            "uncertainties": uncertainties,
            "mean_confidence": sum(confidences) / max(len(confidences), 1),
            "mean_uncertainty": sum(uncertainties) / max(len(uncertainties), 1),
        }
