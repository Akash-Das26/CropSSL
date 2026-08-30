"""
Data augmentation transforms for SSL pre-training and downstream tasks.

Includes multi-crop transforms for DINO-style training,
standard augmentations for SimCLR/MoCo, and reconstruction targets for MAE.
"""

import random
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps

# Handle Pillow versions that lack InterpolationMode
try:
    _BICUBIC = Image.InterpolationMode.BICUBIC
except AttributeError:
    _BICUBIC = Image.BICUBIC


class MultiCropTransform:
    """Multi-crop transform for DINO/DINOv2 style SSL.

    Generates N global crops (large) and M local crops (small)
    with different augmentation strengths.

    Args:
        global_crops_number: Number of global (large) crops.
        local_crops_number: Number of local (small) crops.
        global_crops_scale: Scale range for global crops.
        local_crops_scale: Scale range for local crops.
        global_size: Output size for global crops.
        local_size: Output size for local crops.
    """

    def __init__(
        self,
        global_crops_number: int = 2,
        local_crops_number: int = 8,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        global_size: int = 224,
        local_size: int = 96,
    ):
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_size = global_size
        self.local_size = local_size

        # Strong augmentation for global crops
        self.global_transform = T.Compose([
            T.RandomResizedCrop(
                global_size, scale=global_crops_scale,
                interpolation=_BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Moderate augmentation for local crops
        self.local_transform = T.Compose([
            T.RandomResizedCrop(
                local_size, scale=local_crops_scale,
                interpolation=_BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        """Apply multi-crop transform.

        Returns:
            List of tensors: [global_1, ..., global_N, local_1, ..., local_M]
        """
        crops = []
        for _ in range(self.global_crops_number):
            crops.append(self.global_transform(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


class SimCLRTransform:
    """SimCLR-style contrastive augmentation.

    Generates two augmented views of each image.

    Args:
        size: Output image size.
        jitter_strength: Color jitter strength.
    """

    def __init__(self, size: int = 224, jitter_strength: float = 1.0):
        self.transform = T.Compose([
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(
                    0.8 * jitter_strength,
                    0.8 * jitter_strength,
                    0.8 * jitter_strength,
                    0.2 * jitter_strength,
                )
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(kernel_size=23)], p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns two augmented views."""
        return self.transform(image), self.transform(image)


class MoCoTransform:
    """MoCo v3 augmentation pipeline.

    Similar to SimCLR but with slight differences for queue-based
    contrastive learning.

    Args:
        size: Output image size.
    """

    def __init__(self, size: int = 224):
        self.transform_q = T.Compose([
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.transform_k = T.Compose([
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (query, key) augmented views."""
        return self.transform_q(image), self.transform_k(image)


class MAEReconstructTransform:
    """Transform pipeline for MAE pre-training.

    Returns the original image for reconstruction targets,
    plus a masked version for input.

    Args:
        size: Output image size.
        mask_ratio: Fraction of patches to mask.
    """

    def __init__(self, size: int = 224, mask_ratio: float = 0.75):
        self.size = size
        self.mask_ratio = mask_ratio

        self.transform = T.Compose([
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.target_transform = T.Compose([
            T.Resize(size=(size, size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_tensor, reconstruction_target)."""
        return self.transform(image), self.target_transform(image)


class GaussianBlur:
    """Custom Gaussian blur for augmentation pipelines."""

    def __init__(self, kernel_size: int, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return image.filter(
            ImageFilter.GaussianBlur(radius=sigma)
        )


class Solarization:
    """Solarization augmentation."""

    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(image)
        return image


def get_default_train_transform(
    size: int = 224,
    eval_mode: bool = False,
) -> T.Compose:
    """Default training transform for downstream classification.

    Args:
        size: Output image size.
        eval_mode: If True, uses only resize + normalize (no augmentation).
    """
    if eval_mode:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return T.Compose([
        T.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_test_transform(size: int = 224) -> T.Compose:
    """Standard test/validation transform."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
