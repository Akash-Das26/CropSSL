"""
Self-Supervised Learning Model Registry.

Provides a unified interface to create and manage SSL models.
"""

import torch.nn as nn

from crop_ssl.models.ssl.dino_v2 import DINOv2
from crop_ssl.models.ssl.moco_v3 import MoCoV3
from crop_ssl.models.ssl.simclr import SimCLR
from crop_ssl.models.ssl.mae import MAE


SSL_REGISTRY = {
    "dinov2": DINOv2,
    "moco_v3": MoCoV3,
    "simclr": SimCLR,
    "mae": MAE,
}


def create_ssl_model(
    method: str,
    backbone: str = "vit_base",
    **kwargs,
) -> nn.Module:
    """Create an SSL model by method name.

    Args:
        method: SSL method name ('dinov2', 'moco_v3', 'simclr', 'mae').
        backbone: ViT backbone variant.
        **kwargs: Additional method-specific arguments.

    Returns:
        Instantiated SSL model.

    Raises:
        ValueError: If method is not in registry.
    """
    model_cls = SSL_REGISTRY.get(method)
    if model_cls is None:
        raise ValueError(
            f"Unknown SSL method: {method}. "
            f"Available: {list(SSL_REGISTRY.keys())}"
        )
    return model_cls(backbone=backbone, **kwargs)


def get_ssl_model_info() -> dict:
    """Get information about all available SSL models."""
    return {
        name: {
            "class": cls.__name__,
            "module": cls.__module__,
            "description": cls.__doc__.strip().split("\n")[0]
            if cls.__doc__
            else "No description",
        }
        for name, cls in SSL_REGISTRY.items()
    }
