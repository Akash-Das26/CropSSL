"""
Few-Shot Adaptation Module.

Implements multiple few-shot adaptation strategies:
1. MAML (Model-Agnostic Meta-Learning)
2. LoRA (Low-Rank Adaptation) for efficient fine-tuning
3. Prototypical Networks for metric-based classification
4. Linear probing for baseline comparison
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.

    Freezes the original weights and adds trainable low-rank matrices.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        rank: Low-rank decomposition rank.
        alpha: Scaling factor.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank decomposition
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(dropout)

        # Freeze original layer
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        h = F.linear(x, self.weight, self.bias)
        # Low-rank adaptation
        lora_out = self.dropout(x @ self.lora_A) @ self.lora_B
        return h + lora_out * self.scaling


class LoRAAdapter(nn.Module):
    """LoRA adapter for ViT backbone.

    Replaces attention projection layers with LoRA-modified versions.
    The original weights are frozen; only low-rank matrices are trained.

    Args:
        backbone: Pretrained ViT backbone.
        rank: Low-rank decomposition rank.
        alpha: Scaling factor.
        target_modules: Which module name substrings to adapt.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        backbone: nn.Module,
        rank: int = 8,
        alpha: float = 1.0,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.lora_modules = nn.ModuleDict()

        if target_modules is None:
            target_modules = ["attn.qkv", "attn.proj"]

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace target linear layers with LoRA-wrapped versions
        for name, module in self.backbone.named_modules():
            for target in target_modules:
                if target in name and isinstance(module, nn.Linear):
                    lora_name = name.replace(".", "_")
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    # Copy pretrained weights into LoRALayer
                    lora_layer.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        lora_layer.bias.data.copy_(module.bias.data)
                    self.lora_modules[lora_name] = lora_layer

                    # Replace the module in the backbone
                    self._replace_module(name, lora_layer)

    def _replace_module(self, target_name: str, new_module: nn.Module):
        """Replace a module in the backbone by name path."""
        parts = target_name.split(".")
        parent = self.backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def get_trainable_params(self) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )


class PrototypicalNetwork(nn.Module):
    """Prototypical Network for few-shot classification.

    Classifies based on distance to class prototypes computed
    from support set embeddings.

    Args:
        backbone: Feature encoder backbone.
        metric: Distance metric ('cosine', 'euclidean').
        temperature: Temperature for softmax.
    """

    def __init__(
        self,
        backbone: nn.Module,
        metric: str = "cosine",
        temperature: float = 10.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.metric = metric
        self.temperature = temperature

    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Compute class prototypes from support set.

        Args:
            support_features: (N_support, D) feature embeddings.
            support_labels: (N_support,) class labels.
            n_way: Number of classes.

        Returns:
            Prototypes (n_way, D).
        """
        prototypes = torch.zeros(
            n_way,
            support_features.shape[1],
            device=support_features.device,
        )
        for c in range(n_way):
            mask = support_labels == c
            if mask.sum() > 0:
                prototypes[c] = support_features[mask].mean(dim=0)
        return prototypes

    def forward(
        self,
        query_images: torch.Tensor,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> Dict[str, torch.Tensor]:
        """Classify query images using prototypical networks.

        Args:
            query_images: (B, C, H, W) query images.
            support_images: (S, C, H, W) support images.
            support_labels: (S,) support labels.
            n_way: Number of classes.

        Returns:
            Dict with 'logits', 'prototypes', 'features'.
        """
        # Encode support and query
        support_features = self.backbone.forward_features(support_images)
        query_features = self.backbone.forward_features(query_images)

        # L2 normalize
        support_features = F.normalize(support_features, dim=1)
        query_features = F.normalize(query_features, dim=1)

        # Compute prototypes
        prototypes = self.compute_prototypes(
            support_features, support_labels, n_way
        )

        # Compute distances
        if self.metric == "cosine":
            logits = query_features @ prototypes.T * self.temperature
        elif self.metric == "euclidean":
            dists = torch.cdist(query_features, prototypes)
            logits = -dists * self.temperature
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return {
            "logits": logits,
            "prototypes": prototypes,
            "query_features": query_features,
            "support_features": support_features,
        }


class FewShotAdapter(nn.Module):
    """Unified few-shot adaptation wrapper.

    Provides multiple adaptation strategies in a single interface.

    Args:
        backbone: Pretrained ViT backbone.
        num_classes: Number of target classes.
        adaptation_method: One of 'linear', 'lora', 'maml', 'prototypical'.
        rank: LoRA rank (for 'lora' method).
        hidden_dim: Hidden dimension for classifier head.
        dropout: Dropout rate.
    """

    METHODS = ["linear", "lora", "maml", "prototypical"]

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        adaptation_method: str = "linear",
        rank: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.adaptation_method = adaptation_method

        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out_dim = self.backbone.forward_features(dummy).shape[-1]

        if adaptation_method == "linear":
            self._setup_linear(out_dim, num_classes, dropout)
        elif adaptation_method == "lora":
            self._setup_lora(out_dim, num_classes, rank, dropout)
        elif adaptation_method == "maml":
            self._setup_maml(out_dim, num_classes, hidden_dim, dropout)
        elif adaptation_method == "prototypical":
            self._setup_prototypical(out_dim)
        else:
            raise ValueError(
                f"Unknown adaptation method: {adaptation_method}. "
                f"Available: {self.METHODS}"
            )

    def _setup_linear(
        self, in_dim: int, num_classes: int, dropout: float
    ):
        """Simple linear probing."""
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def _setup_lora(
        self,
        in_dim: int,
        num_classes: int,
        rank: int,
        dropout: float,
    ):
        """LoRA fine-tuning — replaces attention layers in-place.

        LoRA injects new layers into the backbone it is given. To keep the
        caller's backbone pristine (it may be reused for other methods), we
        first deep-copy it, then let LoRA mutate only the private copy.
        """
        import copy as _copy
        self.backbone = _copy.deepcopy(self.backbone)
        # LoRAAdapter freezes backbone and replaces attn layers on the copy
        self.lora_adapter = LoRAAdapter(
            self.backbone,
            rank=rank,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def _setup_maml(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
    ):
        """MAML-compatible setup (all params trainable)."""
        # MAML fine-tunes the whole model, so explicitly unfreeze the backbone
        # in case a previous adapter (e.g. linear) froze it on the shared object.
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _setup_prototypical(self, in_dim: int):
        """Prototypical network setup."""
        self.proto_net = PrototypicalNetwork(self.backbone)

    def forward(
        self,
        x: torch.Tensor,
        support_images: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None,
        n_way: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Query/test images (B, C, H, W).
            support_images: Support images for prototypical.
            support_labels: Support labels for prototypical.
            n_way: Number of classes for prototypical.

        Returns:
            Dict with 'logits' and optional 'prototypes'.
        """
        if self.adaptation_method == "prototypical":
            if support_images is None or support_labels is None:
                raise ValueError(
                    "PrototypicalNetwork requires support_images and support_labels. "
                    "Call forward(x, support_images=..., support_labels=..., n_way=...)"
                )
            return self.proto_net(
                query_images=x,
                support_images=support_images,
                support_labels=support_labels,
                n_way=n_way or self.num_classes,
            )

        # For linear, lora, maml: standard forward through (modified) backbone
        # LoRA modifies the backbone in-place, so forward_features goes through LoRA layers
        features = self.backbone.forward_features(x)
        logits = self.classifier(features)

        return {"logits": logits}

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
