"""
Advanced Training Utilities.

Includes mixed precision training, early stopping, learning rate finder,
EMA for all models, and CutMix/MixUp augmentations.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors a validation metric and stops training when it stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for accuracy.
        restore_best: Whether to restore best model weights.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation metric.
            model: Optional model to save best weights.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            if model and self.restore_best:
                self.best_model_state = copy.deepcopy(model.state_dict())
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if model and self.restore_best:
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def restore_best_model(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class ModelEMA:
    """Exponential Moving Average for any model.

    Maintains a shadow copy of model parameters updated via EMA.

    Args:
        model: Model to track.
        decay: EMA decay rate (0.999 typical).
        warmup: Number of warmup steps before full decay.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup: int = 0,
    ):
        self.model = model
        self.decay = decay
        self.warmup = warmup
        self.step = 0

        # Create shadow copy
        self.shadow = copy.deepcopy(model)
        self.backup = None

        # Freeze shadow
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with EMA."""
        self.step += 1
        # Linear warmup
        if self.step <= self.warmup:
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        else:
            decay = self.decay

        for shadow_param, model_param in zip(
            self.shadow.parameters(), self.model.parameters()
        ):
            shadow_param.data.mul_(decay).add_(
                model_param.data, alpha=1 - decay
            )

    def forward(self, *args, **kwargs):
        """Forward pass using shadow parameters."""
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        """Return shadow model state dict."""
        return self.shadow.state_dict()

    def store(self):
        """Store current model parameters."""
        self.backup = copy.deepcopy(self.model.state_dict())

    def restore(self):
        """Restore stored model parameters."""
        if self.backup is not None:
            self.model.load_state_dict(self.backup)
            self.backup = None


class LRFinder:
    """Learning rate range test.

    Finds optimal learning rate by training with exponentially
    increasing LR and monitoring the loss.

    Args:
        model: Model to train.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device for computation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.best_loss = float("inf")

    def range_test(
        self,
        dataloader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_steps: int = 100,
        smooth_beta: float = 0.98,
    ) -> Dict[str, float]:
        """Run learning rate range test.

        Args:
            dataloader: Training data loader.
            start_lr: Starting learning rate.
            end_lr: Ending learning rate.
            num_steps: Number of steps to test.
            smooth_beta: Smoothing factor for loss.

        Returns:
            Dict with recommended LR and test results.
        """
        self.model.train()
        lr_schedule = self._exp_range(start_lr, end_lr, num_steps)

        avg_loss = 0.0
        best_lr = start_lr

        for step, (images, labels) in enumerate(dataloader):
            if step >= num_steps:
                break

            lr = next(lr_schedule)
            self._set_lr(lr)

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward + backward
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            # Smoothed loss
            avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss.item()
            smoothed_loss = avg_loss / (1 - smooth_beta ** (step + 1))

            self.lrs.append(lr)
            self.losses.append(smoothed_loss)

            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss
                best_lr = lr

            # Stop if loss explodes
            if smoothed_loss > 4 * self.best_loss:
                break

        return {
            "best_lr": best_lr,
            "start_lr": start_lr,
            "end_lr": end_lr,
            "best_loss": self.best_loss,
        }

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _exp_range(self, start, end, num_steps):
        factor = (end / start) ** (1 / num_steps)
        lr = start
        for _ in range(num_steps):
            yield lr
            lr *= factor


class CutMix:
    """CutMix augmentation.

    Cuts a patch from one image and pastes it onto another,
    mixing the labels proportionally.

    Args:
        num_classes: Number of classes.
        alpha: Beta distribution parameter.
        prob: Probability of applying CutMix.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        prob: float = 0.5,
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix.

        Args:
            images: Batch of images (B, C, H, W).
            labels: Batch of labels (B,).

        Returns:
            Mixed images and soft labels.
        """
        if torch.rand(1).item() > self.prob:
            return images, F.one_hot(labels, self.num_classes).float()

        B, C, H, W = images.shape

        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        # Random permutation for mixing
        perm = torch.randperm(B, device=images.device)

        # Generate random bounding box
        cut_ratio = math.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        # Mix images
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

        # Adjust lambda by actual area ratio
        lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)

        # Soft labels
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[perm]

        return mixed_images, mixed_labels


class MixUp:
    """MixUp augmentation.

    Linearly interpolates between pairs of images and labels.

    Args:
        num_classes: Number of classes.
        alpha: Beta distribution parameter.
        prob: Probability of applying MixUp.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.2,
        prob: float = 0.5,
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp."""
        if torch.rand(1).item() > self.prob:
            return images, F.one_hot(labels, self.num_classes).float()

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        perm = torch.randperm(images.shape[0], device=images.device)

        mixed_images = lam * images + (1 - lam) * images[perm]

        labels_onehot = F.one_hot(labels, self.num_classes).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[perm]

        return mixed_images, mixed_labels


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup.

    Args:
        optimizer: Optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
        min_lr: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (
                (self.current_epoch - self.warmup_epochs)
                / max(self.total_epochs - self.warmup_epochs, 1)
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]
