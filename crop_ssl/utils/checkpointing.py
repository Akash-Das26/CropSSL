"""
Checkpointing utilities for model saving and loading.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None,
):
    """Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch.
        metrics: Current metrics.
        save_path: Path to save checkpoint.
        additional_info: Extra data to save.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, save_path)

    # Also save metrics separately for easy access
    metrics_file = save_path.with_suffix(".metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to map weights to.

    Returns:
        Dict with 'epoch', 'metrics', and any additional info.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def save_best_model(
    model: nn.Module,
    metric_value: float,
    metric_name: str = "accuracy",
    save_dir: str = "./checkpoints",
    model_name: str = "best_model",
):
    """Save model if it's the best so far.

    Args:
        model: Model to save.
        metric_value: Current metric value.
        metric_name: Name of the metric.
        save_dir: Directory to save to.
        model_name: Base name for the checkpoint file.

    Returns:
        True if model was saved.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if there's a previous best
    best_file = save_dir / f"{model_name}.best.json"
    is_best = True

    if best_file.exists():
        with open(best_file) as f:
            best_info = json.load(f)
        previous_best = best_info.get("value", 0)
        is_best = metric_value > previous_best

    if is_best:
        save_path = save_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), save_path)

        with open(best_file, "w") as f:
            json.dump({
                "value": metric_value,
                "metric": metric_name,
                "path": str(save_path),
            }, f, indent=2)

        print(
            f"New best {metric_name}: {metric_value:.4f} "
            f"(saved to {save_path})"
        )
        return True

    return False
