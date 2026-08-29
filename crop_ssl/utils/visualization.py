"""
Visualization utilities for CropSSL.

Generates publication-quality plots for:
- Attention map visualization
- Confusion matrices
- Domain shift analysis
- Feature space t-SNE/UMAP
- Training curves
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def plot_confusion_matrix(
    cm: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
):
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix (num_classes, num_classes).
        class_names: List of class names.
        save_path: Path to save the figure.
        normalize: Whether to normalize by row (true labels).
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    if normalize:
        cm_float = cm.float() / cm.sum(dim=1, keepdim=True).clamp(min=1)
    else:
        cm_float = cm.float()

    num_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.8), max(6, num_classes * 0.6)))

    im = ax.imshow(cm_float.numpy(), cmap=plt.cm.Blues, vmin=0, vmax=1 if normalize else None)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = 0.5 if normalize else cm_float.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_float[i, j].item()
            color = "white" if val > thresh else "black"
            ax.text(
                j, i, format(val, fmt),
                ha="center", va="center",
                color=color, fontsize=7,
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_domain_shift_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Cross-Domain Accuracy Comparison",
):
    """Plot accuracy comparison across source/target domains.

    Args:
        results: Dict mapping method name to
            {'source_acc': float, 'target_acc': float}.
        save_path: Path to save the figure.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    methods = list(results.keys())
    source_accs = [results[m]["source_acc"] for m in methods]
    target_accs = [results[m]["target_acc"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, source_accs, width, label="Source Domain",
                    color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, target_accs, width, label="Target Domain",
                    color="#FF5722", alpha=0.85)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Domain shift comparison saved to {save_path}")
    plt.close()


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
):
    """Plot training metrics over epochs.

    Args:
        metrics: Dict mapping metric name to list of values per epoch.
        save_path: Path to save the figure.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    num_plots = len(metrics)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, "b-", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    plt.close()


def visualize_attention_maps(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    num_heads: int = 6,
):
    """Visualize attention maps from ViT.

    Args:
        model: ViT model with get_attention_maps method.
        image: Input image tensor (1, C, H, W).
        layer_indices: Which layers to visualize.
        save_path: Path to save the figure.
        num_heads: Number of attention heads to show.
    """
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        attn_maps = model.get_attention_maps(image)

    if layer_indices is None:
        layer_indices = [0, len(attn_maps) // 2, len(attn_maps) - 1]

    fig, axes = plt.subplots(
        len(layer_indices), num_heads,
        figsize=(2.5 * num_heads, 2.5 * len(layer_indices)),
    )

    for i, layer_idx in enumerate(layer_indices):
        attn = attn_maps[layer_idx][0]  # First batch, all heads
        # Use CLS token attention to all patches
        cls_attn = attn[:num_heads, 0, 1:]  # (H, N)

        num_patches = cls_attn.shape[1]
        h = int(np.sqrt(num_patches))

        for j in range(num_heads):
            attn_map = cls_attn[j].reshape(h, h).numpy()
            axes[i, j].imshow(attn_map, cmap="viridis")
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(f"Layer {layer_idx}", fontsize=9)

    plt.suptitle("Attention Maps (CLS Token)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Attention maps saved to {save_path}")
    plt.close()


def plot_robustness_heatmap(
    results: Dict[str, Dict[str, float]],
    domain_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Cross-Domain Robustness Heatmap",
):
    """Plot heatmap of accuracy across domain pairs.

    Args:
        results: Dict mapping 'source->target' to {'accuracy': float}.
        domain_names: List of domain names.
        save_path: Path to save the figure.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    n = len(domain_names)
    matrix = np.zeros((n, n))

    for pair, metrics in results.items():
        if "->" not in pair:
            continue
        src, tgt = pair.split("->")
        if src in domain_names and tgt in domain_names:
            i = domain_names.index(src)
            j = domain_names.index(tgt)
            matrix[i, j] = metrics.get("accuracy", 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(domain_names, rotation=45, ha="right")
    ax.set_yticklabels(domain_names)
    ax.set_xlabel("Target Domain", fontsize=12)
    ax.set_ylabel("Source Domain", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Robustness heatmap saved to {save_path}")
    plt.close()
