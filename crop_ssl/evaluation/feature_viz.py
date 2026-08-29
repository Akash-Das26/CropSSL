"""
Feature Space Visualization.

t-SNE and UMAP projections for analyzing learned representations
across domains and classes.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np


def extract_features(
    model: torch.nn.Module,
    dataloader,
    device: str = "cpu",
    max_samples: int = 5000,
) -> Dict[str, np.ndarray]:
    """Extract features from a dataset using a model.

    Args:
        model: Model with encode() or forward_features() method.
        dataloader: Data loader.
        device: Device for inference.
        max_samples: Maximum number of samples to extract.

    Returns:
        Dict with 'features', 'labels', 'predictions'.
    """
    model.eval()
    all_features = []
    all_labels = []
    all_preds = []
    count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            if count >= max_samples:
                break

            images = images.to(device)
            if hasattr(model, "encode"):
                features = model.encode(images)
            elif hasattr(model, "forward_features"):
                features = model.forward_features(images)
            else:
                features = model(images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

            if hasattr(model, "head") and hasattr(model.head, "weight"):
                preds = model.head(features).argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())

            count += images.shape[0]

    return {
        "features": np.concatenate(all_features, axis=0)[:max_samples],
        "labels": np.concatenate(all_labels, axis=0)[:max_samples],
        "predictions": (
            np.concatenate(all_preds, axis=0)[:max_samples]
            if all_preds else None
        ),
    }


def compute_tsne(
    features: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Compute t-SNE embedding.

    Args:
        features: Feature matrix (N, D).
        n_components: Number of output dimensions.
        perplexity: Perplexity parameter.
        random_state: Random seed.

    Returns:
        Embedding matrix (N, n_components).
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    return tsne.fit_transform(features)


def compute_umap(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP embedding.

    Args:
        features: Feature matrix (N, D).
        n_components: Number of output dimensions.
        n_neighbors: Number of neighbors.
        min_dist: Minimum distance.
        random_state: Random seed.

    Returns:
        Embedding matrix (N, n_components).
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        return reducer.fit_transform(features)
    except ImportError:
        print("UMAP not installed. Falling back to t-SNE.")
        return compute_tsne(features, n_components)


def plot_feature_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Feature Space Visualization",
    save_path: Optional[str] = None,
    method: str = "t-SNE",
    figsize: tuple = (10, 8),
):
    """Plot 2D feature space visualization.

    Args:
        embeddings: 2D embeddings (N, 2).
        labels: Class labels (N,).
        class_names: Optional class names.
        title: Plot title.
        save_path: Path to save figure.
        method: 't-SNE' or 'UMAP'.
        figsize: Figure size.
    """
    import matplotlib.pyplot as plt

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Use tab20 colormap for many classes
    if n_classes <= 10:
        cmap = plt.cm.get_cmap("tab10", n_classes)
    elif n_classes <= 20:
        cmap = plt.cm.get_cmap("tab20", n_classes)
    else:
        cmap = plt.cm.get_cmap("hsv", n_classes)

    fig, ax = plt.subplots(figsize=figsize)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[i] if class_names and i < len(class_names) else f"Class {label}"
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[cmap(i)],
            label=name,
            s=10,
            alpha=0.7,
        )

    ax.set_title(f"{title} ({method})", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method} Dimension 1", fontsize=11)
    ax.set_ylabel(f"{method} Dimension 2", fontsize=11)

    if n_classes <= 20:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
            markerscale=2,
        )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Feature space plot saved to {save_path}")
    plt.close()


def plot_domain_comparison(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    title: str = "Domain Comparison",
    save_path: Optional[str] = None,
):
    """Plot source vs target domain feature distributions.

    Args:
        source_embeddings: Source domain embeddings (N, 2).
        target_embeddings: Target domain embeddings (M, 2).
        source_labels: Source domain labels.
        target_labels: Target domain labels.
        title: Plot title.
        save_path: Path to save.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Source domain
    scatter1 = axes[0].scatter(
        source_embeddings[:, 0],
        source_embeddings[:, 1],
        c=source_labels,
        cmap="tab20",
        s=10,
        alpha=0.7,
    )
    axes[0].set_title("Source Domain", fontsize=12)
    plt.colorbar(scatter1, ax=axes[0])

    # Target domain
    scatter2 = axes[1].scatter(
        target_embeddings[:, 0],
        target_embeddings[:, 1],
        c=target_labels,
        cmap="tab20",
        s=10,
        alpha=0.7,
    )
    axes[1].set_title("Target Domain", fontsize=12)
    plt.colorbar(scatter2, ax=axes[1])

    # Combined
    all_emb = np.vstack([source_embeddings, target_embeddings])
    domain_labels = np.array(
        [0] * len(source_embeddings) + [1] * len(target_embeddings)
    )
    scatter3 = axes[2].scatter(
        all_emb[:, 0],
        all_emb[:, 1],
        c=domain_labels,
        cmap="coolwarm",
        s=10,
        alpha=0.5,
    )
    axes[2].set_title("Combined (Blue=Source, Red=Target)", fontsize=12)
    plt.colorbar(scatter3, ax=axes[2])

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Domain comparison saved to {save_path}")
    plt.close()
