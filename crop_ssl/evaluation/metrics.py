"""
Evaluation Metrics for Crop Disease Detection.

Includes standard classification metrics plus cross-domain
robustness-specific metrics:
- Accuracy, Precision, Recall, F1 (per-class and macro)
- Domain shift metrics (accuracy drop, relative performance)
- Calibration metrics (ECE, MCE)
- Confusion matrix analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> Dict[str, float]:
    """Compute top-k accuracy.

    Args:
        predictions: Model output logits (N, C).
        labels: Ground truth labels (N,).
        topk: Tuple of k values for top-k accuracy.

    Returns:
        Dict mapping 'top_{k}_acc' to accuracy values.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.shape[0]

        _, pred_topk = predictions.topk(maxk, dim=1, largest=True, sorted=True)
        pred_topk = pred_topk.t()

        correct = pred_topk.eq(labels.view(1, -1).expand_as(pred_topk))

        results = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results[f"top_{k}_acc"] = (correct_k / batch_size * 100).item()

        return results


def compute_per_class_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1.

    Args:
        predictions: Model output logits (N, C).
        labels: Ground truth labels (N,).
        num_classes: Number of classes.
        class_names: Optional list of class names.

    Returns:
        Dict mapping class name/index to metrics dict.
    """
    with torch.no_grad():
        preds = predictions.argmax(dim=1)
        results = {}

        for c in range(num_classes):
            # True positives, false positives, false negatives
            tp = ((preds == c) & (labels == c)).sum().float()
            fp = ((preds == c) & (labels != c)).sum().float()
            fn = ((preds != c) & (labels == c)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            name = class_names[c] if class_names else f"class_{c}"
            results[name] = {
                "precision": precision.item() * 100,
                "recall": recall.item() * 100,
                "f1": f1.item() * 100,
                "support": int((labels == c).sum().item()),
            }

        return results


def compute_macro_metrics(
    per_class: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute macro-averaged metrics from per-class results.

    Args:
        per_class: Output from compute_per_class_metrics.

    Returns:
        Dict with 'macro_precision', 'macro_recall', 'macro_f1'.
    """
    precisions = [v["precision"] for v in per_class.values()]
    recalls = [v["recall"] for v in per_class.values()]
    f1s = [v["f1"] for v in per_class.values()]

    return {
        "macro_precision": np.mean(precisions),
        "macro_recall": np.mean(recalls),
        "macro_f1": np.mean(f1s),
    }


def compute_domain_shift_metrics(
    source_accuracy: float,
    target_accuracy: float,
    source_per_class: Optional[Dict[str, float]] = None,
    target_per_class: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute domain shift robustness metrics.

    Args:
        source_accuracy: Accuracy on source domain (%).
        target_accuracy: Accuracy on target domain (%).
        source_per_class: Per-class accuracy on source.
        target_per_class: Per-class accuracy on target.

    Returns:
        Dict with robustness metrics.
    """
    # Absolute accuracy drop
    absolute_drop = source_accuracy - target_accuracy

    # Relative accuracy drop (%)
    relative_drop = (
        (source_accuracy - target_accuracy) / max(source_accuracy, 1e-8)
    ) * 100

    # Robustness score (0-1, higher is better)
    robustness_score = target_accuracy / max(source_accuracy, 1e-8)

    results = {
        "source_accuracy": source_accuracy,
        "target_accuracy": target_accuracy,
        "absolute_accuracy_drop": absolute_drop,
        "relative_accuracy_drop": relative_drop,
        "robustness_score": robustness_score,
    }

    # Per-class analysis if available
    if source_per_class and target_per_class:
        common_classes = set(source_per_class) & set(target_per_class)
        class_drops = []
        for cls in common_classes:
            drop = source_per_class[cls] - target_per_class[cls]
            class_drops.append(drop)

        results["max_class_accuracy_drop"] = max(class_drops)
        results["mean_class_accuracy_drop"] = np.mean(class_drops)
        results["std_class_accuracy_drop"] = np.std(class_drops)

    return results


def compute_calibration_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute calibration metrics (ECE, MCE).

    Expected Calibration Error (ECE) and Maximum Calibration
    Error (MCE) measure how well predicted probabilities
    align with actual accuracy.

    Args:
        predictions: Model output logits (N, C).
        labels: Ground truth labels (N,).
        n_bins: Number of bins for calibration.

    Returns:
        Dict with 'ece', 'mce', and per-bin statistics.
    """
    with torch.no_grad():
        probs = F.softmax(predictions, dim=1)
        max_probs, preds = probs.max(dim=1)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            in_bin = (max_probs > lower) & (max_probs <= upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                avg_confidence = max_probs[in_bin].mean()
                accuracy_in_bin = (
                    preds[in_bin] == labels[in_bin]
                ).float().mean()

                calibration_error = abs(accuracy_in_bin - avg_confidence)
                ece += calibration_error * prop_in_bin
                mce = max(mce, calibration_error.item())

        return {
            "ece": ece.item() * 100,
            "mce": mce * 100,
        }


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute confusion matrix.

    Args:
        predictions: Model output logits (N, C).
        labels: Ground truth labels (N,).
        num_classes: Number of classes.

    Returns:
        Confusion matrix (num_classes, num_classes).
    """
    with torch.no_grad():
        preds = predictions.argmax(dim=1)
        cm = torch.zeros(
            num_classes, num_classes, dtype=torch.long, device=preds.device
        )
        for t, p in zip(labels, preds):
            cm[t][p] += 1
        return cm


def compute_fisher_discriminant_ratio(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute Fisher Discriminant Ratio (FDR).

    Measures class separability in feature space.
    Higher FDR indicates better class separation.

    Args:
        features: Feature embeddings (N, D).
        labels: Class labels (N,).
        num_classes: Number of classes.

    Returns:
        Fisher Discriminant Ratio (scalar).
    """
    with torch.no_grad():
        # Compute class means
        means = []
        counts = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                means.append(features[mask].mean(dim=0))
                counts.append(mask.sum().float())

        if len(means) < 2:
            return 0.0

        means = torch.stack(means)
        counts = torch.stack(counts)

        # Global mean
        global_mean = features.mean(dim=0)

        # Between-class scatter
        Sb = torch.zeros(features.shape[1], features.shape[1])
        for i, (m, n) in enumerate(zip(means, counts)):
            diff = (m - global_mean).unsqueeze(1)
            Sb += n * (diff @ diff.T)

        # Within-class scatter
        Sw = torch.zeros(features.shape[1], features.shape[1])
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_features = features[mask]
                class_mean = class_features.mean(dim=0)
                diff = class_features - class_mean
                Sw += diff.T @ diff

        # FDR
        Sw_inv = torch.linalg.pinv(Sw)
        fdr = torch.trace(Sw_inv @ Sb).item()

        return fdr


class EvaluationSuite:
    """Comprehensive evaluation suite for crop disease detection.

    Aggregates all metrics and provides a unified evaluation interface.

    Args:
        num_classes: Number of classification classes.
        class_names: Optional list of class names.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names

        # Accumulators
        self.all_predictions: List[torch.Tensor] = []
        self.all_labels: List[torch.Tensor] = []

    def update(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ):
        """Add a batch of predictions and labels."""
        self.all_predictions.append(predictions.cpu())
        self.all_labels.append(labels.cpu())

    def reset(self):
        """Reset accumulators."""
        self.all_predictions.clear()
        self.all_labels.clear()

    def compute(self) -> Dict:
        """Compute all metrics from accumulated predictions.

        Returns:
            Dict with comprehensive evaluation results.
        """
        predictions = torch.cat(self.all_predictions, dim=0)
        labels = torch.cat(self.all_labels, dim=0)

        # Accuracy
        acc = compute_accuracy(predictions, labels, topk=(1, 3, 5))

        # Per-class metrics
        per_class = compute_per_class_metrics(
            predictions, labels, self.num_classes, self.class_names
        )

        # Macro metrics
        macro = compute_macro_metrics(per_class)

        # Calibration
        calibration = compute_calibration_metrics(predictions, labels)

        # Confusion matrix
        cm = compute_confusion_matrix(
            predictions, labels, self.num_classes
        )

        # Feature quality (FDR)
        fdr = compute_fisher_discriminant_ratio(
            predictions, labels, self.num_classes
        )

        return {
            **acc,
            **macro,
            **calibration,
            "fisher_discriminant_ratio": fdr,
            "confusion_matrix": cm,
            "per_class_metrics": per_class,
            "total_samples": len(labels),
        }
