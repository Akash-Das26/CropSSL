"""
Cross-Domain Evaluation Framework.

Evaluates model robustness across multiple source-target domain pairs
and generates comprehensive robustness reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crop_ssl.evaluation.metrics import (
    EvaluationSuite,
    compute_domain_shift_metrics,
)


class CrossDomainEvaluator:
    """Evaluates models across multiple domain pairs.

    Args:
        model: Trained model with encode() method.
        num_classes: Number of classification classes.
        class_names: Optional class names for per-class analysis.
        device: Device for evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device

    @torch.no_grad()
    def evaluate_single_domain(
        self,
        dataloader: DataLoader,
        classifier: Optional[nn.Module] = None,
    ) -> Dict:
        """Evaluate on a single domain.

        Args:
            dataloader: Data loader for evaluation.
            classifier: Optional classification head.
                If None, expects model to have classifier attribute.

        Returns:
            Evaluation metrics dict.
        """
        eval_suite = EvaluationSuite(self.num_classes, self.class_names)

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            features = self.model.encode(images)

            if classifier is not None:
                logits = classifier(features)
            elif hasattr(self.model, "classifier"):
                logits = self.model.classifier(features)
            else:
                raise ValueError("No classifier provided or found on model")

            eval_suite.update(logits, labels)

        return eval_suite.compute()

    def evaluate_cross_domain(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        classifier: Optional[nn.Module] = None,
        source_name: str = "source",
        target_name: str = "target",
    ) -> Dict:
        """Evaluate cross-domain robustness.

        Args:
            source_loader: Source domain data loader.
            target_loader: Target domain data loader.
            classifier: Classification head.
            source_name: Name for source domain.
            target_name: Name for target domain.

        Returns:
            Cross-domain evaluation results.
        """
        # Evaluate on source
        source_results = self.evaluate_single_domain(
            source_loader, classifier
        )

        # Evaluate on target
        target_results = self.evaluate_single_domain(
            target_loader, classifier
        )

        # Compute domain shift metrics
        shift_metrics = compute_domain_shift_metrics(
            source_accuracy=source_results["top_1_acc"],
            target_accuracy=target_results["top_1_acc"],
            source_per_class={
                k: v.get("accuracy", v["f1"])
                for k, v in source_results["per_class_metrics"].items()
            },
            target_per_class={
                k: v.get("accuracy", v["f1"])
                for k, v in target_results["per_class_metrics"].items()
            },
        )

        return {
            "source_domain": source_name,
            "target_domain": target_name,
            "source_results": source_results,
            "target_results": target_results,
            "shift_metrics": shift_metrics,
        }

    def evaluate_all_domain_pairs(
        self,
        domain_loaders: Dict[str, DataLoader],
        classifier: Optional[nn.Module] = None,
    ) -> Dict:
        """Evaluate all source-target domain pairs.

        Args:
            domain_loaders: Dict mapping domain name to DataLoader.
            classifier: Classification head.

        Returns:
            Complete cross-domain evaluation results.
        """
        all_results = {}
        domain_names = list(domain_loaders.keys())

        for source_name in domain_names:
            for target_name in domain_names:
                if source_name == target_name:
                    continue

                pair_key = f"{source_name}->{target_name}"
                print(f"Evaluating: {pair_key}")

                result = self.evaluate_cross_domain(
                    source_loader=domain_loaders[source_name],
                    target_loader=domain_loaders[target_name],
                    classifier=classifier,
                    source_name=source_name,
                    target_name=target_name,
                )
                all_results[pair_key] = result

        # Compute average robustness
        avg_robustness = self._compute_average_robustness(all_results)
        all_results["average_robustness"] = avg_robustness

        return all_results

    def _compute_average_robustness(
        self, results: Dict
    ) -> Dict[str, float]:
        """Compute average robustness across all domain pairs."""
        drops = []
        for key, val in results.items():
            if key == "average_robustness":
                continue
            if isinstance(val, dict) and "shift_metrics" in val:
                drops.append(
                    val["shift_metrics"]["relative_accuracy_drop"]
                )

        if not drops:
            return {}

        return {
            "mean_relative_drop": sum(drops) / len(drops),
            "max_relative_drop": max(drops),
            "min_relative_drop": min(drops),
            "num_domain_pairs": len(drops),
        }

    def save_results(
        self,
        results: Dict,
        save_path: str,
    ):
        """Save evaluation results to JSON.

        Args:
            results: Evaluation results dict.
            save_path: Path to save JSON file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to lists for JSON serialization
        serializable = self._make_serializable(results)

        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"Results saved to {save_path}")

    def _make_serializable(self, obj):
        """Convert tensors and other non-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    def generate_report(
        self, results: Dict
    ) -> str:
        """Generate a human-readable evaluation report.

        Args:
            results: Cross-domain evaluation results.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 70,
            "CROSS-DOMAIN ROBUSTNESS EVALUATION REPORT",
            "=" * 70,
            "",
        ]

        for key, val in results.items():
            if key == "average_robustness":
                continue
            if not isinstance(val, dict):
                continue

            lines.append(f"Domain Pair: {key}")
            lines.append("-" * 40)

            if "source_results" in val:
                lines.append(
                    f"  Source Accuracy: "
                    f"{val['source_results']['top_1_acc']:.2f}%"
                )
            if "target_results" in val:
                lines.append(
                    f"  Target Accuracy: "
                    f"{val['target_results']['top_1_acc']:.2f}%"
                )
            if "shift_metrics" in val:
                sm = val["shift_metrics"]
                lines.append(
                    f"  Absolute Drop:   "
                    f"{sm['absolute_accuracy_drop']:.2f}%"
                )
                lines.append(
                    f"  Relative Drop:   "
                    f"{sm['relative_accuracy_drop']:.2f}%"
                )
                lines.append(
                    f"  Robustness:      "
                    f"{sm['robustness_score']:.4f}"
                )
            lines.append("")

        if "average_robustness" in results:
            ar = results["average_robustness"]
            lines.extend([
                "=" * 70,
                "AVERAGE ROBUSTNESS SUMMARY",
                "=" * 70,
                f"  Domain Pairs Evaluated: {ar.get('num_domain_pairs', 0)}",
                f"  Mean Relative Drop:     "
                f"{ar.get('mean_relative_drop', 0):.2f}%",
                f"  Max Relative Drop:      "
                f"{ar.get('max_relative_drop', 0):.2f}%",
                f"  Min Relative Drop:      "
                f"{ar.get('min_relative_drop', 0):.2f}%",
                "=" * 70,
            ])

        return "\n".join(lines)
