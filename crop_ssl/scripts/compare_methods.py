#!/usr/bin/env python3
"""
Method Comparison & Benchmarking Script.

Systematically compares all SSL methods, adaptation strategies,
and domain pairs to produce a comprehensive benchmark report.

Usage:
    python -m crop_ssl.scripts.compare_methods \
        --data_root ./data \
        --output_dir ./benchmarks
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from crop_ssl.models.ssl import create_ssl_model
from crop_ssl.models.adaptation.few_shot_adapter import FewShotAdapter
from crop_ssl.evaluation.metrics import (
    EvaluationSuite,
)
from crop_ssl.utils.reproducibility import set_seed


SSL_METHODS = ["dinov2", "moco_v3", "simclr", "mae"]
ADAPTATION_METHODS = ["linear", "lora", "prototypical"]
BACKBONES = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}


def create_synthetic_data(
    num_samples: int = 200,
    num_classes: int = 5,
    image_size: int = 224,
    num_domains: int = 2,
) -> Dict[str, DataLoader]:
    """Create synthetic data for benchmarking."""
    datasets = {}
    for domain in range(num_domains):
        images = torch.randn(num_samples, 3, image_size, image_size)
        labels = torch.randint(0, num_classes, (num_samples,))
        ds = TensorDataset(images, labels)
        datasets[f"domain_{domain}"] = DataLoader(
            ds, batch_size=32, shuffle=True
        )
    return datasets


def benchmark_ssl_method(
    method: str,
    backbone: str,
    embed_dim: int,
    dataloader: DataLoader,
    num_classes: int,
    device: str,
    epochs: int = 5,
) -> Dict:
    """Benchmark a single SSL method."""
    set_seed(42)

    start_time = time.time()

    # Create model
    model = create_ssl_model(method, backbone=backbone, embed_dim=embed_dim)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())

    # Quick training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)

            if method == "dinov2":
                # DINOv2 semantics: 1 global crop (224) + local crops (96).
                # Cap the batch for multi-crop so CPU benchmarking stays tractable.
                multi_crop_batch = images[:8]
                crops = [multi_crop_batch] + [
                    torch.randn(multi_crop_batch.shape[0], 3, 96, 96, device=device)
                    for _ in range(9)
                ]
                result = model(crops)
            elif method in ("simclr", "moco_v3"):
                result = model(images, torch.randn_like(images))
            elif method == "mae":
                result = model(images)
            else:
                result = model(images)

            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            print(
                f"    epoch {epoch + 1}/{epochs} batch {batch_idx + 1}/{len(dataloader)} "
                f"loss={loss.item():.4f} ({time.time() - epoch_start:.1f}s)",
                flush=True,
            )

        losses.append(epoch_loss / max(n_batches, 1))

    training_time = time.time() - start_time

    return {
        "method": method,
        "backbone": backbone,
        "params": params,
        "training_time": training_time,
        "final_loss": losses[-1] if losses else 0,
        "losses": losses,
    }


def benchmark_adaptation(
    ssl_method: str,
    backbone: str,
    embed_dim: int,
    adaptation: str,
    source_loader: DataLoader,
    target_loader: DataLoader,
    num_classes: int,
    device: str,
) -> Dict:
    """Benchmark adaptation strategy."""
    set_seed(42)

    # Create and train SSL model briefly
    model = create_ssl_model(ssl_method, backbone=backbone, embed_dim=embed_dim)
    model = model.to(device)

    # Quick pre-train (single step)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for images, _ in source_loader:
        images = images.to(device)
        if ssl_method in ("simclr", "moco_v3"):
            result = model(images, torch.randn_like(images))
        elif ssl_method == "mae":
            result = model(images)
        else:
            multi_crop_batch = images[:8]
            crops = [multi_crop_batch] + [
                torch.randn(multi_crop_batch.shape[0], 3, 96, 96, device=device)
                for _ in range(9)
            ]
            result = model(crops)
        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        break

    # Create adapter
    backbone_module = (
        model.student_backbone if hasattr(model, "student_backbone")
        else model.encoder if hasattr(model, "encoder")
        else model.query_encoder if hasattr(model, "query_encoder")
        else model
    )

    adapter = FewShotAdapter(
        backbone=backbone_module,
        num_classes=num_classes,
        adaptation_method=adaptation,
        rank=4,
    )
    adapter = adapter.to(device)

    # Fine-tune
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, adapter.parameters()), lr=1e-3
    )

    # Prototypical networks need labeled support examples to form class prototypes
    if adaptation == "prototypical":
        support_images = torch.randn(num_classes * 4, 3, 224, 224).to(device)
        support_labels = torch.arange(num_classes).repeat(4).to(device)

    adapter.train()
    for images, labels in source_loader:
        images, labels = images.to(device), labels.to(device)
        if adaptation == "prototypical":
            result = adapter(
                images, support_images=support_images,
                support_labels=support_labels, n_way=num_classes,
            )
        else:
            result = adapter(images)
        loss = criterion(result["logits"], labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    eval_suite = EvaluationSuite(num_classes)
    adapter.eval()
    with torch.no_grad():
        for images, labels in target_loader:
            images, labels = images.to(device), labels.to(device)
            if adaptation == "prototypical":
                result = adapter(
                    images, support_images=support_images,
                    support_labels=support_labels, n_way=num_classes,
                )
            else:
                result = adapter(images)
            eval_suite.update(result["logits"], labels)

    metrics = eval_suite.compute()

    return {
        "ssl_method": ssl_method,
        "adaptation": adaptation,
        "target_acc": metrics["top_1_acc"],
        "macro_f1": metrics["macro_f1"],
        "ece": metrics["ece"],
    }


def run_benchmark(
    data_root: str,
    output_dir: str,
    device: str = "cpu",
    backbone: str = "vit_small",
    quick: bool = True,
):
    """Run full benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embed_dim = BACKBONES[backbone]
    num_classes = 5
    epochs = 2 if quick else 10

    print("=" * 60)
    print("CropSSL Benchmark Suite")
    print("=" * 60)
    print(f"Backbone: {backbone} (dim={embed_dim})")
    print(f"Device: {device}")
    print(f"Quick mode: {quick}")
    print()

    # Create data
    print("Creating synthetic data...")
    dataloaders = create_synthetic_data(
        num_samples=100, num_classes=num_classes
    )
    source_loader = dataloaders["domain_0"]
    target_loader = dataloaders["domain_1"]

    # 1. SSL Method Comparison
    print("\n--- SSL Method Comparison ---")
    ssl_results = []
    for method in SSL_METHODS:
        print(f"  Benchmarking {method}...", end=" ", flush=True)
        result = benchmark_ssl_method(
            method, backbone, embed_dim, source_loader,
            num_classes, device, epochs,
        )
        ssl_results.append(result)
        print(f"loss={result['final_loss']:.4f}, time={result['training_time']:.1f}s")

    # 2. Adaptation Comparison
    print("\n--- Adaptation Strategy Comparison ---")
    adapt_results = []
    for method in SSL_METHODS[:2]:  # Test with top 2 SSL methods
        for adaptation in ADAPTATION_METHODS:
            print(f"  {method} + {adaptation}...", end=" ", flush=True)
            result = benchmark_adaptation(
                method, backbone, embed_dim, adaptation,
                source_loader, target_loader, num_classes, device,
            )
            adapt_results.append(result)
            print(f"acc={result['target_acc']:.1f}%, f1={result['macro_f1']:.1f}%")

    # Save results
    all_results = {
        "ssl_comparison": ssl_results,
        "adaptation_comparison": adapt_results,
        "config": {
            "backbone": backbone,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "device": device,
        },
    }

    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    print("\nSSL Methods (by final loss):")
    for r in sorted(ssl_results, key=lambda x: x["final_loss"]):
        print(f"  {r['method']:12s}  loss={r['final_loss']:.4f}  "
              f"time={r['training_time']:.1f}s  params={r['params']:,}")

    print("\nAdaptation Strategies (by target accuracy):")
    for r in sorted(adapt_results, key=lambda x: -x["target_acc"]):
        print(f"  {r['ssl_method']}+{r['adaptation']:12s}  "
              f"acc={r['target_acc']:.1f}%  f1={r['macro_f1']:.1f}%")

    print(f"\nResults saved to {results_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CropSSL Benchmark")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./benchmarks")
    parser.add_argument("--backbone", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true",
                        help="Quick benchmark (fewer epochs)")
    args = parser.parse_args()

    run_benchmark(
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        backbone=args.backbone,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
