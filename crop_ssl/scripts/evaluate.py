#!/usr/bin/env python3
"""
Cross-Domain Evaluation Script.

Evaluates SSL-pretrained models on source and target domains,
measuring cross-domain robustness.

Usage:
    python -m crop_ssl.scripts.evaluate \\
        --checkpoint ./outputs/ssl_dinov2_base/best_ssl.pth \\
        --source_dataset plantvillage \\
        --target_dataset plantdoc \\
        --data_root ./data \\
        --adaptation_method lora \\
        --k_shot 5
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crop_ssl.data.transforms.augmentations import get_default_train_transform, get_test_transform
from crop_ssl.models.ssl import create_ssl_model
from crop_ssl.models.adaptation.few_shot_adapter import FewShotAdapter
from crop_ssl.evaluation.metrics import (
    compute_domain_shift_metrics,
    EvaluationSuite,
)
from crop_ssl.utils.reproducibility import set_seed
from crop_ssl.utils.logging import ExperimentLogger


def load_model(
    checkpoint_path: str,
    method: str,
    backbone: str,
    num_classes: int,
    adaptation_method: str,
    device: str,
    k_shot: int = 5,
):
    """Load pretrained SSL model and add adaptation head."""
    # Create model
    embed_dim = {
        "vit_small": 384, "vit_base": 768, "vit_large": 1024
    }[backbone]

    model = create_ssl_model(
        method,
        backbone=backbone,
        embed_dim=embed_dim,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "student_backbone" in ckpt:
        model.student_backbone.load_state_dict(ckpt["student_backbone"])
    elif "encoder" in ckpt:
        model.encoder.load_state_dict(ckpt["encoder"])
    elif "query_encoder" in ckpt:
        model.query_encoder.load_state_dict(ckpt["query_encoder"])
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)

    # Wrap with adaptation head
    adapter = FewShotAdapter(
        backbone=model.student_backbone if hasattr(model, "student_backbone") else model.encoder,
        num_classes=num_classes,
        adaptation_method=adaptation_method,
        rank=8,
    )
    adapter = adapter.to(device)

    return adapter, model


def train_few_shot(
    adapter: nn.Module,
    train_loader: DataLoader,
    device: str,
    epochs: int = 50,
    lr: float = 1e-3,
):
    """Train adaptation head on source domain."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, adapter.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()

    adapter.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            result = adapter(images)
            logits = result["logits"]

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            acc = correct / total * 100
            avg_loss = total_loss / len(train_loader)
            print(
                f"Fine-tune Epoch {epoch+1}/{epochs}: "
                f"loss={avg_loss:.4f}, acc={acc:.2f}%"
            )

    return adapter


@torch.no_grad()
def evaluate_model(
    adapter: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    class_names=None,
):
    """Evaluate model on a dataset."""
    eval_suite = EvaluationSuite(num_classes, class_names)
    adapter.eval()

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        result = adapter(images)
        eval_suite.update(result["logits"], labels)

    return eval_suite.compute()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Domain Evaluation for Crop Disease Detection"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--method", type=str, default="dinov2",
        choices=["dinov2", "moco_v3", "simclr", "mae"],
    )
    parser.add_argument("--backbone", type=str, default="vit_base")
    parser.add_argument(
        "--source_dataset", type=str, default="plantvillage",
        choices=["plantvillage", "plantdoc", "rice_leaf", "coffee_leaf",
                 "plant_pathology", "icassava_2019", "new_plant_diseases", "cassava_leaf",
                 "plant_seg", "field_plant", "diamos_plant", "bracol"],
    )
    parser.add_argument(
        "--target_dataset", type=str, default="plantdoc",
        choices=["plantvillage", "plantdoc", "rice_leaf", "coffee_leaf",
                 "plant_pathology", "icassava_2019", "new_plant_diseases", "cassava_leaf",
                 "plant_seg", "field_plant", "diamos_plant", "bracol"],
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--adaptation_method", type=str, default="linear",
        choices=["linear", "lora", "maml", "prototypical"],
    )
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")

    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(args.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(
        log_dir=str(output_dir),
        experiment_name=f"eval_{args.source_dataset}_to_{args.target_dataset}",
    )

    # Load datasets
    train_transform = get_default_train_transform(args.image_size)
    test_transform = get_test_transform(args.image_size)

    print(f"Loading source: {args.source_dataset}...")

    from crop_ssl.data.datasets import (
        PlantVillageDataset, PlantDocDataset,
        RiceLeafDataset, CoffeeLeafDataset,
        PlantPathologyDataset, ICassava2019Dataset,
        NewPlantDiseasesDataset, CassavaLeafDataset,
    )
    from crop_ssl.data.datasets.plant_seg import PlantSegDataset
    from crop_ssl.data.datasets.field_plant import FieldPlantDataset
    from crop_ssl.data.datasets.diamos_plant import DiaMOSPlantDataset
    from crop_ssl.data.datasets.bracol import BRACOLDataset

    dataset_map = {
        "plantvillage": PlantVillageDataset,
        "plantdoc": PlantDocDataset,
        "rice_leaf": RiceLeafDataset,
        "coffee_leaf": CoffeeLeafDataset,
        "plant_pathology": PlantPathologyDataset,
        "icassava_2019": ICassava2019Dataset,
        "new_plant_diseases": NewPlantDiseasesDataset,
        "cassava_leaf": CassavaLeafDataset,
        "plant_seg": PlantSegDataset,
        "field_plant": FieldPlantDataset,
        "diamos_plant": DiaMOSPlantDataset,
        "bracol": BRACOLDataset,
    }

    # Only PlantVillageDataset supports download=True
    download_datasets = {"plantvillage"}

    source_dataset = dataset_map[args.source_dataset](
        root=args.data_root, split="train", transform=train_transform,
        **({"download": True} if args.source_dataset in download_datasets else {}),
    )
    target_dataset = dataset_map[args.target_dataset](
        root=args.data_root, split="test", transform=test_transform,
        **({"download": True} if args.target_dataset in download_datasets else {}),
    )

    num_classes = max(source_dataset.num_classes, target_dataset.num_classes)
    class_names = getattr(source_dataset, "CLASS_NAMES", None) or getattr(
        source_dataset, "classes", None
    )

    print(f"Source: {len(source_dataset)} samples")
    print(f"Target: {len(target_dataset)} samples")
    print(f"Classes: {num_classes}")

    source_loader = DataLoader(
        source_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    target_loader = DataLoader(
        target_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Load model
    print(f"\nLoading {args.method} checkpoint: {args.checkpoint}")
    adapter, ssl_model = load_model(
        checkpoint_path=args.checkpoint,
        method=args.method,
        backbone=args.backbone,
        num_classes=num_classes,
        adaptation_method=args.adaptation_method,
        device=device,
    )

    print(f"Adaptation method: {args.adaptation_method}")
    print(f"Trainable params: {adapter.get_trainable_params():,}")

    # Fine-tune on source domain
    print("\nFine-tuning on source domain...")
    adapter = train_few_shot(
        adapter, source_loader, device, epochs=50, lr=1e-3
    )

    # Evaluate on source
    print("\nEvaluating on source domain...")
    source_results = evaluate_model(
        adapter, source_loader, device, num_classes, class_names
    )
    print(f"Source Accuracy: {source_results['top_1_acc']:.2f}%")

    # Evaluate on target
    print("\nEvaluating on target domain...")
    target_results = evaluate_model(
        adapter, target_loader, device, num_classes, class_names
    )
    print(f"Target Accuracy: {target_results['top_1_acc']:.2f}%")

    # Cross-domain analysis
    shift = compute_domain_shift_metrics(
        source_accuracy=source_results["top_1_acc"],
        target_accuracy=target_results["top_1_acc"],
    )

    print("\n" + "=" * 60)
    print("CROSS-DOMAIN ROBUSTNESS RESULTS")
    print("=" * 60)
    print(f"  Source ({args.source_dataset}):  {shift['source_accuracy']:.2f}%")
    print(f"  Target ({args.target_dataset}):  {shift['target_accuracy']:.2f}%")
    print(f"  Absolute Drop:     {shift['absolute_accuracy_drop']:.2f}%")
    print(f"  Relative Drop:     {shift['relative_accuracy_drop']:.2f}%")
    print(f"  Robustness Score:  {shift['robustness_score']:.4f}")
    print("=" * 60)

    # Save results
    results = {
        "config": vars(args),
        "source_results": source_results,
        "target_results": target_results,
        "shift_metrics": shift,
    }

    # Remove non-serializable items
    results_clean = {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, torch.Tensor)}
        if isinstance(v, dict) else v
        for k, v in results.items()
    }

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")
    logger.close()


if __name__ == "__main__":
    main()
