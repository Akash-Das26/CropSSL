#!/usr/bin/env python3
"""
Full End-to-End Pipeline: CropSSL.

Runs the complete pipeline:
1. Data preparation (synthetic for testing)
2. SSL pre-training (SimCLR on source domain)
3. Few-shot adaptation (LoRA, ProtoNet)
4. Cross-domain evaluation
5. Results report

Usage:
    python -m crop_ssl.scripts.run_pipeline --epochs 3 --device cpu
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from crop_ssl.models.ssl import create_ssl_model
from crop_ssl.models.adaptation.few_shot_adapter import FewShotAdapter
from crop_ssl.models.adaptation.domain_adapter import DomainAdaptationModule
from crop_ssl.models.backbones.vit import vit_small_patch16
from crop_ssl.evaluation.metrics import (
    compute_accuracy,
    compute_per_class_metrics,
    compute_calibration_metrics,
    compute_domain_shift_metrics,
    EvaluationSuite,
)
from crop_ssl.utils.reproducibility import set_seed
from crop_ssl.utils.checkpointing import save_checkpoint
from crop_ssl.utils.training import CosineWarmupScheduler
from crop_ssl.utils.export import model_summary, count_parameters
from crop_ssl.utils.logging import ExperimentLogger, Timer


def create_synthetic_dataset(
    num_samples: int, num_classes: int, image_size: int = 224
) -> TensorDataset:
    """Create a synthetic dataset for pipeline testing."""
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


def stage_1_ssl_pretraining(
    method: str,
    backbone: str,
    embed_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    logger: ExperimentLogger,
    timer: Timer,
) -> nn.Module:
    """Stage 1: SSL pre-training."""
    print("\n" + "=" * 60)
    print("STAGE 1: SSL PRE-TRAINING")
    print("=" * 60)

    model = create_ssl_model(method, backbone=backbone, embed_dim=embed_dim)
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {method} + {backbone}")
    print(f"  Parameters: {params:,}")
    print(f"  Epochs: {epochs}, LR: {lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.04)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=max(1, epochs // 5), total_epochs=epochs)

    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        n = 0
        timer.start("train_epoch")

        for images, _ in train_loader:
            images = images.to(device)

            if method in ("simclr", "moco_v3"):
                # Create two augmented views (simplified: use random augment)
                view2 = torch.randn_like(images)
                result = model(images, view2)
            elif method == "mae":
                result = model(images)
            else:
                # DINOv2: create multi-crop list
                crops = [images] + [torch.randn_like(images) for _ in range(9)]
                result = model(crops)

            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n += 1

            # DINO-specific updates
            if method == "dinov2":
                model.update_center(result["teacher_out"])
                model.update_teacher()

        scheduler.step()
        epoch_time = timer.stop("train_epoch")
        avg_train_loss = train_loss / max(n, 1)
        history["train_loss"].append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                if method in ("simclr", "moco_v3"):
                    view2 = torch.randn_like(images)
                    result = model(images, view2)
                elif method == "mae":
                    result = model(images)
                else:
                    crops = [images] + [torch.randn_like(images) for _ in range(9)]
                    result = model(crops)
                val_loss += result["loss"].item()
                n_val += 1
        avg_val_loss = val_loss / max(n_val, 1)
        history["val_loss"].append(avg_val_loss)

        print(
            f"  Epoch {epoch+1}/{epochs}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

        logger.log_scalar("ssl/train_loss", avg_train_loss, epoch)
        logger.log_scalar("ssl/val_loss", avg_val_loss, epoch)

    print(f"  ✅ Pre-training complete. Best val loss: {best_loss:.4f}")
    return model


def stage_2_few_shot_adaptation(
    backbone_model: nn.Module,
    num_classes: int,
    device: str,
    methods: List[str],
) -> Dict[str, Any]:
    """Stage 2: Few-shot adaptation with multiple methods."""
    print("\n" + "=" * 60)
    print("STAGE 2: FEW-SHOT ADAPTATION")
    print("=" * 60)

    results = {}

    for method in methods:
        print(f"\n  --- {method.upper()} ---")
        adapter = FewShotAdapter(
            backbone_model, num_classes=num_classes,
            adaptation_method=method,
            rank=8 if method == "lora" else 4,
        )
        adapter.to(device)

        trainable = adapter.get_trainable_params()
        total = adapter.get_total_params()
        pct = 100 * trainable / total if total > 0 else 0
        print(f"  Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

        # Simulate forward pass
        x = torch.randn(8, 3, 224, 224).to(device)
        if method in ("prototypical", "maml"):
            support = torch.randn(num_classes * 2, 3, 224, 224).to(device)
            support_labels = torch.arange(num_classes).repeat(2).to(device)
            result = adapter(x, support_images=support, support_labels=support_labels, n_way=num_classes)
        else:
            result = adapter(x)
        assert "logits" in result
        assert result["logits"].shape == (8, num_classes)

        results[method] = {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": pct,
            "output_shape": list(result["logits"].shape),
        }
        print(f"  ✅ {method} adaptation OK")

    return results


def stage_3_domain_adaptation(
    backbone_model: nn.Module,
    num_classes: int,
    device: str,
) -> Dict[str, Any]:
    """Stage 3: Domain adaptation evaluation."""
    print("\n" + "=" * 60)
    print("STAGE 3: DOMAIN ADAPTATION")
    print("=" * 60)

    methods = ["dann", "mmd", "coral"]
    results = {}

    for method in methods:
        print(f"\n  --- {method.upper()} ---")
        adapter = DomainAdaptationModule(
            backbone_model, num_classes=num_classes,
            adaptation_type=method, input_dim=384,
        )
        adapter.to(device)

        src = torch.randn(8, 3, 224, 224).to(device)
        tgt = torch.randn(8, 3, 224, 224).to(device)
        result = adapter(src, tgt)

        domain_loss = result["domain_loss"].item()
        results[method] = {
            "domain_loss": domain_loss,
            "source_logits_shape": list(result["source_logits"].shape),
            "target_logits_shape": list(result["target_logits"].shape),
        }
        print(f"  Domain loss: {domain_loss:.4f}")
        print(f"  ✅ {method} adaptation OK")

    return results


def stage_4_evaluation(
    backbone_model: nn.Module,
    num_classes: int,
    device: str,
) -> Dict[str, Any]:
    """Stage 4: Comprehensive evaluation metrics."""
    print("\n" + "=" * 60)
    print("STAGE 4: EVALUATION METRICS")
    print("=" * 60)

    backbone_model.eval()
    results = {}

    # Simulate predictions
    x = torch.randn(64, 3, 224, 224).to(device)
    with torch.no_grad():
        feat = backbone_model.forward_features(x)
        # Random classifier head for testing
        head = nn.Linear(feat.shape[-1], num_classes).to(device)
        logits = head(feat)

    labels = torch.randint(0, num_classes, (64,)).to(device)

    # Accuracy
    acc = compute_accuracy(logits, labels, topk=(1, 3, 5))
    results["accuracy"] = acc
    print(f"  Top-1 Accuracy: {acc['top_1_acc']:.2f}%")
    print(f"  Top-3 Accuracy: {acc['top_3_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {acc['top_5_acc']:.2f}%")

    # Per-class metrics
    pcm = compute_per_class_metrics(logits, labels, num_classes)
    results["per_class_count"] = len(pcm)
    print(f"  Per-class metrics: {len(pcm)} classes")

    # Calibration
    cal = compute_calibration_metrics(logits, labels)
    results["calibration"] = cal
    print(f"  ECE: {cal['ece']:.2f}%, MCE: {cal['mce']:.2f}%")

    # Domain shift (simulated)
    ds = compute_domain_shift_metrics(acc["top_1_acc"], acc["top_1_acc"] * 0.8)
    results["domain_shift"] = ds
    print(f"  Domain shift drop: {ds['absolute_accuracy_drop']:.2f}%")
    print(f"  Robustness score: {ds['robustness_score']:.4f}")

    # Evaluation suite
    suite = EvaluationSuite(num_classes=num_classes)
    for _ in range(5):
        batch_logits = torch.randn(32, num_classes)
        batch_labels = torch.randint(0, num_classes, (32,))
        suite.update(batch_logits, batch_labels)
    suite_result = suite.compute()
    results["suite_total_samples"] = suite_result["total_samples"]
    print(f"  Evaluation suite: {suite_result['total_samples']} samples processed")

    print("  ✅ Evaluation complete")
    return results


def stage_5_advanced_features(device: str) -> Dict[str, Any]:
    """Stage 5: Advanced features (ensemble, calibration, active learning)."""
    print("\n" + "=" * 60)
    print("STAGE 5: ADVANCED FEATURES")
    print("=" * 60)

    results = {}

    # Model ensemble
    from crop_ssl.evaluation.ensemble import ModelEnsemble
    m1 = vit_small_patch16(num_classes=10).to(device)
    m2 = vit_small_patch16(num_classes=10).to(device)
    ensemble = ModelEnsemble([(m1, 0.5), (m2, 0.5)], num_classes=10)
    x = torch.randn(4, 3, 224, 224).to(device)
    ens_result = ensemble(x)
    results["ensemble"] = {"predictions": ens_result["pred"].tolist()}
    print(f"  Ensemble predictions: {ens_result['pred'].tolist()}")

    # Temperature scaling
    from crop_ssl.evaluation.calibration import TemperatureScaling
    ts = TemperatureScaling()
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    cal_result = ts.calibrate(logits, labels)
    results["temperature_scaling"] = {
        "temperature": cal_result["temperature"],
        "ece_before": cal_result["ece_before"],
        "ece_after": cal_result["ece_after"],
    }
    print(f"  Temperature scaling: T={cal_result['temperature']:.4f}")

    # GradCAM
    from crop_ssl.evaluation.grad_cam import GradCAM
    gc = GradCAM(m1)
    cam = gc.generate(torch.randn(1, 3, 224, 224).to(device))
    results["gradcam"] = {"shape": list(cam.shape)}
    print(f"  GradCAM heatmap: {cam.shape}")

    # Active learning
    from crop_ssl.evaluation.active_learning import ActiveLearner
    from torch.utils.data import DataLoader
    al = ActiveLearner(m1)
    unlabeled = TensorDataset(torch.randn(30, 3, 224, 224), torch.zeros(30))
    loader = DataLoader(unlabeled, batch_size=10)
    selected = al.uncertainty_sampling(loader, n_samples=5)
    results["active_learning"] = {"selected": selected}
    print(f"  Active learning: selected {len(selected)} samples")

    # Feature visualization
    from crop_ssl.evaluation.feature_viz import extract_features
    feat_loader = DataLoader(TensorDataset(torch.randn(16, 3, 224, 224), torch.randint(0, 5, (16,))), batch_size=8)
    feat_result = extract_features(m1, feat_loader, max_samples=16)
    results["feature_extraction"] = {
        "features_shape": list(feat_result["features"].shape)
    }
    print(f"  Feature extraction: {feat_result['features'].shape}")

    print("  ✅ Advanced features complete")
    return results


def generate_report(all_results: Dict[str, Any], output_path: str):
    """Generate a final pipeline report."""
    print("\n" + "=" * 60)
    print("PIPELINE REPORT")
    print("=" * 60)

    report = {
        "pipeline": "CropSSL End-to-End",
        "status": "complete",
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print(f"\n  Summary:")
    if "ssl_pretraining" in all_results:
        print(f"    SSL pre-training: ✅")
    if "few_shot" in all_results:
        methods = list(all_results["few_shot"].keys())
        print(f"    Few-shot methods: {', '.join(methods)}")
    if "domain_adaptation" in all_results:
        methods = list(all_results["domain_adaptation"].keys())
        print(f"    Domain adaptation: {', '.join(methods)}")
    if "evaluation" in all_results:
        acc = all_results["evaluation"].get("accuracy", {})
        print(f"    Top-1 accuracy: {acc.get('top_1_acc', 'N/A')}%")
    if "advanced" in all_results:
        print(f"    Advanced features: ✅")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CropSSL Full Pipeline")
    parser.add_argument("--method", type=str, default="simclr",
                        choices=["simclr", "dinov2", "moco_v3", "mae"])
    parser.add_argument("--backbone", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="./outputs/pipeline")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        device = "cpu"

    embed_dims = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}
    embed_dim = embed_dims[args.backbone]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_dir=str(output_dir), experiment_name="pipeline")
    timer = Timer()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         CropSSL — Full Pipeline Execution               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Method: {args.method}")
    print(f"  Backbone: {args.backbone} ({embed_dim}d)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")

    # Create data
    print("\nPreparing synthetic datasets...")
    train_dataset = create_synthetic_dataset(200, args.num_classes)
    val_dataset = create_synthetic_dataset(40, args.num_classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    all_results = {}

    # Stage 1: SSL Pre-training
    timer.start("total")
    ssl_model = stage_1_ssl_pretraining(
        method=args.method,
        backbone=args.backbone,
        embed_dim=embed_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        logger=logger,
        timer=timer,
    )
    all_results["ssl_pretraining"] = {"method": args.method, "backbone": args.backbone}

    # Save SSL checkpoint
    ckpt_path = str(output_dir / "ssl_checkpoint.pth")
    tmp_optimizer = torch.optim.Adam(ssl_model.parameters(), lr=1e-4)
    save_checkpoint(ssl_model, tmp_optimizer, epoch=args.epochs, metrics={"loss": 0.0}, save_path=ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Extract raw backbone from SSL model
    if hasattr(ssl_model, "encoder"):
        backbone = ssl_model.encoder
    elif hasattr(ssl_model, "student_backbone"):
        backbone = ssl_model.student_backbone
    elif hasattr(ssl_model, "query_encoder"):
        backbone = ssl_model.query_encoder
    else:
        backbone = ssl_model

    # Stage 2: Few-shot adaptation
    fs_results = stage_2_few_shot_adaptation(
        backbone_model=backbone,
        num_classes=args.num_classes,
        device=device,
        methods=["linear", "lora", "prototypical"],
    )
    all_results["few_shot"] = fs_results

    # Stage 3: Domain adaptation
    da_results = stage_3_domain_adaptation(
        backbone_model=backbone,
        num_classes=args.num_classes,
        device=device,
    )
    all_results["domain_adaptation"] = da_results

    # Stage 4: Evaluation
    eval_results = stage_4_evaluation(
        backbone_model=backbone,
        num_classes=args.num_classes,
        device=device,
    )
    all_results["evaluation"] = eval_results

    # Stage 5: Advanced features
    adv_results = stage_5_advanced_features(device=device)
    all_results["advanced"] = adv_results

    total_time = timer.stop("total")
    all_results["total_time_seconds"] = total_time

    # Model summary
    summary = model_summary(ssl_model)
    params = count_parameters(ssl_model)
    all_results["model_summary"] = {
        "total_params": params["total"],
        "trainable_params": params["trainable"],
    }

    # Report
    report_path = str(output_dir / "pipeline_report.json")
    generate_report(all_results, report_path)

    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print(f"  Timer summary:\n{timer.summary()}")
    logger.close()
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
