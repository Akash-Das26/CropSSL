#!/usr/bin/env python3
"""
SSL Pre-training Script.

Trains self-supervised vision models on source domain data
(PlantVillage) before downstream adaptation.

Usage:
    python -m crop_ssl.scripts.train_ssl \\
        --method dinov2 \\
        --backbone vit_base \\
        --data_root ./data \\
        --epochs 100 \\
        --batch_size 64 \\
        --lr 1e-4
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crop_ssl.data.datasets.plantvillage import PlantVillageDataset
from crop_ssl.data.transforms.augmentations import (
    MultiCropTransform,
    SimCLRTransform,
    MoCoTransform,
    MAEReconstructTransform,
    get_default_train_transform,
)
from crop_ssl.models.ssl import create_ssl_model
from crop_ssl.utils.logging import ExperimentLogger, Timer
from crop_ssl.utils.checkpointing import save_checkpoint
from crop_ssl.utils.reproducibility import set_seed


def get_ssl_transforms(method: str, image_size: int = 224):
    """Get appropriate transforms for SSL method."""
    if method == "dinov2":
        return MultiCropTransform(
            global_crops_number=2,
            local_crops_number=8,
            global_size=image_size,
            local_size=96,
        )
    elif method == "simclr":
        return SimCLRTransform(size=image_size)
    elif method == "moco_v3":
        return MoCoTransform(size=image_size)
    elif method == "mae":
        return MAEReconstructTransform(size=image_size)
    else:
        return get_default_train_transform(image_size)


def train_one_epoch_ssl(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    method: str,
    epoch: int,
    logger: ExperimentLogger,
    timer: Timer,
):
    """Train for one epoch with SSL loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, _) in enumerate(dataloader):
        timer.start("forward")

        if method == "dinov2":
            # Multi-crop: each image produces multiple crops
            # images should be a list of tensors from the dataset
            if isinstance(images, list):
                crops = [img.to(device) for img in images]
            else:
                # Handle single tensor input
                crops = [images.to(device)]

            result = model(crops)
            loss = result["loss"]

        elif method in ("simclr", "moco_v3"):
            view_1, view_2 = images[0].to(device), images[1].to(device)
            result = model(view_1, view_2)
            loss = result["loss"]

        elif method == "mae":
            imgs = images[0].to(device) if isinstance(images, tuple) else images.to(device)
            result = model(imgs)
            loss = result["loss"]

        else:
            imgs = images.to(device) if torch.is_tensor(images) else images[0].to(device)
            result = model(imgs)
            loss = result["loss"]

        timer.stop("forward")

        # Backward
        timer.start("backward")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        timer.stop("backward")

        total_loss += loss.item()
        num_batches += 1

        # Log
        if batch_idx % 50 == 0:
            avg_loss = total_loss / num_batches
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f})"
            )
            logger.log_scalar(
                "train/loss_step", loss.item(),
                epoch * len(dataloader) + batch_idx,
            )

        # Update teacher for DINO (center BEFORE teacher EMA)
        if method == "dinov2":
            model.update_center(result["teacher_out"])
            model.update_teacher()

    avg_loss = total_loss / max(num_batches, 1)
    logger.log_scalar("train/loss_epoch", avg_loss, epoch)

    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="SSL Pre-training for Crop Disease Detection"
    )
    parser.add_argument(
        "--method", type=str, default="dinov2",
        choices=["dinov2", "moco_v3", "simclr", "mae"],
        help="SSL method to use",
    )
    parser.add_argument(
        "--backbone", type=str, default="vit_base",
        choices=["vit_small", "vit_base", "vit_large"],
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"

    # Create output directory
    output_dir = Path(args.output_dir) / f"ssl_{args.method}_{args.backbone}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = ExperimentLogger(
        log_dir=str(output_dir),
        experiment_name="ssl_pretrain",
    )
    timer = Timer()

    # Log config
    config = vars(args)
    logger.log_config(config)

    # Dataset
    print(f"Loading {args.data_root}/PlantVillage...")
    transform = get_ssl_transforms(args.method, args.image_size)
    dataset = PlantVillageDataset(
        root=args.data_root,
        split="train",
        transform=transform,
        download=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images, {dataset.num_classes} classes")

    # Model
    print(f"Creating {args.method} model with {args.backbone}...")
    ssl_config = {
        "backbone": args.backbone,
        "embed_dim": {
            "vit_small": 384, "vit_base": 768, "vit_large": 1024
        }[args.backbone],
    }
    model = create_ssl_model(args.method, **ssl_config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Parameters: {num_params:,} total, {trainable:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler with linear warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(
            args.epochs - args.warmup_epochs, 1
        )
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    # Training loop
    print(f"\nStarting SSL pre-training for {args.epochs} epochs...")
    print("=" * 60)

    best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        timer.start("epoch")
        avg_loss = train_one_epoch_ssl(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            method=args.method,
            epoch=epoch,
            logger=logger,
            timer=timer,
        )
        epoch_time = timer.stop("epoch")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1} complete: "
            f"loss={avg_loss:.4f}, lr={current_lr:.2e}, "
            f"time={epoch_time:.1f}s"
        )

        logger.log_scalar("train/lr", current_lr, epoch)
        logger.log_scalar("train/epoch_time", epoch_time, epoch)

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"loss": avg_loss},
                save_path=str(output_dir / "best_ssl.pth"),
            )
            print(f"  New best model saved (loss: {avg_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"loss": avg_loss},
                save_path=str(output_dir / f"checkpoint_epoch_{epoch+1}.pth"),
            )

    print("\n" + "=" * 60)
    print("SSL pre-training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(timer.summary())

    logger.close()


if __name__ == "__main__":
    main()
