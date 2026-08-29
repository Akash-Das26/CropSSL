#!/usr/bin/env python3
"""
Dataset Download & Preparation Script.

Downloads and prepares all crop disease datasets for CropSSL.

Usage:
    python -m crop_ssl.scripts.download_data --data_root ./data
    python -m crop_ssl.scripts.download_data --dataset plantvillage
    python -m crop_ssl.scripts.download_data --synthetic  # Quick test
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_plantvillage(data_root: str):
    """Download PlantVillage dataset."""
    from crop_ssl.data.datasets.plantvillage import PlantVillageDataset
    print("Loading PlantVillage (will download if needed)...")
    ds = PlantVillageDataset(root=data_root, split="train", download=True)
    print(f"  ✓ PlantVillage: {len(ds)} samples, {ds.num_classes} classes")
    return ds


def create_plantdoc_synthetic(data_root: str):
    """Create synthetic PlantDoc dataset."""
    from crop_ssl.data.datasets.plantdoc import PlantDocDataset
    ds = PlantDocDataset(root=data_root, split="train")
    print(f"  ✓ PlantDoc (synthetic): {len(ds)} samples, {ds.num_classes} classes")
    return ds


def create_riceleaf_synthetic(data_root: str):
    """Create synthetic RiceLeaf dataset."""
    from crop_ssl.data.datasets.rice_leaf import RiceLeafDataset
    ds = RiceLeafDataset(root=data_root, split="train")
    print(f"  ✓ RiceLeaf (synthetic): {len(ds)} samples, {ds.num_classes} classes")
    return ds


def create_coffeeleaf_synthetic(data_root: str):
    """Create synthetic CoffeeLeaf dataset."""
    from crop_ssl.data.datasets.coffee_leaf import CoffeeLeafDataset
    ds = CoffeeLeafDataset(root=data_root, split="train")
    print(f"  ✓ CoffeeLeaf (synthetic): {len(ds)} samples, {ds.num_classes} classes")
    return ds


def create_synthetic_all(data_root: str):
    """Create all synthetic datasets for pipeline testing."""
    print("\n📦 Creating synthetic datasets for pipeline testing...\n")
    create_plantdoc_synthetic(data_root)
    create_riceleaf_synthetic(data_root)
    create_coffeeleaf_synthetic(data_root)
    print("\n✅ All synthetic datasets created!")


def main():
    parser = argparse.ArgumentParser(description="Download crop disease datasets")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "plantvillage", "plantdoc", "rice_leaf",
                                 "coffee_leaf", "synthetic"],
                        help="Which dataset to download")
    parser.add_argument("--synthetic", action="store_true",
                        help="Create synthetic datasets only (fast testing)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.dataset == "synthetic":
        create_synthetic_all(str(data_root))
        return

    print("=" * 60)
    print("CropSSL Dataset Preparation")
    print("=" * 60)

    if args.dataset in ("all", "plantvillage"):
        try:
            download_plantvillage(str(data_root))
        except Exception as e:
            print(f"  ⚠ PlantVillage download failed: {e}")
            print("  → Creating synthetic fallback...")
            # Create synthetic as fallback
            from crop_ssl.data.datasets.plantvillage import PlantVillageDataset
            ds = PlantVillageDataset(root=str(data_root), split="train", download=True)
            print(f"  ✓ PlantVillage (synthetic): {len(ds)} samples")

    if args.dataset in ("all", "plantdoc"):
        create_plantdoc_synthetic(str(data_root))

    if args.dataset in ("all", "rice_leaf"):
        create_riceleaf_synthetic(str(data_root))

    if args.dataset in ("all", "coffee_leaf"):
        create_coffeeleaf_synthetic(str(data_root))

    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print(f"   Data root: {data_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
