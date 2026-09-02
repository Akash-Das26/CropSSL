#!/usr/bin/env python3
"""
Dataset Download & Preparation Script.

Downloads and prepares all crop disease datasets for CropSSL.

Usage:
    python -m crop_ssl.scripts.download_data --data_root ./data
    python -m crop_ssl.scripts.download_data --dataset plantvillage
    python -m crop_ssl.scripts.download_data --synthetic  # Quick test
    python -m crop_ssl.scripts.download_data --dataset all  # Download everything

Supported datasets and their primary sources:
    plantvillage      — HuggingFace (mohanty/PlantVillage) / Mendeley / GitHub
    plantdoc          — Synthetic fallback (manual Kaggle download required)
    cassava_leaf      — HuggingFace (pufanyi/cassava-leaf-disease-classification)
    plant_pathology   — Synthetic fallback (manual Kaggle download required)
    icassava_2019     — Synthetic fallback (manual Kaggle download required)
    rice_leaf         — Synthetic fallback (manual Kaggle download required)
    coffee_leaf       — Synthetic fallback
    new_plant_diseases — Synthetic fallback (Kaggle augmented PlantVillage)
    plant_seg         — Synthetic fallback (Zenodo: https://doi.org/10.5281/zenodo.XXX)
    field_plant       — Synthetic fallback (Roboflow: https://universe.roboflow.com/plant-disease-detection/fieldplant)
    diamos_plant      — Synthetic fallback (Zenodo: https://doi.org/10.5281/zenodo.5557313)
    bracol            — Synthetic fallback (Mendeley: https://data.mendeley.com/datasets/yy2k5y8mxg/1)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Dataset download functions ────────────────────────────────────────────

def download_plantvillage(data_root: str):
    """Download PlantVillage dataset (HuggingFace preferred)."""
    from crop_ssl.data.datasets.plantvillage import PlantVillageDataset
    print("Loading PlantVillage (will download if needed)...")
    ds = PlantVillageDataset(root=data_root, split="train", download=True)
    print(f"  ✓ PlantVillage: {len(ds)} samples, {ds.num_classes} classes")
    return ds


def download_cassava_leaf(data_root: str):
    """Download Cassava Leaf Disease from HuggingFace."""
    from crop_ssl.data.datasets.cassava_leaf import CassavaLeafDataset
    print("Loading CassavaLeaf (will download from HuggingFace if needed)...")
    ds = CassavaLeafDataset(root=data_root, split="train")
    print(f"  ✓ CassavaLeaf: {len(ds)} samples, {ds.num_classes} classes")
    return ds


def create_plantdoc_synthetic(data_root: str):
    """Create PlantDoc dataset (synthetic or manual download)."""
    from crop_ssl.data.datasets.plantdoc import PlantDocDataset
    print("Loading PlantDoc (synthetic fallback — manual download from GitHub for real data)...")
    ds = PlantDocDataset(root=data_root, split="train")
    print(f"  ✓ PlantDoc: {len(ds)} samples, {ds.num_classes} classes")
    print("    → Real data: https://github.com/pratikkayal/PlantDoc-Dataset")
    return ds


def create_plant_pathology_synthetic(data_root: str):
    """Create Plant Pathology 2020 dataset (synthetic or manual download)."""
    from crop_ssl.data.datasets.plant_pathology import PlantPathologyDataset
    print("Loading PlantPathology2020 (synthetic fallback — manual download from Kaggle for real data)...")
    ds = PlantPathologyDataset(root=data_root, split="train")
    print(f"  ✓ PlantPathology: {len(ds)} samples, {ds.num_classes} classes")
    print("    → Real data: https://www.kaggle.com/c/plant-pathology-2020-fgvc7")
    return ds


def create_icassava_synthetic(data_root: str):
    """Create iCassava 2019 dataset (synthetic or manual download)."""
    from crop_ssl.data.datasets.icassava_2019 import ICassava2019Dataset
    print("Loading iCassava2019 (synthetic fallback — manual download from Kaggle for real data)...")
    ds = ICassava2019Dataset(root=data_root, split="train")
    print(f"  ✓ iCassava2019: {len(ds)} samples, {ds.num_classes} classes")
    print("    → Real data: https://www.kaggle.com/c/cassava-disease")
    return ds


def create_riceleaf_synthetic(data_root: str):
    """Create RiceLeaf dataset (synthetic or manual download)."""
    from crop_ssl.data.datasets.rice_leaf import RiceLeafDataset
    print("Loading RiceLeaf (synthetic fallback)...")
    ds = RiceLeafDataset(root=data_root, split="train")
    print(f"  ✓ RiceLeaf: {len(ds)} samples, {ds.num_classes} classes")
    return ds


def create_coffeeleaf_synthetic(data_root: str):
    """Create CoffeeLeaf dataset (synthetic or manual download)."""
    from crop_ssl.data.datasets.coffee_leaf import CoffeeLeafDataset
    print("Loading CoffeeLeaf (synthetic fallback)...")
    ds = CoffeeLeafDataset(root=data_root, split="train")
    print(f"  ✓ CoffeeLeaf: {len(ds)} samples, {ds.num_classes} classes")
    return ds


def create_new_plant_diseases_synthetic(data_root: str):
    """Create NewPlantDiseases dataset (synthetic or Kaggle)."""
    from crop_ssl.data.datasets.new_plant_diseases import NewPlantDiseasesDataset
    print("Loading NewPlantDiseases (synthetic fallback)...")
    ds = NewPlantDiseasesDataset(root=data_root, split="train")
    print(f"  ✓ NewPlantDiseases: {len(ds)} samples, {ds.num_classes} classes")
    print("    → Real data: https://www.kaggle.com/datasets/emmarex/plantdisease")
    return ds


def create_plant_seg_synthetic(data_root: str):
    """Create PlantSeg dataset (synthetic or Zenodo download)."""
    from crop_ssl.data.datasets.plant_seg import PlantSegDataset
    print("Loading PlantSeg (synthetic fallback — segmentation dataset)...")
    ds = PlantSegDataset(root=data_root, split="train")
    print(f"  ✓ PlantSeg: {len(ds)} samples, {ds.num_classes} classes (segmentation)")
    print("    → Real data: https://github.com/tqwei05/PlantSeg")
    return ds


def create_field_plant_synthetic(data_root: str):
    """Create FieldPlant dataset (synthetic or Roboflow download)."""
    from crop_ssl.data.datasets.field_plant import FieldPlantDataset
    print("Loading FieldPlant (synthetic fallback — real plantation images)...")
    ds = FieldPlantDataset(root=data_root, split="train")
    print(f"  ✓ FieldPlant: {len(ds)} samples, {ds.num_classes} classes")
    print("    → Real data: https://universe.roboflow.com/plant-disease-detection/fieldplant")
    return ds


def create_diamos_plant_synthetic(data_root: str):
    """Create DiaMOS Plant dataset (synthetic or Zenodo download)."""
    from crop_ssl.data.datasets.diamos_plant import DiaMOSPlantDataset
    print("Loading DiaMOSPlant (synthetic fallback — severity regression)...")
    ds = DiaMOSPlantDataset(root=data_root, split="train")
    print(f"  ✓ DiaMOSPlant: {len(ds)} samples, {ds.num_classes} classes + severity")
    print("    → Real data: https://doi.org/10.5281/zenodo.5557313")
    return ds


def create_bracol_synthetic(data_root: str):
    """Create BRACOL dataset (synthetic or Mendeley download)."""
    from crop_ssl.data.datasets.bracol import BRACOLDataset
    print("Loading BRACOL (synthetic fallback — multi-phone coffee disease)...")
    ds = BRACOLDataset(root=data_root, split="train")
    print(f"  ✓ BRACOL: {len(ds)} samples, {ds.num_classes} classes, {ds.num_phone_models} phone models")
    print("    → Real data: https://data.mendeley.com/datasets/yy2k5y8mxg/1")
    return ds


def create_synthetic_all(data_root: str):
    """Create all synthetic datasets for pipeline testing."""
    print("\n📦 Creating synthetic datasets for pipeline testing...\n")
    create_plantdoc_synthetic(data_root)
    create_plant_pathology_synthetic(data_root)
    create_icassava_synthetic(data_root)
    create_riceleaf_synthetic(data_root)
    create_coffeeleaf_synthetic(data_root)
    create_new_plant_diseases_synthetic(data_root)
    create_plant_seg_synthetic(data_root)
    create_field_plant_synthetic(data_root)
    create_diamos_plant_synthetic(data_root)
    create_bracol_synthetic(data_root)
    print("\n✅ All synthetic datasets created!")


# ── Main ──────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    "plantvillage", "plantdoc", "cassava_leaf", "plant_pathology",
    "icassava_2019", "rice_leaf", "coffee_leaf", "new_plant_diseases",
    "plant_seg", "field_plant", "diamos_plant", "bracol",
]

DATASET_DESCRIPTIONS = {
    "plantvillage": "54,309 images, 38 classes, controlled lab (HuggingFace auto-download)",
    "plantdoc": "2,598 images, 27 classes, real field (manual download from GitHub)",
    "cassava_leaf": "21,397 images, 5 classes, African fields (HuggingFace auto-download)",
    "plant_pathology": "1,821 images, 4 classes, apple foliar disease (manual download from Kaggle)",
    "icassava_2019": "5,656 images, 5 classes, cassava disease (manual download from Kaggle)",
    "rice_leaf": "~5,000 images, 7 classes, rice disease (manual download from Kaggle)",
    "coffee_leaf": "~5,000 images, 5 classes, coffee disease",
    "new_plant_diseases": "87,848 images, 38 classes, augmented PlantVillage (manual download from Kaggle)",
    "plant_seg": "11,400+ images, 115 classes, segmentation masks (Zenodo download)",
    "field_plant": "5,170 images, 27 classes, real plantation-shot (Roboflow download)",
    "diamos_plant": "3,505 images, 10 classes + severity 0-100%, pear (Zenodo download)",
    "bracol": "1,747 images, 5 classes, 5 phone models, coffee (Mendeley download)",
}


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare crop disease datasets for CropSSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with synthetic data (instant)
  python -m crop_ssl.scripts.download_data --synthetic

  # Download all datasets (auto-downloads what's possible)
  python -m crop_ssl.scripts.download_data --data_root ./data

  # Download specific dataset
  python -m crop_ssl.scripts.download_data --dataset plantvillage
  python -m crop_ssl.scripts.download_data --dataset cassava_leaf

  # List all available datasets with sources
  python -m crop_ssl.scripts.download_data --list
""",
    )
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all"] + ALL_DATASETS + ["synthetic"],
                        help="Which dataset to download")
    parser.add_argument("--synthetic", action="store_true",
                        help="Create synthetic datasets only (fast testing)")
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets with sources")
    args = parser.parse_args()

    if args.list:
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║              CropSSL — Available Datasets                  ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        for name, desc in DATASET_DESCRIPTIONS.items():
            print(f"║  {name:20s} │ {desc}")
        print("╚══════════════════════════════════════════════════════════════╝")
        return

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.dataset == "synthetic":
        create_synthetic_all(str(data_root))
        return

    print("=" * 64)
    print("  CropSSL Dataset Preparation")
    print("=" * 64)

    target = args.dataset
    downloaders = {
        "plantvillage": download_plantvillage,
        "plantdoc": create_plantdoc_synthetic,
        "cassava_leaf": download_cassava_leaf,
        "plant_pathology": create_plant_pathology_synthetic,
        "icassava_2019": create_icassava_synthetic,
        "rice_leaf": create_riceleaf_synthetic,
        "coffee_leaf": create_coffeeleaf_synthetic,
        "new_plant_diseases": create_new_plant_diseases_synthetic,
        "plant_seg": create_plant_seg_synthetic,
        "field_plant": create_field_plant_synthetic,
        "diamos_plant": create_diamos_plant_synthetic,
        "bracol": create_bracol_synthetic,
    }

    if target == "all":
        for ds_name in ALL_DATASETS:
            try:
                downloaders[ds_name](str(data_root))
            except Exception as e:
                print(f"  ⚠ {ds_name} failed: {e}")
            print()
    else:
        try:
            downloaders[target](str(data_root))
        except Exception as e:
            print(f"  ⚠ {target} failed: {e}")

    print("=" * 64)
    print(f"  ✅ Dataset preparation complete! Data root: {data_root}")
    print("=" * 64)


if __name__ == "__main__":
    main()
