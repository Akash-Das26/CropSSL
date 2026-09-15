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
    plant_seg         — Synthetic fallback (GitHub: https://github.com/tqwei05/PlantSeg; data via Zenodo, linked in repo; paper: https://www.nature.com/articles/s41597-025-06513-4)
    field_plant       — Synthetic fallback (Roboflow: https://universe.roboflow.com/plant-disease-detection/fieldplant)
    diamos_plant      — Synthetic fallback (Zenodo: https://doi.org/10.5281/zenodo.5557313)
    bracol            — Synthetic fallback (Mendeley: https://data.mendeley.com/datasets/yy2k5y8mxg/1)
"""

import argparse
import shutil
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


# ── Zip import: arrange manual downloads into loader-expected layouts ────

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Where each dataset's manually-downloaded archive must land, and how to
# recognize its content inside the zip. Layouts mirror the loaders exactly
# (see each dataset module's "Expected directory structure" docstring).
IMPORT_SPECS = {
    "plantvillage": {"target": "PlantVillage", "style": "pv_buckets"},
    "plantdoc": {"target": "PlantDoc", "style": "class_folders"},
    "rice_leaf": {"target": "RiceLeaf", "style": "class_folders"},
    "coffee_leaf": {"target": "CoffeeLeaf", "style": "class_folders"},
    "domainnet_plant": {"target": "DomainNetPlant", "style": "class_folders"},
    "new_plant_diseases": {"target": "plant-disease", "style": "class_folders"},
    "icassava_2019": {"target": "iCassava2019/train", "style": "class_folders"},
    "field_plant": {"target": "FieldPlant/train", "style": "roboflow_csv"},
    "plant_pathology": {"target": "PlantPathology", "style": "images_plus_csv",
                        "csv": "train.csv", "images_dir": "images"},
    "cassava_leaf": {"target": "cassava-leaf-disease", "style": "images_plus_csv",
                     "csv": "train.csv", "images_dir": "train_images"},
    "diamos_plant": {"target": "DiaMOSPlant", "style": "images_plus_csv",
                     "csv": "annotations.csv", "images_dir": "images"},
    "bracol": {"target": "BRACOL", "style": "images_plus_csv",
               "csv": "metadata.csv", "images_dir": "images"},
    "plant_seg": {"target": "PlantSeg", "style": "images_plus_masks"},
}


def _safe_extract(zip_path: Path, staging: Path) -> Path:
    """Extract a zip (zip-slip guarded) and return its content root."""
    import zipfile

    staging_resolved = staging.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            dest = (staging / member).resolve()
            if not dest.is_relative_to(staging_resolved):
                raise ValueError(f"unsafe path in zip: {member}")
        zf.extractall(staging)

    entries = list(staging.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]  # Kaggle-style single wrapper folder
    return staging


def _is_image(f: Path) -> bool:
    return f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES


def _clean_synthetic_fallback(target: Path) -> int:
    """Remove leftover synthetic_* fallback files so they can't mix with
    real data under real class labels (audit finding). Class folders the
    removal leaves completely empty are removed too — otherwise they linger
    as empty classes in every later audit — while genuinely empty real
    class folders (a real data problem worth surfacing) are left alone."""
    removed = 0
    emptied = []
    if target.exists():
        for f in sorted(target.rglob("synthetic_*")):
            if f.is_file():
                parent = f.parent
                f.unlink()
                removed += 1
                emptied.append(parent)
        for d in sorted(set(emptied), key=lambda p: len(p.parts), reverse=True):
            while target != d and target in d.parents and d.exists() \
                    and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
    return removed


def _move_file(src: Path, dest_dir: Path, moved: list, skipped: list):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        skipped.append(src.name)
        return
    shutil.move(str(src), str(dest))
    moved.append(dest)


def _merge_class_dirs(root: Path, target: Path,
                      loose_class_name: str = None) -> tuple:
    """Merge every directory that directly holds images into
    target/<dirname>/ — handles train/+test/ splits of the same classes,
    and DomainNet's domain/<class> nesting (domain prefix is reported).

    loose_class_name: when the zip was a single wrapper folder whose name
    IS the class (images sit directly inside it), loose images at the
    content root are filed under that name. Only passed when we unwrapped
    a single folder — loose images at a multi-entry zip root carry no
    inferable class and are left alone.
    """
    moved, skipped = [], []
    class_dirs = [
        d for d in sorted(root.rglob("*"))
        if d.is_dir() and any(_is_image(f) for f in d.iterdir())
    ]
    for cd in class_dirs:
        rel = cd.relative_to(root)
        if len(rel.parts) > 1:
            print(f"    note: merging {rel} → {cd.name}/ (domain/level structure flattened)")
        for f in sorted(cd.iterdir()):
            if _is_image(f):
                _move_file(f, target / cd.name, moved, skipped)
    if loose_class_name:
        for f in sorted(root.iterdir()):
            if _is_image(f):
                _move_file(f, target / loose_class_name, moved, skipped)
    return moved, skipped


def _import_roboflow_csv(root: Path, target: Path) -> tuple:
    """FieldPlant Roboflow export: _annotations.csv + images beside it."""
    moved, skipped = [], []
    csvs = [p for p in sorted(root.rglob("_annotations.csv"))]
    if not csvs:
        return moved, skipped
    csv_src = csvs[0]
    _move_file(csv_src, target, moved, skipped)
    for f in sorted(csv_src.parent.iterdir()):
        if _is_image(f):
            _move_file(f, target, moved, skipped)
    return moved, skipped


def _import_images_plus_csv(root: Path, target: Path, spec: dict) -> tuple:
    moved, skipped = [], []
    csv_name = spec.get("csv")
    csvs = [p for p in sorted(root.rglob("*.csv"))
            if csv_name is None or p.name == csv_name]
    if not csvs and csv_name:
        csvs = [p for p in sorted(root.rglob("*.csv"))]  # any csv, warn below
    if csvs:
        if csvs[0].name != csv_name:
            print(f"    note: expected {csv_name}, found {csvs[0].name} — using it")
        _move_file(csvs[0], target, moved, skipped)
    images_dir = spec.get("images_dir", "images")
    img_dir = root / images_dir
    if not img_dir.is_dir():
        img_dir = next(
            (d for d in sorted(root.rglob("*"))
             if d.is_dir() and any(_is_image(f) for f in d.iterdir())),
            None,
        )
    n_img = 0
    if img_dir is not None:
        for f in sorted(img_dir.iterdir()):
            if _is_image(f):
                _move_file(f, target / images_dir, moved, skipped)
                n_img += 1
    return moved, skipped


def _import_pv(root: Path, target: Path) -> tuple:
    """PlantVillage archives, two real-world shapes:

    1. mohanty/Mendeley bundles with named image-type buckets at any depth
       (colored/ or color/, grayscale/ or gray/, segmented/ or segmentation/),
       each holding <class>/*.jpg — each bucket maps to its loader path
       (PlantVillage/colored|grayscale|segmented).
    2. Bare class folders (e.g. spMohanty color.zip extracts) — filed into
       PlantVillage/colored/.
    """
    moved, skipped = [], []
    bucket_names = {
        "colored": "colored", "color": "colored",
        "grayscale": "grayscale", "gray": "grayscale",
        "segmented": "segmented", "segmentation": "segmented",
    }
    consumed = set()
    bucket_dirs = [
        d for d in sorted(root.rglob("*"))
        if d.is_dir() and d.name.lower() in bucket_names
    ]
    for bd in bucket_dirs:
        consumed.add(bd)
        dest_name = bucket_names[bd.name.lower()]
        for cd in sorted(bd.rglob("*")):
            if cd.is_dir() and any(_is_image(f) for f in cd.iterdir()):
                consumed.add(cd)
                for f in sorted(cd.iterdir()):
                    if _is_image(f):
                        _move_file(f, target / dest_name / cd.name,
                                   moved, skipped)
    # Class folders outside any bucket belong in colored/
    for cd in sorted(root.rglob("*")):
        if (cd in consumed or not cd.is_dir()
                or not any(_is_image(f) for f in cd.iterdir())):
            continue
        if any(p in consumed for p in cd.parents):
            continue
        for f in sorted(cd.iterdir()):
            if _is_image(f):
                _move_file(f, target / "colored" / cd.name, moved, skipped)
    return moved, skipped


def _import_images_plus_masks(root: Path, target: Path) -> tuple:
    moved, skipped = [], []
    dirs = [d for d in sorted(root.rglob("*")) if d.is_dir()]
    img_dir = next((d for d in dirs if "image" in d.name.lower()), None)
    mask_dir = next((d for d in dirs if "mask" in d.name.lower()), None)
    for src_dir, dest_name in ((img_dir, "images"), (mask_dir, "masks")):
        if src_dir is not None:
            for f in sorted(src_dir.rglob("*")):
                if f.is_file():
                    _move_file(f, target / dest_name, moved, skipped)
    class_map = next((p for p in root.rglob("class_map.json")), None)
    if class_map is not None:
        _move_file(class_map, target, moved, skipped)
    return moved, skipped


def import_zip(name: str, zip_path: str, data_root: str) -> None:
    """Extract a manually-downloaded dataset zip and arrange it into the
    layout its loader expects. Prints what landed where; follow with
    validate_datasets to audit the import."""
    import tempfile

    spec = IMPORT_SPECS.get(name)
    if spec is None:
        raise ValueError(
            f"No import layout defined for '{name}'. "
            f"Supported: {sorted(IMPORT_SPECS)}"
        )
    src = Path(zip_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"zip not found: {src}")

    root_path = Path(data_root)
    target = root_path / spec["target"]
    removed = _clean_synthetic_fallback(target)
    if removed:
        print(f"  removed {removed} leftover synthetic_* fallback file(s) from {target}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "x"
        content = _safe_extract(src, staging)
        # Single wrapper folder unwrapped? Its name may itself be the class.
        loose_class = content.name if content != staging else None
        style = spec["style"]
        if style == "class_folders":
            moved, skipped = _merge_class_dirs(
                content, target, loose_class_name=loose_class
            )
        elif style == "roboflow_csv":
            moved, skipped = _import_roboflow_csv(content, target)
        elif style == "images_plus_csv":
            moved, skipped = _import_images_plus_csv(content, target, spec)
        elif style == "images_plus_masks":
            moved, skipped = _import_images_plus_masks(content, target)
        elif style == "pv_buckets":
            moved, skipped = _import_pv(content, target)
        else:  # pragma: no cover
            raise ValueError(f"unknown import style: {style}")

    print(f"  ✓ {name}: {len(moved)} file(s) → {target}"
          + (f", {len(skipped)} skipped (name collisions)" if skipped else ""))
    if moved:
        print(f"    verify: python3 -m crop_ssl.scripts.validate_datasets "
              f"--root {data_root} --datasets {name}")


def verify_import(name: str, data_root: str) -> bool:
    """Run the dataset integrity validator on one just-imported dataset and
    print a compact summary. Returns True when the import looks healthy.

    Failure conditions (also surfaced by exit code 2 from the CLI):
    data missing, empty classes, tiny/zero-byte images, cross-split leakage.
    Corrupt files quarantined by the loader and duplicate-content samples
    are flagged as warnings without failing the import.
    """
    from crop_ssl.scripts.validate_datasets import audit_dataset

    report = audit_dataset(name, str(data_root), do_hashes=True)
    if not report.get("data_present"):
        print(f"  ⚠ {name}: no data found at {data_root} — import may have failed")
        return False

    ok = True
    print(f"  ✓ {name}: {report['samples_total']} samples "
          f"({report['samples_per_split']}), {report['num_classes']} classes")
    if report["empty_classes"]:
        ok = False
        print(f"  ⚠ empty classes (no samples discovered): "
              f"{report['empty_classes']}")
    n_quarantined = len(report["quarantined_corrupt"])
    if n_quarantined:
        print(f"  ⚠ {n_quarantined} corrupt file(s) quarantined — "
              f"inspect <dataset>.quarantined")
    stats = report["image_stats"]
    if stats["tiny"] or stats["zero_byte"]:
        ok = False
        print(f"  ⚠ suspicious images: {len(stats['tiny'])} tiny (<=1px), "
              f"{len(stats['zero_byte'])} zero-byte")
    n_dupes = report.get("duplicate_samples") or 0
    if isinstance(n_dupes, int) and n_dupes:
        print(f"  ⚠ {n_dupes} duplicate-content sample(s) "
              f"in {report['duplicate_content_groups']} group(s)")
    leaks = report.get("cross_split_leakage") or {}
    if isinstance(leaks, dict) and leaks:
        ok = False
        print(f"  ⚠ cross-split leakage (same bytes in multiple splits): "
              f"{leaks}")
    if ok and not (isinstance(n_dupes, int) and n_dupes) and not n_quarantined:
        print("    integrity: no corrupt files, no duplicates, no split leakage")
    return ok


def dedupe_dataset(name: str, data_root: str) -> int:
    """Remove exact-duplicate image files (same md5) from an imported
    dataset directory. The first occurrence in sorted path order is kept,
    so behavior is deterministic; every removal is printed and logged to
    ``<data_root>/<name>-dedupe.log``. Class directories emptied by the
    removal are pruned so no empty classes linger.

    Note on cross-CLASS duplicates: one image appearing under two class
    folders is a label ambiguity as well as a duplicate; this keeps the
    alphabetically-first path and logs both, so the decision can be
    re-adjudicated manually from the log.

    Returns the number of files removed.
    """
    from crop_ssl.scripts.validate_datasets import md5

    spec = IMPORT_SPECS.get(name)
    if spec is None:
        raise ValueError(
            f"No import layout defined for '{name}'. "
            f"Supported: {sorted(IMPORT_SPECS)}"
        )
    target = Path(data_root) / spec["target"]
    if not target.exists():
        print(f"  ⚠ {name}: nothing to dedupe (no data at {target})")
        return 0

    files = sorted(
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    kept: dict = {}
    removed: list = []
    for p in files:
        h = md5(p)
        if h in kept:
            removed.append((p, kept[h]))
        else:
            kept[h] = p

    if not removed:
        print(f"  ✓ {name}: no duplicate-content files")
        return 0

    freed = sum(p.stat().st_size for p, _ in removed)
    log_path = Path(data_root) / f"{name}-dedupe.log"
    with open(log_path, "w") as log:
        for dup, original in removed:
            dup.unlink()
            print(f"    removed duplicate: {dup.relative_to(target)} "
                  f"(identical to {original.relative_to(target)})")
            log.write(f"REMOVED\t{dup}\tidentical to\t{original}\n")

    # Prune class directories the removal left empty (deepest first), so
    # no empty classes linger — same rule as the synthetic-fallback cleanup.
    emptied = {dup.parent for dup, _ in removed}
    for d in sorted(emptied, key=lambda p: len(p.parts), reverse=True):
        while target != d and target in d.parents and d.exists() \
                and not any(d.iterdir()):
            d.rmdir()
            d = d.parent

    print(f"  ✓ {name}: removed {len(removed)} duplicate file(s), "
          f"freed {freed / (1024 * 1024):.1f} MB — log: {log_path}")
    return len(removed)


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
    parser.add_argument("--import-zip", nargs=2, metavar=("DATASET", "ZIP"),
                        action="append",
                        help="Import a manually-downloaded zip into the layout "
                             "its loader expects (repeatable)")
    parser.add_argument("--verify", action="store_true",
                        help="After each --import-zip, run the dataset "
                             "integrity validator and print a summary "
                             "(exit code 2 if problems are flagged)")
    parser.add_argument("--dedupe", action="store_true",
                        help="Remove duplicate-content image files (same bytes): "
                             "keeps the first occurrence in sorted path order, "
                             "logs removals to <root>/<dataset>-dedupe.log. "
                             "Runs after each --import-zip, or standalone with "
                             "--dataset <name> on already-imported data")
    args = parser.parse_args()

    if args.list:
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║              CropSSL — Available Datasets                  ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        for name, desc in DATASET_DESCRIPTIONS.items():
            print(f"║  {name:20s} │ {desc}")
        print("╚══════════════════════════════════════════════════════════════╝")
        return

    if args.import_zip:
        flagged = False
        for ds_name, zip_path in args.import_zip:
            try:
                import_zip(ds_name, zip_path, args.data_root)
            except Exception as e:
                print(f"  ⚠ import failed for {ds_name}: {e}")
                sys.exit(1)
            if args.dedupe:
                dedupe_dataset(ds_name, args.data_root)
            if args.verify and not verify_import(ds_name, args.data_root):
                flagged = True
        if flagged:
            sys.exit(2)  # imports landed, but verification flagged problems
        return

    if args.dedupe:
        if args.dataset in ("all", "synthetic"):
            print("--dedupe requires a single --dataset name")
            sys.exit(1)
        try:
            dedupe_dataset(args.dataset, args.data_root)
        except Exception as e:
            print(f"  ⚠ dedupe failed for {args.dataset}: {e}")
            sys.exit(1)
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
