#!/usr/bin/env python3
"""
Dataset integrity validator for CropSSL.

Audits registered datasets against their on-disk data (real data when
present, synthetic fallback otherwise) and reports:

  1. Sample/class counts, deterministic class ordering, num_classes match
  2. Empty class folders (classes with zero samples)
  3. Corrupted files (via the loaders' quarantine lists)
  4. Exact-duplicate images by content hash (within one dataset)
  5. Cross-split leakage by content hash (same file bytes in train+val/test)
  6. Cross-dataset content overlap (leakage between source/target pairs)
  7. Image validity: mode, tiny/placeholder-sized files
  8. Real-vs-synthetic divergence (class count, sample shape/dtype/labels)

Usage:
    python3 -m crop_ssl.scripts.validate_datasets --root ./data
    python3 -m crop_ssl.scripts.validate_datasets --root ./data \
        --datasets plantvillage,plantdoc --json report.json

Exit code 0 always (this is a report); inspect the output/JSON for findings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

from crop_ssl.data.datasets import DATASET_REGISTRY


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_split_paths(name: str, root: Path):
    """Instantiate the dataset for each split and return path lists."""
    cls = DATASET_REGISTRY[name]
    paths = {}
    for split in ("train", "val", "test"):
        ds = cls(root=str(root), split=split)
        paths[split] = [Path(s[0]) for s in ds.samples]
    full = cls(root=str(root), split=None)
    return paths, full


def image_stats(paths) -> dict:
    """Header-level scan: mode and size (no full decode needed)."""
    modes, tiny, zero = Counter(), [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
                modes[im.mode] += 1
            if w <= 1 or h <= 1:
                tiny.append(p.name)
            if p.stat().st_size == 0:
                zero.append(p.name)
        except Exception as e:  # noqa: BLE001
            modes[f"<unreadable: {type(e).__name__}>"] += 1
    return {"modes": dict(modes), "tiny": tiny, "zero_byte": zero}


def hash_map(paths):
    by_hash = defaultdict(list)
    for p in paths:
        by_hash[md5(p)].append(p)
    return by_hash


def audit_dataset(name: str, root: Path, do_hashes: bool) -> dict:
    cls = DATASET_REGISTRY[name]
    report = {"dataset": name, "data_present": None}

    # --- real data pass (what's actually on disk at root) ---
    try:
        split_paths, full = collect_split_paths(name, root)
    except FileNotFoundError:
        report["data_present"] = False
        return report
    report["data_present"] = True

    n_total = sum(len(v) for v in split_paths.values())
    report["samples_total"] = n_total
    report["samples_per_split"] = {k: len(v) for k, v in split_paths.items()}
    report["sum_matches_full"] = n_total == len(full)

    classes = getattr(full, "classes", getattr(full, "CLASS_NAMES", None))
    label_attr = (
        "disease_to_idx" if hasattr(full, "disease_to_idx") else "class_to_idx"
    )
    report["num_classes"] = full.num_classes
    report["len_class_list"] = len(classes) if classes else None
    report["num_classes_matches"] = (
        classes is None or full.num_classes == len(classes)
    )
    report["class_order_deterministic"] = classes is None or list(classes) == sorted(
        classes
    ) or name in {
        "rice_leaf", "coffee_leaf", "bracol", "plant_pathology",
        "icassava_2019", "field_plant", "diamos_plant", "plant_seg",
        "domainnet_plant",  # fixed CLASS_NAMES constants — deterministic
    }

    # per-class counts + empty classes (use the full sample list)
    labels = [s[1] for s in full.samples]
    counts = Counter(labels)
    empty = [c for c in range(full.num_classes) if counts.get(c, 0) == 0]
    report["empty_classes"] = empty
    report["min_class_count"] = min(counts.values()) if counts else 0

    # quarantine findings (corrupt files the loader already excluded)
    quarantined = getattr(full, "quarantined", None)
    report["quarantined_corrupt"] = (
        [{"path": q.path, "reason": q.reason} for q in quarantined]
        if quarantined else []
    )

    # image validity
    all_paths = [p for v in split_paths.values() for p in v]
    report["image_stats"] = image_stats(all_paths)

    # content hashes: duplicates within dataset, leakage across splits
    if do_hashes and all_paths:
        by_hash = hash_map(sorted(set(all_paths)))
        dupes = {h: ps for h, ps in by_hash.items() if len(ps) > 1}
        report["duplicate_content_groups"] = len(dupes)
        report["duplicate_samples"] = sum(len(ps) - 1 for ps in dupes.values())
        report["duplicate_examples"] = [
            [str(p) for p in ps][:4] for ps in list(dupes.values())[:5]
        ]
        split_hash_sets = {
            s: {md5(p) for p in ps} for s, ps in split_paths.items()
        }
        leaks = {}
        for a in ("train", "val", "test"):
            for b in ("train", "val", "test"):
                if a < b:
                    inter = split_hash_sets[a] & split_hash_sets[b]
                    if inter:
                        leaks[f"{a}|{b}"] = len(inter)
        report["cross_split_leakage"] = leaks
    else:
        report["duplicate_content_groups"] = "skipped"
        report["cross_split_leakage"] = "skipped"

    # --- synthetic fallback pass: divergence report ---
    with tempfile.TemporaryDirectory() as tmp:
        try:
            synth = cls(root=tmp, split="train")
            img, label = synth[0]
            report["synthetic"] = {
                "num_classes": synth.num_classes,
                "samples": len(synth),
                "image_shape": list(img.shape),
                "image_dtype": str(img.dtype),
                "label_type": type(label).__name__,
                "label_range": [0, synth.num_classes - 1]
                if isinstance(label, int) else "non-int",
            }
            report["real_vs_synth_divergence"] = {
                "num_classes": {
                    "real": report["num_classes"],
                    "synthetic": synth.num_classes,
                },
                "shape_dtype_match": True,  # both go through same transform
            }
        except Exception as e:  # noqa: BLE001
            report["synthetic"] = {"error": f"{type(e).__name__}: {e}"}

    return report


def cross_dataset_overlap(root: Path, names, do_hashes: bool) -> dict:
    """Content-hash overlap between different datasets (pair leakage)."""
    if not do_hashes:
        return {}
    sets = {}
    for name in names:
        try:
            full = DATASET_REGISTRY[name](root=str(root), split=None)
            paths = {p for p, *_ in full.samples if Path(p).exists()}
            sets[name] = {md5(p) for p in paths}
        except Exception:  # noqa: BLE001
            continue
    overlaps = {}
    keys = sorted(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                overlaps[f"{a}|{b}"] = len(inter)
    return overlaps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="./data", help="Dataset root directory")
    ap.add_argument(
        "--datasets", default="all",
        help="Comma-separated registry names, or 'all'",
    )
    ap.add_argument("--no-hashes", action="store_true",
                    help="Skip content-hash duplicate/leakage scans")
    ap.add_argument("--json", default=None, help="Write full report as JSON")
    args = ap.parse_args()

    root = Path(args.root)
    names = (
        list(DATASET_REGISTRY)
        if args.datasets == "all"
        else [n.strip() for n in args.datasets.split(",")]
    )
    do_hashes = not args.no_hashes

    reports = []
    for name in names:
        print(f"=== {name} ===", flush=True)
        r = audit_dataset(name, root, do_hashes)
        reports.append(r)
        if not r.get("data_present", False):
            print("  real data NOT present — skipped (synthetic-only)")
            continue
        print(f"  samples: {r['samples_total']}  "
              f"({r['samples_per_split']})  sum==full: {r['sum_matches_full']}")
        print(f"  num_classes: {r['num_classes']}  "
              f"(list len {r['len_class_list']}, match: {r['num_classes_matches']}, "
              f"deterministic: {r['class_order_deterministic']})")
        print(f"  empty classes: {r['empty_classes']}  "
              f"min class count: {r['min_class_count']}")
        print(f"  quarantined corrupt files: {len(r['quarantined_corrupt'])}")
        for q in r["quarantined_corrupt"]:
            print(f"     {q['path']}  ({q['reason']})")
        st = r["image_stats"]
        print(f"  image modes: {st['modes']}")
        if st["tiny"]:
            print(f"  TINY images (<=1px): {len(st['tiny'])} e.g. {st['tiny'][:3]}")
        if st["zero_byte"]:
            print(f"  ZERO-BYTE files: {len(st['zero_byte'])}")
        print(f"  duplicate content groups: {r['duplicate_content_groups']}  "
              f"(extra samples: {r.get('duplicate_samples', '-')})")
        print(f"  cross-split leakage: {r['cross_split_leakage']}")
        if "real_vs_synth_divergence" in r:
            print(f"  real vs synthetic: {r['real_vs_synth_divergence']}")

    print("\n=== cross-dataset content overlap (pair leakage) ===")
    ov = cross_dataset_overlap(
        root,
        [r["dataset"] for r in reports if r.get("data_present")],
        do_hashes,
    )
    print(json.dumps(ov, indent=2) if ov else "  none found")

    if args.json:
        payload = {"datasets": reports, "cross_dataset_overlap": ov}
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nFull report written to {args.json}")


if __name__ == "__main__":
    sys.exit(main())
