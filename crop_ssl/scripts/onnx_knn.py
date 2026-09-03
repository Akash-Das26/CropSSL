#!/usr/bin/env python3
"""
Few-Shot Field Classifier on SSL Embeddings (PyTorch or ONNX).

Trains a **nearest-centroid / k-NN** classifier on top of self-supervised
embeddings — the classic non-parametric few-shot recipe that needs no
gradient updates at all. Works against:

* a live PyTorch SSL model (`create_ssl_model` + optional checkpoint), or
* an exported **ONNX** model (see ``POST /models/{name}/export``) via
  ``onnxruntime``, so it can be run entirely on-device.

Usage (PyTorch model):
    python3 -m crop_ssl.scripts.onnx_knn \\
        --method simclr --backbone vit_small \\
        --data_root ./data --shots 5 --k 5 --num-classes 10

Usage (ONNX embeddings):
    python3 -m crop_ssl.scripts.onnx_knn \\
        --onnx model_exports/simclr_vit_small.onnx \\
        --data_root ./data --shots 5 --k 5 --num-classes 10

Output: per-shot accuracy + a confusion-matrix-style per-class report.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from crop_ssl.utils.reproducibility import set_seed


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(images: np.ndarray) -> np.ndarray:
    """Normalize (N, C, H, W) float32 images in [0, 1] to ImageNet stats."""
    return (images - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]


def get_embedding_fn(args, device="cpu"):
    """Build an embedding function from either a PyTorch model or ONNX file."""
    if args.onnx:
        try:
            import onnxruntime as ort
        except ImportError:
            sys.exit("onnxruntime is required for --onnx mode: pip install onnxruntime")

        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        def embed(images: np.ndarray) -> np.ndarray:
            """images: (N, C, H, W) float32 in [0, 1]; normalized internally."""
            out = sess.run(None, {input_name: normalize(images)})[0]
            return np.asarray(out)

        print(f"  ONNX session ready: {Path(args.onnx).name} ({sess.get_inputs()[0].shape})")
        return embed

    # --- PyTorch path ---
    from crop_ssl.models.ssl import create_ssl_model

    model = create_ssl_model(args.method, backbone=args.backbone, embed_dim=384)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected and not any(k.startswith("model_state_dict") for k in state):
            print(f"  WARNING: checkpoint loaded with {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys (may be mismatched)")
    model.eval().to(device)

    def embed(images: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(normalize(images)).float().to(device)
        with torch.no_grad():
            feats = model.encode(x)
        return feats.cpu().numpy()

    print(f"  PyTorch model ready: {args.method}/{args.backbone} ({device})")
    return embed


def load_fewshot_split(root: Path, num_classes: int, shots: int, image_size: int,
                       split_ratio: float = 0.5, seed: int = 0):
    """Load structured images as (support, query) with disjoint classes.

    Uses the repo's deterministic synthetic generator so the script runs
    out-of-the-box with zero downloads; pass ``--data_root`` pointing at a
    real dataset folder to use real images instead.
    """
    from PIL import Image

    rng = np.random.RandomState(seed)

    # Prefer real data if present: expect folders named by class under
    # data_root/<split>/<class>/. Fall back to structured synthetic images.
    def to_chw(img):
        """PIL image → (C, H, W) float32 in [0, 1]."""
        arr = np.asarray(img.convert("RGB").resize((image_size, image_size)), dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)

    train_dir = root / "train"
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()]) if train_dir.exists() else []
    if len(class_dirs) >= num_classes:
        class_dirs = class_dirs[:num_classes]
        support, query = [], []
        for c, d in enumerate(class_dirs):
            files = sorted(list(d.iterdir()))[:200]
            rng.shuffle(files)
            sup = files[:shots]
            qry = files[shots: shots + max(shots, 20)]
            for f in sup:
                support.append((to_chw(Image.open(f)), c))
            for f in qry:
                query.append((to_chw(Image.open(f)), c))
        print(f"  Real data: {len(class_dirs)} classes x {shots} shots from {train_dir}")
        return support, query

    # --- Structured synthetic fallback (dominant-color blocks per class) ---
    print("  No real data found — using structured synthetic images "
          "(dominant color per class)")
    n_per_class = shots + max(shots, 20)
    support, query = [], []
    for c in range(num_classes):
        base = np.array([(c * 55) % 255, (c * 97) % 255, (c * 41) % 255], dtype=np.float32)
        for i in range(n_per_class):
            img = np.zeros((image_size, image_size, 3), dtype=np.float32)
            img[:, :] = base
            # class-distinctive corner block so classes are separable
            block = image_size // 4
            img[:block, :block] = np.clip(base + (i % 5) * 12, 0, 255)
            item = ((img / 255.0).transpose(2, 0, 1).astype(np.float32), c)
            (support if i < shots else query).append(item)
    return support, query


def nearest_centroid(support_feats: np.ndarray, support_labels: np.ndarray,
                     query_feats: np.ndarray, k: int = 0) -> np.ndarray:
    """Classify query embeddings.

    ``k=0`` → nearest-centroid (one prototype per class).
    ``k>0`` → k-NN vote over the support set.
    """
    if k > 0:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(query_feats, support_feats)          # (Q, S)
        preds = []
        for row in sim:
            top = np.argsort(row)[::-1][:k]
            labels = support_labels[top]
            preds.append(np.bincount(labels).argmax())
        return np.array(preds)

    # nearest-centroid
    prototypes = []
    for c in np.unique(support_labels):
        feats = support_feats[support_labels == c]
        prototypes.append(feats.mean(axis=0))
    prototypes = np.stack(prototypes)
    sim = _cosine(query_feats, prototypes)
    return sim.argmax(axis=1)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-8)
    return a_n @ b_n.T


def run_eval(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    embed = get_embedding_fn(args, device=device)

    support, query = load_fewshot_split(
        Path(args.data_root), args.num_classes, args.shots,
        image_size=args.image_size, seed=args.seed,
    )
    print(f"  Support: {len(support)} | Query: {len(query)}")

    def to_batch(items):
        xs = np.stack([x for x, _ in items])
        ys = np.array([y for _, y in items])
        feats = []
        for i in range(0, len(xs), args.batch_size):
            feats.append(embed(xs[i:i + args.batch_size]))
        return np.concatenate(feats, axis=0), ys

    sup_feats, sup_labels = to_batch(support)
    qry_feats, qry_labels = to_batch(query)

    preds = nearest_centroid(sup_feats, sup_labels, qry_feats, k=args.k)
    acc = float((preds == qry_labels).mean())
    mode = f"k-NN (k={args.k})" if args.k > 0 else "nearest-centroid"
    print(f"\n  {mode} accuracy: {acc * 100:.2f}%  ({len(query)} query samples)")

    # per-class report
    print("\n  Per-class accuracy:")
    for c in np.unique(qry_labels):
        mask = qry_labels == c
        ca = float((preds[mask] == c).mean())
        print(f"    class {c:>2}: {ca * 100:5.1f}%  (n={mask.sum()})")
    return acc


def main():
    p = argparse.ArgumentParser(description="Few-shot classifier on SSL embeddings (PyTorch or ONNX)")
    p.add_argument("--method", default="simclr", choices=["simclr", "dinov2", "moco_v3", "mae"])
    p.add_argument("--backbone", default="vit_small", choices=["vit_small", "vit_base", "vit_large"])
    p.add_argument("--checkpoint", default=None, help="Optional trained .pth checkpoint")
    p.add_argument("--onnx", default=None, help="Exported .onnx model (overrides --method/--backbone)")
    p.add_argument("--data-root", default="./data", help="Dataset root (train/<class>/ layout or synthetic fallback)")
    p.add_argument("--shots", type=int, default=5, help="Support shots per class")
    p.add_argument("--k", type=int, default=5, help="k for k-NN; 0 = nearest centroid")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = p.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()