#!/usr/bin/env python3
"""
Controlled Covariate-Shift Experiment (the "why it works" evidence).

Simulates the lab→field problem on structured synthetic data so the claim can be
tested end-to-end on CPU in minutes, through the *real* pipeline components:

* SSL backbones come from ``crop_ssl.models.ssl.create_ssl_model`` (MoCo v3 / ViT-S)
* Few-shot adapters come from ``crop_ssl.models.adaptation.few_shot_adapter``
* The SSL backbone is pre-trained on the unlabeled SOURCE (lab) data,
  exactly as the CropSSL pipeline prescribes — labels are never used
  for pre-training.

Data: 6 "disease" classes encoded by colour + leaf-like gradient. The FIELD
domain is the same classes shot by a different camera: independent per-channel
white-balance gain, non-linear gamma warp, sensor noise and clutter patches.
A linear head fit on lab embeddings cannot invert this shift (covariate shift).

What is measured (all numbers are actual run output, saved to --out as JSON):

  naive   head fit on LAB data only, deployed to FIELD   (the cliff)
  linear  head re-fit on k FIELD shots per class
  lora    LoRA(r=8) on k FIELD shots per class
  proto   prototypical network on k FIELD shots (0 trainable params)
  oracle  head fit directly on FIELD (upper bound)

Run on the SSL pre-trained backbone AND a random-init backbone of the same
architecture, so the "does pre-training help?" question is answered on
identical data and budget.

Usage:
    python3 -m crop_ssl.scripts.covariate_shift_exp --out results/covariate_shift.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from crop_ssl.utils.reproducibility import set_seed

SIZE, N_CLS = 64, 6


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def make_data(n, field, seed=1, noise=0.06):
    """Class-coded colours; field=True applies WB + gamma + noise + clutter."""
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    colors = np.array(
        [[220, 60, 60], [60, 200, 80], [70, 80, 220], [230, 180, 40],
         [170, 60, 190], [40, 190, 210]],
        dtype=np.float32,
    )
    for _ in range(n):
        c = rng.randint(N_CLS)
        img = np.zeros((SIZE, SIZE, 3), dtype=np.float32) + colors[c]
        yy = np.linspace(0, 1, SIZE)[:, None]
        xx = np.linspace(0, 1, SIZE)[None, :]
        img = img * (0.75 + 0.25 * np.sin(3 * xx) * np.cos(2 * yy))[..., None]
        if field:
            gains = rng.uniform(0.55, 1.35, size=3).astype(np.float32)
            img = img * gains                       # white-balance error
            img = np.sign(img) * (np.abs(img) ** 0.8)  # gamma warp
            if noise:
                img = img + rng.randn(*img.shape) * noise
            for _ in range(6):                      # clutter (soil/shadow)
                px, py = rng.randint(0, SIZE - 8), rng.randint(0, SIZE - 8)
                img[py:py + 8, px:px + 8] *= rng.uniform(0.3, 0.9)
        img = np.clip(img / 255.0, 0, 1).astype(np.float32)
        xs.append(img.transpose(2, 0, 1))
        ys.append(c)
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys)


def augment(x):
    """SSL augmentation matching the deployment shift: per-channel white-balance
    gain, non-linear gamma, sensor noise, flip. Teaching the encoder invariance
    to exactly the perturbations the FIELD domain applies is the mechanism by
    which SSL pre-training is expected to absorb covariate shift."""
    x = x.clone()
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])
    g = torch.empty(3, 1, 1).uniform_(0.6, 1.4)   # white-balance jitter
    x = x * g
    x = torch.sign(x) * (torch.abs(x) ** torch.empty(1).uniform_(0.75, 1.1).item())
    x = x + torch.randn_like(x) * 0.02            # sensor noise
    return x


# --------------------------------------------------------------------------
# SSL pre-training on unlabeled source data
# --------------------------------------------------------------------------
def pretrain_ssl(steps: int = 120, batch_size: int = 32, lr: float = 8e-4,
                 device: str = "cpu") -> torch.nn.Module:
    """MoCo v3 ViT-S/16 pre-trained on unlabeled lab data (the CropSSL premise)."""
    from crop_ssl.models.ssl import create_ssl_model
    model = create_ssl_model("moco_v3", backbone="vit_small", embed_dim=384)
    # MoCo with a 4096 queue is stable at small batch; keep LR moderate and decay.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    for step in range(steps):
        x1, _ = make_data(batch_size, field=False, seed=step % 7)
        x2, _ = make_data(batch_size, field=False, seed=(step + 100) % 7)
        out = model(augment(x1), augment(x2))
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        sched.step()
    model.eval()
    return model.query_encoder  # the ViT backbone


def encode(backbone, x, bs=96):
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            feats.append(backbone.forward_features(x[i:i + bs]))
    return torch.cat(feats)


# --------------------------------------------------------------------------
# evaluations
# --------------------------------------------------------------------------
def linear_head(fit_f, fit_y, test_f, test_y, seed=0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=800, C=1.0, random_state=seed)
    clf.fit(fit_f.numpy(), fit_y.numpy())
    return float((clf.predict(test_f.numpy()) == test_y.numpy()).mean())


def fresh_backbone(backbone):
    """Deep-copy the backbone.

    FewShotAdapter's LoRA path injects adapters into the backbone IN-PLACE, so
    every method must receive its own copy or later methods would run on an
    already-adapted (contaminated) network.
    """
    import copy
    return copy.deepcopy(backbone)


def adapt(backbone, method, s_x, s_y, q_x, q_y, steps=100, lr=5e-3):
    """Fit a FewShotAdapter on k shots, evaluate on the query set.

    The backbone is deep-copied per call because LoRA mutates it in-place;
    sharing one instance across methods would silently contaminate results.
    """
    from crop_ssl.models.adaptation.few_shot_adapter import FewShotAdapter
    backbone = fresh_backbone(backbone)
    adapter = FewShotAdapter(backbone, num_classes=N_CLS,
                             adaptation_method=method, rank=8)
    if method == "prototypical":
        adapter.eval()
        with torch.no_grad():
            out = adapter(q_x, support_images=s_x, support_labels=s_y, n_way=N_CLS)
            acc = float((out["logits"].argmax(1) == q_y).float().mean())
        return acc, 0
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    adapter.train()
    for _ in range(steps):
        idx = torch.randperm(len(s_x))[: min(16, len(s_x))]
        out = adapter(s_x[idx])
        loss = lossf(out["logits"], s_y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    adapter.eval()
    with torch.no_grad():
        out = adapter(q_x)
        acc = float((out["logits"].argmax(1) == q_y).float().mean())
    return acc, adapter.get_trainable_params()


def calibration_diagnosis(backbone, src_x, src_y, tgt_x, tgt_y) -> dict:
    """Does lab-calibrated confidence stay calibrated under the field shift?

    Fits a logistic head on LAB embeddings, learns a temperature on LAB
    logits (TemperatureScaling), then measures the Expected Calibration
    Error of those lab-calibrated predictions on FIELD data. High field ECE
    = confidence that does not transfer across the shift (overconfident
    errors in the field).

    Returns:
        Dict with 'T', 'ece_field_raw', 'ece_field_cal', 'acc_field'.
    """
    from sklearn.linear_model import LogisticRegression
    from crop_ssl.evaluation.calibration import TemperatureScaling

    src_f = encode(backbone, src_x)
    tgt_f = encode(backbone, tgt_x)
    clf = LogisticRegression(max_iter=800, C=1.0, random_state=0)
    clf.fit(src_f.numpy(), src_y.numpy())
    lab_logits = torch.from_numpy(clf.predict_log_proba(src_f.numpy())).float()
    fld_logits = torch.from_numpy(clf.predict_log_proba(tgt_f.numpy())).float()

    ts = TemperatureScaling(init_temperature=1.5)
    cal = ts.calibrate(lab_logits, src_y)
    ece_raw = ts._compute_ece(fld_logits, tgt_y)
    ece_cal = ts._compute_ece(ts.forward(fld_logits), tgt_y)
    acc = float((clf.predict(tgt_f.numpy()) == tgt_y.numpy()).mean())
    return {
        "temperature": float(cal["temperature"]),
        "ece_field_raw": ece_raw,
        "ece_field_cal": ece_cal,
        "acc_field": acc,
    }


def run_table(backbone, tag, k: int = 5, seed: int = 0) -> dict:
    """Full lab→field table for one backbone."""
    set_seed(seed)
    src_x, src_y = make_data(500, field=False, seed=1)
    tgt_x, tgt_y = make_data(800, field=True, seed=2)

    src_f = encode(backbone, src_x)
    tgt_f = encode(backbone, tgt_x)

    # Note: instance-level CKA (lab vs field embeddings) is intentionally NOT
    # reported here. The synthetic source has only N_CLS distinct images
    # (identical per-class copies), so per-instance Gram geometry is degenerate
    # and CKA is uninformative on this toy data. CKA is the diagnosis tool for
    # real structured domain gaps (PlantVillage -> PlantDoc) where instance
    # diversity is genuine -- see crop_ssl.evaluation.cka.

    # Calibration transfer diagnosis: does lab-calibrated confidence survive
    # the shift? (ECE on FIELD with a temperature learned on LAB.)
    cal = calibration_diagnosis(backbone, src_x, src_y, tgt_x, tgt_y)
    print(f"  field ECE (lab-calibrated)          : {cal['ece_field_cal']*100:5.1f}%")

    rng = np.random.RandomState(0)
    shots, query = [], []
    for c in range(N_CLS):
        ci = np.where(tgt_y.numpy() == c)[0]
        rng.shuffle(ci)
        shots += list(ci[:k])
        query += list(ci[k:])
    s_idx = np.array(shots)
    q_idx = np.array(query)
    s_x, s_y = tgt_x[s_idx], tgt_y[s_idx]
    q_x, q_y = tgt_x[q_idx], tgt_y[q_idx]

    row = {"tag": tag, "k_shots": k, "query_size": int(len(q_idx))}
    row["ece_field_raw"] = cal["ece_field_raw"]
    row["ece_field_cal"] = cal["ece_field_cal"]
    row["temperature"] = cal["temperature"]
    row["acc_field"] = cal["acc_field"]
    row["naive"] = linear_head(src_f, src_y, tgt_f[q_idx], tgt_y[q_idx])
    print(f"  naive (lab head -> field)          : {row['naive']*100:5.1f}%")
    for method in ["linear", "lora", "prototypical"]:
        acc, tp = adapt(backbone, method, s_x, s_y, q_x, q_y)
        row[method] = acc
        row[f"{method}_params"] = tp
        print(f"  {method:<13} ({tp:>7,} trainable)  : {acc*100:5.1f}%")
    row["oracle"] = linear_head(tgt_f[q_idx], tgt_y[q_idx],
                                tgt_f[q_idx], tgt_y[q_idx])
    print(f"  oracle (head fit on field)         : {row['oracle']*100:5.1f}%")
    return row


def main():
    p = argparse.ArgumentParser(description="Controlled covariate-shift experiment")
    p.add_argument("--out", default=None, help="Save JSON of results")
    p.add_argument("--k", type=int, default=5, help="Field shots per class")
    p.add_argument("--k-sweep", action="store_true",
                   help="Sweep k in [1, 2, 5, 10, 20] (reproduces the paper table)")
    p.add_argument("--ssl-steps", type=int, default=120, help="SSL pre-train steps")
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--skip-ssl", action="store_true",
                   help="Use a random backbone instead of pre-training (speed)")
    args = p.parse_args()

    set_seed(0)
    torch.set_num_threads(4)

    from crop_ssl.models.ssl import create_ssl_model
    if args.skip_ssl:
        ssl = create_ssl_model("moco_v3", backbone="vit_small",
                               embed_dim=384).query_encoder.eval()
        print("Skipping SSL pre-training (random backbone).")
    else:
        print("Pre-training SSL encoder (MoCo v3 / ViT-S) on unlabeled lab data...")
        ssl = pretrain_ssl(steps=args.ssl_steps, lr=args.lr, device=args.device)
    rnd = create_ssl_model("moco_v3", backbone="vit_small",
                           embed_dim=384).query_encoder.eval()

    print("\nBackbone A: SSL pre-trained | Backbone B: random init (same arch)")
    if args.k_sweep:
        rows = []
        for k in [1, 2, 5, 10, 20]:
            for tag, bb in [("ssl_pretrained", ssl), ("random_init", rnd)]:
                print(f"\n### k={k} {tag}")
                rows.append(run_table(bb, tag, k=k))
    else:
        rows = [run_table(ssl, "ssl_pretrained", k=args.k),
                run_table(rnd, "random_init", k=args.k)]

    print("\n================ SUMMARY (measured) ================")
    if args.k_sweep:
        print(f"{'backbone':<15}{'k':>3}{'naive':>9}{'linear':>9}{'lora':>9}{'proto':>9}{'oracle':>9}")
        for r in rows:
            print(f"{r['tag']:<15}{r['k_shots']:>3}{r['naive']*100:>8.1f}%{r['linear']*100:>8.1f}%"
                  f"{r['lora']*100:>8.1f}%{r['prototypical']*100:>8.1f}%{r['oracle']*100:>8.1f}%")
    else:
        print(f"{'backbone':<15}{'naive':>8}{'linear':>9}{'lora':>9}{'proto':>9}{'oracle':>9}")
        for r in rows:
            print(f"{r['tag']:<15}{r['naive']*100:>7.1f}%{r['linear']*100:>8.1f}%"
                  f"{r['lora']*100:>8.1f}%{r['prototypical']*100:>8.1f}%"
                  f"{r['oracle']*100:>8.1f}%")
    print("\nCalibration transfer (temperature learned on LAB, ECE on FIELD):")
    print(f"{'backbone':<15}{'T':>7}{'field ECE raw':>15}{'field ECE cal':>16}{'field acc':>11}")
    # rows alternate [ssl_k1, rnd_k1, ssl_k2, rnd_k2, ...] -> first row of each
    # backbone via stride slicing (robust to any number of k values).
    seen = set()
    for r in rows:
        if r["tag"] in seen:
            continue
        seen.add(r["tag"])
        print(f"{r['tag']:<15}{r['temperature']:>7.3f}{r['ece_field_raw']*100:>14.1f}%"
              f"{r['ece_field_cal']*100:>15.1f}%{r['acc_field']*100:>10.1f}%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()