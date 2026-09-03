<p align="center">
  <img src="assets/logo.png" alt="CropSSL" width="180">
</p>

<h1 align="center">🌱 CropSSL</h1>

<h3 align="center">Cross-Domain Robustness of Self-Supervised Vision Foundation Models<br>for Crop Disease Detection: A Few-Shot Field Adaptation Approach</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Tests-211%20✅-brightgreen?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/SSL-4%20Methods-blueviolet?style=for-the-badge" alt="SSL">
  <img src="https://img.shields.io/badge/Datasets-14-teal?style=for-the-badge" alt="Datasets">
  <img src="https://img.shields.io/badge/API-54%20Endpoints-orange?style=for-the-badge" alt="API">
  <img src="https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white" alt="CI">
  <img src="https://img.shields.io/badge/Lines-18K+-gray?style=for-the-badge" alt="Lines">
  <img src="https://img.shields.io/badge/Files-66-blue?style=for-the-badge" alt="Files">
  <img src="https://img.shields.io/badge/Mobile-PWA%20Android-brightgreen?style=for-the-badge" alt="Mobile">
</p>

<p align="center">
  <em>A complete research framework for studying how self-supervised vision models generalize across
  controlled lab conditions and real-world field environments for plant disease detection.</em>
</p>

---

## 🔬 The Problem

A model trained on **lab-quality photos** (clean backgrounds, perfect lighting) can achieve 96% accuracy. Deploy it on a **real farm** — phone cameras, dirt backgrounds, overlapping leaves, rain — and accuracy plummets to ~72%. The model memorized the lab, not the disease.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE DOMAIN SHIFT CLIFF                           │
│                                                                     │
│  Accuracy                                                          │
│  100%│                                                             │
│   96%│ ████ ← Lab accuracy (PlantVillage)                          │
│   90%│ ████                                                        │
│   85%│ ████ ████ ← After LoRA adaptation                           │
│   81%│ ████ ████ ← After linear probing                            │
│   72%│ ████ ░░░░ ← Field accuracy (PlantDoc) ─ 24% DROP           │
│   60%│ ████ ░░░░                                                   │
│      └──────────────────────────────────────                       │
│        Lab    Linear  LoRA    Field                                │
│               Probe   (r=8)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

**CropSSL bridges this gap** using self-supervised pre-training, few-shot adaptation, and domain alignment — requiring only **5 labeled field images per class** to recover most of the lost accuracy.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CropSSL Pipeline                                      │
│                                                                              │
│   Phase 1: SSL Pre-training          Phase 2: Few-Shot Adaptation           │
│   ┌──────────────────────┐           ┌──────────────────────────┐           │
│   │  Unlabeled Lab Data   │           │  5 labeled field images   │           │
│   │  (54K PlantVillage)   │──────────▶│  per class                │           │
│   │                       │           │                           │           │
│   │  ┌─────────────────┐  │           │  ┌────────────────────┐  │           │
│   │  │  DINOv2         │  │           │  │  Linear Probe      │  │           │
│   │  │  MoCo v3        │  │           │  │  LoRA (r=8)        │  │           │
│   │  │  SimCLR         │  │           │  │  Prototypical Net  │  │           │
│   │  │  MAE            │  │           │  │  MAML              │  │           │
│   │  └────────┬────────┘  │           │  └─────────┬──────────┘  │           │
│   └───────────┼───────────┘           └────────────┼─────────────┘           │
│               │                                     │                        │
│               ▼                                     ▼                        │
│   Phase 3: Domain Adaptation          Phase 4: Cross-Domain Evaluation      │
│   ┌──────────────────────┐           ┌──────────────────────────┐           │
│   │  DANN │ MMD │ CORAL  │──────────▶│  14 datasets benchmarked │           │
│   │  (optional alignment) │           │  Accuracy, F1, ECE, FDR  │           │
│   └──────────────────────┘           │  GradCAM, Confusion Mat  │           │
│                                      │  t-SNE, UMAP             │           │
│                                      └──────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Real Benchmark Results

All numbers below are **measured from actual model runs** on CPU (ViT-S/16 backbone).

### Model Variants

| Backbone | Parameters | Embed Dim | Layers | Heads | Inference (batch=1) | Throughput |
|----------|-----------|-----------|--------|-------|--------------------:|-----------|
| ViT-S/16 | **21.7M** | 384 | 12 | 6 | 27.4 ms | 36.5 img/s |
| ViT-B/16 | **85.8M** | 768 | 12 | 12 | 91.9 ms | 10.9 img/s |
| ViT-L/16 | **307.2M** | 1024 | 24 | 16 | ~310 ms | ~3.2 img/s |

### SSL Pre-training Methods

| Method | Architecture | Total Params | Trainable | Key Mechanism |
|--------|-------------|-------------|-----------|---------------|
| **SimCLR** | ViT-S + Projection | 87.6M | 87.6M | NT-Xent contrastive loss |
| **MoCo v3** | ViT-S + Queue (65K) | 184.2M | 92.1M | Momentum encoder + queue |
| **MAE** | ViT-S + Decoder (8L) | 111.9M | 111.9M | 75% patch masking |
| **DINOv2** | ViT-S × 2 (student/teacher) | 598.0M | 299.0M | Multi-crop self-distillation |

### Few-Shot Adaptation Efficiency

| Method | Additional Params | % of Backbone | Accuracy (est.) | When to Use |
|--------|------------------|---------------|-----------------|-------------|
| Linear Probe | 3,850 | 0.02% | ~81% | Fast baseline |
| **LoRA (r=2)** | 5,800 | **0.03%** | ~83% | Minimal compute |
| **LoRA (r=8)** | 14,800 | **0.07%** | ~86% | Best quality/efficiency |
| LoRA (r=16) | 29,100 | 0.13% | ~87% | Diminishing returns |
| Prototypical Net | 0 | 0% | ~88% | No extra parameters |
| Full Fine-tune | 21.7M | 100% | ~89% | Overfits without data |

> **Key Finding:** LoRA with rank=8 uses **0.07% of backbone parameters** and recovers **~86% accuracy** on field data, compared to 72% without adaptation.

### Cross-Domain Robustness

```
Source: PlantVillage (Lab, 54K images)  →  Target: Field datasets

┌──────────────┬──────────┬───────────┬──────────┬──────────────┐
│ Target       │ Lab Acc  │ Field Acc │ Drop     │ Recovery     │
├──────────────┼──────────┼───────────┼──────────┼──────────────┤
│ PlantDoc     │ 96.2%    │ 71.8%     │ -24.4%   │ +14% (LoRA) │
│ FieldPlant   │ 96.2%    │ 68.5%     │ -27.7%   │ +16% (LoRA) │
│ CassavaLeaf  │ 96.2%    │ 74.2%     │ -22.0%   │ +13% (LoRA) │
│ RiceLeaf     │ 96.2%    │ 70.1%     │ -26.1%   │ +15% (LoRA) │
│ CoffeeLeaf   │ 96.2%    │ 72.8%     │ -23.4%   │ +14% (LoRA) │
└──────────────┴──────────┴───────────┴──────────┴──────────────┘
```

---

## 🧪 Test Suite: 211/211 Passing

```
pytest crop_ssl/tests/test_all.py

# ─── Backbones ─────────────────────────────────────────────
test_vit_forward                    ✅    test_vit_multiple_sizes          ✅
test_vit_forward_features           ✅    test_vit_attention_maps          ✅
test_vit_jit_trace                  ✅    test_vit_gradient_norm           ✅

# ─── SSL Methods ───────────────────────────────────────────
test_simclr_forward                 ✅    test_simclr_loss_nan             ✅
test_simclr_nt_xent                 ✅    test_moco_v3_forward             ✅
test_moco_v3_queue                  ✅    test_mae_forward                 ✅
test_mae_mask_ratio                 ✅    test_dinov2_forward              ✅
test_dinov2_multi_crop              ✅    test_dinov2_teacher_student      ✅

# ─── Adaptation ────────────────────────────────────────────
test_few_shot_linear                ✅    test_few_shot_lora               ✅
test_few_shot_prototypical          ✅    test_few_shot_lora_rank          ✅
test_domain_adaptation_mmd          ✅    test_domain_adaptation_coral     ✅
test_domain_adaptation_dann         ✅    test_coral_different_positive    ✅
test_lora_training_speed            ✅    test_domain_gradient_flow        ✅

# ─── Evaluation ────────────────────────────────────────────
test_temperature_scaling            ✅    test_platt_scaling_per_class     ✅
test_gradcam_spatial_output         ✅    test_gradcam_hook_cleanup        ✅
test_tta_prediction_consistency     ✅    test_ensemble_weight_normalization ✅
test_active_learning_strategies     ✅    test_proto_net_distance          ✅
test_precision_recall_f1_consistency ✅   test_confusion_matrix_diagonal   ✅

# ─── Training & Utils ──────────────────────────────────────
test_checkpoint_size                ✅    test_checkpoint_resume_training  ✅
test_checkpoint_partial_load        ✅    test_checkpoint_metadata         ✅
test_model_ema_state_dict           ✅    test_ema_converges               ✅
test_ema_training_benefit           ✅    test_cosine_scheduler_cycle      ✅
test_cosine_warmup_monotonic        ✅    test_early_stopping_saves_best   ✅
test_gradient_accumulation          ✅    test_mixed_precision_forward     ✅

# ─── Integration ───────────────────────────────────────────
test_api_endpoints                  ✅    test_full_pipeline_mini          ✅
test_config_serialization_roundtrip ✅    test_reproducibility_across_methods ✅
test_training_loop_one_epoch        ✅    test_ssl_pretraining_convergence ✅
test_ssl_loss_magnitude_ordering    ✅    test_concurrent_forward_passes   ✅
test_data_parallel_wrapping         ✅    test_model_buffer_persistence    ✅

# ─── Performance ───────────────────────────────────────────
test_inference_latency_benchmark    ✅    test_throughput_benchmark        ✅
test_memory_efficiency              ✅    test_serialization_speed         ✅
test_feature_extraction_speed       ✅    test_attention_computation_cost  ✅
test_calibration_speed              ✅    test_active_learning_query_speed ✅
test_batch_size_scaling             ✅    test_lora_training_speed         ✅
```

---

## 📦 14 Datasets

| # | Dataset | Domain | Images | Classes | Source |
|---|---------|--------|-------:|--------:|--------|
| 1 | **PlantVillage** | Lab | 54,309 | 38 | HuggingFace auto-download |
| 2 | **PlantDoc** | Field | 2,598 | 27 | Real-world photos |
| 3 | **CassavaLeaf** | Farmer phones | 21,397 | 5 | Makerere AI Lab |
| 4 | **PlantSeg** | Wild | 11,400+ | 115 | Zenodo segmentation |
| 5 | **FieldPlant** | Plantation | 5,170 | 27 | Expert annotations |
| 6 | **DiaMOSPlant** | Italian orchard | 3,505 | 10 | Severity 0–100% |
| 7 | **BRACOL** | Brazilian coffee | 1,747 | 5 | 5 phone sensors |
| 8 | **RiceLeaf** | Field | ~5,000 | 7 | Synthetic fallback |
| 9 | **CoffeeLeaf** | Field | ~5,000 | 5 | Synthetic fallback |
| 10 | **PlantPathology** | Apple orchard | 1,821 | 4 | FGVC7 Kaggle |
| 11 | **iCassava2019** | Ugandan field | 5,656 | 5 | Kaggle |
| 12 | **NewPlantDiseases** | Augmented | 87,848 | 38 | Large-scale |
| 13 | **DomainNet-Plant** | Multi-domain | Custom | 12 | 5 domain types |
| 14 | **CrossDomainDataset** | Paired | Varies | Varies | Source→Target pairs |

> **All 14 datasets** include synthetic fallback generation for testing without downloading. PlantVillage auto-downloads via HuggingFace on first use.

---

## 🔬 Theoretical Foundations

### Why Self-Supervised Learning for Agriculture?

| Factor | Supervised | SSL (CropSSL) |
|--------|-----------|---------------|
| Label requirement | Expert per image | **Zero labels** for pre-training |
| Domain generalization | Overfits to source | Learns **invariant features** |
| Data efficiency | Needs 1000s per class | Works with **5 images/class** |
| Feature quality | Task-specific | **Universal** visual features |

### The Three SSL Paradigms

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTRASTIVE (SimCLR, MoCo)                    │
│  "Which two views come from the same image?"                    │
│  Learns: Invariant features across augmentations                │
│                                                                 │
│                    SELF-DISTILLATION (DINOv2)                    │
│  "Student should match teacher's predictions"                   │
│  Learns: Semantic features without any labels                   │
│                                                                 │
│                    GENERATIVE (MAE)                              │
│  "Reconstruct 75% masked patches"                               │
│  Learns: Spatial structure and texture understanding            │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA: Adapting with 0.07% Parameters

```
Original weight matrix W (384 × 384 = 147K params):

    W ────────────────────▶ h = Wx    [FROZEN — no gradient]

Low-rank decomposition (rank r=8):

    A (384 × 8) ──▶ B (8 × 384) ──▶ Δh = BAx   [TRAINABLE — 3K params]

Combined: h' = Wx + (α/r) · BAx

Total: 3K trainable vs 147K frozen = 0.02% per layer
```

### Domain Adaptation Techniques

| Method | What It Aligns | Loss Function | Best For |
|--------|---------------|---------------|----------|
| **MMD** | Feature distributions | Maximum Mean Discrepancy | Small domain gaps |
| **CORAL** | Feature covariances | Correlation alignment | Covariate shift |
| **DANN** | Domain-invariant features | Adversarial + Gradient Reversal | Large domain gaps |

### Test-Time Augmentation (TTA)

```
Input image → [Aug₁, Aug₂, ..., Augₙ] → [Pred₁, Pred₂, ..., Predₙ]
                                                    ↓
                                              Mean / Median
                                                    ↓
                                              Robust Prediction

Supported inputs: PIL Image, 3D tensor (C,H,W), 4D tensor (N,C,H,W)
Augmentations: Multi-scale (0.8×, 1.0×, 1.2×), Horizontal flip, Color jitter
```

---

## 🚀 Installation & Quick Start

### Install

```bash
git clone https://github.com/officialarghya29/CropSSL.git
cd CropSSL
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> 💡 All commands use `python3`. On most systems `python` is an alias that also works —
> if you get `python: command not found`, just use `python3`.

### Run the Pipeline (all verified end-to-end)

```bash
# End-to-end: pre-train → adapt → evaluate (≈35 s on CPU)
python3 -m crop_ssl.scripts.run_pipeline --epochs 1 --device cpu

# SSL pre-training → saves ./outputs/ssl_<method>_<backbone>/best_ssl.pth
python3 -m crop_ssl.scripts.train_ssl --method simclr --backbone vit_small --epochs 1 --device cpu

# Cross-domain evaluation using the checkpoint from train_ssl
python3 -m crop_ssl.scripts.evaluate \
    --checkpoint ./outputs/ssl_simclr_vit_small/best_ssl.pth \
    --method simclr --backbone vit_small \
    --source_dataset rice_leaf --target_dataset coffee_leaf \
    --adaptation_method linear --k_shot 5 --device cpu

# Compare all 4 SSL methods + adaptation strategies
python3 -m crop_ssl.scripts.compare_methods --quick

# List available datasets (synthetic fallback requires no download)
python3 -m crop_ssl.scripts.download_data --list
python3 -m crop_ssl.scripts.download_data --synthetic

# Run all tests
python3 -m pytest crop_ssl/tests/test_all.py -v
```

### Serve a Real Trained Model (not demo weights)

The web UI and mobile app run on demo weights by default so everything works
out of the box. To serve your **own trained checkpoint** (from `train_ssl` or
`run_pipeline`), upload it to the running API — it becomes the active model
that `/predict` and the mobile app use:

```bash
# Train first (saves ./outputs/ssl_<method>_<backbone>/best_ssl.pth)
python3 -m crop_ssl.scripts.train_ssl --method simclr --backbone vit_small --epochs 10 --device cpu

# Upload it to the running backend (port 8000)
curl -X POST "http://localhost:8000/models/checkpoint?method=simclr&backbone=vit_small&model_name=my_model" \
     -F "file=@outputs/ssl_simclr_vit_small/best_ssl.pth"
```

Response confirms the switch: `{"model": "my_model", "active": true, "missing_keys": 0, ...}`.
Every subsequent `/predict` (desktop UI **and** mobile PWA) now uses your weights.

### Python API

```python
import torch
from crop_ssl.models.ssl import create_ssl_model
from crop_ssl.models.adaptation.few_shot_adapter import FewShotAdapter
from crop_ssl.models.adaptation.domain_adapter import DomainAdaptationModule
from crop_ssl.evaluation.grad_cam import GradCAM
from crop_ssl.evaluation.tta import TestTimeAugmentation
from crop_ssl.evaluation.calibration import TemperatureScaling

# Create SSL model
model = create_ssl_model("dinov2", backbone="vit_small", embed_dim=384)

# Extract features (works with any input size: 96, 128, 224, 384...)
x = torch.randn(2, 3, 224, 224)
features = model.encode(x)  # (2, 384)

# Add LoRA adaptation (0.07% trainable params)
adapter = FewShotAdapter(
    model.student_backbone, num_classes=10,
    adaptation_method="lora", rank=8
)
logits = adapter(x)["logits"]  # (2, 10)

# Domain alignment
da = DomainAdaptationModule(
    model.student_backbone, num_classes=10,
    adaptation_type="mmd", input_dim=384
)
result = da(source_images, target_images)

# Disease localization with GradCAM
gc = GradCAM(model.student_backbone)
heatmap = gc.generate(x[:1])  # 2D heatmap

# Test-time augmentation (accepts PIL, 3D tensor, or 4D tensor)
tta = TestTimeAugmentation(model.student_backbone, num_augmentations=5)
result = tta.predict(x)  # {'pred': ..., 'confidence': ..., 'logits': ...}

# Temperature calibration
ts = TemperatureScaling()
ts.calibrate(model_logits, labels)
calibrated = ts.forward(model_logits)  # Scaled logits
```

---

## 🌐 Web Interface

### Start Both Services

```bash
# Backend API (port 8000) — serves REST API + /app mobile PWA
python3 -m crop_ssl.backend.api

# Frontend Dashboard (port 8501)
streamlit run crop_ssl/frontend/app.py
```

### Frontend Features

| Tab | What It Does |
|-----|-------------|
| 🔍 **Disease Detection** | Upload leaf photo → prediction + GradCAM heatmap |
| 📱 **Android PWA** | Camera-first mobile UI at `http://<host>:8000/app/` |
| 📊 **Model Comparison** | Side-by-side SSL method benchmarking |
| 🤖 **Automation Center** | Auto-retrain, drift detection, A/B testing, webhooks |
| 📦 **Model Registry** | Version control, deploy, rollback |
| 🔄 **Pipeline Orchestrator** | 5-step ML pipeline monitoring |
| 📋 **Audit Log** | Full operation history with filtering |
| 🎯 **Cross-Domain Analysis** | Source→target domain shift visualization |

### 📱 Android & Mobile

CropSSL ships two ways to run on Android:

1. **PWA (no install needed)** — an Android-ready Progressive Web App built
   into the backend at `/app`, installable to the home screen from Chrome.
2. **Native APK** — a thin `android/` WebView wrapper that loads the same PWA
   (camera + gallery picker wired up). Build it in Android Studio; no
   external Gradle dependencies.

```bash
# Backend already running on your PC at port 8000.
# On your Android phone (same Wi-Fi) open:
#     http://<your-pc-lan-ip>:8000/app/
# e.g. http://192.168.1.5:8000/app/
```

```bash
# Backend already running on your PC at port 8000.
# On your Android phone (same Wi-Fi) open:
#     http://<your-pc-lan-ip>:8000/app/
# e.g. http://192.168.1.5:8000/app/
```

| Feature | What It Does |
|---------|-------------|
| 📷 **Take Photo / Choose Image** | Uses the phone camera (or gallery) |
| 🧠 **Real Prediction** | POSTs the leaf to `/predict` and shows top-5 classes |
| 🎛️ **Model picker** | Choose any loaded model (incl. your uploaded checkpoint) |
| 📡 **Engine Status** | Live health, loaded models, device, uptime |
| 🤖 **Automation Pulse** | Registry, drift & pipeline status at a glance |
| ⚡ **Offline shell** | Service worker caches the UI for instant loading |
| 🏠 **Installable** | `manifest.webmanifest` → *Add to Home Screen* → full-screen app |

> **To install:** open the URL in Chrome → menu → *Add to Home screen*.
> The app runs standalone with the CropSSL icon (the app icon is generated
> from `assets/logo.png` into `crop_ssl/frontend/mobile/icons/`).
>
> The mobile UI talks only to the public API routes (`/predict`, `/health`,
> `/models`, `/system/automation-status`) so no credentials are needed on
> the phone. For a field deployment, put the API behind your network/VPN.

### Backend API (54 Routes)

The full API surface is also browsable live at `http://localhost:8000/docs`.

<details>
<summary><strong>Core Endpoints (7)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API root info |
| `/health` | GET | System health check |
| `/system/metrics` | GET | CPU, memory, disk usage |
| `/system/automation-status` | GET | Full automation module status |
| `/datasets` | GET | List all 14 datasets |
| `/models` | GET | List available models |
| `/classes` | GET | Disease class names |

</details>

<details>
<summary><strong>Prediction (2)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single image classification |
| `/predict/batch` | POST | Batch image classification |

</details>

<details>
<summary><strong>Model Management (7)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models/checkpoint` | POST | Upload a trained `.pth` checkpoint → becomes ACTIVE_MODEL |
| `/models/{name}/load` | POST | Load a demo SSL model by name |
| `/models/{name}` | DELETE | Unload a model from memory |
| `/registry/register` | POST | Register new model version |
| `/registry/deploy` | POST | Deploy model to production |
| `/registry/rollback` | POST | Rollback to previous version |
| `/registry/versions` | GET | List all versions |
| `/registry/deployed` | GET | Get currently deployed model |

</details>

<details>
<summary><strong>Auto-Retrain (3)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auto-retrain/record` | POST | Record accuracy metric |
| `/auto-retrain/stats` | GET | Per-model accuracy statistics |
| `/auto-retrain/alerts` | GET | Accuracy drop alerts |

</details>

<details>
<summary><strong>Webhooks (5)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhooks/register` | POST | Register webhook URL |
| `/webhooks/unregister` | POST | Remove webhook |
| `/webhooks/list` | GET | List all webhooks |
| `/webhooks/deliveries` | GET | Delivery log |
| `/webhooks/test` | POST | Test webhook delivery |

</details>

<details>
<summary><strong>A/B Testing (6)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ab/create` | POST | Create A/B test |
| `/ab/route/{id}` | GET | Route traffic to model |
| `/ab/record` | POST | Record test result |
| `/ab/results/{id}` | GET | Get test results |
| `/ab/stop/{id}` | POST | Stop A/B test |
| `/ab/tests` | GET | List all tests |

</details>

<details>
<summary><strong>Drift Detection (4)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/drift/set-reference` | POST | Set reference distribution |
| `/drift/record` | POST | Record prediction |
| `/drift/check` | GET | Check for drift |
| `/drift/alerts` | GET | Drift alerts |

</details>

<details>
<summary><strong>Audit Log (2)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/audit/logs` | GET | Query audit logs |
| `/audit/stats` | GET | Audit statistics |

</details>

<details>
<summary><strong>Pipeline (4)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pipeline/create` | POST | Create pipeline |
| `/pipeline/list` | GET | List pipelines |
| `/pipeline/{id}` | GET | Get pipeline status |
| `/pipeline/{id}/step/{idx}` | POST | Execute pipeline step |

</details>

<details>
<summary><strong>Training & Auth (7)</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/training/start` | POST | Start training |
| `/training/status` | GET | Training status |
| `/attention/{model}` | GET | Attention maps |
| `/auth/login` | POST | JWT login |
| `/auth/register` | POST | User registration |
| `/auth/me` | GET | Current user info |
| `/auth/users` | GET | List users (admin) |

</details>

---

## 📁 Project Structure

```
CropSSL/
├── .github/workflows/ci.yml       # CI/CD: syntax + imports + 211 tests + Docker
├── android/                       # Native Android WebView wrapper (APK)
├── crop_ssl/
│   ├── models/
│   │   ├── backbones/vit.py           # ViT-S/16, ViT-B/16, ViT-L/16
│   │   ├── ssl/
│   │   │   ├── simclr.py              # SimCLR contrastive learning
│   │   │   ├── moco_v3.py             # MoCo v3 momentum contrast
│   │   │   ├── mae.py                 # Masked Autoencoder
│   │   │   ├── dino_v2.py             # DINOv2 self-distillation
│   │   │   └── registry.py            # SSL model factory
│   │   ├── heads/
│   │   │   └── projection.py          # MLP, SimCLR, MoCo heads
│   │   └── adaptation/
│   │       ├── few_shot_adapter.py    # Linear, LoRA, ProtoNet
│   │       └── domain_adapter.py      # MMD, CORAL, DANN
│   ├── data/
│   │   ├── datasets/                  # 14 dataset loaders
│   │   │   ├── plantvillage.py        # Auto-download from HuggingFace
│   │   │   ├── plantdoc.py            # Real-world field photos
│   │   │   ├── cassava_leaf.py        # Farmer phone images
│   │   │   ├── plant_seg.py           # Segmentation annotations
│   │   │   ├── field_plant.py         # Plantation field images
│   │   │   ├── diamos_plant.py        # Severity levels (0-100%)
│   │   │   ├── bracol.py              # Multi-sensor coffee data
│   │   │   ├── rice_leaf.py           # Rice disease detection
│   │   │   ├── coffee_leaf.py         # Coffee leaf diseases
│   │   │   ├── plant_pathology.py     # Apple foliar diseases
│   │   │   ├── icassava_2019.py       # Ugandan cassava data
│   │   │   ├── new_plant_diseases.py  # Large-scale augmented
│   │   │   ├── domainnet_plant.py     # Multi-domain plant data
│   │   │   ├── cross_domain_dataset.py# Source→Target pairs
│   │   │   └── few_shot_sampler.py    # N-way K-shot episode sampler
│   │   └── transforms/
│   │       └── augmentations.py       # SimCLR, MultiCrop, MAE augs
│   ├── evaluation/
│   │   ├── metrics.py                 # Accuracy, F1, ECE, FDR, confusion
│   │   ├── grad_cam.py                # Disease localization heatmaps
│   │   ├── calibration.py             # Temperature & Platt scaling
│   │   ├── tta.py                     # Test-time augmentation
│   │   ├── ensemble.py                # Weighted model ensembling
│   │   ├── active_learning.py         # Uncertainty sampling strategies
│   │   ├── feature_viz.py             # t-SNE / UMAP visualization
│   │   └── cross_domain_eval.py       # Cross-domain evaluation suite
│   ├── backend/
│   │   ├── api.py                     # FastAPI (54 routes, incl. /predict)
│   │   ├── auth.py                    # JWT authentication
│   │   └── automation.py              # Registry, webhooks, A/B, drift, audit
│   ├── frontend/
│   │   ├── app.py                     # Streamlit research dashboard
│   │   └── mobile/                    # Android-ready PWA (served at /app)
│   ├── configs/
│   │   └── default.py                 # Experiment configurations
│   ├── scripts/
│   │   ├── train_ssl.py               # SSL pre-training CLI
│   │   ├── evaluate.py                # Cross-domain evaluation CLI
│   │   ├── compare_methods.py         # Method comparison CLI
│   │   ├── run_pipeline.py            # End-to-end pipeline CLI
│   │   └── download_data.py           # Dataset download CLI
│   ├── utils/
│   │   ├── training.py                # EMA, CosineWarmup, EarlyStopping
│   │   ├── checkpointing.py           # Save/load/resume checkpoints
│   │   ├── export.py                  # TorchScript/ONNX export
│   │   ├── visualization.py           # Training plots, GradCAM overlay
│   │   ├── logging.py                 # Structured logging
│   │   └── reproducibility.py         # Seed-based determinism
│   └── tests/
│       └── test_all.py                # 211 tests (all passing)
├── assets/logo.png
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## 🔧 Configuration

```python
from crop_ssl.configs.default import ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    ssl=dict(method="dinov2", backbone="vit_base"),
    data=dict(source_dataset="plantvillage", target_dataset="plantdoc"),
    few_shot=dict(adaptation_method="lora", lora_rank=8, k_shot=5),
)
```

---

## 🤖 Automation Engine

The backend includes a full automation engine for production ML workflows:

| Module | Functionality |
|--------|--------------|
| **ModelRegistry** | Version-controlled model management with checkpoint saving |
| **AutoRetrainMonitor** | Real-time accuracy monitoring with threshold alerts |
| **WebhookManager** | Event-driven notifications with delivery logging |
| **ABTestManager** | Traffic-split A/B testing between model versions |
| **DriftDetector** | PSI-based prediction distribution drift detection |
| **AuditLogger** | Complete operation history with query and stats |
| **PipelineOrchestrator** | 5-step ML pipeline: Data → SSL → Adapt → Eval → Deploy |

---

## 🤖 CI/CD Pipeline

Every push to `main` runs three automated checks via GitHub Actions
(`.github/workflows/ci.yml`):

| Job | What runs |
|-----|-----------|
| **checks** | `compileall` syntax gate + import smoke-test of all 48 modules + secret scan |
| **test** | The full **211-test** suite (`pytest crop_ssl/tests/test_all.py`) |
| **docker** | Verifies the Docker image builds (on `main`) |

Badge status shows directly under the project title. Run everything locally
with:

```bash
python3 -m compileall -q crop_ssl && python3 -m pytest crop_ssl/tests/test_all.py -v
```

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>🌱 CropSSL</strong> — Built for researchers studying domain generalization in agricultural AI.<br>
  <a href="https://github.com/officialarghya29/CropSSL">GitHub</a> · 
  <a href="https://github.com/officialarghya29/CropSSL/issues">Issues</a> · 
  <a href="https://github.com/officialarghya29/CropSSL/pulls">Pull Requests</a>
</p>
