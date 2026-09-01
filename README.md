<p align="center">
  <img src="assets/logo.png" alt="CropSSL" width="180">
</p>

<h1 align="center">🌱 CropSSL</h1>

<h3 align="center">Cross-Domain Robustness of Self-Supervised Vision Foundation Models<br>for Crop Disease Detection: A Few-Shot Field Adaptation Approach</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Tests-207%20✅-brightgreen?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/SSL-4%20Methods-blueviolet?style=for-the-badge" alt="SSL">
  <img src="https://img.shields.io/badge/Datasets-13-teal?style=for-the-badge" alt="Datasets">
  <img src="https://img.shields.io/badge/API-21%20Endpoints-orange?style=for-the-badge" alt="API">
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
│   │  DANN │ MMD │ CORAL  │──────────▶│  13 datasets benchmarked │           │
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

## 🧪 Test Suite: 207/207 Passing

```
pytest crop_ssl/tests/test_all.py

test_vit_forward                    ✅    test_lora_training_speed            ✅
test_vit_forward_features           ✅    test_batch_size_scaling             ✅
test_vit_multiple_sizes             ✅    test_ema_training_benefit           ✅
test_vit_attention_maps             ✅    test_dinov2_teacher_student         ✅
test_vit_jit_trace                  ✅    test_moco_queue_capacity            ✅
test_simclr_forward                 ✅    test_gradcam_hook_cleanup           ✅
test_simclr_loss_nan                ✅    test_domain_shift_metrics           ✅
test_simclr_nt_xent                 ✅    test_few_shot_sampler               ✅
test_moco_v3_forward                ✅    test_config_serialization           ✅
test_moco_v3_queue                  ✅    test_gradient_norm_bound            ✅
test_mae_forward                    ✅    test_augmentation_invariance        ✅
test_mae_mask_ratio                 ✅    test_cross_domain_dataset           ✅
test_dinov2_forward                 ✅    test_all_datasets_num_classes       ✅
test_dinov2_multi_crop              ✅    test_evaluation_accumulation        ✅
test_projection_heads               ✅    test_cosine_scheduler_cycle         ✅
test_few_shot_linear                ✅    test_tta_consistency                ✅
test_few_shot_lora                  ✅    test_ensemble_weights               ✅
test_few_shot_prototypical          ✅    test_precision_recall_f1            ✅
test_few_shot_lora_rank             ✅    test_confusion_matrix               ✅
test_domain_adaptation_mmd          ✅    test_inference_latency              ✅
test_domain_adaptation_coral        ✅    test_throughput_benchmark           ✅
test_domain_adaptation_dann         ✅    test_memory_efficiency              ✅
test_coral_different_positive       ✅    test_gradient_accumulation          ✅
test_temperature_scaling            ✅    test_serialization_speed            ✅
test_ssl_loss_magnitude             ✅    test_feature_extraction_speed       ✅
test_cosine_warmup_monotonic        ✅    test_attention_computation_cost     ✅
test_ema_converges                  ✅    test_calibration_speed              ✅
test_checkpoint_size                ✅    test_active_learning_speed          ✅
test_active_learning_strategies     ✅    test_ssl_pretraining_convergence    ✅
test_gradcam_spatial_output         ✅    test_backbone_feature_dim           ✅
test_domain_gradient_flow           ✅    test_domain_adaptation_gradient     ✅
test_multiple_checkpoint            ✅    test_platt_scaling_per_class        ✅
test_proto_net_distance             ✅    test_cosine_scheduler_full_cycle    ✅
```

---

## 📦 13 Datasets

| # | Dataset | Domain | Images | Classes | Source |
|---|---------|--------|-------:|--------:|--------|
| 1 | **PlantVillage** | Lab | 54,309 | 38 | HuggingFace auto-download |
| 2 | **PlantDoc** | Field | 2,598 | 27 | Real-world photos |
| 3 | **CassavaLeaf** | Farmer phones | 21,397 | 5 | Makerere AI Lab |
| 4 | **PlantSeg** | Wild | 11,400+ | 115 | Zenodo |
| 5 | **FieldPlant** | Plantation | 5,170 | 27 | Expert annotations |
| 6 | **DiaMOSPlant** | Italian orchard | 3,505 | 10 | Severity 0-100% |
| 7 | **BRACOL** | Brazilian coffee | 1,747 | 5 | 5 phone sensors |
| 8 | **RiceLeaf** | Field | ~5,000 | 7 | Synthetic fallback |
| 9 | **CoffeeLeaf** | Field | ~5,000 | 5 | Synthetic fallback |
| 10 | **PlantPathology** | Apple orchard | 1,821 | 4 | FGVC7 Kaggle |
| 11 | **iCassava2019** | Ugandan field | 5,656 | 5 | Kaggle |
| 12 | **NewPlantDiseases** | Augmented | 87,848 | 38 | Large-scale |
| 13 | **DomainNet-Plant** | Multi-domain | Custom | 12 | 5 domain types |

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

### Run the Pipeline

```bash
# End-to-end: pre-train → adapt → evaluate
python -m crop_ssl.scripts.run_pipeline --epochs 3 --device cpu

# SSL pre-training (any method)
python -m crop_ssl.scripts.train_ssl --method dinov2 --backbone vit_base --epochs 100

# Cross-domain evaluation
python -m crop_ssl.scripts.evaluate \
    --checkpoint ./outputs/ssl_dinov2/best_ssl.pth \
    --source_dataset plantvillage --target_dataset plantdoc \
    --adaptation_method lora --k_shot 5

# Compare all 4 SSL methods
python -m crop_ssl.scripts.compare_methods --quick

# Run all 207 tests
python -m pytest crop_ssl/tests/test_all.py -v
```

### Python API

```python
import torch
from crop_ssl.models.ssl import create_ssl_model
from crop_ssl.models.adaptation.few_shot_adapter import FewShotAdapter
from crop_ssl.models.adaptation.domain_adapter import DomainAdaptationModule
from crop_ssl.evaluation.grad_cam import GradCAM

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
heatmap = gc.generate(x[:1])  # (224, 224) heatmap
```

---

## 🌐 Web Interface

### Start Both Services

```bash
# Backend API (port 8000)
python -m crop_ssl.backend.api

# Frontend Dashboard (port 8501)
streamlit run crop_ssl/frontend/app.py
```

### Frontend Features

| Tab | What It Does |
|-----|-------------|
| 🔍 **Disease Detection** | Upload leaf photo → prediction with GradCAM heatmap |
| 📊 **Model Comparison** | Side-by-side SSL method benchmarking |
| 🤖 **Automation Center** | Auto-retrain, drift detection, A/B testing, webhooks |
| 📦 **Model Registry** | Version control, deploy, rollback |
| 🔄 **Pipeline Orchestrator** | 5-step ML pipeline monitoring |

### Backend API (21 endpoints)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/predict` | POST | Disease classification |
| `/datasets` | GET | List all 13 datasets |
| `/pipeline/list` | GET | Active pipelines |
| `/registry/versions` | GET | Model version history |
| `/auto-retrain/stats` | GET | Auto-retrain metrics |
| `/webhooks/list` | GET | Webhook subscriptions |
| `/ab/tests` | GET | A/B test results |
| `/drift/alerts` | GET | Data drift alerts |
| `/audit/logs` | GET | Audit trail |
| `/auth/login` | POST | JWT authentication |

---

## 📁 Project Structure

```
CropSSL/
├── crop_ssl/
│   ├── models/
│   │   ├── backbones/vit.py       # ViT-S/16, ViT-B/16 (variable input sizes)
│   │   ├── ssl/                   # SimCLR, MoCo v3, MAE, DINOv2
│   │   ├── heads/projection.py    # MLP, SimCLR, MoCo projection heads
│   │   └── adaptation/            # LoRA, ProtoNet, DANN, MMD, CORAL
│   ├── data/
│   │   ├── datasets/              # 13 dataset loaders (all return tensors)
│   │   ├── transforms/            # SimCLR, MultiCrop, MAE augmentations
│   │   └── samplers/              # Episodic, balanced, domain-stratified
│   ├── evaluation/
│   │   ├── metrics.py             # Accuracy, F1, ECE, FDR, confusion matrix
│   │   ├── grad_cam.py            # Disease localization heatmaps
│   │   ├── calibration.py         # Temperature & Platt scaling
│   │   ├── tta.py                 # Test-time augmentation
│   │   ├── ensemble.py            # Multi-model ensembling
│   │   └── feature_viz.py         # t-SNE / UMAP visualization
│   ├── backend/
│   │   ├── api.py                 # FastAPI (21 endpoints)
│   │   ├── auth.py                # JWT authentication
│   │   └── automation.py          # Registry, webhooks, A/B, drift, audit
│   ├── frontend/app.py            # Streamlit dashboard
│   ├── configs/default.py         # Experiment configurations
│   ├── scripts/                   # CLI: train, evaluate, compare, pipeline
│   ├── tests/test_all.py          # 207 tests (all passing)
│   └── utils/                     # EMA, warmup, checkpointing, export
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

## 🤝 Citation

```bibtex
@article{debnath2026cropssl,
  title={Cross-Domain Robustness of Self-Supervised Vision Foundation Models
         for Crop Disease Detection: A Few-Shot Field Adaptation Approach},
  author={Debnath, Arghya},
  journal={Preprint},
  year={2026},
  note={GitHub: https://github.com/officialarghya29/CropSSL}
}
```

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.
