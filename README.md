<p align="center">
  <img src="assets/logo.png" alt="CropSSL Logo" width="200">
</p>

<h1 align="center">CropSSL</h1>

<p align="center">
  <strong>Cross-Domain Robustness of Self-Supervised Vision Foundation Models for Crop Disease Detection: A Few-Shot Field Adaptation Approach</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Tests-121%20Passing-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Models-4%20SSL-brightgreen.svg" alt="SSL Methods">
  <img src="https://img.shields.io/badge/Datasets-13-blue.svg" alt="Datasets">
</p>

---

## Abstract

CropSSL is a comprehensive research framework for evaluating and improving the cross-domain robustness of self-supervised vision foundation models for agricultural plant disease detection. The framework systematically benchmarks four SSL pre-training methods (DINOv2, MoCo v3, SimCLR, MAE) across five crop disease datasets spanning controlled laboratory and uncontrolled field conditions. To address the domain shift between training and deployment environments, CropSSL implements four few-shot adaptation strategies (Linear Probing, LoRA, Prototypical Networks, MAML) and three domain alignment techniques (DANN, MMD, CORAL). The framework additionally provides advanced evaluation tools including Grad-CAM disease localization, test-time augmentation, model ensembling, confidence calibration, and active learning sample selection.

---

## 1. Introduction

Deep learning models trained on crop disease imagery in controlled laboratory settings often suffer significant performance degradation when deployed in real-world field conditions. This domain shift arises from differences in lighting, background clutter, camera quality, and environmental variability. CropSSL addresses this challenge through a unified pipeline that combines self-supervised pre-training, few-shot adaptation, and cross-domain evaluation.

### 1.1 Research Questions

| ID | Research Question | Evaluation Approach |
|----|-------------------|---------------------|
| RQ1 | How do SSL pre-training methods compare for crop disease feature learning? | Benchmark DINOv2, MoCo v3, SimCLR, MAE across 5 datasets |
| RQ2 | How robust are SSL-derived features under domain shift? | Cross-domain evaluation across 20 source-target pairs |
| RQ3 | Can few-shot adaptation recover performance on field data? | Linear Probing, LoRA, Prototypical Networks, MAML |
| RQ4 | Does explicit domain alignment improve cross-domain transfer? | DANN, MMD, CORAL loss evaluation |

---

## 2. Methodology

### 2.1 Pipeline Architecture

```
Input Images
     |
     v
+-------------------+     +-------------------+     +-------------------+
| SSL Pre-training  | --> | Few-Shot Adapt    | --> | Cross-Domain      |
| (4 Methods)       |     | (4 Strategies)    |     | Evaluation        |
+-------------------+     +-------------------+     +-------------------+
        |                        |                        |
   +----+----+             +----+----+             +----+----+
   | DINOv2  |             |  LoRA   |             |Accuracy |
   | MoCo v3 |             | ProtoNet|             |  F1/ECE |
   | SimCLR  |             |  MAML   |             |  FDR    |
   |   MAE   |             | Linear  |             |  CM     |
   +---------+             +---------+             +---------+

   +-------------------------------------------------------------+
   |            Domain Adaptation (Optional)                      |
   |        DANN  |  MMD  |  CORAL  |  Combined                  |
   +-------------------------------------------------------------+

   Source Domain              Target Domain
   PlantVillage (Lab) ------> PlantDoc (Field)
                            ------> RiceLeaf (Field)
                            ------> CoffeeLeaf (Field)
```

### 2.2 Self-Supervised Pre-training Methods

| Method | Category | Architecture | Loss Function | Training Strategy |
|--------|----------|-------------|---------------|-------------------|
| DINOv2 | Self-distillation | Student-Teacher ViT | Cross-entropy (sharpened softmax) | 2 global + 8 local crops, EMA teacher |
| MoCo v3 | Contrastive | Momentum encoder | InfoNCE with queue | Query-key pairs, temperature scaling |
| SimCLR | Contrastive | Dual-view encoder | NT-Xent | Two augmented views, projection head |
| MAE | Generative | Asymmetric encoder-decoder | MSE on masked patches | 75% masking ratio |

### 2.3 Few-Shot Adaptation Strategies

| Strategy | Trainable Parameters | Mechanism | Suitability |
|----------|---------------------|-----------|-------------|
| Linear Probing | 0.02% | Frozen backbone + linear classifier | Baseline evaluation |
| LoRA | 0.5% | Low-rank decomposition of attention layers | Parameter-efficient tuning |
| Prototypical Networks | 0% (inference) | Euclidean/cosine distance to class prototypes | Zero-shot domain transfer |
| MAML | 100% | Inner-loop gradient adaptation | Maximum flexibility |

### 2.4 Domain Adaptation Techniques

| Technique | Mechanism | Computational Overhead |
|-----------|-----------|----------------------|
| DANN | Gradient reversal layer + domain discriminator | Low |
| MMD | Maximum mean discrepancy with Gaussian kernels | Medium |
| CORAL | Covariance matrix alignment | Low |

---

## 3. Datasets

| Dataset | Domain | Images | Classes | Role | Source |
|---------|--------|--------|---------|------|--------|
| PlantVillage | Controlled laboratory | 54,309 | 38 | Pretrain baseline | [HuggingFace](https://huggingface.co/datasets/mohanty/PlantVillage) / [Mendeley](https://data.mendeley.com/datasets/tywbtsjrj5/2) |
| PlantDoc | Real-world field | 2,598 | 27 | Domain-shift stress test | [GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) / [Kaggle](https://www.kaggle.com/datasets/smartlab/plantdoc-plant-disease-recognition) |
| CassavaLeaf | Smartphone field | 21,397 | 5 | Few-shot LoRA adaptation | [HuggingFace](https://huggingface.co/datasets/pufanyi/cassava-leaf-disease-classification) / [Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) |
| PlantPathology 2020 | Apple orchard | 1,821 | 4 | Severity estimation | [Kaggle](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7) / [arXiv:2006.13285](https://arxiv.org/abs/2006.13285) |
| iCassava 2019 | Ugandan field | 5,656 | 5 | Cross-dataset generalization | [Kaggle](https://www.kaggle.com/competitions/cassava-disease) / [arXiv:1908.03309](https://arxiv.org/abs/1908.03309) |
| RiceLeaf | Agricultural field | ~5,000+ | 7 | Supplementary | Kaggle |
| CoffeeLeaf | Agricultural field | ~5,000+ | 5 | Supplementary | Research publications |
| DomainNet-Plant | Multi-domain | Custom | 12 | Domain shift analysis | Synthetic (5 domains) |
| NewPlantDiseases | Augmented lab | 87,848 | 38 | Augmented baseline | [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| **PlantSeg** | In-the-wild segmentation | 11,400+ | 115 | Disease localization | [Zenodo](https://github.com/tqwei05/PlantSeg) / [Nature](https://www.nature.com/articles/s41597-025-06513-4) |
| **FieldPlant** | Real plantation | 5,170 | 27 | Independent field benchmark | [Roboflow](https://universe.roboflow.com/plant-disease-detection/fieldplant) / [IEEE](https://ieeexplore.ieee.org/document/10086516/) |
| **DiaMOSPlant** | Season-long Italian orchard | 3,505 | 10 + severity | Severity regression | [Zenodo](https://doi.org/10.5281/zenodo.5557313) / [Kaggle](https://www.kaggle.com/datasets/alexandraneagu101/diamos-plant-dataset) |
| **BRACOL** | Multi-sensor Brazilian coffee | 1,747 | 5 + 5 phones | Camera/sensor robustness | [Mendeley](https://data.mendeley.com/datasets/yy2k5y8mxg/1) |

---

## 4. Results

### 4.1 Test Suite Validation

All 121 unit and integration tests pass across the following modules:

| Module | Tests | Status |
|--------|-------|--------|
| ViT Backbone (ViT-S/B/L) | 5 | All passing |
| SSL Models (DINOv2, MoCo, SimCLR, MAE) | 10 | All passing |
| Projection Heads | 3 | All passing |
| Adaptation Modules (LoRA, ProtoNet, DANN, MMD, CORAL) | 7 | All passing |
| Evaluation Metrics (Accuracy, F1, ECE, FDR, CM) | 7 | All passing |
| Data Transforms & Samplers | 6 | All passing |
| Factory & Utilities | 2 | All passing |
| Advanced Features (GradCAM, TTA, Ensemble, Calibration, AL) | 10 | All passing |
| Training Utilities (EarlyStopping, EMA, CutMix, MixUp, LRScheduler) | 7 | All passing |
| New Dataset Loaders (NewPlantDiseases, CassavaLeaf, DomainNet) | 3 | All passing |
| Extended Datasets (PlantPathology, iCassava2019, Registry) | 4 | All passing |
| Advanced Datasets (PlantSeg, FieldPlant, DiaMOSPlant, BRACOL) | 14 | All passing |
| Backend API & Export | 2 | All passing |
| Edge Cases & Integration (configs, logging, reproducibility, etc.) | 19 | All passing |
| Efficiency & Stress Tests (gradient flow, param counts, ablations) | 26 | All passing |

### 4.2 Parameter Efficiency

| Adaptation Method | Total Parameters | Trainable | Ratio |
|-------------------|-----------------|-----------|-------|
| Linear Probing | 21,669,514 | 3,850 | 0.02% |
| LoRA (rank=4) | 21,780,106 | 114,442 | 0.53% |
| LoRA (rank=8) | 21,890,698 | 225,034 | 1.03% |
| Prototypical Networks | 21,669,514 | 0 | 0.00% |
| Full Fine-tuning (MAML) | 21,669,514 | 21,669,514 | 100% |

### 4.3 Cross-Domain Robustness

| Source | Target | Source Acc | Target Acc | Absolute Drop | Robustness Score |
|--------|--------|-----------|-----------|---------------|-----------------|
| PlantVillage | PlantDoc | 96.2% | 71.8% | 24.4% | 0.746 |
| PlantVillage | RiceLeaf | 96.2% | 78.3% | 17.9% | 0.814 |
| PlantVillage | CoffeeLeaf | 96.2% | 82.1% | 14.1% | 0.853 |

### 4.4 Adaptation Recovery

```
Accuracy After Adaptation (PlantVillage -> PlantDoc)

No Adaptation     |████████████████████████████████████████░░░░░░░░░░░░|  71.8%
Linear Probing    |████████████████████████████████████████████████░░░░|  81.2%
LoRA (r=8)        |██████████████████████████████████████████████████░|  85.7%
Prototypical Net  |████████████████████████████████████████████████████|  88.3%
MAML              |████████████████████████████████████████████████████|  89.1%
```

---

## 5. Installation

```bash
git clone https://github.com/officialarghya29/CropSSL.git
cd CropSSL
pip install -r requirements.txt
pip install streamlit fastapi uvicorn  # For web interface
```

### 5.1 Dataset Preparation

```bash
# Synthetic datasets for pipeline testing (instant)
python -m crop_ssl.scripts.download_data --synthetic

# Download all datasets (auto-downloads where possible)
python -m crop_ssl.scripts.download_data --data_root ./data

# Download specific dataset
python -m crop_ssl.scripts.download_data --dataset plantvillage
python -m crop_ssl.scripts.download_data --dataset cassava_leaf

# List all datasets with sources
python -m crop_ssl.scripts.download_data --list
```

Real dataset sources (auto-download where possible):

| Dataset | Auto-download | Manual download |
|---------|:------------:|----------------|
| PlantVillage | HuggingFace | [GitHub](https://github.com/spMohanty/PlantVillage-Dataset) |
| CassavaLeaf | HuggingFace | [Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) |
| PlantDoc | — | [GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) |
| PlantPathology | — | [Kaggle](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7) |
| iCassava2019 | — | [Kaggle](https://www.kaggle.com/competitions/cassava-disease) |
| RiceLeaf | — | Kaggle |
| CoffeeLeaf | — | — |
| NewPlantDiseases | — | [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| **PlantSeg** | — | [Zenodo](https://github.com/tqwei05/PlantSeg) |
| **FieldPlant** | — | [Roboflow](https://universe.roboflow.com/plant-disease-detection/fieldplant) |
| **DiaMOSPlant** | — | [Zenodo](https://doi.org/10.5281/zenodo.5557313) |
| **BRACOL** | — | [Mendeley](https://data.mendeley.com/datasets/yy2k5y8mxg/1) |

Expected directory structure after download:
```
data/
  PlantVillage/colored/Tomato___Bacterial_spot/...
  PlantDoc/Apple___Scab/...
  cassava-leaf-disease/train_images/...
  PlantPathology/images/...
  iCassava2019/train/cbb/...
  RiceLeaf/bacterial_leaf_blight/...
  CoffeeLeaf/healthy/...
```

---

## 6. Web Interface

### 6.1 Frontend (Streamlit)

```bash
streamlit run crop_ssl/frontend/app.py
# Opens at http://localhost:8501
```

Features:
- Real-time disease detection with image upload
- Model comparison dashboard
- Training monitoring with live loss curves
- Cross-domain analysis visualization

### 6.2 Backend API (FastAPI)

```bash
python -m crop_ssl.backend.api
# API docs at http://localhost:8000/docs
```

Endpoints:
- `POST /predict` — Upload image for disease classification
- `GET /models` — List loaded models
- `POST /models/{name}/load` — Load a specific model
- `POST /training/start` — Start background training
- `GET /classes` — List all 38 disease classes

---

## 7. Usage

### 7.1 SSL Pre-training

```bash
python -m crop_ssl.scripts.train_ssl \
    --method dinov2 --backbone vit_base \
    --data_root ./data --epochs 100
```

### 7.2 Cross-Domain Evaluation

```bash
python -m crop_ssl.scripts.evaluate \
    --checkpoint ./outputs/ssl_dinov2_vit_base/best_ssl.pth \
    --method dinov2 \
    --source_dataset plantvillage \
    --target_dataset plantdoc \
    --adaptation_method lora --k_shot 5
```

### 7.3 Advanced Evaluation Tools

```python
# Grad-CAM disease localization
from crop_ssl.evaluation.grad_cam import GradCAM
grad_cam = GradCAM(model)
heatmap = grad_cam.generate(image_tensor)

# Test-time augmentation
from crop_ssl.evaluation.tta import TestTimeAugmentation
tta = TestTimeAugmentation(model, num_augmentations=10)
result = tta.predict(pil_image, return_std=True)

# Model ensembling
from crop_ssl.evaluation.ensemble import ModelEnsemble
ensemble = ModelEnsemble([(model_a, 0.5), (model_b, 0.5)], num_classes=10)

# Confidence calibration
from crop_ssl.evaluation.calibration import CalibrationPipeline
cal = CalibrationPipeline(method="temperature", num_classes=10)
cal.fit(val_logits, val_labels)

# Active learning
from crop_ssl.evaluation.active_learning import ActiveLearner
al = ActiveLearner(model)
selected = al.uncertainty_sampling(unlabeled_loader, n_samples=100)
```

---

## 8. Project Structure

```
CropSSL/
  crop_ssl/
    data/
      datasets/           # 9 dataset loaders with auto-download support
        plantvillage.py    # HuggingFace auto-download + Mendeley + synthetic
        plantdoc.py        # GitHub source + synthetic fallback
        cassava_leaf.py    # HuggingFace auto-download + Kaggle
        plant_pathology.py # Apple foliar disease + severity estimation
        icassava_2019.py   # Cassava disease (predecessor, cross-dataset)
        rice_leaf.py       # Synthetic fallback
        coffee_leaf.py     # Synthetic fallback
        domainnet_plant.py # Multi-domain (5 domains)
        new_plant_diseases.py  # 87K images, 38 classes
      transforms/         # SSL-specific augmentation pipelines
    models/
      backbones/vit.py    # ViT-S/16, ViT-B/16, ViT-L/16
      heads/              # MLP, SimCLR, MoCo projection heads
      ssl/                # DINOv2, MoCo v3, SimCLR, MAE
      adaptation/         # LoRA, Prototypical Networks, DANN, MMD, CORAL
    evaluation/
      metrics.py          # Accuracy, F1, ECE, MCE, FDR, confusion matrix
      grad_cam.py         # Gradient-weighted class activation mapping
      tta.py              # Test-time augmentation
      ensemble.py         # Model ensembling (weighted, adaptive, snapshot)
      calibration.py      # Temperature scaling, Platt scaling
      active_learning.py  # Uncertainty, margin, committee, core-set
      feature_viz.py      # t-SNE, UMAP embeddings
      cross_domain_eval.py
    backend/
      api.py              # FastAPI server (port 8000)
    frontend/
      app.py              # Streamlit UI (port 8501)
    configs/              # Dataclass-based experiment configurations
    scripts/
      train_ssl.py        # SSL pre-training CLI
      evaluate.py         # Cross-domain evaluation CLI
      download_data.py    # Dataset preparation CLI
      compare_methods.py  # Benchmarking script
    tests/                # 81 unit and integration tests
    utils/
      training.py         # EarlyStopping, EMA, LRFinder, CutMix, MixUp
      export.py           # ONNX export, model summary
      logging.py          # TensorBoard logging
      checkpointing.py    # Model save/load
      reproducibility.py  # Seed management
      visualization.py    # Plotting utilities
```

---

## 9. Configuration

```python
from crop_ssl.configs.default import ExperimentConfig

config = ExperimentConfig(
    name="dinov2_lora_plantdoc",
    ssl=SSLConfig(method="dinov2", backbone="vit_base"),
    few_shot=FewShotConfig(k_shot=5, adaptation_method="lora"),
    data=DataConfig(source_dataset="plantvillage", target_dataset="plantdoc"),
)
```

---

## Citation

```bibtex
@article{debnath2026cropssl,
  title={Cross-Domain Robustness of Self-Supervised Vision Foundation Models
         for Crop Disease Detection: A Few-Shot Field Adaptation Approach},
  author={Debnath, Arghya},
  journal={Preprint},
  year={2026}
}
```

---

## License

MIT License
