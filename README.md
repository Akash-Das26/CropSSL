# 🌿 CropSSL

## Cross-Domain Robustness of Self-Supervised Vision Foundation Models for Crop Disease Detection: A Few-Shot Field Adaptation Approach

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv">
</p>

---

## 📋 Overview

CropSSL is a comprehensive research framework for evaluating and improving the **cross-domain robustness** of self-supervised vision foundation models for **agricultural plant disease detection**. It addresses the critical challenge of domain shift when models trained in controlled lab environments are deployed in real-world field conditions.

### Key Contributions

1. **Systematic Evaluation** of 4 SSL methods (DINOv2, MoCo v3, SimCLR, MAE) across 5 crop disease datasets with different domain characteristics
2. **Few-Shot Field Adaptation** strategies (LoRA, Prototypical Networks, MAML, Linear Probing) for efficient deployment with minimal labeled field data
3. **Cross-Domain Robustness Metrics** including accuracy drop analysis, calibration error, and Fisher Discriminant Ratio
4. **Domain Adaptation Techniques** (DANN, MMD, CORAL) for improving field robustness

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CropSSL Framework                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   SSL Pre-   │    │  Few-Shot    │    │   Cross-     │  │
│  │   training   │───▶│  Adaptation  │───▶│   Domain     │  │
│  │   Module     │    │  Module      │    │   Eval       │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│        │                   │                   │            │
│   ┌────┴────┐        ┌────┴────┐        ┌────┴────┐       │
│   │ DINOv2  │        │  LoRA   │        │Accuracy │       │
│   │ MoCo v3 │        │  MAML   │        │  F1     │       │
│   │ SimCLR  │        │  Proto  │        │  ECE    │       │
│   │  MAE    │        │ Linear  │        │  FDR    │       │
│   └─────────┘        └─────────┘        └─────────┘       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Domain Adaptation Layer                  │  │
│  │         DANN  |  MMD  |  CORAL  |  Combined         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Datasets: PlantVillage → PlantDoc → RiceLeaf → CoffeeLeaf │
│            (Lab)  (Field)  (Field)    (Field)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Datasets

| Dataset | Domain | Images | Classes | Characteristics |
|---------|--------|--------|---------|-----------------|
| **PlantVillage** | Lab/Studio | 54,309 | 38 | Controlled lighting, clean backgrounds |
| **PlantDoc** | Real-world | 2,598 | 27 | Uncontrolled environments, varying quality |
| **RiceLeaf** | Field | ~5,000+ | 7 | Natural weather conditions |
| **CoffeeLeaf** | Field | ~5,000+ | 5 | Varying geographic conditions |
| **DomainNet-Plant** | Multi-domain | Custom | 12 | 5 domains (studio, greenhouse, field, mobile, aerial) |

### Domain Shift Analysis

```
Source Domain (Lab)              Target Domain (Field)
┌─────────────────┐              ┌─────────────────┐
│  PlantVillage   │   Domain     │   PlantDoc      │
│  - Clean BG     │   Shift      │   - Noisy BG    │
│  - Even light   │  ────────▶   │   - Var. light  │
│  - High res     │              │   - Low res     │
│  - One crop     │              │   - Multi-crop  │
└─────────────────┘              └─────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/officialarghya29/CropSSL.git
cd CropSSL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 1. SSL Pre-training

```bash
# Pre-train with DINOv2 on PlantVillage
python -m crop_ssl.scripts.train_ssl \
    --method dinov2 \
    --backbone vit_base \
    --data_root ./data \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4

# Pre-train with MoCo v3
python -m crop_ssl.scripts.train_ssl \
    --method moco_v3 \
    --backbone vit_base \
    --data_root ./data \
    --epochs 100
```

### 2. Cross-Domain Evaluation

```bash
# Evaluate with LoRA adaptation
python -m crop_ssl.scripts.evaluate \
    --checkpoint ./outputs/ssl_dinov2_vit_base/best_ssl.pth \
    --method dinov2 \
    --source_dataset plantvillage \
    --target_dataset plantdoc \
    --adaptation_method lora \
    --k_shot 5

# Evaluate with Prototypical Networks
python -m crop_ssl.scripts.evaluate \
    --checkpoint ./outputs/ssl_dinov2_vit_base/best_ssl.pth \
    --method dinov2 \
    --source_dataset plantvillage \
    --target_dataset rice_leaf \
    --adaptation_method prototypical \
    --k_shot 5
```

### 3. Few-Shot Evaluation

```bash
# 5-way 1-shot evaluation
python -m crop_ssl.scripts.evaluate \
    --checkpoint ./outputs/ssl_dinov2_vit_base/best_ssl.pth \
    --method dinov2 \
    --adaptation_method prototypical \
    --k_shot 1
```

---

## 📁 Project Structure

```
CropSSL/
├── crop_ssl/
│   ├── __init__.py
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── plantvillage.py        # PlantVillage loader
│   │   │   ├── plantdoc.py            # PlantDoc loader
│   │   │   ├── rice_leaf.py           # Rice leaf loader
│   │   │   ├── coffee_leaf.py         # Coffee leaf loader
│   │   │   ├── domainnet_plant.py     # Multi-domain dataset
│   │   │   ├── cross_domain_dataset.py # Cross-domain wrapper
│   │   │   └── few_shot_sampler.py    # Episodic & balanced samplers
│   │   └── transforms/
│   │       └── augmentations.py       # SSL-specific augmentations
│   ├── models/
│   │   ├── backbones/
│   │   │   └── vit.py                 # Vision Transformer (ViT-S/B/L)
│   │   ├── heads/
│   │   │   └── projection.py          # Projection heads for SSL
│   │   ├── ssl/
│   │   │   ├── dino_v2.py             # DINOv2 implementation
│   │   │   ├── moco_v3.py             # MoCo v3 implementation
│   │   │   ├── simclr.py              # SimCLR implementation
│   │   │   └── mae.py                 # MAE implementation
│   │   └── adaptation/
│   │       ├── few_shot_adapter.py    # LoRA, MAML, ProtoNet
│   │       └── domain_adapter.py      # DANN, MMD, CORAL
│   ├── evaluation/
│   │   ├── metrics.py                 # Accuracy, F1, ECE, FDR
│   │   └── cross_domain_eval.py       # Multi-domain evaluation
│   ├── configs/
│   │   └── default.py                 # Experiment configurations
│   ├── scripts/
│   │   ├── train_ssl.py               # SSL pre-training script
│   │   └── evaluate.py                # Cross-domain evaluation
│   └── utils/
│       ├── logging.py                 # TensorBoard logging
│       ├── checkpointing.py           # Model save/load
│       ├── reproducibility.py         # Seed management
│       └── visualization.py           # Plotting & analysis
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Methods

### Self-Supervised Pre-training

| Method | Type | Key Idea | Loss Function |
|--------|------|----------|---------------|
| **DINOv2** | Self-distillation | Student-teacher with EMA + multi-crop | Cross-entropy (sharpened softmax) |
| **MoCo v3** | Contrastive | Momentum queue + asymmetric learning | InfoNCE |
| **SimCLR** | Contrastive | Two-view augmented contrastive pairs | NT-Xent |
| **MAE** | Generative | 75% masked patches → reconstruction | MSE on patches |

### Few-Shot Adaptation

| Method | Approach | Trainable Params | Notes |
|--------|----------|-----------------|-------|
| **Linear Probe** | Freeze backbone + linear head | ~0.3M | Baseline |
| **LoRA** | Low-rank adaptation of attention | ~1-5M | Efficient fine-tuning |
| **MAML** | Meta-learning inner loop | All params | Fast adaptation |
| **Prototypical Nets** | Distance-based classification | 0 (inference only) | No fine-tuning needed |

### Domain Adaptation

| Method | Mechanism | Loss Component |
|--------|-----------|----------------|
| **DANN** | Gradient reversal + domain discriminator | Adversarial CE |
| **MMD** | Maximum Mean Discrepancy alignment | Kernel MMD |
| **CORAL** | Second-order statistics alignment | Covariance distance |

---

## 📈 Evaluation Metrics

- **Top-1/3/5 Accuracy**: Standard classification accuracy
- **Macro Precision/Recall/F1**: Balanced class-wise metrics
- **ECE (Expected Calibration Error)**: Prediction confidence calibration
- **Accuracy Drop**: Absolute and relative performance decrease across domains
- **Robustness Score**: Target accuracy / Source accuracy (0-1)
- **Fisher Discriminant Ratio**: Feature class separability measure

---

## 📜 Citation

If you find this work useful, please cite:

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

## 🛠️ Configuration

All experiments are configured via dataclasses in `crop_ssl/configs/default.py`:

```python
from crop_ssl.configs.default import ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    ssl=SSLConfig(method="dinov2", backbone="vit_base"),
    few_shot=FewShotConfig(k_shot=5, adaptation_method="lora"),
    data=DataConfig(source_dataset="plantvillage", target_dataset="plantdoc"),
)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- [PlantVillage Dataset](https://plantvillage.psu.edu/)
- [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- Vision Transformer (ViT) by Dosovitskiy et al.
- PyTorch team for the excellent framework
