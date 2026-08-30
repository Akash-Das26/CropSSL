"""
Default Configuration for CropSSL experiments.

Provides structured configuration for:
- SSL pre-training
- Few-shot adaptation
- Cross-domain evaluation
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data configuration."""
    source_dataset: str = "plantvillage"
    target_dataset: str = "plantdoc"
    source_root: str = "./data"
    target_root: str = "./data"
    image_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class SSLConfig:
    """Self-supervised learning configuration."""
    method: str = "dinov2"  # dinov2, moco_v3, simclr, mae
    backbone: str = "vit_base"  # vit_small, vit_base, vit_large
    embed_dim: int = 768
    patch_size: int = 16
    out_dim: int = 65536
    temperature: float = 0.07
    momentum_teacher: float = 0.996
    teacher_temp: float = 0.04
    student_temp: float = 0.1
    mask_ratio: float = 0.75  # For MAE
    local_crops_number: int = 8  # For DINO


@dataclass
class FewShotConfig:
    """Few-shot adaptation configuration."""
    k_shot: int = 5
    n_way: int = 10
    q_query: int = 15
    num_episodes: int = 100
    adaptation_method: str = "lora"  # linear, lora, maml, prototypical
    lora_rank: int = 8
    lora_alpha: float = 1.0


@dataclass
class DomainAdaptationConfig:
    """Domain adaptation configuration."""
    method: str = "none"  # none, dann, mmd, coral, combined
    domain_weight: float = 1.0
    adversarial_alpha: float = 1.0


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.04
    warmup_epochs: int = 10
    total_epochs: int = 100
    min_lr: float = 1e-6
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int = 128
    eval_frequency: int = 5
    save_best: bool = True
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "f1", "ece", "robustness"]
    )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "cropssl_dinov2_base"
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"

    data: DataConfig = field(default_factory=DataConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    domain_adaptation: DomainAdaptationConfig = field(
        default_factory=DomainAdaptationConfig
    )
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        """Convert to nested dictionary."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create from nested dictionary."""
        data_cfg = DataConfig(**d.get("data", {}))
        ssl_cfg = SSLConfig(**d.get("ssl", {}))
        fs_cfg = FewShotConfig(**d.get("few_shot", {}))
        da_cfg = DomainAdaptationConfig(**d.get("domain_adaptation", {}))
        train_cfg = TrainConfig(**d.get("train", {}))
        eval_cfg = EvalConfig(**d.get("eval", d.get("evaluation", {})))

        return cls(
            name=d.get("name", "cropssl"),
            seed=d.get("seed", 42),
            device=d.get("device", "cuda"),
            output_dir=d.get("output_dir", "./outputs"),
            log_dir=d.get("log_dir", "./logs"),
            data=data_cfg,
            ssl=ssl_cfg,
            few_shot=fs_cfg,
            domain_adaptation=da_cfg,
            train=train_cfg,
            evaluation=eval_cfg,
        )


# Pre-defined experiment configurations
DINOV2_PLANTVILLAGE_TO_PLANTDOC = ExperimentConfig(
    name="dinov2_plantvillage_to_plantdoc",
    ssl=SSLConfig(method="dinov2", backbone="vit_base"),
    data=DataConfig(
        source_dataset="plantvillage",
        target_dataset="plantdoc",
    ),
)

MOCO_PLANTVILLAGE_TO_PLANTDOC = ExperimentConfig(
    name="moco_v3_plantvillage_to_plantdoc",
    ssl=SSLConfig(method="moco_v3", backbone="vit_base"),
    data=DataConfig(
        source_dataset="plantvillage",
        target_dataset="plantdoc",
    ),
)

SIMCLR_PLANTVILLAGE_TO_PLANTDOC = ExperimentConfig(
    name="simclr_plantvillage_to_plantdoc",
    ssl=SSLConfig(method="simclr", backbone="vit_base"),
    data=DataConfig(
        source_dataset="plantvillage",
        target_dataset="plantdoc",
    ),
)

MAE_PLANTVILLAGE_TO_PLANTDOC = ExperimentConfig(
    name="mae_plantvillage_to_plantdoc",
    ssl=SSLConfig(method="mae", backbone="vit_base"),
    data=DataConfig(
        source_dataset="plantvillage",
        target_dataset="plantdoc",
    ),
)

FEW_SHOT_5WAY_5SHOT = ExperimentConfig(
    name="few_shot_5way_5shot",
    few_shot=FewShotConfig(n_way=5, k_shot=5),
)

FEW_SHOT_5WAY_1SHOT = ExperimentConfig(
    name="few_shot_5way_1shot",
    few_shot=FewShotConfig(n_way=5, k_shot=1),
)
