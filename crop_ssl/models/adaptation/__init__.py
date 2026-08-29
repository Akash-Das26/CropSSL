"""Adaptation modules for cross-domain robustness."""

from crop_ssl.models.adaptation.few_shot_adapter import (
    FewShotAdapter,
    LoRAAdapter,
    PrototypicalNetwork,
)
from crop_ssl.models.adaptation.domain_adapter import (
    DomainAdaptationModule,
    MMDLoss,
    CORALLoss,
)

__all__ = [
    "FewShotAdapter",
    "LoRAAdapter",
    "PrototypicalNetwork",
    "DomainAdaptationModule",
    "MMDLoss",
    "CORALLoss",
]
