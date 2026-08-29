"""Evaluation modules for CropSSL."""

from crop_ssl.evaluation.metrics import EvaluationSuite, compute_accuracy, compute_domain_shift_metrics
from crop_ssl.evaluation.cross_domain_eval import CrossDomainEvaluator
from crop_ssl.evaluation.grad_cam import GradCAM
from crop_ssl.evaluation.tta import TestTimeAugmentation
from crop_ssl.evaluation.ensemble import ModelEnsemble, SnapshotEnsemble, AdaptiveEnsemble
from crop_ssl.evaluation.calibration import TemperatureScaling, PlattScaling, CalibrationPipeline
from crop_ssl.evaluation.active_learning import ActiveLearner

__all__ = [
    "EvaluationSuite",
    "compute_accuracy",
    "compute_domain_shift_metrics",
    "CrossDomainEvaluator",
    "GradCAM",
    "TestTimeAugmentation",
    "ModelEnsemble",
    "SnapshotEnsemble",
    "AdaptiveEnsemble",
    "TemperatureScaling",
    "PlattScaling",
    "CalibrationPipeline",
    "ActiveLearner",
]
