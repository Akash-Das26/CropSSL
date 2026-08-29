"""Evaluation modules for CropSSL."""

from crop_ssl.evaluation.metrics import EvaluationSuite, compute_accuracy, compute_domain_shift_metrics
from crop_ssl.evaluation.cross_domain_eval import CrossDomainEvaluator

__all__ = [
    "EvaluationSuite",
    "compute_accuracy",
    "compute_domain_shift_metrics",
    "CrossDomainEvaluator",
]
