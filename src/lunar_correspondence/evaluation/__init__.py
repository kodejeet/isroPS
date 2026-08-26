"""Evaluation metrics and benchmarking package."""

from lunar_correspondence.evaluation.benchmark import summarize_benchmark_results
from lunar_correspondence.evaluation.ground_truth import compute_ground_truth_error
from lunar_correspondence.evaluation.metrics import (
    compute_grid_coverage,
    evaluate_registration,
)

__all__ = [
    "compute_grid_coverage",
    "compute_ground_truth_error",
    "evaluate_registration",
    "summarize_benchmark_results",
]
