"""Benchmark aggregation utilities across dataset collections."""

from typing import Any

import numpy as np

from lunar_correspondence.io.metadata import EvaluationResult


def summarize_benchmark_results(results: list[EvaluationResult]) -> dict[str, Any]:
    """Summarize evaluation results across multiple image pair runs.

    Args:
        results: List of EvaluationResult instances.

    Returns:
        Summary metrics dictionary (mean RMSE, mean inlier ratio, mean coverage).
    """
    if not results:
        return {}

    rmses = [r.rmse_pixels for r in results if r.rmse_pixels is not None]
    inlier_ratios = [r.inlier_ratio for r in results]
    coverages = [r.coverage for r in results]

    return {
        "num_pairs_evaluated": len(results),
        "mean_rmse_pixels": float(np.mean(rmses)) if rmses else None,
        "mean_inlier_ratio": float(np.mean(inlier_ratios)),
        "mean_coverage_pct": float(np.mean(coverages)),
    }
