"""Quantitative evaluation metrics computation module.

Calculates Total Matches, Inlier Matches, Inlier Ratio, RMSE, Median Error,
and Grid-Cell Spatial Coverage Percentage as expected by ISRO PS standards.
"""

import numpy as np

from lunar_correspondence.io.metadata import EvaluationResult, GeometricModel, MatchSet


def compute_grid_coverage(
    inlier_points_xy: np.ndarray,
    image_shape: tuple[int, int],
    grid_rows: int = 4,
    grid_cols: int = 4,
) -> float:
    """Compute grid-cell spatial coverage percentage across image canvas.

    Divides image canvas (height, width) into grid_rows x grid_cols equal cells,
    and calculates what percentage of cells contain at least one inlier match point.

    Args:
        inlier_points_xy: (N, 2) array of (x, y) point coordinates.
        image_shape: (height, width) tuple of canvas.
        grid_rows: Number of grid rows.
        grid_cols: Number of grid columns.

    Returns:
        Coverage percentage float in range [0.0, 100.0].
    """
    if len(inlier_points_xy) == 0:
        return 0.0

    h, w = image_shape
    total_cells = grid_rows * grid_cols
    occupied_cells = set()

    cell_h = max(1.0, h / float(grid_rows))
    cell_w = max(1.0, w / float(grid_cols))

    for pt in inlier_points_xy:
        x, y = pt[0], pt[1]
        col = int(np.clip(x / cell_w, 0, grid_cols - 1))
        row = int(np.clip(y / cell_h, 0, grid_rows - 1))
        occupied_cells.add((row, col))

    coverage_pct = (len(occupied_cells) / float(total_cells)) * 100.0
    return float(coverage_pct)


def evaluate_registration(
    match_set: MatchSet,
    geometric_model: GeometricModel,
    reference_shape: tuple[int, int],
    grid_rows: int = 4,
    grid_cols: int = 4,
    processing_time_seconds: float = 0.0,
    random_seed: int = 42,
) -> EvaluationResult:
    """Evaluate registration quality and compute ISRO PS metric deliverables.

    Args:
        match_set: MatchSet containing raw matches and point coordinates.
        geometric_model: Estimated GeometricModel containing inlier mask and reprojection errors.
        reference_shape: (height, width) of reference image canvas.
        grid_rows: Number of grid rows for coverage evaluation.
        grid_cols: Number of grid columns for coverage evaluation.
        processing_time_seconds: Elapsed execution time.
        random_seed: Random seed used during pipeline run.

    Returns:
        EvaluationResult populated with quantitative metrics.
    """
    total_matches = len(match_set.source_points)
    inliers_mask = geometric_model.inlier_mask
    inlier_matches = int(np.sum(inliers_mask)) if inliers_mask is not None else 0

    inlier_ratio = (
        (float(inlier_matches) / float(total_matches)) if total_matches > 0 else 0.0
    )

    if inlier_matches > 0 and geometric_model.reprojection_errors is not None:
        inlier_errors = geometric_model.reprojection_errors[inliers_mask]
        rmse = (
            float(np.sqrt(np.mean(inlier_errors**2)))
            if len(inlier_errors) > 0
            else None
        )
        median_err = float(np.median(inlier_errors)) if len(inlier_errors) > 0 else None
        inlier_pts_ref = match_set.reference_points[inliers_mask]
    else:
        rmse = None
        median_err = None
        inlier_pts_ref = np.zeros((0, 2), dtype=np.float32)

    coverage = compute_grid_coverage(
        inlier_points_xy=inlier_pts_ref,
        image_shape=reference_shape,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )

    spatial_uniformity = coverage / 100.0  # Normalized 0..1 uniformity score

    return EvaluationResult(
        total_matches=total_matches,
        inlier_matches=inlier_matches,
        inlier_ratio=inlier_ratio,
        rmse_pixels=rmse,
        median_error_pixels=median_err,
        coverage=coverage,
        spatial_uniformity=spatial_uniformity,
        processing_time_seconds=processing_time_seconds,
        random_seed=random_seed,
    )
