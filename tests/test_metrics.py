"""Unit tests for grid coverage and evaluation metrics calculation."""

import numpy as np

from lunar_correspondence.evaluation.metrics import (
    compute_grid_coverage,
    evaluate_registration,
)
from lunar_correspondence.io.metadata import EvaluationResult, GeometricModel, MatchSet


def test_grid_coverage_all_cells():
    # 4x4 grid over 100x100 canvas (cells are 25x25)
    # Generate 1 point inside each of the 16 grid cells
    pts = []
    for r in range(4):
        for c in range(4):
            pts.append([c * 25 + 12.5, r * 25 + 12.5])
    pts_arr = np.array(pts, dtype=np.float32)

    coverage = compute_grid_coverage(
        pts_arr, image_shape=(100, 100), grid_rows=4, grid_cols=4
    )
    assert coverage == 100.0


def test_grid_coverage_single_cell():
    # Points clustered in top-left cell only
    pts_arr = np.array([[5, 5], [10, 10], [15, 15]], dtype=np.float32)
    coverage = compute_grid_coverage(
        pts_arr, image_shape=(100, 100), grid_rows=4, grid_cols=4
    )
    # 1 out of 16 cells occupied = 6.25%
    assert coverage == 6.25


def test_evaluate_registration_summary():
    pts_src = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    pts_ref = pts_src.copy()
    inliers = np.array([True, True, True, False])
    errors = np.array([0.2, 0.4, 0.6, 10.0], dtype=np.float32)

    match_set = MatchSet(
        source_points=pts_src, reference_points=pts_ref, inlier_mask=inliers
    )
    geo_model = GeometricModel(
        transform_matrix=np.eye(3),
        model_type="homography",
        inlier_mask=inliers,
        reprojection_errors=errors,
    )

    eval_res = evaluate_registration(match_set, geo_model, reference_shape=(100, 100))

    assert isinstance(eval_res, EvaluationResult)
    assert eval_res.total_matches == 4
    assert eval_res.inlier_matches == 3
    assert eval_res.inlier_ratio == 0.75
    assert eval_res.rmse_pixels is not None
    assert eval_res.rmse_pixels < 1.0
    assert eval_res.coverage > 0.0


def test_evaluate_registration_self_match():
    """Self-match (identical coordinates) must yield RMSE ≈ 0.0, Median ≈ 0.0 (not None)."""
    pts_src = np.array(
        [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float32
    )
    pts_ref = pts_src.copy()
    inliers = np.ones(5, dtype=bool)
    errors = np.zeros(5, dtype=np.float32)

    match_set = MatchSet(
        source_points=pts_src, reference_points=pts_ref, inlier_mask=inliers
    )
    geo_model = GeometricModel(
        transform_matrix=np.eye(3),
        model_type="homography",
        inlier_mask=inliers,
        reprojection_errors=errors,
    )

    eval_res = evaluate_registration(match_set, geo_model, reference_shape=(100, 100))

    assert eval_res.total_matches == 5
    assert eval_res.inlier_matches == 5
    assert eval_res.inlier_ratio == 1.0
    assert eval_res.rmse_pixels is not None
    assert eval_res.median_error_pixels is not None
    np.testing.assert_allclose(eval_res.rmse_pixels, 0.0, atol=1e-6)
    np.testing.assert_allclose(eval_res.median_error_pixels, 0.0, atol=1e-6)


def test_evaluate_registration_with_refinement_fields():
    """Verify pre_refinement_rmse_pixels and post_refinement_rmse_pixels are populated."""
    pts_src = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    pts_ref = pts_src.copy()
    inliers = np.array([True, True, True, True])
    errors = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    match_set = MatchSet(
        source_points=pts_src, reference_points=pts_ref, inlier_mask=inliers
    )
    geo_model = GeometricModel(
        transform_matrix=np.eye(3),
        model_type="homography",
        inlier_mask=inliers,
        reprojection_errors=errors,
    )

    eval_res = evaluate_registration(
        match_set,
        geo_model,
        reference_shape=(100, 100),
        pre_refinement_rmse_pixels=0.55,
        post_refinement_rmse_pixels=0.48,
    )

    assert eval_res.pre_refinement_rmse_pixels == 0.55
    assert eval_res.post_refinement_rmse_pixels == 0.48
