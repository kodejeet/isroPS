"""Unit tests for geometry, homography, RANSAC, and warping."""

import numpy as np

from lunar_correspondence.geometry.homography import compute_reprojection_errors
from lunar_correspondence.geometry.ransac import estimate_geometric_model
from lunar_correspondence.geometry.transforms import warp_image
from lunar_correspondence.io.metadata import GeometricModel, MatchSet


def test_homography_reprojection_error_zero_for_identity():
    pts_src = np.array([[10, 10], [50, 20], [30, 80], [90, 90]], dtype=np.float32)
    pts_ref = pts_src.copy()
    H_eye = np.eye(3, dtype=np.float32)

    errors = compute_reprojection_errors(pts_src, pts_ref, H_eye)
    assert len(errors) == 4
    np.testing.assert_allclose(errors, 0.0, atol=1e-5)


def test_ransac_estimation_on_known_affine():
    # Construct exact translation + scaling transform: x' = 1.1*x + 10, y' = 1.1*y - 5
    pts_src = np.array(
        [[20, 30], [100, 50], [40, 150], [200, 200], [80, 120]], dtype=np.float32
    )
    pts_ref = pts_src * 1.1 + np.array([10.0, -5.0], dtype=np.float32)

    match_set = MatchSet(source_points=pts_src, reference_points=pts_ref)
    model = estimate_geometric_model(
        match_set, model_type="homography", reproj_threshold=2.0
    )

    assert isinstance(model, GeometricModel)
    assert model.inlier_mask.sum() == 5
    assert np.mean(model.reprojection_errors) < 1.0


def test_warp_image():
    img_src = np.zeros((100, 100, 1), dtype=np.uint8)
    img_src[40:60, 40:60] = 255
    H_eye = np.eye(3, dtype=np.float32)

    geo_model = GeometricModel(
        transform_matrix=H_eye,
        model_type="homography",
        inlier_mask=np.array([True]),
        reprojection_errors=np.array([0.0]),
    )

    warped = warp_image(img_src, geo_model, output_shape=(100, 100))
    assert warped.shape == (100, 100, 1)
    assert np.array_equal(warped, img_src)


def test_ransac_self_match_all_inliers_zero_error():
    """RANSAC on identical points should find identity transform with 100% inliers and 0 error."""
    pts_src = np.array(
        [[10, 10], [50, 20], [30, 80], [90, 90], [120, 150], [200, 210]],
        dtype=np.float32,
    )
    pts_ref = pts_src.copy()

    match_set = MatchSet(source_points=pts_src, reference_points=pts_ref)
    model = estimate_geometric_model(
        match_set, model_type="homography", reproj_threshold=2.0
    )

    assert model.inlier_mask.sum() == 6
    np.testing.assert_allclose(model.reprojection_errors, 0.0, atol=1e-5)
    np.testing.assert_allclose(model.transform_matrix, np.eye(3), atol=1e-4)


def test_ransac_insufficient_points():
    """RANSAC with fewer than 4 points returns identity with 0 inliers."""
    pts_src = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    pts_ref = pts_src.copy()

    match_set = MatchSet(source_points=pts_src, reference_points=pts_ref)
    model = estimate_geometric_model(match_set, model_type="homography")

    assert model.inlier_mask.sum() == 0
    assert len(model.reprojection_errors) == 3
