"""Regression tests for frozen core data contracts.

These tests enforce that the field names, types, array shapes, and semantics
of FeatureSet, MatchSet, GeometricModel, and EvaluationResult remain stable.
Any accidental change to these structures will be caught here.

CRITICAL: Do NOT weaken or remove these tests.
"""

from dataclasses import fields

import numpy as np

from lunar_correspondence.io.metadata import (
    EvaluationResult,
    FeatureSet,
    GeometricModel,
    ImageData,
    ImageMetadata,
    MatchSet,
)


class TestFeatureSetContract:
    """FeatureSet must have keypoints (N,2) in (x,y), descriptors, scores, method."""

    def test_required_fields_exist(self):
        field_names = {f.name for f in fields(FeatureSet)}
        assert "keypoints" in field_names
        assert "descriptors" in field_names
        assert "scores" in field_names
        assert "method" in field_names

    def test_keypoints_shape_n2(self):
        kps = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        fs = FeatureSet(keypoints=kps, method="test")
        assert fs.keypoints.shape == (2, 2)
        assert fs.keypoints.shape[1] == 2, "keypoints must be (N, 2)"

    def test_empty_keypoints_shape(self):
        kps = np.zeros((0, 2), dtype=np.float32)
        fs = FeatureSet(keypoints=kps, method="test")
        assert fs.keypoints.shape == (0, 2)

    def test_descriptors_optional(self):
        kps = np.zeros((3, 2), dtype=np.float32)
        fs = FeatureSet(keypoints=kps, method="test")
        assert fs.descriptors is None

    def test_scores_optional(self):
        kps = np.zeros((3, 2), dtype=np.float32)
        fs = FeatureSet(keypoints=kps, method="test")
        assert fs.scores is None

    def test_method_default(self):
        kps = np.zeros((3, 2), dtype=np.float32)
        fs = FeatureSet(keypoints=kps)
        assert fs.method == "unknown"


class TestMatchSetContract:
    """MatchSet must have source_points and reference_points (N,2) in (x,y)."""

    def test_required_fields_exist(self):
        field_names = {f.name for f in fields(MatchSet)}
        assert "source_points" in field_names
        assert "reference_points" in field_names
        assert "confidence" in field_names
        assert "inlier_mask" in field_names

    def test_points_shape_n2(self):
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ref = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        ms = MatchSet(source_points=src, reference_points=ref)
        assert ms.source_points.shape == (2, 2)
        assert ms.reference_points.shape == (2, 2)
        assert ms.source_points.shape[1] == 2, "source_points must be (N, 2)"
        assert ms.reference_points.shape[1] == 2, "reference_points must be (N, 2)"

    def test_empty_points_shape(self):
        src = np.zeros((0, 2), dtype=np.float32)
        ref = np.zeros((0, 2), dtype=np.float32)
        ms = MatchSet(source_points=src, reference_points=ref)
        assert ms.source_points.shape == (0, 2)
        assert ms.reference_points.shape == (0, 2)

    def test_confidence_optional(self):
        src = np.zeros((3, 2), dtype=np.float32)
        ref = np.zeros((3, 2), dtype=np.float32)
        ms = MatchSet(source_points=src, reference_points=ref)
        assert ms.confidence is None

    def test_inlier_mask_optional(self):
        src = np.zeros((3, 2), dtype=np.float32)
        ref = np.zeros((3, 2), dtype=np.float32)
        ms = MatchSet(source_points=src, reference_points=ref)
        assert ms.inlier_mask is None


class TestGeometricModelContract:
    """GeometricModel must have transform_matrix, model_type, inlier_mask, reprojection_errors."""

    def test_required_fields_exist(self):
        field_names = {f.name for f in fields(GeometricModel)}
        assert "transform_matrix" in field_names
        assert "model_type" in field_names
        assert "inlier_mask" in field_names
        assert "reprojection_errors" in field_names

    def test_construction_with_valid_data(self):
        H = np.eye(3, dtype=np.float32)
        mask = np.array([True, False, True], dtype=bool)
        errors = np.array([0.1, 5.0, 0.2], dtype=np.float32)
        gm = GeometricModel(
            transform_matrix=H,
            model_type="homography",
            inlier_mask=mask,
            reprojection_errors=errors,
        )
        assert gm.transform_matrix.shape == (3, 3)
        assert gm.model_type == "homography"
        assert gm.inlier_mask.dtype == bool
        assert gm.reprojection_errors.shape == (3,)

    def test_affine_model_type(self):
        H = np.eye(3, dtype=np.float32)
        gm = GeometricModel(
            transform_matrix=H,
            model_type="affine",
            inlier_mask=np.array([True]),
            reprojection_errors=np.array([0.0]),
        )
        assert gm.model_type == "affine"


class TestEvaluationResultContract:
    """EvaluationResult must have all expected fields including scale_factor."""

    def test_required_fields_exist(self):
        field_names = {f.name for f in fields(EvaluationResult)}
        expected = {
            "total_matches",
            "inlier_matches",
            "inlier_ratio",
            "rmse_pixels",
            "median_error_pixels",
            "coverage",
            "spatial_uniformity",
            "processing_time_seconds",
            "random_seed",
            "scale_factor",
        }
        assert expected.issubset(field_names), f"Missing fields: {expected - field_names}"

    def test_scale_factor_default(self):
        er = EvaluationResult(
            total_matches=10,
            inlier_matches=8,
            inlier_ratio=0.8,
            rmse_pixels=1.0,
            median_error_pixels=0.9,
            coverage=50.0,
            spatial_uniformity=0.5,
            processing_time_seconds=1.0,
            random_seed=42,
        )
        assert er.scale_factor == 1.0

    def test_scale_factor_mutable(self):
        """scale_factor can be set after construction (used by run_registration)."""
        er = EvaluationResult(
            total_matches=10,
            inlier_matches=8,
            inlier_ratio=0.8,
            rmse_pixels=1.0,
            median_error_pixels=0.9,
            coverage=50.0,
            spatial_uniformity=0.5,
            processing_time_seconds=1.0,
            random_seed=42,
        )
        er.scale_factor = 0.5
        assert er.scale_factor == 0.5


class TestImageDataContract:
    """ImageData.array must always be (H, W, C)."""

    def test_hwc_shape_properties(self):
        arr = np.zeros((100, 120, 3), dtype=np.uint8)
        meta = ImageMetadata(instrument="TEST")
        img = ImageData(array=arr, path="test.png", metadata=meta)
        assert img.height == 100
        assert img.width == 120
        assert img.channels == 3

    def test_single_channel_properties(self):
        arr = np.zeros((64, 80, 1), dtype=np.uint8)
        meta = ImageMetadata(instrument="TEST")
        img = ImageData(array=arr, path="test.png", metadata=meta)
        assert img.channels == 1


class TestDataclassesAreNotFrozen:
    """Verify dataclasses are standard (not frozen) so pipeline can mutate fields like inlier_mask."""

    def test_matchset_inlier_mask_mutable(self):
        ms = MatchSet(
            source_points=np.zeros((3, 2)),
            reference_points=np.zeros((3, 2)),
        )
        ms.inlier_mask = np.array([True, False, True])
        assert ms.inlier_mask is not None

    def test_evaluation_result_scale_factor_mutable(self):
        er = EvaluationResult(
            total_matches=0,
            inlier_matches=0,
            inlier_ratio=0.0,
            rmse_pixels=None,
            median_error_pixels=None,
            coverage=0.0,
            spatial_uniformity=None,
            processing_time_seconds=0.0,
            random_seed=42,
        )
        er.scale_factor = 2.0
        assert er.scale_factor == 2.0
