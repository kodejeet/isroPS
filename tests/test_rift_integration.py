"""RIFT2 integration tests — feature extraction, matching, and full pipeline.

Verifies:
- RIFTFeatureExtractor properly extends FeatureExtractor and produces valid FeatureSet.
- RIFTMatcher properly extends Matcher and produces valid MatchSet.
- RIFT2 features → matching → RANSAC pipeline path works end-to-end.
- SIFT and RIFT2 are interchangeable at the pipeline interface.
- All data contracts (FeatureSet, MatchSet) are preserved.

Note: RIFT2 is the rotation-invariant successor in the RIFT algorithmic lineage.
The integrated implementation adapts the upstream Python RIFT2 repository
rather than reproducing the paper from scratch.
"""

import os

import cv2
import numpy as np
import pytest

from lunar_correspondence.features.base import FeatureExtractor
from lunar_correspondence.features.rift_features import RIFTFeatureExtractor
from lunar_correspondence.features.sift_features import SIFTFeatureExtractor
from lunar_correspondence.io.metadata import (
    FeatureSet,
    ImageData,
    ImageMetadata,
    MatchSet,
)
from lunar_correspondence.matching.base import Matcher
from lunar_correspondence.matching.descriptor_matcher import DescriptorMatcher
from lunar_correspondence.matching.rift_matcher import RIFTMatcher
from lunar_correspondence.pipeline import RegistrationPipeline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_image_256():
    """Generate a 256x256 synthetic lunar-like grayscale image."""
    np.random.seed(42)
    h, w = 256, 256
    surface = np.random.normal(120, 20, (h, w)).astype(np.float32)
    for _ in range(15):
        cx = np.random.randint(30, w - 30)
        cy = np.random.randint(30, h - 30)
        r = np.random.randint(8, 35)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        surface[dist <= (r * 0.7)] -= 40.0
        surface[(dist > (r * 0.7)) & (dist <= r)] += 50.0

    img = np.clip(surface, 0, 255).astype(np.uint8)
    return img


@pytest.fixture(scope="module")
def synthetic_pair(synthetic_image_256):
    """Create a source/reference pair with known geometric transform."""
    src = synthetic_image_256
    h, w = src.shape

    # Apply mild affine transform + illumination change
    angle = np.radians(8)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = w / 2.0, h / 2.0
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    R = np.array(
        [[cos_a, -sin_a, 10], [sin_a, cos_a, -8], [0, 0, 1]], dtype=np.float32
    )
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
    H = T2 @ R @ T1
    ref = cv2.warpPerspective(src, H, (w, h))
    # Add illumination gradient
    X = np.linspace(0.85, 1.15, w)
    ref = np.clip(ref.astype(np.float32) * X[np.newaxis, :], 0, 255).astype(
        np.uint8
    )

    src_data = ImageData(
        array=src[:, :, np.newaxis],
        path="synthetic_src.png",
        metadata=ImageMetadata(instrument="TEST_OHRC"),
    )
    ref_data = ImageData(
        array=ref[:, :, np.newaxis],
        path="synthetic_ref.png",
        metadata=ImageMetadata(instrument="TEST_TMC2"),
    )
    return src_data, ref_data


# ===========================================================================
# Test 1 — Feature adapter architecture
# ===========================================================================


class TestRIFTFeatureExtractorArchitecture:
    """RIFTFeatureExtractor must be a FeatureExtractor returning FeatureSet."""

    def test_is_subclass_of_feature_extractor(self):
        assert issubclass(RIFTFeatureExtractor, FeatureExtractor)

    def test_instance_is_feature_extractor(self):
        extractor = RIFTFeatureExtractor()
        assert isinstance(extractor, FeatureExtractor)

    def test_extract_signature_matches_base(self):
        """Ensure extract() accepts the same arguments as the base class."""
        import inspect

        base_sig = inspect.signature(FeatureExtractor.extract)
        rift_sig = inspect.signature(RIFTFeatureExtractor.extract)
        base_params = list(base_sig.parameters.keys())
        rift_params = list(rift_sig.parameters.keys())
        assert base_params == rift_params

    def test_return_annotation_is_featureset(self):
        """Ensure extract() return type annotation is FeatureSet."""
        import inspect

        sig = inspect.signature(RIFTFeatureExtractor.extract)
        assert sig.return_annotation is FeatureSet

    def test_instantiation_with_config(self):
        """Instantiation with custom config does not raise."""
        extractor = RIFTFeatureExtractor({"npt": 1000, "patch_size": 48})
        assert isinstance(extractor, FeatureExtractor)


# ===========================================================================
# Test 2 — Feature extraction
# ===========================================================================


class TestRIFTFeatureExtraction:
    """Verify RIFT2 feature extraction produces valid FeatureSet."""

    def test_extraction_succeeds(self, synthetic_pair):
        """Extraction on a synthetic image completes without error."""
        src_data, _ = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs = extractor.extract(src_data)
        assert isinstance(fs, FeatureSet)

    def test_output_is_valid_featureset(self, synthetic_pair):
        """Output has correct field types and shapes."""
        src_data, _ = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs = extractor.extract(src_data)

        # Keypoints must be (N, 2)
        assert fs.keypoints.ndim == 2
        assert fs.keypoints.shape[1] == 2
        assert fs.keypoints.dtype == np.float32

        # Descriptors must be (N, D)
        assert fs.descriptors is not None
        assert fs.descriptors.ndim == 2
        assert fs.descriptors.shape[0] == fs.keypoints.shape[0]
        assert fs.descriptors.dtype == np.float32

        # Scores must be (N,)
        assert fs.scores is not None
        assert fs.scores.shape == (fs.keypoints.shape[0],)

    def test_descriptor_count_matches_keypoint_count(self, synthetic_pair):
        """Number of descriptors equals number of keypoints."""
        src_data, _ = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs = extractor.extract(src_data)
        assert fs.descriptors.shape[0] == fs.keypoints.shape[0]

    def test_values_are_finite(self, synthetic_pair):
        """All output values are finite (no NaN/Inf)."""
        src_data, _ = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs = extractor.extract(src_data)
        assert np.all(np.isfinite(fs.keypoints))
        assert np.all(np.isfinite(fs.descriptors))
        assert np.all(np.isfinite(fs.scores))

    def test_method_label(self, synthetic_pair):
        """FeatureSet.method is 'RIFT2'."""
        src_data, _ = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs = extractor.extract(src_data)
        assert fs.method == "RIFT2"

    def test_keypoints_within_image_bounds(self, synthetic_pair):
        """All keypoint coordinates are within the image dimensions."""
        src_data, _ = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs = extractor.extract(src_data)
        if len(fs.keypoints) > 0:
            assert np.all(fs.keypoints[:, 0] >= 0)
            assert np.all(fs.keypoints[:, 0] < src_data.width)
            assert np.all(fs.keypoints[:, 1] >= 0)
            assert np.all(fs.keypoints[:, 1] < src_data.height)

    def test_empty_image_handling(self):
        """Extraction on a blank image returns empty FeatureSet safely."""
        blank = ImageData(
            array=np.zeros((64, 64, 1), dtype=np.uint8),
            path="blank.png",
            metadata=ImageMetadata(instrument="TEST"),
        )
        extractor = RIFTFeatureExtractor({"npt": 100, "patch_size": 24})
        fs = extractor.extract(blank)
        assert isinstance(fs, FeatureSet)
        assert fs.keypoints.shape[1] == 2
        # Should handle gracefully (may produce 0 or few features)
        assert fs.descriptors is not None


# ===========================================================================
# Test 3 — Matcher
# ===========================================================================


class TestRIFTMatcher:
    """Verify RIFT2 matcher produces valid MatchSet."""

    def test_is_subclass_of_matcher(self):
        assert issubclass(RIFTMatcher, Matcher)

    def test_instance_is_matcher(self):
        matcher = RIFTMatcher()
        assert isinstance(matcher, Matcher)

    def test_match_signature_matches_base(self):
        """Ensure match() accepts the same arguments as the base class."""
        import inspect

        base_sig = inspect.signature(Matcher.match)
        rift_sig = inspect.signature(RIFTMatcher.match)
        base_params = list(base_sig.parameters.keys())
        rift_params = list(rift_sig.parameters.keys())
        assert base_params == rift_params

    def test_matcher_executes(self, synthetic_pair):
        """Matcher runs without error on RIFT2 features."""
        src_data, ref_data = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs_src = extractor.extract(src_data)
        fs_ref = extractor.extract(ref_data)

        matcher = RIFTMatcher()
        ms = matcher.match(fs_src, fs_ref)
        assert isinstance(ms, MatchSet)

    def test_output_is_valid_matchset(self, synthetic_pair):
        """MatchSet output has correct field types and shapes."""
        src_data, ref_data = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs_src = extractor.extract(src_data)
        fs_ref = extractor.extract(ref_data)

        matcher = RIFTMatcher()
        ms = matcher.match(fs_src, fs_ref)

        # Source and reference points must be (N, 2)
        assert ms.source_points.ndim == 2
        assert ms.source_points.shape[1] == 2
        assert ms.reference_points.ndim == 2
        assert ms.reference_points.shape[1] == 2

    def test_correspondence_lengths_agree(self, synthetic_pair):
        """Source and reference point arrays have equal length."""
        src_data, ref_data = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs_src = extractor.extract(src_data)
        fs_ref = extractor.extract(ref_data)

        matcher = RIFTMatcher()
        ms = matcher.match(fs_src, fs_ref)
        assert ms.source_points.shape[0] == ms.reference_points.shape[0]

    def test_confidence_is_valid(self, synthetic_pair):
        """Confidence array has correct shape and finite values."""
        src_data, ref_data = synthetic_pair
        extractor = RIFTFeatureExtractor()
        fs_src = extractor.extract(src_data)
        fs_ref = extractor.extract(ref_data)

        matcher = RIFTMatcher()
        ms = matcher.match(fs_src, fs_ref)
        assert ms.confidence is not None
        assert ms.confidence.shape == (ms.source_points.shape[0],)
        assert np.all(np.isfinite(ms.confidence))

    def test_empty_feature_handling(self):
        """Matcher returns empty MatchSet when given empty features."""
        empty_fs = FeatureSet(
            keypoints=np.zeros((0, 2), dtype=np.float32),
            descriptors=np.zeros((0, 216), dtype=np.float32),
            method="RIFT2",
        )
        matcher = RIFTMatcher()
        ms = matcher.match(empty_fs, empty_fs)
        assert isinstance(ms, MatchSet)
        assert ms.source_points.shape == (0, 2)
        assert ms.reference_points.shape == (0, 2)


# ===========================================================================
# Test 4 — End-to-end RIFT feature → match
# ===========================================================================


class TestRIFTFeatureToMatch:
    """Verify RIFT2 features → RIFT2 matching works as a combined path."""

    def test_feature_to_match_pipeline(self, synthetic_pair):
        """Image → RIFT features → RIFT matching works without errors."""
        src_data, ref_data = synthetic_pair
        extractor = RIFTFeatureExtractor()
        matcher = RIFTMatcher()

        fs_src = extractor.extract(src_data)
        fs_ref = extractor.extract(ref_data)
        ms = matcher.match(fs_src, fs_ref)

        assert isinstance(ms, MatchSet)
        # Should produce some matches on this synthetic pair
        assert len(ms.source_points) > 0, (
            "Expected at least some matches on the synthetic pair"
        )

    def test_coordinates_preserve_xy_convention(self, synthetic_pair):
        """Matched points preserve (x, y) = (col, row) convention."""
        src_data, ref_data = synthetic_pair
        extractor = RIFTFeatureExtractor()
        matcher = RIFTMatcher()

        fs_src = extractor.extract(src_data)
        fs_ref = extractor.extract(ref_data)
        ms = matcher.match(fs_src, fs_ref)

        if len(ms.source_points) > 0:
            # x (col) must be < width, y (row) must be < height
            assert np.all(ms.source_points[:, 0] < src_data.width)
            assert np.all(ms.source_points[:, 1] < src_data.height)
            assert np.all(ms.reference_points[:, 0] < ref_data.width)
            assert np.all(ms.reference_points[:, 1] < ref_data.height)


# ===========================================================================
# Test 5 — Full registration via pipeline
# ===========================================================================


class TestRIFTFullRegistration:
    """Verify the complete RIFT2 path through the registration pipeline."""

    def test_pipeline_instantiation_with_rift_config(self):
        """Pipeline successfully instantiates with RIFT feature+matcher config."""
        config = {
            "pipeline": {"random_seed": 42},
            "feature_extraction": {"method": "rift"},
            "matching": {"method": "rift"},
        }
        pipeline = RegistrationPipeline(config)
        assert isinstance(pipeline.feature_extractor, RIFTFeatureExtractor)
        assert isinstance(pipeline.matcher, RIFTMatcher)

    def test_rift_pipeline_run(self, synthetic_pair):
        """Full pipeline run with RIFT2 produces valid registration result."""
        src_data, ref_data = synthetic_pair
        config = {
            "pipeline": {"random_seed": 42},
            "feature_extraction": {
                "method": "rift",
                "rift": {"npt": 3000, "patch_size": 48},
            },
            "matching": {"method": "rift", "rift": {"ratio_test_threshold": 0.92}},
            "geometry": {
                "model_type": "homography",
                "ransac": {
                    "reproj_threshold": 5.0,
                    "max_iters": 2000,
                    "confidence": 0.99,
                },
            },
            "evaluation": {"grid_rows": 4, "grid_cols": 4},
        }
        pipeline = RegistrationPipeline(config)
        reg_result, eval_result = pipeline.run(src_data, ref_data)

        # Registration must produce a valid result
        assert reg_result.registered_image is not None
        assert reg_result.geometric_model is not None
        assert reg_result.match_set is not None

        # Evaluation must compute metrics
        assert eval_result.total_matches >= 0
        assert eval_result.processing_time_seconds > 0

    def test_sift_pipeline_still_works(self, synthetic_pair):
        """SIFT pipeline path is unaffected by RIFT2 integration."""
        src_data, ref_data = synthetic_pair
        config = {
            "pipeline": {"random_seed": 42},
            "feature_extraction": {"method": "sift", "sift": {"nfeatures": 1000}},
            "matching": {
                "method": "descriptor",
                "descriptor": {"ratio_test_threshold": 0.75},
            },
            "geometry": {
                "model_type": "homography",
                "ransac": {
                    "reproj_threshold": 3.0,
                    "max_iters": 2000,
                    "confidence": 0.99,
                },
            },
            "evaluation": {"grid_rows": 4, "grid_cols": 4},
        }
        pipeline = RegistrationPipeline(config)
        reg_result, eval_result = pipeline.run(src_data, ref_data)
        assert reg_result.registered_image is not None
        assert eval_result.total_matches > 0
        assert eval_result.inlier_matches > 0


# ===========================================================================
# Interchangeability
# ===========================================================================


class TestRIFTSIFTInterchangeability:
    """SIFT and RIFT2 must be interchangeable at the pipeline interface."""

    def test_sift_extractor_is_feature_extractor(self):
        assert issubclass(SIFTFeatureExtractor, FeatureExtractor)

    def test_rift_extractor_is_feature_extractor(self):
        assert issubclass(RIFTFeatureExtractor, FeatureExtractor)

    def test_descriptor_matcher_is_matcher(self):
        assert issubclass(DescriptorMatcher, Matcher)

    def test_rift_matcher_is_matcher(self):
        assert issubclass(RIFTMatcher, Matcher)

    def test_pipeline_instantiates_with_rift_feature_config(self):
        """Pipeline should successfully instantiate with rift feature method."""
        config = {
            "pipeline": {"random_seed": 42},
            "feature_extraction": {"method": "rift"},
            "matching": {"method": "descriptor"},
        }
        pipeline = RegistrationPipeline(config)
        assert isinstance(pipeline.feature_extractor, RIFTFeatureExtractor)
        assert isinstance(pipeline.matcher, DescriptorMatcher)

    def test_pipeline_instantiates_with_rift_matcher_config(self):
        """Pipeline should successfully instantiate with rift matcher method."""
        config = {
            "pipeline": {"random_seed": 42},
            "feature_extraction": {"method": "sift"},
            "matching": {"method": "rift"},
        }
        pipeline = RegistrationPipeline(config)
        assert isinstance(pipeline.feature_extractor, SIFTFeatureExtractor)
        assert isinstance(pipeline.matcher, RIFTMatcher)
