"""RIFT integration and architecture readiness tests.

Verifies:
- RIFT feature extractor and matcher properly extend the abstract base classes.
- Both raise NotImplementedError with descriptive messages.
- RIFT integrates through the existing contracts (FeatureSet, MatchSet).
- The pipeline can instantiate with RIFT config.
- SIFT and RIFT are interchangeable at the pipeline interface.
"""

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


class TestRIFTFeatureExtractorArchitecture:
    """RIFTFeatureExtractor must be a FeatureExtractor returning FeatureSet."""

    def test_is_subclass_of_feature_extractor(self):
        assert issubclass(RIFTFeatureExtractor, FeatureExtractor)

    def test_instance_is_feature_extractor(self):
        extractor = RIFTFeatureExtractor()
        assert isinstance(extractor, FeatureExtractor)

    def test_extract_raises_not_implemented(self):
        extractor = RIFTFeatureExtractor()
        dummy_image = ImageData(
            array=np.zeros((32, 32, 1), dtype=np.uint8),
            path="dummy.png",
            metadata=ImageMetadata(instrument="TEST"),
        )
        with pytest.raises(NotImplementedError, match="RIFT feature extractor"):
            extractor.extract(dummy_image)

    def test_extract_signature_matches_base(self):
        """Ensure extract() accepts the same arguments as the base class."""
        import inspect

        base_sig = inspect.signature(FeatureExtractor.extract)
        rift_sig = inspect.signature(RIFTFeatureExtractor.extract)
        # Both should accept (self, image, **kwargs)
        base_params = list(base_sig.parameters.keys())
        rift_params = list(rift_sig.parameters.keys())
        assert base_params == rift_params

    def test_return_annotation_is_featureset(self):
        """Ensure extract() return type annotation is FeatureSet."""
        import inspect

        sig = inspect.signature(RIFTFeatureExtractor.extract)
        assert sig.return_annotation is FeatureSet


class TestRIFTMatcherArchitecture:
    """RIFTMatcher must be a Matcher returning MatchSet."""

    def test_is_subclass_of_matcher(self):
        assert issubclass(RIFTMatcher, Matcher)

    def test_instance_is_matcher(self):
        matcher = RIFTMatcher()
        assert isinstance(matcher, Matcher)

    def test_match_raises_not_implemented(self):
        matcher = RIFTMatcher()
        dummy_fs = FeatureSet(
            keypoints=np.zeros((0, 2), dtype=np.float32),
            method="test",
        )
        with pytest.raises(NotImplementedError, match="RIFT feature matcher"):
            matcher.match(dummy_fs, dummy_fs)

    def test_match_signature_matches_base(self):
        """Ensure match() accepts the same arguments as the base class."""
        import inspect

        base_sig = inspect.signature(Matcher.match)
        rift_sig = inspect.signature(RIFTMatcher.match)
        base_params = list(base_sig.parameters.keys())
        rift_params = list(rift_sig.parameters.keys())
        assert base_params == rift_params


class TestRIFTSIFTInterchangeability:
    """SIFT and RIFT must be interchangeable at the pipeline interface."""

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

    def test_pipeline_rift_run_raises_not_implemented(self):
        """Pipeline.run() with RIFT feature method should raise NotImplementedError."""
        config = {
            "pipeline": {"random_seed": 42},
            "feature_extraction": {"method": "rift"},
            "matching": {"method": "descriptor"},
        }
        pipeline = RegistrationPipeline(config)

        dummy_image = ImageData(
            array=np.zeros((32, 32, 1), dtype=np.uint8),
            path="dummy.png",
            metadata=ImageMetadata(instrument="TEST"),
        )

        with pytest.raises(NotImplementedError, match="RIFT"):
            pipeline.run(dummy_image, dummy_image)
