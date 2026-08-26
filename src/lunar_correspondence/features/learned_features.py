"""Learned feature extractor stub (SuperPoint / DISK / ALIKED).

STATUS: Stub pending P2 implementation. See docs/research_notes.md.
"""

from typing import Any

from lunar_correspondence.features.base import FeatureExtractor
from lunar_correspondence.io.metadata import FeatureSet, ImageData


class LearnedFeatureExtractor(FeatureExtractor):
    """Learned feature extractor adapter stub (Pending P2)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def extract(self, image: ImageData, **kwargs) -> FeatureSet:
        """Stub extract method raising clear pending notification."""
        raise NotImplementedError(
            "Learned feature extractor (SuperPoint / ALIKED) is pending P2 development. "
            "For Day-1 execution, please use SIFT (feature_extraction.method: 'sift')."
        )
