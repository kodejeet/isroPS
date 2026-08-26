"""RIFT (Rotation and Illumination Invariant Feature Transform) feature extractor stub.

Gloss:
- RIFT: Feature extraction algorithm using phase congruency and Maximum Index Maps (MIM)
  to achieve solar illumination and shadow-reversal invariance on lunar imagery.

STATUS: Stub pending P1 implementation. See docs/research_notes.md.
"""

from typing import Any

from lunar_correspondence.features.base import FeatureExtractor
from lunar_correspondence.io.metadata import FeatureSet, ImageData


class RIFTFeatureExtractor(FeatureExtractor):
    """RIFT feature extractor adapter stub (Pending P1)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def extract(self, image: ImageData, **kwargs) -> FeatureSet:
        """Stub extract method raising clear pending notification."""
        raise NotImplementedError(
            "RIFT feature extractor is pending P1 development. "
            "For Day-1 execution, please configure feature_extraction.method: 'sift'. "
            "See docs/research_notes.md for RIFT design specifications."
        )
