"""Abstract Base Class for feature extractors.

CRITICAL CONVENTION:
All extracted keypoints in FeatureSet.keypoints strictly follow (x, y) = (col, row).
"""

from abc import ABC, abstractmethod

from lunar_correspondence.io.metadata import FeatureSet, ImageData


class FeatureExtractor(ABC):
    """Abstract Base Class for all feature detection and description implementations."""

    @abstractmethod
    def extract(self, image: ImageData, **kwargs) -> FeatureSet:
        """Extract keypoints and descriptors from an image.

        Args:
            image: ImageData structure holding (H, W, C) array and metadata.
            **kwargs: Config-driven optional parameters for future extensions (RIFT/Learned adapters).

        Returns:
            FeatureSet containing keypoints of shape (N, 2) in (x, y) coordinates and descriptors.
        """
