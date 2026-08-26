"""Abstract Base Class for feature matchers.

CRITICAL CONVENTION:
Returned MatchSet.source_points and MatchSet.reference_points strictly enforce (x, y) coordinates.
"""

from abc import ABC, abstractmethod

from lunar_correspondence.io.metadata import FeatureSet, MatchSet


class Matcher(ABC):
    """Abstract Base Class for feature descriptor matchers."""

    @abstractmethod
    def match(
        self, features_src: FeatureSet, features_ref: FeatureSet, **kwargs
    ) -> MatchSet:
        """Establish correspondences between source and reference feature sets.

        Args:
            features_src: Extracted FeatureSet from source image.
            features_ref: Extracted FeatureSet from reference image.
            **kwargs: Config-driven optional parameters for future extensions.

        Returns:
            MatchSet containing paired point coordinates in (x, y) format and confidence scores.
        """
