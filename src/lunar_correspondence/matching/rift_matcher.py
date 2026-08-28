"""RIFT2 feature matcher adapter.

Implements the project's Matcher interface for RIFT2 descriptors.
Uses BFMatcher with L2 norm and Lowe's ratio test, consistent with the
upstream RIFT2 matching approach (see src/matcher_functions.py in upstream).

The upstream demo uses BFMatcher + knnMatch + Lowe's ratio test at 0.95,
which is appropriate for the MIM histogram descriptors produced by RIFT2.

STATUS: Implemented — produces MatchSet compatible with the existing pipeline.
"""

from typing import Any

import cv2
import numpy as np

from lunar_correspondence.io.metadata import FeatureSet, MatchSet
from lunar_correspondence.matching.base import Matcher


class RIFTMatcher(Matcher):
    """RIFT2 descriptor matcher using BFMatcher with Lowe's ratio test.

    Consumes FeatureSet outputs from RIFTFeatureExtractor and produces
    MatchSet compatible with the rest of the pipeline (spatial selection,
    RANSAC, subpixel refinement, evaluation).

    The default ratio test threshold is set to 0.90, which is slightly
    more permissive than the SIFT default (0.75), following the upstream
    RIFT2 recommendation for MIM histogram descriptors.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.ratio_threshold = cfg.get("ratio_test_threshold", 0.90)
        norm_name = cfg.get("norm_type", "L2")

        norm_map = {
            "L2": cv2.NORM_L2,
            "L1": cv2.NORM_L1,
            "HAMMING": cv2.NORM_HAMMING,
        }
        self.norm_type = norm_map.get(norm_name.upper(), cv2.NORM_L2)
        self.bf_matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)

    def match(
        self, features_src: FeatureSet, features_ref: FeatureSet, **kwargs
    ) -> MatchSet:
        """Match RIFT2 features between source and reference sets.

        Args:
            features_src: Source FeatureSet from RIFTFeatureExtractor.
            features_ref: Reference FeatureSet from RIFTFeatureExtractor.
            **kwargs: Extra parameters compatibility.

        Returns:
            MatchSet containing matched (N, 2) points in (x, y) coordinates,
            with confidence derived from Lowe's ratio test.
        """
        desc1 = features_src.descriptors
        desc2 = features_ref.descriptors

        # Handle empty/missing descriptor cases safely
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return self._empty_match_set()

        # Ensure float32 for BFMatcher
        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)

        # Need at least 2 reference descriptors for knnMatch k=2
        if len(desc2) < 2:
            return self._empty_match_set()

        # KNN Match k=2 for Lowe's Ratio Test
        try:
            knn_matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return self._empty_match_set()

        pts_src = []
        pts_ref = []
        confidences = []

        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    pts_src.append(features_src.keypoints[m.queryIdx])
                    pts_ref.append(features_ref.keypoints[m.trainIdx])
                    # Higher confidence for lower ratio distance
                    conf = 1.0 - (m.distance / (n.distance + 1e-6))
                    confidences.append(conf)

        if len(pts_src) == 0:
            return self._empty_match_set()

        return MatchSet(
            source_points=np.array(pts_src, dtype=np.float32),
            reference_points=np.array(pts_ref, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            inlier_mask=None,
        )

    @staticmethod
    def _empty_match_set() -> MatchSet:
        """Return a valid empty MatchSet for zero-match edge cases."""
        return MatchSet(
            source_points=np.zeros((0, 2), dtype=np.float32),
            reference_points=np.zeros((0, 2), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            inlier_mask=None,
        )
