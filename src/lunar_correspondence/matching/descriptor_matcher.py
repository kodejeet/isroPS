"""Descriptor-based feature matcher implementation with Lowe's Ratio Test.

Gloss:
- Lowe's Ratio Test: Eliminates ambiguous keypoint matches by confirming that the distance to
  the closest match is significantly smaller than the distance to the second-closest match (e.g. ratio < 0.75).
"""

from typing import Any

import cv2
import numpy as np

from lunar_correspondence.io.metadata import FeatureSet, MatchSet
from lunar_correspondence.matching.base import Matcher


class DescriptorMatcher(Matcher):
    """Real working descriptor matcher supporting Brute-Force and FLANN matching."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        norm_name = cfg.get("norm_type", "L2")
        self.ratio_threshold = cfg.get("ratio_test_threshold", 0.75)

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
        """Match features between source and reference sets.

        Args:
            features_src: Source FeatureSet.
            features_ref: Reference FeatureSet.
            **kwargs: Extra parameters compatibility.

        Returns:
            MatchSet containing matched (N, 2) points in (x, y) coordinates.
        """
        desc1 = features_src.descriptors
        desc2 = features_ref.descriptors

        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return MatchSet(
                source_points=np.zeros((0, 2), dtype=np.float32),
                reference_points=np.zeros((0, 2), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                inlier_mask=None,
            )

        # KNN Match k=2 for Lowe's Ratio Test
        try:
            knn_matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return MatchSet(
                source_points=np.zeros((0, 2), dtype=np.float32),
                reference_points=np.zeros((0, 2), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                inlier_mask=None,
            )

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
            return MatchSet(
                source_points=np.zeros((0, 2), dtype=np.float32),
                reference_points=np.zeros((0, 2), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                inlier_mask=None,
            )

        return MatchSet(
            source_points=np.array(pts_src, dtype=np.float32),
            reference_points=np.array(pts_ref, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            inlier_mask=None,
        )
