"""SIFT (Scale-Invariant Feature Transform) feature extractor implementation.

Gloss:
- SIFT: Scale-Invariant Feature Transform - standard computer vision feature detector that identifies
  scale-space blob keypoints and computes 128-dimensional local gradient histograms.
"""

from typing import Any

import cv2
import numpy as np

from lunar_correspondence.features.base import FeatureExtractor
from lunar_correspondence.io.metadata import FeatureSet, ImageData
from lunar_correspondence.preprocessing.normalization import to_grayscale


class SIFTFeatureExtractor(FeatureExtractor):
    """Real working SIFT feature extractor.

    Converts multi-channel ImageData arrays into single-band grayscale before extracting keypoints.
    Returned keypoints shape is (N, 2) enforcing (x, y) = (col, row) coordinates.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        nfeatures = cfg.get("nfeatures", 2000)
        nOctaveLayers = cfg.get("nOctaveLayers", 3)
        contrastThreshold = cfg.get("contrastThreshold", 0.04)
        edgeThreshold = cfg.get("edgeThreshold", 10)
        sigma = cfg.get("sigma", 1.6)

        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
        )

    def extract(self, image: ImageData, **kwargs) -> FeatureSet:
        """Extract SIFT keypoints and descriptors.

        Args:
            image: Input ImageData containing (H, W, C) array.
            **kwargs: Signature compatibility for extra parameters.

        Returns:
            FeatureSet containing (N, 2) keypoints in (x, y) coordinates and 128D descriptors.
        """
        # Convert multi-band array to single-channel uint8 grayscale
        gray = to_grayscale(image.array)

        cv_kps, descriptors = self.sift.detectAndCompute(gray, None)

        if cv_kps is None or len(cv_kps) == 0:
            return FeatureSet(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, 128), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                method="SIFT",
            )

        # cv_kp.pt returns (x, y) = (col, row)
        kps_xy = np.array([kp.pt for kp in cv_kps], dtype=np.float32)
        scores = np.array([kp.response for kp in cv_kps], dtype=np.float32)

        return FeatureSet(
            keypoints=kps_xy,
            descriptors=(
                descriptors.astype(np.float32) if descriptors is not None else None
            ),
            scores=scores,
            method="SIFT",
        )
