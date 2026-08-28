"""RIFT2 (Rotation and Illumination Invariant Feature Transform) feature extractor adapter.

Wraps the vendored RIFT2 implementation to satisfy the project's FeatureExtractor
interface, producing FeatureSet outputs compatible with the existing pipeline.

Gloss:
- RIFT2: Successor to RIFT, using phase congruency and Maximum Index Maps (MIM)
  to achieve solar illumination and rotation invariance on multimodal imagery.
- Phase Congruency: Contrast-invariant edge/corner measure computed via
  Log-Gabor filter banks in the frequency domain.
- MIM (Maximum Index Map): Illumination-invariant orientation map derived from
  the Log-Gabor filter response with maximum energy at each pixel.

Upstream implementation: https://github.com/canyagmur/RIFT2-multimodal-matching-rotation-python
Original paper: Li, Jiayuan, Qingwu Hu, and Mingyao Ai. "RIFT2: Speeding-up
  RIFT with A New Rotation-Invariance Technique" (2023). arXiv:2303.00319

STATUS: Implemented — adapts vendored RIFT2 core for the SIH26166 pipeline.
"""

from typing import Any

import numpy as np

from lunar_correspondence.features._vendor.rift2.core import RIFT2Core
from lunar_correspondence.features.base import FeatureExtractor
from lunar_correspondence.io.metadata import FeatureSet, ImageData
from lunar_correspondence.preprocessing.normalization import to_grayscale


class RIFTFeatureExtractor(FeatureExtractor):
    """RIFT2 feature extractor adapter.

    Converts ImageData to grayscale, runs the vendored RIFT2 core algorithm
    (phase congruency detection → orientation assignment → MIM descriptor),
    and returns a standard FeatureSet.

    Returned keypoints shape is (N, 2) enforcing (x, y) = (col, row) coordinates.
    Descriptor dimensionality is no * no * nbin (default: 6*6*6 = 216).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        # Map project-level config keys to RIFT2Core config keys
        rift2_config = {
            "nscale": cfg.get("nscale", 4),
            "norient": cfg.get("norient", 6),
            "npt": cfg.get("npt", 5000),
            "minWaveLength": cfg.get("minWaveLength", 3),
            "mult": cfg.get("mult", 1.6),
            "sigmaOnf": cfg.get("sigmaOnf", 0.75),
            "g": cfg.get("g", 3),
            "k": cfg.get("k", 1),
            "patch_size": cfg.get("patch_size", 96),
            "no": cfg.get("no", 6),
            "nbin": cfg.get("nbin", 6),
            "is_ori": cfg.get("is_ori", 1),
            "ori_peak_ratio": cfg.get("ori_peak_ratio", 0.8),
        }
        self._rift2 = RIFT2Core(rift2_config)
        self._desc_dim = rift2_config["no"] * rift2_config["no"] * rift2_config["nbin"]

    def extract(self, image: ImageData, **kwargs) -> FeatureSet:
        """Extract RIFT2 keypoints and descriptors from an image.

        Args:
            image: Input ImageData containing (H, W, C) array.
            **kwargs: Signature compatibility for extra parameters.

        Returns:
            FeatureSet containing (N, 2) keypoints in (x, y) coordinates
            and RIFT2 MIM histogram descriptors of shape (N, D).
        """
        # Convert multi-band array to single-channel uint8 grayscale
        gray = to_grayscale(image.array)

        # Run RIFT2 detection and description
        keypoints_xy, descriptors = self._rift2.detect_and_describe(gray)

        # Handle empty/no-feature case
        if len(keypoints_xy) == 0:
            return FeatureSet(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, self._desc_dim), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                method="RIFT2",
            )

        # Generate scores from descriptor norms (higher norm = stronger feature)
        scores = np.linalg.norm(descriptors, axis=1).astype(np.float32)

        return FeatureSet(
            keypoints=keypoints_xy,
            descriptors=descriptors,
            scores=scores,
            method="RIFT2",
        )
