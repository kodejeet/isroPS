"""Contrast enhancement and local illumination preprocessing.

Note: Local contrast enhancement (such as CLAHE) boosts visual contrast in deep shadows,
but does NOT solve solar illumination invariance (shadow reversal). Illumination invariance
requires phase-congruency feature extraction (e.g. RIFT).
"""

import cv2
import numpy as np

from lunar_correspondence.preprocessing.normalization import to_grayscale


def apply_clahe(
    array: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to single-channel or RGB image.

    Gloss:
    - CLAHE: Contrast Limited Adaptive Histogram Equalization - divides the image into small contextual
      tiles and equalizes histograms locally to improve visibility in dark crater shadows.

    Args:
        array: Input image array (H, W, C) or (H, W).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced image array of same dimensions.
    """
    gray = to_grayscale(array)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    if array.ndim == 3:
        return np.expand_dims(enhanced, axis=-1)
    return enhanced
