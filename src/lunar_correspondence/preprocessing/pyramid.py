"""Multiscale image pyramid generation utilities."""

import cv2
import numpy as np


def build_gaussian_pyramid(array: np.ndarray, levels: int = 3) -> list[np.ndarray]:
    """Build a multi-level Gaussian pyramid for scale-space image analysis.

    Args:
        array: Input image array.
        levels: Number of pyramid octave levels to construct.

    Returns:
        List of downsampled image arrays from fine (level 0) to coarse.
    """
    pyramid = [array]
    current = array.copy()
    for _ in range(1, levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid
