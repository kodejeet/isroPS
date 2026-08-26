"""Affine transformation utilities."""

import cv2
import numpy as np


def fit_affine_transform(pts_src: np.ndarray, pts_ref: np.ndarray) -> np.ndarray:
    """Fit a 6-degree-of-freedom 2x3 affine matrix between source and reference point sets.

    Args:
        pts_src: (N, 2) source points in (x, y) coordinates.
        pts_ref: (N, 2) reference points in (x, y) coordinates.

    Returns:
        3x3 homogeneous matrix representation of estimated affine transform.
    """
    if len(pts_src) < 3:
        return np.eye(3, dtype=np.float32)

    M, _ = cv2.estimateAffine2D(pts_src, pts_ref)
    if M is None:
        return np.eye(3, dtype=np.float32)

    return np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float32)
