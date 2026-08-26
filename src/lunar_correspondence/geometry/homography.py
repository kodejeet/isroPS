"""Homography geometric matrix estimation module.

Gloss:
- Homography: A 3x3 projective transformation matrix that maps 2D coordinates from one image plane
  to another image plane assuming planar terrain or a rotating perspective camera.

CRITICAL CONVENTION:
All 2D points strictly follow (x, y) = (col, row) coordinates.
"""

import numpy as np


def compute_reprojection_errors(
    pts_src: np.ndarray, pts_ref: np.ndarray, H: np.ndarray
) -> np.ndarray:
    """Compute reprojection error distances (in pixels) for all point pairs under homography H.

    Args:
        pts_src: (N, 2) array of (x, y) source coordinates.
        pts_ref: (N, 2) array of (x, y) reference coordinates.
        H: 3x3 Homography transformation matrix.

    Returns:
        (N,) array of Euclidean error distances in pixels.
    """
    if len(pts_src) == 0:
        return np.zeros((0,), dtype=np.float32)

    # Convert to homogeneous coordinates (N, 3)
    ones = np.ones((len(pts_src), 1), dtype=np.float32)
    pts_src_h = np.hstack([pts_src, ones])  # (N, 3)

    # Transform points: H @ p_src^T -> (3, N)
    pts_proj_h = (H @ pts_src_h.T).T  # (N, 3)

    # Normalize by z coordinate
    z = pts_proj_h[:, 2:3]
    z[np.abs(z) < 1e-8] = 1e-8
    pts_proj = pts_proj_h[:, :2] / z  # (N, 2) in (x, y)

    # Euclidean distance between projected source points and ground truth reference points
    errors = np.linalg.norm(pts_proj - pts_ref, axis=1)
    return errors.astype(np.float32)
