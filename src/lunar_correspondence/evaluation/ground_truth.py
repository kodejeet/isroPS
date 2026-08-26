"""Ground truth verification helper utilities."""

import numpy as np


def compute_ground_truth_error(
    estimated_H: np.ndarray, ground_truth_H: np.ndarray, image_shape: tuple
) -> float:
    """Compute Mean Corner Error (MCE) in pixels between estimated homography and ground-truth homography.

    Args:
        estimated_H: 3x3 estimated homography matrix.
        ground_truth_H: 3x3 ground-truth homography matrix.
        image_shape: (height, width).

    Returns:
        Mean corner reprojection error distance in pixels.
    """
    h, w = image_shape
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [w - 1.0, 0.0, 1.0],
            [w - 1.0, h - 1.0, 1.0],
            [0.0, h - 1.0, 1.0],
        ],
        dtype=np.float32,
    ).T

    # Transform corners under estimated homography
    est_proj = estimated_H @ corners
    est_proj = est_proj[:2] / est_proj[2]

    # Transform corners under ground truth homography
    gt_proj = ground_truth_H @ corners
    gt_proj = gt_proj[:2] / gt_proj[2]

    mean_corner_error = float(np.mean(np.linalg.norm(est_proj - gt_proj, axis=0)))
    return mean_corner_error
