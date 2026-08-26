"""Sub-pixel keypoint refinement stub module.

Gloss:
- Sub-pixel Refinement: Fine-tuning integer pixel corner locations to sub-pixel precision
  (e.g., using cv2.cornerSubPix or optical flow intensity surface fitting) as required by ISRO PS expectations.

STATUS: Interface defined; real subpixel optimization stubbed for P2.
"""

import numpy as np


def refine_subpixel(
    image_array: np.ndarray,
    keypoints_xy: np.ndarray,
    win_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Refine keypoint coordinates to sub-pixel accuracy.

    Args:
        image_array: Input grayscale image array.
        keypoints_xy: (N, 2) keypoints in (x, y) coordinates.
        win_size: Half of side length of search window.

    Returns:
        (N, 2) array of sub-pixel refined keypoint coordinates.
    """
    if len(keypoints_xy) == 0:
        return keypoints_xy.copy()

    # Pass through for Day-1 baseline skeleton
    return keypoints_xy.copy()
