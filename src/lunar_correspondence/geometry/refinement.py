"""Sub-pixel keypoint refinement module.

Refines integer/approximate keypoint locations to sub-pixel precision using
intensity gradient covariance fitting (cv2.cornerSubPix).

CRITICAL CONVENTION:
All 2D points strictly follow (x, y) = (col, row) coordinates.
x: 0 to width - 1 (columns)
y: 0 to height - 1 (rows)
"""

import cv2
import numpy as np


def refine_subpixel(
    image_array: np.ndarray,
    keypoints_xy: np.ndarray,
    win_size: tuple[int, int] = (5, 5),
    zero_zone: tuple[int, int] = (-1, -1),
    criteria: tuple[int, int, float] | None = None,
) -> np.ndarray:
    """Refine keypoint coordinates to sub-pixel accuracy using cv2.cornerSubPix.

    Operates on grayscale image representations. Safely handles image borders,
    empty point sets, and enforces finite coordinate bounds in (x, y) format.

    Args:
        image_array: Input image array (uint8 or convertible, shape (H, W) or (H, W, C)).
        keypoints_xy: (N, 2) array of keypoints in (x, y) coordinates.
        win_size: Half of search window size (width, height), e.g. (5, 5).
        zero_zone: Half of dead zone size (-1, -1 for none).
        criteria: Termination criteria tuple (type, max_iters, epsilon).

    Returns:
        (N, 2) np.float32 array of sub-pixel refined keypoint coordinates.
    """
    if len(keypoints_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # 1. Prepare single-channel uint8 grayscale image
    img = image_array
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = img[:, :, 0]

    if img.dtype != np.uint8:
        # Scale/clip safely to uint8
        img_min, img_max = float(img.min()), float(img.max())
        if img_max > img_min:
            img_uint8 = np.clip(
                ((img.astype(np.float32) - img_min) / (img_max - img_min)) * 255.0,
                0,
                255,
            ).astype(np.uint8)
        else:
            img_uint8 = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        img_uint8 = img.copy()

    h, w = img_uint8.shape[:2]
    win_w, win_h = win_size

    # Ensure contiguous float32 copy of keypoints
    pts = np.array(keypoints_xy, dtype=np.float32).copy()

    # 2. Filter keypoints with adequate margin from image boundaries
    # Points too close to border cannot fit the (2*win_w + 1, 2*win_h + 1) window
    valid_mask = (
        (pts[:, 0] >= win_w)
        & (pts[:, 0] <= (w - 1 - win_w))
        & (pts[:, 1] >= win_h)
        & (pts[:, 1] <= (h - 1 - win_h))
        & np.isfinite(pts[:, 0])
        & np.isfinite(pts[:, 1])
    )

    if not np.any(valid_mask):
        return pts

    term_criteria = criteria or (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # Reshape valid points to (M, 1, 2) for cv2.cornerSubPix
    valid_pts = pts[valid_mask].reshape(-1, 1, 2).copy()

    try:
        refined_valid = cv2.cornerSubPix(
            image=img_uint8,
            corners=valid_pts,
            winSize=(int(win_w), int(win_h)),
            zeroZone=(int(zero_zone[0]), int(zero_zone[1])),
            criteria=term_criteria,
        )
        refined_pts = refined_valid.reshape(-1, 2)

        # Sanity check: keep only finite refined points within image bounds
        finite_mask = (
            np.isfinite(refined_pts[:, 0])
            & np.isfinite(refined_pts[:, 1])
            & (refined_pts[:, 0] >= 0)
            & (refined_pts[:, 0] <= w - 1)
            & (refined_pts[:, 1] >= 0)
            & (refined_pts[:, 1] <= h - 1)
        )

        # Update points that were successfully refined
        indices = np.where(valid_mask)[0]
        pts[indices[finite_mask]] = refined_pts[finite_mask]
    except cv2.error:
        # Graceful fallback to unrefined points on unexpected OpenCV error
        pass

    return pts.astype(np.float32)
