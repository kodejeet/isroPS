"""Feature correspondence match line visualization utilities."""

import os

import cv2
import numpy as np

from lunar_correspondence.io.metadata import MatchSet
from lunar_correspondence.preprocessing.normalization import (
    to_grayscale,
)


def draw_match_lines(
    image_src: np.ndarray,
    image_ref: np.ndarray,
    match_set: MatchSet,
    max_matches_to_show: int = 100,
    output_path: str | None = None,
) -> np.ndarray:
    """Draw side-by-side match line visualization showing inlier and outlier feature correspondences.

    CRITICAL CONVENTION:
    match_set points strictly follow (x, y) = (col, row) coordinates.
    Inliers are drawn in bright green lines; outliers are drawn in red lines.

    Args:
        image_src: Source image array.
        image_ref: Reference image array.
        match_set: MatchSet containing source_points, reference_points, and inlier_mask.
        max_matches_to_show: Maximum match lines to render to avoid clutter.
        output_path: Optional path to save figure file.

    Returns:
        RGB numpy array of side-by-side visualization canvas.
    """
    gray_src = to_grayscale(image_src)
    gray_ref = to_grayscale(image_ref)

    h1, w1 = gray_src.shape[:2]
    h2, w2 = gray_ref.shape[:2]

    # Create canvas side-by-side
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Fill left and right panels
    canvas[:h1, :w1] = cv2.cvtColor(gray_src, cv2.COLOR_GRAY2RGB)
    canvas[:h2, w1 : w1 + w2] = cv2.cvtColor(gray_ref, cv2.COLOR_GRAY2RGB)

    pts_src = match_set.source_points
    pts_ref = match_set.reference_points
    inliers = match_set.inlier_mask

    if inliers is None:
        inliers = np.ones(len(pts_src), dtype=bool)

    num_matches = len(pts_src)
    if num_matches > max_matches_to_show:
        # Subsample indices evenly
        indices = np.linspace(0, num_matches - 1, max_matches_to_show, dtype=int)
    else:
        indices = np.arange(num_matches)

    for idx in indices:
        x1, y1 = int(round(pts_src[idx][0])), int(round(pts_src[idx][1]))
        x2, y2 = int(round(pts_ref[idx][0])) + w1, int(round(pts_ref[idx][1]))

        is_inlier = bool(inliers[idx])
        color = (
            (0, 255, 0) if is_inlier else (255, 0, 0)
        )  # Green for inliers, Red for outliers
        thickness = 2 if is_inlier else 1

        cv2.circle(canvas, (x1, y1), 4, color, -1)
        cv2.circle(canvas, (x2, y2), 4, color, -1)
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return canvas
