"""Image geometric transformation and un-warping functions."""

import cv2
import numpy as np

from lunar_correspondence.io.metadata import GeometricModel


def warp_image(
    image_array: np.ndarray,
    geometric_model: GeometricModel,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp source image array into reference image coordinate frame using GeometricModel.

    Args:
        image_array: Source image array (H, W, C) or (H, W).
        geometric_model: Estimated homography or affine transformation matrix.
        output_shape: Target canvas dimensions (height, width).

    Returns:
        Warped image array aligned to reference frame.
    """
    H = geometric_model.transform_matrix
    target_h, target_w = output_shape

    if image_array.ndim == 3:
        # Multi-channel array: warp each channel independently or collectively
        channels = [
            cv2.warpPerspective(image_array[:, :, c], H, (target_w, target_h))
            for c in range(image_array.shape[2])
        ]
        return np.stack(channels, axis=-1)
    else:
        return cv2.warpPerspective(image_array, H, (target_w, target_h))
