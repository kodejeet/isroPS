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

    # Check if transformation is affine or near-affine
    is_affine = (
        abs(H[2, 0]) < 1e-6 and abs(H[2, 1]) < 1e-6 and abs(H[2, 2] - 1.0) < 1e-4
    )

    if not is_affine:
        # Validate that the denominator does not cross zero or cause singularity rays
        h33 = H[2, 2] if abs(H[2, 2]) > 1e-9 else 1.0
        h_row = H[2, :] / h33
        corners = np.array(
            [[0, 0, 1], [target_w, 0, 1], [0, target_h, 1], [target_w, target_h, 1]],
            dtype=np.float32,
        )
        denoms = corners @ h_row
        if np.any(denoms <= 0.15) or np.any(denoms >= 8.0):
            # Degenerate perspective warp detected: fall back to affine portion to guarantee no fan-line explosion
            is_affine = True

    if is_affine:
        M = H[:2, :].astype(np.float32)
        if image_array.ndim == 3:
            channels = [
                cv2.warpAffine(
                    image_array[:, :, c],
                    M,
                    (target_w, target_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                for c in range(image_array.shape[2])
            ]
            return np.stack(channels, axis=-1)
        else:
            return cv2.warpAffine(
                image_array,
                M,
                (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    else:
        H_32 = H.astype(np.float32)
        if image_array.ndim == 3:
            channels = [
                cv2.warpPerspective(
                    image_array[:, :, c],
                    H_32,
                    (target_w, target_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                for c in range(image_array.shape[2])
            ]
            return np.stack(channels, axis=-1)
        else:
            return cv2.warpPerspective(
                image_array,
                H_32,
                (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
