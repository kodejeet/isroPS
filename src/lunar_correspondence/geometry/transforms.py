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
    # Check if geometric model has block-based along-track decomposition
    block_model = getattr(geometric_model, "block_model", None)
    if block_model is not None and getattr(block_model, "is_continuous", False) and len(block_model.blocks) > 1:
        valid_blocks = [b for b in block_model.blocks if b.is_valid]
        if len(valid_blocks) > 1:
            try:
                return _warp_block_geometry(image_array, valid_blocks, output_shape)
            except Exception:
                # Fall back to global matrix
                pass

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


def _warp_block_geometry(
    image_array: np.ndarray,
    valid_blocks: list[Any],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Piecewise along-track warping and blending across consecutive blocks."""
    target_h, target_w = output_shape
    is_3d = image_array.ndim == 3
    num_channels = image_array.shape[2] if is_3d else 1

    accum = np.zeros((target_h, target_w, num_channels), dtype=np.float32)
    weights = np.zeros((target_h, target_w, 1), dtype=np.float32)

    for block in valid_blocks:
        y0 = int(round(block.y_start))
        y1 = min(image_array.shape[0], int(round(block.y_end)))
        if y1 <= y0:
            continue

        block_sub = image_array[y0:y1, :]
        bh, bw = block_sub.shape[:2]
        if bh == 0 or bw == 0:
            continue

        # Sub-image coordinates: y_global = y_local + y0
        # Transformation: H_local = H_global @ [[1, 0, 0], [0, 1, y0], [0, 0, 1]]
        T_offset = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]], dtype=np.float32)
        H_local = (block.transform_matrix @ T_offset).astype(np.float32)

        # Create feathering weights along along-track dimension
        feather = np.ones((bh, bw, 1), dtype=np.float32)
        margin = min(100, bh // 4)
        if margin > 0:
            ramp_up = np.linspace(0.0, 1.0, margin, dtype=np.float32)[:, np.newaxis, np.newaxis]
            ramp_down = np.linspace(1.0, 0.0, margin, dtype=np.float32)[:, np.newaxis, np.newaxis]
            feather[:margin, :, :] = np.minimum(feather[:margin, :, :], ramp_up)
            feather[bh - margin :, :, :] = np.minimum(feather[bh - margin :, :, :], ramp_down)

        M = H_local[:2, :].astype(np.float32)
        # Warp sub-image
        if is_3d:
            warped_sub = np.stack(
                [
                    cv2.warpAffine(block_sub[:, :, c].astype(np.float32), M, (target_w, target_h), flags=cv2.INTER_LINEAR)
                    for c in range(num_channels)
                ],
                axis=-1,
            )
        else:
            warped_2d = cv2.warpAffine(block_sub.astype(np.float32), M, (target_w, target_h), flags=cv2.INTER_LINEAR)
            warped_sub = warped_2d[:, :, np.newaxis]

        warped_w = cv2.warpAffine(feather[:, :, 0], M, (target_w, target_h), flags=cv2.INTER_LINEAR)[:, :, np.newaxis]

        accum += warped_sub * warped_w
        weights += warped_w

    valid_mask = weights > 1e-4
    np.divide(accum, weights, out=accum, where=valid_mask)

    if np.issubdtype(image_array.dtype, np.integer):
        accum = np.clip(accum, 0, 255).astype(image_array.dtype)
    else:
        accum = accum.astype(image_array.dtype)

    return accum if is_3d else accum[:, :, 0]

