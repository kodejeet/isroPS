"""Image array normalization utilities."""

import numpy as np


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize input numeric array to 8-bit uint8 range [0, 255].

    Handles 16-bit raster or floating point reflectance values safely.
    """
    if array.dtype == np.uint8:
        return array.copy()

    arr = array.astype(np.float32)
    min_val, max_val = np.min(arr), np.max(arr)

    if max_val > min_val:
        normalized = (arr - min_val) / (max_val - min_val) * 255.0
    else:
        normalized = np.zeros_like(arr)

    return np.clip(normalized, 0, 255).astype(np.uint8)


def to_grayscale(array: np.ndarray) -> np.ndarray:
    """Convert (H, W, C) image array into 2D single-channel grayscale array.

    For hyperspectral/multi-channel cubes (C > 3), selects the first band.
    """
    arr_8u = normalize_to_uint8(array)

    if arr_8u.ndim == 2:
        return arr_8u
    elif arr_8u.ndim == 3:
        if arr_8u.shape[2] == 1:
            return arr_8u[:, :, 0]
        elif arr_8u.shape[2] == 3:
            import cv2

            return cv2.cvtColor(arr_8u, cv2.COLOR_RGB2GRAY)
        elif arr_8u.shape[2] == 4:
            import cv2

            return cv2.cvtColor(arr_8u[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            # Hyperspectral cube: select band 0
            return arr_8u[:, :, 0]
    return arr_8u
