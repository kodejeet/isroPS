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


def create_feature_image(
    array: np.ndarray,
    method: str = "percentile",
    p_low: float = 0.5,
    p_high: float = 99.5,
    mask_negatives: bool = False,
    apply_clahe: bool = False,
    clahe_clip_limit: float = 3.0,
    clahe_grid_size: tuple[int, int] = (16, 16),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 2D uint8 feature extraction image and a boolean validity mask.

    Preserves the input array untouched.

    Args:
        array: Input raster array of shape (H, W) or (H, W, C).
        method: Normalization method ("percentile", "minmax", "zscore", "log").
        p_low: Lower percentile cutoff (e.g. 0.5%).
        p_high: Upper percentile cutoff (e.g. 99.5%).
        mask_negatives: If True, treats values < 0 as invalid in the mask.
        apply_clahe: If True, applies CLAHE to the normalized 8-bit result.
        clahe_clip_limit: CLAHE clip limit.
        clahe_grid_size: CLAHE tile grid size (rows, cols).

    Returns:
        Tuple of (feature_image_uint8, validity_mask_bool).
    """
    import cv2

    # Extract 2D single band if 3D
    if array.ndim == 2:
        band_2d = array
    elif array.ndim == 3:
        if array.shape[2] == 1:
            band_2d = array[:, :, 0]
        elif array.shape[2] in [3, 4] and array.dtype == np.uint8:
            band_2d = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            band_2d = array[:, :, 0]
    else:
        raise ValueError(f"Unsupported array dimensions: {array.ndim}")

    arr_f = band_2d.astype(np.float32)

    # Establish validity mask
    valid = np.isfinite(arr_f)
    if mask_negatives:
        valid = valid & (arr_f >= 0)

    if not np.any(valid):
        return np.zeros(arr_f.shape, dtype=np.uint8), valid

    valid_vals = arr_f[valid]

    if method == "minmax":
        min_v = float(np.min(valid_vals))
        max_v = float(np.max(valid_vals))
        if max_v > min_v:
            norm = np.clip((arr_f - min_v) / (max_v - min_v) * 255.0, 0, 255)
        else:
            norm = np.zeros_like(arr_f)
    elif method == "zscore":
        mean_v = float(np.mean(valid_vals))
        std_v = float(np.std(valid_vals))
        std_v = std_v if std_v > 1e-6 else 1.0
        # Map [mean - 2.5*std, mean + 2.5*std] to [0, 255]
        norm = np.clip((arr_f - (mean_v - 2.5 * std_v)) / (5.0 * std_v) * 255.0, 0, 255)
    elif method == "log":
        min_v = float(np.min(valid_vals))
        shifted = np.maximum(arr_f - min_v + 1.0, 1.0)
        log_vals = np.log(shifted)
        valid_log = log_vals[valid]
        p1, p2 = np.percentile(valid_log, [p_low, p_high])
        if p2 > p1:
            norm = np.clip((log_vals - p1) / (p2 - p1) * 255.0, 0, 255)
        else:
            norm = np.zeros_like(arr_f)
    else:  # Default: "percentile"
        p1, p2 = np.percentile(valid_vals, [p_low, p_high])
        if p2 > p1:
            norm = np.clip((arr_f - p1) / (p2 - p1) * 255.0, 0, 255)
        else:
            norm = np.zeros_like(arr_f)

    res_8u = norm.astype(np.uint8)

    # Optional CLAHE on feature image
    if apply_clahe:
        grid_r = min(clahe_grid_size[0], max(1, res_8u.shape[0] // 8))
        grid_c = min(clahe_grid_size[1], max(1, res_8u.shape[1] // 8))
        grid_r = max(2, grid_r)
        grid_c = max(2, grid_c)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(grid_c, grid_r))
        res_8u = clahe.apply(res_8u)

    return res_8u, valid


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

