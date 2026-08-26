"""Image grid tiling utilities for handling ultra-large satellite rasters."""

import numpy as np


def generate_tiles(
    array: np.ndarray, tile_size: tuple[int, int] = (512, 512), overlap: int = 64
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Generate sub-image tiles with spatial overlap for processing high-resolution scenes (e.g. OHRC).

    Args:
        array: Input image array (H, W, C).
        tile_size: (tile_height, tile_width).
        overlap: Border overlap in pixels between adjacent tiles.

    Returns:
        List of tuples: ((min_y, min_x, max_y, max_x), tile_array).
    """
    h, w = array.shape[:2]
    th, tw = tile_size
    tiles = []

    y_step = th - overlap
    x_step = tw - overlap

    for y in range(0, h, y_step):
        for x in range(0, w, x_step):
            y_end = min(y + th, h)
            x_end = min(x + tw, w)
            tile_crop = array[y:y_end, x:x_end]
            tiles.append(((y, x, y_end, x_end), tile_crop))
            if x_end == w:
                break
        if y_end == h:
            break

    return tiles
