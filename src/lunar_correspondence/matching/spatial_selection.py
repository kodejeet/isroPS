"""Spatial Match Selection module.

Divides the source image plane into a uniform grid (default 8x8) and selects
the top-K (default 4) most confident candidate matches per cell prior to RANSAC.
This prevents spatial clustering of correspondences in high-texture areas and ensures
broad geometric constraint distribution across the entire lunar surface canvas.

CRITICAL CONVENTION:
All 2D point coordinates strictly follow (x, y) = (column, row).
x: 0 to width - 1 (columns)
y: 0 to height - 1 (rows)
"""

from collections import defaultdict

import numpy as np

from lunar_correspondence.io.metadata import MatchSet


def select_spatial_matches(
    match_set: MatchSet,
    image_shape: tuple[int, int],
    grid_rows: int = 8,
    grid_cols: int = 8,
    top_k: int = 4,
) -> MatchSet:
    """Select a spatially distributed subset of candidate feature matches.

    Divides the canvas into grid_rows x grid_cols equal cells based on source
    point (x, y) coordinates, sorts matches within each cell by descending confidence,
    and retains at most top_k matches per cell.

    Args:
        match_set: Candidate MatchSet containing source and reference points in (x, y).
        image_shape: (height, width) tuple of the source canvas.
        grid_rows: Number of grid cell rows (default 8).
        grid_cols: Number of grid cell columns (default 8).
        top_k: Maximum number of matches to retain per cell (default 4).

    Returns:
        Filtered MatchSet preserving data contracts and (x, y) coordinates.
    """
    pts_src = match_set.source_points
    pts_ref = match_set.reference_points
    n_matches = len(pts_src)

    if n_matches == 0:
        return MatchSet(
            source_points=np.zeros((0, 2), dtype=np.float32),
            reference_points=np.zeros((0, 2), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32)
            if match_set.confidence is not None
            else None,
            inlier_mask=np.zeros((0,), dtype=bool)
            if match_set.inlier_mask is not None
            else None,
        )

    height, width = image_shape
    cell_h = max(1.0, height / float(grid_rows))
    cell_w = max(1.0, width / float(grid_cols))

    conf = match_set.confidence

    # Map each match index into its grid cell (row_idx, col_idx)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)

    for i in range(n_matches):
        x, y = pts_src[i, 0], pts_src[i, 1]
        col_idx = int(np.clip(x / cell_w, 0, grid_cols - 1))
        row_idx = int(np.clip(y / cell_h, 0, grid_rows - 1))
        cells[(row_idx, col_idx)].append(i)

    selected_indices: list[int] = []

    # Sort each cell's matches by confidence (descending) and retain top_k
    for cell_key in sorted(cells.keys()):
        idx_list = cells[cell_key]
        if conf is not None and len(conf) == n_matches:
            # Sort by descending confidence score
            idx_list.sort(key=lambda idx: conf[idx], reverse=True)
        # Retain top_k
        selected_indices.extend(idx_list[:top_k])

    selected_indices.sort()  # Keep deterministic index order
    sel_arr = np.array(selected_indices, dtype=np.int64)

    sel_src = pts_src[sel_arr]
    sel_ref = pts_ref[sel_arr]
    sel_conf = conf[sel_arr] if conf is not None else None
    sel_mask = (
        match_set.inlier_mask[sel_arr]
        if match_set.inlier_mask is not None
        else None
    )

    return MatchSet(
        source_points=sel_src.astype(np.float32),
        reference_points=sel_ref.astype(np.float32),
        confidence=sel_conf.astype(np.float32) if sel_conf is not None else None,
        inlier_mask=sel_mask.astype(bool) if sel_mask is not None else None,
    )
