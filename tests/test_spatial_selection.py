"""Unit tests for spatial match selection module."""

import numpy as np

from lunar_correspondence.io.metadata import MatchSet
from lunar_correspondence.matching.spatial_selection import select_spatial_matches


def test_spatial_selection_empty():
    """Empty MatchSet should return empty MatchSet with preserved shapes."""
    empty_ms = MatchSet(
        source_points=np.zeros((0, 2), dtype=np.float32),
        reference_points=np.zeros((0, 2), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
    )
    result = select_spatial_matches(
        empty_ms, image_shape=(512, 512), grid_rows=8, grid_cols=8, top_k=4
    )
    assert len(result.source_points) == 0
    assert len(result.reference_points) == 0
    assert result.source_points.shape == (0, 2)
    assert result.reference_points.shape == (0, 2)


def test_spatial_selection_single_cell_top_k():
    """When a single grid cell contains 10 matches, only top_k=4 highest confidence are kept."""
    # 512x512 image, 8x8 grid -> each cell is 64x64.
    # Put 10 points in cell (0, 0): x in [10..30], y in [10..30].
    pts_src = np.array([[10.0 + i, 10.0 + i] for i in range(10)], dtype=np.float32)
    pts_ref = pts_src + 5.0
    # Confidences 0.1 to 1.0
    conf = np.array([0.1 * (i + 1) for i in range(10)], dtype=np.float32)

    match_set = MatchSet(
        source_points=pts_src, reference_points=pts_ref, confidence=conf
    )

    result = select_spatial_matches(
        match_set, image_shape=(512, 512), grid_rows=8, grid_cols=8, top_k=4
    )

    assert len(result.source_points) == 4
    assert len(result.reference_points) == 4
    assert len(result.confidence) == 4
    # The 4 selected matches must have the highest confidence (0.7, 0.8, 0.9, 1.0)
    expected_conf = {0.7, 0.8, 0.9, 1.0}
    actual_conf = {round(float(c), 2) for c in result.confidence}
    assert actual_conf == expected_conf


def test_spatial_selection_multi_cell_distribution():
    """Matches across multiple grid cells should each retain up to top_k."""
    # Place 3 points in cell (0, 0), 6 points in cell (1, 1), 2 points in cell (7, 7)
    # Cell size for 512x512 with 8x8 is 64x64.
    c0_src = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
    c1_src = np.array([[70.0 + i, 70.0 + i] for i in range(6)], dtype=np.float32)
    c7_src = np.array([[480.0, 480.0], [490.0, 490.0]], dtype=np.float32)

    pts_src = np.vstack([c0_src, c1_src, c7_src])
    pts_ref = pts_src.copy()
    conf = np.ones(len(pts_src), dtype=np.float32) * 0.9

    match_set = MatchSet(
        source_points=pts_src, reference_points=pts_ref, confidence=conf
    )

    result = select_spatial_matches(
        match_set, image_shape=(512, 512), grid_rows=8, grid_cols=8, top_k=4
    )

    # c0: 3 kept (< 4)
    # c1: 4 kept (6 pruned to 4)
    # c7: 2 kept (< 4)
    # Total = 3 + 4 + 2 = 9
    assert len(result.source_points) == 9
    assert len(result.reference_points) == 9


def test_spatial_selection_coordinate_convention():
    """Verify (x, y) = (col, row) coordinate bucketing convention."""
    # Point at x=400 (col ~ 6 in 8 cols of 64px), y=20 (row ~ 0 in 8 rows of 64px)
    # In (x, y) = (col, row): col_idx = 400 // 64 = 6, row_idx = 20 // 64 = 0
    pts_src = np.array([[400.0, 20.0]], dtype=np.float32)
    pts_ref = pts_src.copy()
    ms = MatchSet(source_points=pts_src, reference_points=pts_ref)

    res = select_spatial_matches(
        ms, image_shape=(512, 512), grid_rows=8, grid_cols=8, top_k=4
    )
    assert len(res.source_points) == 1
    np.testing.assert_allclose(res.source_points[0], [400.0, 20.0])


def test_spatial_selection_improves_coverage_on_clustered_distribution():
    """Controlled test proving spatial selection improves spatial coverage when matches are clustered.

    Scenario:
    - 80 matches are concentrated in a single cell (top-left, x in [10..30], y in [10..30])
      with very high confidence (0.95 to 0.99).
    - 15 matches are distributed across 5 other distinct grid cells with slightly lower confidence (0.70 to 0.80).
    - An ungated top-20 selection by confidence would be 100% trapped in the single top-left cell (Coverage = 6.25%).
    - With spatial selection (8x8 grid, top_k=4), the cluster is capped at 4 matches, and matches
      across all 6 cells are retained (Coverage = 6/16 = 37.5%), proving spatial coverage improvement.
    """
    from lunar_correspondence.evaluation.metrics import compute_grid_coverage

    img_shape = (512, 512)

    # 1. Clustered matches in cell (0, 0)
    clustered_src = np.array(
        [[15.0 + (i % 20), 15.0 + (i // 20)] for i in range(80)], dtype=np.float32
    )
    clustered_conf = np.linspace(0.95, 0.99, 80, dtype=np.float32)

    # 2. Dispersed matches across 5 distinct cells: (2, 2), (2, 5), (5, 2), (5, 5), (7, 7)
    # Cell size for 8x8 is 64x64.
    dispersed_coords = [
        [2 * 64 + 32.0, 2 * 64 + 32.0],
        [5 * 64 + 32.0, 2 * 64 + 32.0],
        [2 * 64 + 32.0, 5 * 64 + 32.0],
        [5 * 64 + 32.0, 5 * 64 + 32.0],
        [7 * 64 + 32.0, 7 * 64 + 32.0],
    ]
    dispersed_src_list = []
    dispersed_conf_list = []
    for coord in dispersed_coords:
        for _ in range(3):  # 3 matches per dispersed cell = 15 matches total
            dispersed_src_list.append([coord[0] + np.random.uniform(-5, 5), coord[1] + np.random.uniform(-5, 5)])
            dispersed_conf_list.append(0.75)

    dispersed_src = np.array(dispersed_src_list, dtype=np.float32)
    dispersed_conf = np.array(dispersed_conf_list, dtype=np.float32)

    # Combine into full candidate match set (95 matches total)
    all_src = np.vstack([clustered_src, dispersed_src])
    all_ref = all_src.copy()
    all_conf = np.concatenate([clustered_conf, dispersed_conf])

    candidate_ms = MatchSet(
        source_points=all_src,
        reference_points=all_ref,
        confidence=all_conf,
    )

    # Ungated baseline: if a downstream budget selects top 20 by confidence
    top20_ungated_indices = np.argsort(all_conf)[::-1][:20]
    ungated_pts = all_src[top20_ungated_indices]
    ungated_coverage = compute_grid_coverage(
        inlier_points_xy=ungated_pts,
        image_shape=img_shape,
        grid_rows=4,
        grid_cols=4,
    )

    # Spatial selection applied:
    selected_ms = select_spatial_matches(
        match_set=candidate_ms,
        image_shape=img_shape,
        grid_rows=8,
        grid_cols=8,
        top_k=4,
    )
    # If downstream selects top 20 from spatially filtered set
    sel_conf = selected_ms.confidence
    top20_sel_indices = np.argsort(sel_conf)[::-1][:20]
    selected_pts = selected_ms.source_points[top20_sel_indices]
    selected_coverage = compute_grid_coverage(
        inlier_points_xy=selected_pts,
        image_shape=img_shape,
        grid_rows=4,
        grid_cols=4,
    )

    # Assert that spatial selection demonstrably improved coverage
    assert selected_coverage > ungated_coverage
    assert ungated_coverage == 6.25  # 1 out of 16 cells
    assert selected_coverage >= 37.5  # 6 out of 16 cells (37.5%)
