"""Along-track block-based geometric estimation module for pushbroom sensors.

Partitions elongated along-track point correspondences into overlapping sub-blocks,
fits robust local models (affine or homography) per block, verifies inter-block
geometric continuity, and provides smooth blended coordinate projection.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from lunar_correspondence.geometry.homography import compute_reprojection_errors
from lunar_correspondence.geometry.ransac import estimate_geometric_model
from lunar_correspondence.io.metadata import GeometricModel, MatchSet


@dataclass
class BlockTransform:
    """Transformation model and metadata for a single along-track block."""

    block_index: int
    y_start: float
    y_end: float
    transform_matrix: np.ndarray  # 3x3 homogeneous matrix
    model_type: str
    inlier_count: int
    inlier_mask: np.ndarray
    rmse_pixels: float | None
    is_valid: bool = True


@dataclass
class BlockGeometryModel:
    """Composite along-track piecewise geometric model."""

    blocks: list[BlockTransform]
    block_length: int
    block_overlap: int
    is_continuous: bool
    total_inliers: int
    inlier_mask: np.ndarray
    reprojection_errors: np.ndarray
    continuity_warnings: list[str] = field(default_factory=list)


def estimate_block_geometry(
    match_set: MatchSet,
    block_length: int = 1500,
    block_overlap: int = 300,
    local_model: str = "affine",
    reproj_threshold: float = 3.5,
    max_iters: int = 2000,
    max_translation_jump: float = 25.0,
    max_scale_diff: float = 0.25,
) -> BlockGeometryModel:
    """Fit piecewise along-track models with continuity verification across overlapping blocks.

    Args:
        match_set: MatchSet containing source_points (N, 2) and reference_points (N, 2).
        block_length: Scan-line length (y-axis) per along-track block in pixels.
        block_overlap: Along-track overlap between adjacent blocks in pixels.
        local_model: Model type for each block ("affine" or "homography").
        reproj_threshold: RANSAC inlier threshold in pixels.
        max_iters: Maximum RANSAC iterations per block.
        max_translation_jump: Maximum permissible translation jump (pixels) between neighbors.
        max_scale_diff: Maximum permissible determinant scale difference between neighbors.

    Returns:
        BlockGeometryModel containing per-block transforms and composite metrics.
    """
    src_pts = match_set.source_points
    dst_pts = match_set.reference_points
    n_pts = len(src_pts)

    if n_pts == 0:
        return BlockGeometryModel(
            blocks=[],
            block_length=block_length,
            block_overlap=block_overlap,
            is_continuous=False,
            total_inliers=0,
            inlier_mask=np.zeros(0, dtype=bool),
            reprojection_errors=np.zeros(0, dtype=np.float32),
            continuity_warnings=["NO_POINTS_AVAILABLE"],
        )

    max_y = float(np.max(src_pts[:, 1]))
    step = max(100, block_length - block_overlap)
    y_starts = list(range(0, int(max_y) + 1, step)) if max_y > 0 else [0]

    blocks: list[BlockTransform] = []
    global_inlier_mask = np.zeros(n_pts, dtype=bool)
    global_errors = np.full(n_pts, np.nan, dtype=np.float32)

    min_pts_required = 4 if local_model == "homography" else 3

    for idx, y0 in enumerate(y_starts):
        y1 = y0 + block_length
        pt_indices = np.where((src_pts[:, 1] >= y0) & (src_pts[:, 1] < y1))[0]

        if len(pt_indices) < min_pts_required:
            blocks.append(
                BlockTransform(
                    block_index=idx,
                    y_start=float(y0),
                    y_end=float(y1),
                    transform_matrix=np.eye(3, dtype=np.float32),
                    model_type=local_model,
                    inlier_count=0,
                    inlier_mask=np.zeros(len(pt_indices), dtype=bool),
                    rmse_pixels=None,
                    is_valid=False,
                )
            )
            continue

        b_src = src_pts[pt_indices]
        b_dst = dst_pts[pt_indices]
        b_match = MatchSet(source_points=b_src, reference_points=b_dst)

        geo = estimate_geometric_model(
            b_match,
            model_type=local_model,
            reproj_threshold=reproj_threshold,
            max_iters=max_iters,
        )

        inliers = int(np.sum(geo.inlier_mask)) if geo.inlier_mask is not None else 0
        if inliers >= min_pts_required:
            local_errs = geo.reprojection_errors[geo.inlier_mask]
            b_rmse = float(np.sqrt(np.mean(local_errs**2))) if len(local_errs) > 0 else None

            # Mark global inliers
            inlier_indices = pt_indices[geo.inlier_mask]
            global_inlier_mask[inlier_indices] = True
            global_errors[pt_indices] = geo.reprojection_errors

            blocks.append(
                BlockTransform(
                    block_index=idx,
                    y_start=float(y0),
                    y_end=float(y1),
                    transform_matrix=geo.transform_matrix,
                    model_type=local_model,
                    inlier_count=inliers,
                    inlier_mask=geo.inlier_mask,
                    rmse_pixels=b_rmse,
                    is_valid=True,
                )
            )
        else:
            blocks.append(
                BlockTransform(
                    block_index=idx,
                    y_start=float(y0),
                    y_end=float(y1),
                    transform_matrix=np.eye(3, dtype=np.float32),
                    model_type=local_model,
                    inlier_count=0,
                    inlier_mask=np.zeros(len(pt_indices), dtype=bool),
                    rmse_pixels=None,
                    is_valid=False,
                )
            )

    # Continuity checks between consecutive valid blocks
    continuity_warnings: list[str] = []
    valid_blocks = [b for b in blocks if b.is_valid]

    is_continuous = True
    if len(valid_blocks) >= 2:
        for i in range(len(valid_blocks) - 1):
            b1 = valid_blocks[i]
            b2 = valid_blocks[i + 1]

            t1 = b1.transform_matrix[:2, 2]
            t2 = b2.transform_matrix[:2, 2]
            trans_drift = float(np.linalg.norm(t2 - t1))

            a1 = b1.transform_matrix[:2, :2]
            a2 = b2.transform_matrix[:2, :2]
            scale1 = float(np.sqrt(abs(np.linalg.det(a1)))) if abs(np.linalg.det(a1)) > 0 else 1.0
            scale2 = float(np.sqrt(abs(np.linalg.det(a2)))) if abs(np.linalg.det(a2)) > 0 else 1.0
            scale_diff = abs(scale2 - scale1)

            if trans_drift > max_translation_jump:
                continuity_warnings.append(
                    f"BLOCK_DRIFT: Translation jump {trans_drift:.1f}px between block {b1.block_index} and {b2.block_index}"
                )
                is_continuous = False

            if scale_diff > max_scale_diff:
                continuity_warnings.append(
                    f"SCALE_DISCONTINUITY: Scale difference {scale_diff:.2f} between block {b1.block_index} and {b2.block_index}"
                )
                is_continuous = False
    elif len(valid_blocks) == 0:
        is_continuous = False
        continuity_warnings.append("NO_VALID_BLOCKS")

    total_inliers = int(np.sum(global_inlier_mask))

    return BlockGeometryModel(
        blocks=blocks,
        block_length=block_length,
        block_overlap=block_overlap,
        is_continuous=is_continuous,
        total_inliers=total_inliers,
        inlier_mask=global_inlier_mask,
        reprojection_errors=global_errors,
        continuity_warnings=continuity_warnings,
    )


def project_points_block_geometry(
    points_xy: np.ndarray,
    model: BlockGeometryModel,
) -> np.ndarray:
    """Project source (x, y) coordinates into reference coordinates using smooth blended block models.

    Args:
        points_xy: (N, 2) source points in (x, y) coordinates.
        model: Fitted BlockGeometryModel.

    Returns:
        (N, 2) projected points in reference coordinates.
    """
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    valid_blocks = [b for b in model.blocks if b.is_valid]
    if not valid_blocks:
        # Fallback identity
        return points_xy.copy().astype(np.float32)

    if len(valid_blocks) == 1:
        # Single valid block: apply directly
        H = valid_blocks[0].transform_matrix
        pts_h = np.hstack([points_xy, np.ones((len(points_xy), 1), dtype=np.float32)])
        proj_h = (H @ pts_h.T).T
        proj = proj_h[:, :2] / np.maximum(proj_h[:, 2:], 1e-8)
        return proj.astype(np.float32)

    # Multi-block: for each point, blend adjacent blocks if in overlap zone
    projected = np.zeros_like(points_xy, dtype=np.float32)

    for i, (x, y) in enumerate(points_xy):
        pt_h = np.array([x, y, 1.0], dtype=np.float32)

        # Find candidate blocks containing y
        containing = [b for b in valid_blocks if b.y_start <= y <= b.y_end]

        if not containing:
            # Nearest block
            dists = [abs(y - 0.5 * (b.y_start + b.y_end)) for b in valid_blocks]
            best_b = valid_blocks[int(np.argmin(dists))]
            p_h = best_b.transform_matrix @ pt_h
            projected[i] = p_h[:2] / max(p_h[2], 1e-8)
        elif len(containing) == 1:
            p_h = containing[0].transform_matrix @ pt_h
            projected[i] = p_h[:2] / max(p_h[2], 1e-8)
        else:
            # Blend two overlapping blocks linearly along y
            b1, b2 = containing[0], containing[1]
            overlap_start = b2.y_start
            overlap_end = b1.y_end
            span = max(1.0, overlap_end - overlap_start)
            w2 = np.clip((y - overlap_start) / span, 0.0, 1.0)
            w1 = 1.0 - w2

            p1_h = b1.transform_matrix @ pt_h
            p1 = p1_h[:2] / max(p1_h[2], 1e-8)

            p2_h = b2.transform_matrix @ pt_h
            p2 = p2_h[:2] / max(p2_h[2], 1e-8)

            projected[i] = w1 * p1 + w2 * p2

    return projected
