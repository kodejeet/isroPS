"""RANSAC robust transform matrix estimation.

Gloss:
- RANSAC: Random Sample Consensus - iterative algorithm that estimates a geometric transformation model
  while filtering out outlier matches caused by shadow artifacts or texture repetition.

CRITICAL CONVENTION:
pts_src and pts_ref are shape (N, 2), enforcing (x, y) = (col, row) coordinates.
"""

import cv2
import numpy as np

from lunar_correspondence.geometry.homography import compute_reprojection_errors
from lunar_correspondence.io.metadata import GeometricModel, MatchSet


def is_valid_homography(
    H: np.ndarray | None,
    domain_size: tuple[int, int] = (512, 512),
) -> bool:
    """Validate that homography H does not contain projective singularities or extreme distortion."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    if abs(H[2, 2]) < 1e-9:
        return False

    H_norm = H / H[2, 2]
    w, h = domain_size
    corners = np.array(
        [[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1], [w / 2, h / 2, 1]],
        dtype=np.float32,
    )
    denoms = corners @ H_norm[2, :]
    if np.any(denoms <= 0.15) or np.any(denoms >= 8.0):
        return False

    det = np.linalg.det(H_norm[:2, :2])
    if det <= 0.02 or det >= 50.0:
        return False

    return True


def estimate_geometric_model(
    match_set: MatchSet,
    model_type: str = "homography",
    reproj_threshold: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.99,
    random_seed: int | None = 42,
) -> GeometricModel:
    """Estimate robust GeometricModel (Homography or Affine) using RANSAC.

    Args:
        match_set: MatchSet containing source_points and reference_points in (x, y) coordinates.
        model_type: "homography" or "affine".
        reproj_threshold: Maximum allowable reprojection error in pixels for RANSAC inliers.
        max_iters: Maximum RANSAC iteration count.
        confidence: Desired RANSAC confidence level (0.0 to 1.0).
        random_seed: Optional seed for reproducible RANSAC sampling.

    Returns:
        GeometricModel instance with transform matrix, inlier mask, and reprojection errors.
    """
    pts_src = match_set.source_points
    pts_ref = match_set.reference_points

    if random_seed is not None:
        cv2.setRNGSeed(random_seed)
        np.random.seed(random_seed)

    if len(pts_src) < 4:
        # Insufficient points for geometric estimation
        return GeometricModel(
            transform_matrix=np.eye(3, dtype=np.float32),
            model_type=model_type,
            inlier_mask=np.zeros(len(pts_src), dtype=bool),
            reprojection_errors=np.zeros(len(pts_src), dtype=np.float32),
        )

    if model_type.lower() == "affine":
        matrix, mask = cv2.estimateAffine2D(
            pts_src,
            pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
        if matrix is None:
            H = np.eye(3, dtype=np.float32)
            errors = compute_reprojection_errors(pts_src, pts_ref, H)
            inlier_mask = np.zeros(len(pts_src), dtype=bool)
        else:
            H = np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float32)
            errors = compute_reprojection_errors(pts_src, pts_ref, H)
            inlier_mask = (
                mask.ravel().astype(bool)
                if mask is not None
                else (errors <= reproj_threshold)
            )
    else:
        # Default Homography estimation with Affine fallback
        H, mask = cv2.findHomography(
            pts_src,
            pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )

        h_valid = is_valid_homography(H)
        h_inliers = mask.sum() if (mask is not None and h_valid) else 0

        # If homography is degenerate or poorly supported (<6 inliers), try affine
        if (not h_valid or h_inliers < 6) and len(pts_src) >= 3:
            matrix, aff_mask = cv2.estimateAffine2D(
                pts_src,
                pts_ref,
                method=cv2.RANSAC,
                ransacReprojThreshold=reproj_threshold,
                maxIters=max_iters,
                confidence=confidence,
            )
            aff_inliers = aff_mask.sum() if aff_mask is not None else 0
            if matrix is not None and (aff_inliers >= h_inliers or not h_valid):
                H = np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float32)
                mask = aff_mask
                model_type = "affine"

        if H is None or not np.all(np.isfinite(H)):
            H = np.eye(3, dtype=np.float32)
            errors = compute_reprojection_errors(pts_src, pts_ref, H)
            inlier_mask = np.zeros(len(pts_src), dtype=bool)
        else:
            H = H.astype(np.float32)
            errors = compute_reprojection_errors(pts_src, pts_ref, H)
            inlier_mask = (
                mask.ravel().astype(bool)
                if mask is not None
                else (errors <= reproj_threshold)
            )

    return GeometricModel(
        transform_matrix=H,
        model_type=model_type,
        inlier_mask=inlier_mask,
        reprojection_errors=errors,
    )
