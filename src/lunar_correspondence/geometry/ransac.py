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
        # Insufficient points for homography estimation
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
            # Convert 2x3 affine matrix to 3x3 homogeneous matrix
            H = np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float32)
            errors = compute_reprojection_errors(pts_src, pts_ref, H)
            inlier_mask = (
                mask.ravel().astype(bool)
                if mask is not None
                else (errors <= reproj_threshold)
            )
    else:
        # Default Homography estimation
        H, mask = cv2.findHomography(
            pts_src,
            pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
        if H is None:
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
