"""Data structures and metadata definitions for lunar image correspondence.

CRITICAL CONVENTION:
All 2D point coordinates (FeatureSet.keypoints, MatchSet.source_points, etc.)
strictly enforce the (x, y) = (column, row) convention matching OpenCV.
x: 0 to width - 1 (columns)
y: 0 to height - 1 (rows)
NumPy array indexing: array[row, col] = array[y, x]
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageMetadata:
    """Metadata container for lunar satellite/reference images.

    Gloss:
    - Instrument: The camera sensor name (e.g. OHRC, TMC-2, IIRS, LRO_NAC, SELENE).
    - Sun Azimuth: Compass direction toward the sun in degrees (0..360).
    - Sun Elevation: Angle of the sun above the local horizon in degrees (0..90).
    - Resolution: Ground Sampling Distance (GSD) in meters per pixel.
    """

    instrument: str  # Instrument registry key or "UNKNOWN"
    acquisition_time: str | None = None
    resolution_m_per_px: float | None = None
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    geographic_bounds: tuple[float, float, float, float] | None = (
        None  # (min_lon, min_lat, max_lon, max_lat)
    )
    projection: str | None = None
    source_path: str = ""


@dataclass
class ImageData:
    """Core image representation holding image array data and metadata.

    Array shape is always (H, W, C) with C >= 1.
    Multi-channel arrays (e.g., IIRS hyperspectral cubes with ~256 bands) retain all channels.
    Feature extractors select or reduce channels explicitly.
    """

    array: np.ndarray  # shape (H, W, C), C >= 1
    path: str
    metadata: ImageMetadata

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def channels(self) -> int:
        return self.array.shape[2] if self.array.ndim > 2 else 1


@dataclass
class FeatureSet:
    """Detected keypoints and feature descriptors.

    keypoints array shape is (N, 2), enforcing (x, y) = (col, row) coordinates.
    """

    keypoints: np.ndarray  # shape (N, 2), (x, y)
    descriptors: np.ndarray | None = None  # shape (N, D)
    scores: np.ndarray | None = None  # shape (N,)
    method: str = "unknown"


@dataclass
class MatchSet:
    """Pairwise correspondences established between source and reference keypoints.

    source_points and reference_points shape: (N, 2), (x, y) convention.
    """

    source_points: np.ndarray  # (N, 2), (x, y) in source image
    reference_points: np.ndarray  # (N, 2), (x, y) in reference image
    confidence: np.ndarray | None = None  # (N,) match score/distance
    inlier_mask: np.ndarray | None = (
        None  # (N,) boolean array (True for RANSAC inliers)
    )


@dataclass
class GeometricModel:
    """Estimated geometric transformation matrix (Homography or Affine)."""

    transform_matrix: np.ndarray  # 3x3 for homography, 2x3 or 3x3 for affine
    model_type: str  # "homography" or "affine"
    inlier_mask: np.ndarray  # boolean mask of inliers used
    reprojection_errors: np.ndarray  # (N,) reprojection error per point pair in pixels


@dataclass
class RegistrationResult:
    """Warped source image aligned to reference image frame, plus match geometry."""

    registered_image: np.ndarray
    geometric_model: GeometricModel
    match_set: MatchSet


@dataclass
class EvaluationResult:
    """Quantitative registration performance evaluation metrics.

    Coverage represents grid-cell spatial coverage percentage (0.0 to 100.0%).
    """

    total_matches: int
    inlier_matches: int
    inlier_ratio: float
    rmse_pixels: float | None
    median_error_pixels: float | None
    coverage: float  # grid-cell coverage percentage
    spatial_uniformity: float | None
    processing_time_seconds: float
    random_seed: int
    scale_factor: float = 1.0
