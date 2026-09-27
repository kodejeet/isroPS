"""SensorProfile abstraction module for CHANDRAMAP / SIH 2026.

Defines unified sensor configurations encapsulating:
- Sensor spatial sampling (source GSD vs reference GSD)
- Radiometric conditioning (percentile scaling, mask-aware feature images, CLAHE)
- Feature extraction strategy (SIFT, RIFT2, Fused)
- Deformation & geometric model (Homography, Affine, Block-Affine)
- Quality gates (illumination screening, line-loss thresholds)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RadiometryConfig:
    """Configuration for converting native rasters into feature extraction representations."""

    method: str = "percentile"  # "percentile", "minmax", "zscore", "log"
    p_low: float = 0.5
    p_high: float = 99.5
    mask_negatives: bool = False
    apply_clahe: bool = False
    clahe_clip_limit: float = 3.0
    clahe_grid_size: tuple[int, int] = (16, 16)


@dataclass
class GeometryConfig:
    """Configuration for geometric correspondence modeling and RANSAC."""

    primary_model: str = "homography"  # "homography", "affine", "block_affine"
    reproj_threshold: float = 3.0
    block_length_px: int = 1500
    block_overlap_px: int = 300
    max_iters: int = 2000
    confidence: float = 0.99


@dataclass
class SensorProfile:
    """Top-level profile governing registration pipeline behavior for a sensor pair."""

    name: str
    source_instrument: str
    reference_instrument: str
    approx_source_gsd_m: float
    approx_reference_gsd_m: float
    scale_policy: str = "native"  # "native" or "pyramid_to_reference"
    radiometry: RadiometryConfig = field(default_factory=RadiometryConfig)
    feature_method: str = "fusion"  # "sift", "rift", "fusion"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    min_sun_elevation_deg: float = 0.0
    subpixel_refinement_enabled: bool = False
    description: str = ""


# Built-in Sensor Profiles
PROFILES: dict[str, SensorProfile] = {
    "OHRC_SIH_5M": SensorProfile(
        name="OHRC_SIH_5M",
        source_instrument="OHRC",
        reference_instrument="LRO_NAC",
        approx_source_gsd_m=5.0,
        approx_reference_gsd_m=5.0,
        scale_policy="native",
        radiometry=RadiometryConfig(
            method="minmax",
            apply_clahe=True,
            clahe_clip_limit=3.0,
            clahe_grid_size=(8, 8),
        ),
        feature_method="fusion",
        geometry=GeometryConfig(
            primary_model="homography",
            reproj_threshold=3.0,
        ),
        min_sun_elevation_deg=0.0,
        subpixel_refinement_enabled=True,
        description="Official SIH downsampled OHRC 5m source registered to LRO 5m reference.",
    ),
    "TMC2": SensorProfile(
        name="TMC2",
        source_instrument="TMC-2",
        reference_instrument="LRO_NAC",
        approx_source_gsd_m=5.48,
        approx_reference_gsd_m=1.0,
        scale_policy="native",
        radiometry=RadiometryConfig(
            method="minmax",
            apply_clahe=True,
            clahe_clip_limit=2.5,
            clahe_grid_size=(8, 8),
        ),
        feature_method="fusion",
        geometry=GeometryConfig(
            primary_model="homography",
            reproj_threshold=3.0,
        ),
        min_sun_elevation_deg=0.0,
        subpixel_refinement_enabled=False,
        description="Chandrayaan-2 TMC-2 stereo camera to NASA LROC NAC reference.",
    ),
    "IIRS_EQUATORIAL_WAC": SensorProfile(
        name="IIRS_EQUATORIAL_WAC",
        source_instrument="IIRS",
        reference_instrument="LRO_WAC",
        approx_source_gsd_m=93.74,
        approx_reference_gsd_m=94.74,
        scale_policy="native",
        radiometry=RadiometryConfig(
            method="percentile",
            p_low=0.5,
            p_high=99.5,
            mask_negatives=True,
            apply_clahe=False,
            clahe_clip_limit=3.0,
            clahe_grid_size=(16, 16),
        ),
        feature_method="fusion",
        geometry=GeometryConfig(
            primary_model="block_affine",
            reproj_threshold=4.0,
            block_length_px=1500,
            block_overlap_px=300,
        ),
        min_sun_elevation_deg=5.0,
        subpixel_refinement_enabled=False,
        description="IIRS infrared spectral radiance to LRO WAC equatorial morphology basemap.",
    ),
    "IIRS_SOUTH_POLE_WAC": SensorProfile(
        name="IIRS_SOUTH_POLE_WAC",
        source_instrument="IIRS",
        reference_instrument="LRO_WAC",
        approx_source_gsd_m=83.14,
        approx_reference_gsd_m=200.0,
        scale_policy="native",
        radiometry=RadiometryConfig(
            method="percentile",
            p_low=0.5,
            p_high=99.5,
            mask_negatives=True,
            apply_clahe=False,
            clahe_clip_limit=3.0,
            clahe_grid_size=(16, 16),
        ),
        feature_method="fusion",
        geometry=GeometryConfig(
            primary_model="block_affine",
            reproj_threshold=4.0,
            block_length_px=1200,
            block_overlap_px=200,
        ),
        min_sun_elevation_deg=5.0,
        subpixel_refinement_enabled=False,
        description="IIRS south polar strip to LRO WAC Polar Stereographic basemap (200m).",
    ),
}


def get_sensor_profile(profile_name: str) -> SensorProfile:
    """Retrieve a registered SensorProfile by identifier.

    Args:
        profile_name: Name of the profile (e.g. 'OHRC_SIH_5M', 'IIRS_EQUATORIAL_WAC').

    Returns:
        SensorProfile instance.
    """
    key = profile_name.upper()
    for name, prof in PROFILES.items():
        if name.upper() == key:
            return prof

    raise KeyError(f"Sensor profile '{profile_name}' not found. Available: {list(PROFILES.keys())}")


def detect_sensor_profile(instrument: str, gsd: float | None = None, is_polar: bool = False) -> SensorProfile:
    """Automatically deduce the best SensorProfile from instrument type and metadata.

    Args:
        instrument: Sensor name string (e.g. 'OHRC', 'TMC-2', 'IIRS').
        gsd: Ground sampling distance in meters per pixel.
        is_polar: Whether the scene is in a polar projection.

    Returns:
        SensorProfile instance.
    """
    inst = instrument.upper()
    if "OHR" in inst:
        return PROFILES["OHRC_SIH_5M"]
    elif "TMC" in inst:
        return PROFILES["TMC2"]
    elif "IIR" in inst:
        if is_polar or (gsd and gsd < 88.0):
            return PROFILES["IIRS_SOUTH_POLE_WAC"]
        return PROFILES["IIRS_EQUATORIAL_WAC"]
    else:
        # Fallback to general optical profile
        return PROFILES["OHRC_SIH_5M"]
