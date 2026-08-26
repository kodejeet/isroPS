"""Instrument registry and metadata definitions for lunar sensors.

Provides plain-English descriptions, spatial resolution ranges, spectral types,
and band-processing hints for all supported optical and reference sensors.
"""

from typing import Any

INSTRUMENT_REGISTRY: dict[str, dict[str, Any]] = {
    "OHRC": {
        "name": "Orbiter High Resolution Camera",
        "mission": "Chandrayaan-2 (ISRO)",
        "type": "Panchromatic Optical",
        "approx_resolution_m_per_px": (0.25, 0.32),
        "typical_use": "Primary source image for ultra-high-resolution terrain & hazard mapping",
        "processing_hints": "Single panchromatic band. Sensitive to extreme solar shadow variations.",
        "plain_english": "Chandrayaan-2 ultra-high-res camera (~25cm/px) for detailed surface features.",
    },
    "TMC-2": {
        "name": "Terrain Mapping Camera 2",
        "mission": "Chandrayaan-2 (ISRO)",
        "type": "Panchromatic Stereo",
        "approx_resolution_m_per_px": (5.0, 10.0),
        "typical_use": "Stereo 3D mapping and surface DEM generation",
        "processing_hints": "Wide coverage. Lower spatial resolution compared to OHRC.",
        "plain_english": "Chandrayaan-2 stereo camera (~5m/px) for 3D elevation modeling.",
    },
    "IIRS": {
        "name": "Imaging Infra-Red Spectrometer",
        "mission": "Chandrayaan-2 (ISRO)",
        "type": "Hyperspectral (~256 bands)",
        "approx_resolution_m_per_px": (80.0, 100.0),
        "typical_use": "Mineralogical mapping and hydration analysis",
        "processing_hints": "Hyperspectral cube (~256 bands). Select single representative band (e.g. band 0) or mean before 2D keypoint extraction.",
        "plain_english": "Chandrayaan-2 hyperspectral sensor (~80m/px) measuring infrared spectrum across 256 bands.",
    },
    "LRO_NAC": {
        "name": "Narrow Angle Camera (LROC)",
        "mission": "Lunar Reconnaissance Orbiter (NASA)",
        "type": "Panchromatic Reference",
        "approx_resolution_m_per_px": (0.5, 2.0),
        "typical_use": "High-quality NASA reference image for co-registration",
        "processing_hints": "Panchromatic high-resolution baseline.",
        "plain_english": "NASA LRO narrow-angle reference camera (~0.5-2.0m/px).",
    },
    "SELENE": {
        "name": "Terrain Camera (TC)",
        "mission": "SELENE / Kaguya (JAXA)",
        "type": "Panchromatic Reference",
        "approx_resolution_m_per_px": (10.0, 10.0),
        "typical_use": "Global Japanese lunar reference baseline",
        "processing_hints": "Panchromatic global mapping baseline.",
        "plain_english": "JAXA Kaguya reference camera (~10m/px).",
    },
    "UNKNOWN": {
        "name": "Unknown / Unspecified Instrument",
        "mission": "Unknown",
        "type": "Generic Optical",
        "approx_resolution_m_per_px": None,
        "typical_use": "Fallback for unclassified raster imagery",
        "processing_hints": "No prior assumptions about resolution or bands.",
        "plain_english": "Unrecognized instrument fallback.",
    },
}


def get_instrument_info(instrument_name: str) -> dict[str, Any]:
    """Retrieve metadata information for a registered instrument.

    Returns UNKNOWN entry if instrument_name is not registered.
    """
    key = instrument_name.upper() if instrument_name else "UNKNOWN"
    return INSTRUMENT_REGISTRY.get(key, INSTRUMENT_REGISTRY["UNKNOWN"])
