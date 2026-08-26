"""Format-agnostic image loader module.

Supports standard raster formats (PNG, JPEG, TIFF) natively.
Includes lazy-import stubs for GeoTIFF (via rasterio) and PDS4 formats.
Ensures returned ImageData.array always has shape (H, W, C) with C >= 1.
"""

import os

import cv2
import numpy as np
from PIL import Image

from lunar_correspondence.io.formats import ImageFormat, detect_image_format
from lunar_correspondence.io.metadata import ImageData, ImageMetadata
from lunar_correspondence.planetary.instruments import get_instrument_info


def load_image(
    path: str,
    instrument: str = "UNKNOWN",
    resolution_m_per_px: float | None = None,
    sun_azimuth_deg: float | None = None,
    sun_elevation_deg: float | None = None,
) -> ImageData:
    """Load image from disk and return a unified ImageData structure.

    Args:
        path: Path to target image file.
        instrument: Instrument string identifier (OHRC, TMC-2, IIRS, LRO_NAC, SELENE, UNKNOWN).
        resolution_m_per_px: Optional Ground Sampling Distance (GSD) in meters per pixel.
        sun_azimuth_deg: Optional Sun Azimuth angle in degrees.
        sun_elevation_deg: Optional Sun Elevation angle in degrees.

    Returns:
        ImageData object containing (H, W, C) array and metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    fmt = detect_image_format(path)

    if fmt == ImageFormat.PDS4:
        raise NotImplementedError(
            "PDS4 planetary label reader is pending P1 implementation. "
            "For Day-1 baseline testing, please use calibrated PNG, JPEG, or TIFF products."
        )

    # Attempt loading standard or TIFF raster
    array = None
    if fmt == ImageFormat.TIFF:
        # Check if geo extra rasterio is installed
        try:
            import rasterio

            with rasterio.open(path) as src:
                array = src.read()  # (C, H, W)
                array = np.moveaxis(array, 0, -1)  # reshape to (H, W, C)
        except ImportError:
            # Fallback to OpenCV / tifffile loading
            img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img_cv is not None:
                array = img_cv
        except Exception:
            pass

    if array is None:
        # OpenCV standard read (returns BGR or Grayscale)
        img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_cv is not None:
            if img_cv.ndim == 3 and img_cv.shape[2] == 3:
                # Convert OpenCV BGR to RGB
                array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            else:
                array = img_cv
        else:
            # Fallback to Pillow
            with Image.open(path) as pil_img:
                array = np.array(pil_img)

    if array is None:
        raise ValueError(f"Failed to decode image data from file: {path}")

    # Enforce (H, W, C) shape with C >= 1
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    elif array.ndim == 3 and array.shape[2] not in [1, 3, 4]:
        # Hyperspectral or multi-band format retain shape (H, W, C)
        pass

    inst_info = get_instrument_info(instrument)

    # If GSD is not explicitly provided, default to registry range mid-point if available, else None
    if (
        resolution_m_per_px is None
        and inst_info["approx_resolution_m_per_px"] is not None
    ):
        res_range = inst_info["approx_resolution_m_per_px"]
        resolution_m_per_px = float(np.mean(res_range))

    metadata = ImageMetadata(
        instrument=instrument.upper(),
        acquisition_time=None,
        resolution_m_per_px=resolution_m_per_px,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        geographic_bounds=None,
        projection=None,
        source_path=path,
    )

    return ImageData(array=array, path=path, metadata=metadata)
