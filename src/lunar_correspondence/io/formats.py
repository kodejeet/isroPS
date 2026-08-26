"""Image format detection and classification module."""

import os
from enum import Enum


class ImageFormat(Enum):
    STANDARD = "standard"  # PNG, JPEG, BMP
    TIFF = "tiff"  # TIFF / GeoTIFF
    PDS4 = "pds4"  # Planetary Data System v4 (XML + binary)
    UNKNOWN = "unknown"


def detect_image_format(path: str) -> ImageFormat:
    """Detect image file format based on file extension and header existence.

    Gloss:
    - PDS4: Planetary Data System format combining XML metadata label with binary blob.
    - GeoTIFF: TIFF file containing embedded geospatial projection tags.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        return ImageFormat.STANDARD
    elif ext in [".tif", ".tiff"]:
        return ImageFormat.TIFF
    elif ext in [".xml", ".img", ".lbl", ".dat"]:
        return ImageFormat.PDS4
    else:
        return ImageFormat.UNKNOWN
