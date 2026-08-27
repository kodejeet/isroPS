"""Streamlit Upload Component and File Handler.

Converts Streamlit UploadedFile objects into ImageData structures in-memory.
Caches and handles preview rendering and format checks.
"""

import io
import os

import cv2
import numpy as np
from PIL import Image

from lunar_correspondence.io.metadata import ImageData, ImageMetadata

PDS4_EXTENSIONS = {".xml", ".lbl", ".img", ".qub"}


def load_uploaded_image(uploaded_file, instrument_hint: str = "UNKNOWN") -> ImageData:
    """Convert a Streamlit UploadedFile object into an ImageData structure.

    Args:
        uploaded_file: Streamlit UploadedFile instance.
        instrument_hint: User-selected metadata hint.

    Returns:
        ImageData object holding image array and ImageMetadata.

    Raises:
        ValueError: If a raw PDS4 file is uploaded or format reading fails.
    """
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower()

    if ext in PDS4_EXTENSIONS:
        raise ValueError(
            f"File '{file_name}' appears to be a raw PDS4 product (label + binary pair). "
            "Direct upload of raw PDS4 is not supported in this prototype. "
            "Please convert it to GeoTIFF using GDAL first (see data/README.md for the one-line command)."
        )

    file_bytes = uploaded_file.getvalue()
    array = None

    # 1. Try reading via rasterio MemoryFile if available
    try:
        from rasterio.io import MemoryFile

        with MemoryFile(file_bytes) as memfile, memfile.open() as dataset:
            # Read shape (C, H, W) and transpose to (H, W, C)
            arr = dataset.read()
            if arr.ndim == 3:
                array = np.transpose(arr, (1, 2, 0))
            else:
                array = arr[:, :, np.newaxis]
    except Exception:
        pass

    # 2. Try PIL Image fallback
    if array is None:
        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            arr = np.array(pil_img)
            if arr.ndim == 2:
                array = arr[:, :, np.newaxis]
            elif arr.ndim == 3:
                array = arr
        except Exception:
            pass

    # 3. Try OpenCV decoding fallback
    if array is None:
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img_cv is not None:
                if img_cv.ndim == 2:
                    array = img_cv[:, :, np.newaxis]
                elif img_cv.ndim == 3:
                    # Convert BGR/BGRA to RGB/RGBA
                    if img_cv.shape[2] == 3:
                        array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    elif img_cv.shape[2] == 4:
                        array = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
                    else:
                        array = img_cv
        except Exception:
            pass

    if array is None:
        raise ValueError(
            f"Failed to read image file '{file_name}'. Ensure it is a valid PNG, JPEG, TIFF, or GeoTIFF file."
        )

    metadata = ImageMetadata(instrument=instrument_hint, source_path=file_name)
    return ImageData(array=array, path=file_name, metadata=metadata)


def get_display_preview(
    image_data: ImageData, max_display_dim: int = 1024
) -> np.ndarray:
    """Create a downsized RGB/grayscale display preview array for browser rendering.

    Keeps the full-resolution array untouched for processing.
    """
    arr = image_data.array
    h, w = arr.shape[:2]
    max_dim = max(h, w)

    if max_dim > max_display_dim:
        scale = max_display_dim / float(max_dim)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        preview = cv2.resize(arr, (new_w, new_h))
    else:
        preview = arr.copy()

    if preview.ndim == 3 and preview.shape[2] == 1:
        preview = preview[:, :, 0]
    elif preview.ndim == 3 and preview.shape[2] > 3:
        preview = preview[:, :, :3]  # Truncate hyperspectral bands to RGB for display

    return preview
