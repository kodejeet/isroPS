"""Unit tests for image loading, metadata parsing, and formats."""

import cv2
import numpy as np
import pytest

from lunar_correspondence.io.image_loader import load_image
from lunar_correspondence.io.metadata import ImageData


def test_load_standard_image(tmp_path):
    # Create temporary PNG image
    test_img = np.zeros((100, 120, 3), dtype=np.uint8)
    test_img[20:50, 30:70] = 255
    img_path = str(tmp_path / "test.png")
    cv2.imwrite(img_path, test_img)

    img_data = load_image(img_path, instrument="OHRC")

    assert isinstance(img_data, ImageData)
    assert img_data.height == 100
    assert img_data.width == 120
    assert img_data.array.shape == (100, 120, 3)
    assert img_data.metadata.instrument == "OHRC"
    assert img_data.metadata.resolution_m_per_px is not None


def test_load_grayscale_image_shape(tmp_path):
    test_img = np.zeros((80, 80), dtype=np.uint8)
    img_path = str(tmp_path / "test_gray.png")
    cv2.imwrite(img_path, test_img)

    img_data = load_image(img_path, instrument="TMC-2")
    # Enforces (H, W, C) shape with C=1
    assert img_data.array.shape == (80, 80, 1)


def test_pds4_stub_raises_error(tmp_path):
    for ext in [".xml", ".lbl", ".img", ".qub"]:
        pds4_path = str(tmp_path / f"test_product{ext}")
        with open(pds4_path, "w") as f:
            f.write("DUMMY PDS4 CONTENT")

        with pytest.raises(NotImplementedError) as exc_info:
            load_image(pds4_path)

        assert "Direct raw PDS4-style ingestion" in str(exc_info.value)
        assert "GeoTIFF" in str(exc_info.value)
