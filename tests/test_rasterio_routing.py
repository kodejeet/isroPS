"""Regression tests for TIFF/GeoTIFF loading and Rasterio routing.

Verifies:
- TIFF loading via OpenCV fallback when rasterio is not installed.
- TIFF loading always produces (H, W, C) shape.
- Rasterio error handling emits RuntimeWarning and falls back to OpenCV.
- Rasterio success path (mocked) produces correct (H, W, C) shape.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from lunar_correspondence.io.image_loader import load_image
from lunar_correspondence.io.metadata import ImageData


class TestTIFFLoadingFallback:
    """Tests for TIFF loading when rasterio is NOT available."""

    def test_tiff_grayscale_loads_with_correct_shape(self, tmp_path):
        """Grayscale TIFF must be returned as (H, W, 1)."""
        img = np.random.randint(0, 256, (64, 80), dtype=np.uint8)
        tiff_path = str(tmp_path / "gray.tif")
        cv2.imwrite(tiff_path, img)

        result = load_image(tiff_path)
        assert isinstance(result, ImageData)
        assert result.array.shape == (64, 80, 1)

    def test_tiff_rgb_loads_with_correct_shape(self, tmp_path):
        """3-channel TIFF must be returned as (H, W, 3)."""
        img = np.random.randint(0, 256, (64, 80, 3), dtype=np.uint8)
        tiff_path = str(tmp_path / "rgb.tiff")
        cv2.imwrite(tiff_path, img)

        result = load_image(tiff_path)
        assert isinstance(result, ImageData)
        assert result.array.shape == (64, 80, 3)

    def test_tiff_16bit_loads_with_correct_shape(self, tmp_path):
        """16-bit TIFF must preserve bit depth and have (H, W, C) shape."""
        img = np.random.randint(0, 65535, (48, 64), dtype=np.uint16)
        tiff_path = str(tmp_path / "16bit.tif")
        cv2.imwrite(tiff_path, img)

        result = load_image(tiff_path)
        assert isinstance(result, ImageData)
        assert result.array.ndim == 3
        assert result.array.shape[2] >= 1


class TestRasterioErrorHandling:
    """Tests for rasterio error handling using mocks."""

    def test_rasterio_error_emits_warning_and_falls_back(self, tmp_path):
        """When rasterio is installed but fails, a RuntimeWarning should be
        emitted and the loader should fall back to OpenCV."""
        # Create a valid TIFF so OpenCV fallback works
        img = np.random.randint(0, 256, (32, 40), dtype=np.uint8)
        tiff_path = str(tmp_path / "test.tif")
        cv2.imwrite(tiff_path, img)

        # Mock rasterio to raise an error when opening
        mock_rasterio = MagicMock()
        mock_rasterio.open.side_effect = RuntimeError("Simulated rasterio failure")

        with (
            patch.dict("sys.modules", {"rasterio": mock_rasterio}),
            pytest.warns(RuntimeWarning, match="Rasterio failed to read"),
        ):
            result = load_image(tiff_path)

        # Should still successfully load via OpenCV fallback
        assert isinstance(result, ImageData)
        assert result.array.ndim == 3
        assert result.array.shape == (32, 40, 1)

    def test_rasterio_success_path_produces_correct_shape(self, tmp_path):
        """When rasterio is available and succeeds, the result must be (H, W, C)."""
        tiff_path = str(tmp_path / "geo.tif")
        # Create a dummy file so os.path.exists passes
        with open(tiff_path, "wb") as f:
            f.write(b"\x00" * 100)

        # Mock rasterio to return a (C, H, W) array
        mock_src = MagicMock()
        mock_src.read.return_value = np.random.randint(
            0, 256, (3, 50, 60), dtype=np.uint8
        )
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        with patch.dict("sys.modules", {"rasterio": mock_rasterio}):
            result = load_image(tiff_path)

        assert isinstance(result, ImageData)
        # rasterio returns (C, H, W) which must be transposed to (H, W, C)
        assert result.array.shape == (50, 60, 3)

    def test_rasterio_single_band_produces_hwc(self, tmp_path):
        """Single-band rasterio read must produce (H, W, 1)."""
        tiff_path = str(tmp_path / "single_band.tif")
        with open(tiff_path, "wb") as f:
            f.write(b"\x00" * 100)

        mock_src = MagicMock()
        mock_src.read.return_value = np.random.randint(
            0, 256, (1, 100, 120), dtype=np.uint8
        )
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_src

        with patch.dict("sys.modules", {"rasterio": mock_rasterio}):
            result = load_image(tiff_path)

        assert isinstance(result, ImageData)
        assert result.array.shape == (100, 120, 1)

    def test_rasterio_import_error_falls_back_silently(self, tmp_path):
        """When rasterio is not installed (ImportError), no warning is emitted
        and OpenCV fallback is used."""
        img = np.random.randint(0, 256, (32, 40), dtype=np.uint8)
        tiff_path = str(tmp_path / "no_rasterio.tif")
        cv2.imwrite(tiff_path, img)

        # Ensure rasterio is not importable
        with patch.dict("sys.modules", {"rasterio": None}):
            # Should not raise or warn
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = load_image(tiff_path)

        assert isinstance(result, ImageData)
        assert result.array.ndim == 3
