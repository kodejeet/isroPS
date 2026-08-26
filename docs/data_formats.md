# Data Formats Guide: Lunar Imagery & Planetary Archives

This document outlines the file formats encountered in lunar remote sensing and clarifies how they are ingested within the pipeline.

---

## 1. Standard Optical Formats (PNG / JPEG / TIFF)
- **Extension:** `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
- **Characteristics:** Standard 8-bit or 16-bit uncompressed/compressed 2D raster arrays.
- **Handling:** Fully supported in Day-1 codebase via Pillow, OpenCV, and `tifffile`. Channels are expanded into shape `(H, W, C)` with `C >= 1`.

---

## 2. GeoTIFF / GIS Rasters
- **Extension:** `.tif`, `.tiff` (with embedded GeoTIFF tags)
- **Characteristics:** GeoTIFFs encapsulate spatial coordinate metadata (bounding box, Map Projection, Datum/Ellipsoid) directly within TIFF header tags.
- **Handling:** Optional `[geo]` extra dependency (`rasterio`, `shapely`). When installed, GeoTIFF spatial metadata is populated into `ImageMetadata.geographic_bounds` and `projection`. Stubs gracefully degrade when `rasterio` is not installed.

---

## 3. PDS4 (Planetary Data System v4)
- **Structure:** Pair of files — XML header (`.xml`) + raw binary image blob (`.IMG` or `.DAT`).
- **Characteristics:** PDS4 is the official archival format used by ISRO PRADAN (Chandrayaan-2) and NASA PDS. The `.xml` file defines pixel bit depth, endianness, dimensions, orbital orientation, and instrument calibration constants.
- **Calibrated vs Raw Products:**
  - *Raw Products (`..._RAW`):* Uncalibrated digital numbers directly from sensor CCD. Requires complex dark-current and flat-field radiometric correction.
  - *Calibrated Products (`..._CAL`):* Radiometrically calibrated target reflectance / radiance. **Always prefer calibrated products for registration experiments.**

---

## 4. Hyperspectral Cubes (IIRS)
- **Structure:** 3D array of shape `(H, W, Bands)` where `Bands ≈ 256`.
- **Characteristics:** Each pixel contains a continuous spectral reflectance curve spanning visible to infrared wavelengths.
- **Band Reduction:** Standard 2D keypoint detectors (e.g. SIFT) cannot process 256 bands directly. `ImageData` preserves all channels, but band-selection strategies (e.g., extracting band 0 or computing a mean panchromatic band) must be applied prior to 2D feature extraction.

---

## 5. Kaggle Sample Datasets (Non-Authoritative Warning)
- **Notice:** Public Kaggle datasets (e.g. `piyushsharma5654/ohrc-images-chandaryaan-2`) are convenient for quick interface testing.
- **Warning:** Kaggle uploads often strip critical PDS4 headers, geospatial coordinate metadata, and sensor calibration factors, or apply lossy compression. **Do not rely on Kaggle datasets for official scientific reporting or quantitative spatial error metrics.**
