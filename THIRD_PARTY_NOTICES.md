# Third-Party Notices

This project incorporates or adapts code from the following third-party sources.

---

## 1. RIFT2 Python Implementation

- **Upstream Repository:** https://github.com/canyagmur/RIFT2-multimodal-matching-rotation-python
- **Author:** canyagmur
- **License:** No explicit license file found in the upstream repository as of 2026-08-28.
  The repository is publicly accessible on GitHub. Usage here is for academic/research
  purposes within the SIH 2026 competition context.
- **Original Paper (RIFT2):** Li, Jiayuan, Qingwu Hu, and Mingyao Ai. "RIFT2:
  Speeding-up RIFT with A New Rotation-Invariance Technique" (2023). arXiv:2303.00319.
- **Original Paper (RIFT):** Li, Jiayuan, Qingwu Hu, and Mingyao Ai. "RIFT:
  Multi-modal image matching based on radiation-variation insensitive feature
  transform." IEEE Transactions on Image Processing 29 (2020): 3296–3310.
- **MATLAB Original:** https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation

### Components Reused

The following upstream files were vendored into
`src/lunar_correspondence/features/_vendor/rift2/`:

| Upstream File | Vendored As | Status |
|---|---|---|
| `src/RIFT2.py` | `core.py` | Adapted |
| `src/phase_congruency/phasecong.py` | `phasecong.py` | Adapted |
| `src/phase_congruency/tools.py` | `tools.py` | Adapted |

### Modifications Made

- **`core.py` (adapted from `RIFT2.py`):**
  - Removed `joblib` parallelism dependency; uses serial processing to avoid
    unnecessary dependency.
  - Removed YAML config-file loading; config is passed as a Python dict.
  - Removed print statements; operates silently.
  - Fixed indentation errors present in the upstream source
    (`calculate_orientation_hist`, `feature_description`).
  - Changed relative imports to use vendored phase congruency module.
  - Added `detect_and_describe()` convenience method returning `(keypoints_xy, descriptors)`.
  - Added type hints and docstrings.
  - Renamed class from `RIFT2` to `RIFT2Core` to avoid confusion with the adapter.

- **`phasecong.py` (adapted from `src/phase_congruency/phasecong.py`):**
  - Replaced `scipy.fftpack` imports with `scipy.fft` (the modern API).
  - Removed `pyfftw` optional dependency path.
  - Added division-by-zero guard for blank/uniform images.
  - Minor style adjustments.

- **`tools.py` (adapted from `src/phase_congruency/tools.py`):**
  - Replaced `scipy.fftpack` imports with `scipy.fft`.
  - Removed `pyfftw` optional dependency and associated warning.
  - Removed unused `perfft2` function.
  - Minor style adjustments.

### Newly Written Adapter Code

| File | Description |
|---|---|
| `src/lunar_correspondence/features/rift_features.py` | Adapter implementing `FeatureExtractor` interface for RIFT2 |
| `src/lunar_correspondence/matching/rift_matcher.py` | Adapter implementing `Matcher` interface for RIFT2 descriptors |
| `tests/test_rift_integration.py` | Comprehensive RIFT2 integration tests |

These adapter files are entirely new code written specifically for this project
and do not contain code from the upstream repository.

### Dependencies

The vendored RIFT2 code requires:
- `numpy` (already a project dependency)
- `scipy` (already a project dependency, used for FFT)
- `opencv-python` (already a project dependency, used for FAST detector)

No additional dependencies were introduced.

---

## 2. Phase Congruency (PhasePack)

- **Upstream Repository:** https://github.com/alimuldal/phasepack
- **Author:** Ali Shervin Muldal
- **License:** MIT License
- **Usage:** The phase congruency computation in the vendored RIFT2 code
  is adapted from PhasePack, as acknowledged in the RIFT2 upstream README.

The PhasePack code implements Peter Kovesi's phase congruency algorithms:
- Kovesi, P. "Image Features From Phase Congruency". Videre, MIT Press, 1999.
- Kovesi, P. "Phase Congruency Detects Corners and Edges". DICTA 2003.

---

## 3. OpenCV (cv2)

- **Repository:** https://github.com/opencv/opencv
- **License:** Apache License 2.0
- **Usage:** Used throughout the project for image I/O, SIFT feature detection,
  BFMatcher, FAST feature detector (in RIFT2 pipeline), geometric transforms,
  and RANSAC estimation.

---

## 4. Other Dependencies

Standard Python scientific computing stack used throughout:
- **NumPy** — BSD License
- **SciPy** — BSD License
- **Matplotlib** — PSF-based License (visualization only)
- **PyYAML** — MIT License
- **rasterio** — BSD License (geospatial image I/O)
