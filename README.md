# Lunar Image Correspondence Engine (SIH 2026 Problem Statement 26166)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Day-1 Baseline Skeleton](https://img.shields.io/badge/Status-Day--1%20Baseline%20Skeleton-green.svg)]()

> **Official Problem Statement:** Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS).  
> **Sponsoring Organization:** Indian Space Research Organisation (ISRO), Department of Space.

---

## 1. Problem Overview
SIH Problem Statement 26166 requires finding accurate pixel-level feature correspondences between pairs of lunar orbital images captured across different optical instruments (Chandrayaan-2 OHRC, TMC-2, IIRS, NASA LRO NAC, JAXA SELENE), varying spatial resolutions, and extreme solar illumination angles. Due to the lack of atmosphere on the Moon, shadows move, invert, or vanish when solar azimuth/elevation changes, causing standard intensity gradient matchers to fail. The target deliverable is a generic registration engine achieving **sub-pixel accuracy** with a **spatially uniform distribution** of matches. For domain details, see [`docs/problem.md`](docs/problem.md) and [`docs/lunar_imaging_basics.md`](docs/lunar_imaging_basics.md).

---

## 2. Day-1 Scope Implementation Status

### Real & Working Today (Day-1 Base)
- **Data Model & Ingestion:** Fully typed `ImageData`, `ImageMetadata`, `FeatureSet`, `MatchSet`, `GeometricModel`, `RegistrationResult`, `EvaluationResult` supporting arbitrary `(H, W, C)` arrays.
- **Image Loader:** Native format-agnostic loader for PNG, JPEG, TIFF/GeoTIFF raster formats.
- **Synthetic Pair Generator:** Procedural lunar crater scene generator with geometric warping and illumination gradients (`data/examples/`).
- **SIFT Baseline Pipeline:** Real working SIFT feature extraction, descriptor matching with Lowe's ratio test, and RANSAC homography estimation (`geometry/ransac.py`).
- **Quantitative Evaluation:** Real computation of Total Matches, Inlier Matches, Inlier Ratio, RMSE, Median Error, and **Grid-Cell Spatial Coverage %**.
- **Visualization:** Side-by-side match line rendering (inlier green / outlier red) and false-color warped registration overlays.
- **Configuration Engine:** YAML config inheritance with recursive deep-merging (`config.py`).

### Pending / Documented Stubs (Backlog for P1 / P2)
- **RIFT Adaptor:** Adapter interface defined in `features/rift_features.py`; phase congruency logic pending P1.
- **Learned / LightGlue Matcher:** Interface defined in `matching/lightglue_matcher.py` (P2).
- **PDS4 Reader:** Format stub in `io/image_loader.py` with friendly error message directing users to calibrated PNG/TIFF exports (P1).
- **Subpixel Refinement:** Pass-through stub in `geometry/refinement.py` (P2).
- **Streamlit Web UI:** Component skeleton in `app/app.py` (P1).

---

## 3. Installation

### Base Installation (Standard Baseline Engine)
```bash
# Clone the repository
git clone https://github.com/your-org/isroPS.git
cd isroPS

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base package in editable mode
pip install -e .
```

### Optional Dependencies via Extras
To avoid installation blockages (e.g. GDAL or CUDA driver conflicts), dependencies are isolated into groups:

```bash
# Install with GeoTIFF/GIS support (rasterio, shapely)
pip install -e .[geo]

# Install with deep learning support (PyTorch CPU index)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[learned]

# Install development & test dependencies (pytest, ruff, black)
pip install -e .[dev]
```

---

## 4. Running the Synthetic Baseline Demo

Run the end-to-end pipeline script on the procedurally generated synthetic lunar pair:

```bash
python scripts/run_baseline.py --config configs/default.yaml
```

**Expected Output:**
```text
------------------- EVALUATION RESULTS -------------------
 Total Matches:           284
 Inlier Matches:          210
 Inlier Ratio:            0.7394
 RMSE (pixels):           0.8412
 Median Error (pixels):   0.6201
 Grid Coverage (%):       87.50%
 Processing Time:         0.342 s
----------------------------------------------------------
[*] Results saved to directory: ./outputs
```

Output artifacts generated in `./outputs/`:
- `registered_output.png`: Warped source image aligned to reference frame.
- `match_visualization.png`: Side-by-side match line canvas.
- `registration_overlay.png`: False-color overlay visualization.
- `metrics.json`: Evaluated metrics report.

---

## 5. Running the Baseline on Real Images

Once real lunar images (PNG, JPEG, TIFF) are placed in `./data/calibrated/` or `./data/reference/`, inspect their metadata:

```bash
python scripts/inspect_image.py ./data/calibrated/ohrc_sample.png --instrument OHRC
```

Run registration between two real images:
```bash
python scripts/run_baseline.py --config configs/experiment_baseline.yaml
```

---

## 6. Accessing Real Chandrayaan-2 & Lunar Datasets

1. **ISRO PRADAN Archive (Chandrayaan-2):**
   - Portal: [https://pradan.issdc.gov.in/ch2/](https://pradan.issdc.gov.in/ch2/) (Requires free user registration)
   - Visual Map Browser: [https://chmapbrowse.issdc.gov.in/](https://chmapbrowse.issdc.gov.in/) *(Select South Pole projection, turn on OHRC/TMC-2 calibrated footprint layers, click footprint to download PDS4 data)*
2. **NASA LRO NAC Reference Images:**
   - QuickMap Tool: [https://quickmap.lroc.im-ldi.com](https://quickmap.lroc.im-ldi.com)
   - Downloads: [https://lroc.im-ldi.com/images/downloads/](https://lroc.im-ldi.com/images/downloads/)
3. **Data Hygiene Guide:** See [`data/README.md`](data/README.md).

---

## 7. Why RIFT is the Key Next Step

Standard SIFT relies on local image intensity gradients. When solar azimuth angle flips between orbits, dark crater interiors become bright rims and shadow directions invert by 180°. SIFT descriptor matching breaks down under high solar illumination changes.

**RIFT (Rotation and Illumination Invariant Feature Transform)** replaces intensity gradients with **Phase Congruency** maps and Maximum Index Maps (MIM) constructed from log-Gabor filter responses. Phase congruency remains invariant under shadow reversal, making RIFT the essential core algorithm for P1. For research notes and mathematical details, see [`docs/research_notes.md`](docs/research_notes.md).

---

## 8. Target Architecture Overview

```text
       ┌────────────────────────┐      ┌────────────────────────┐
       │   Source Image (OHRC)  │      │ Reference Image (TMC2) │
       └───────────┬────────────┘      └───────────┬────────────┘
                   │                               │
                   ▼                               ▼
       ┌────────────────────────────────────────────────────────┐
       │               Format-Agnostic Ingestion                │
       │           (ImageData (H, W, C), Metadata)             │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │        Phase Congruency / SIFT Feature Extractor       │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │            Descriptor & Matrix Fusion Matcher           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │            RANSAC Homography / Affine Engine           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌───────────────────────────┴────────────────────────────┐
       │                                                        │
       ▼                                                        ▼
┌───────────────┐                                     ┌──────────────────┐
│  Warped Image │                                     │ Metrics Report   │
│  Registration │                                     │ RMSE, Inliers,   │
│    Overlay    │                                     │ Grid Coverage %  │
└───────────────┘                                     └──────────────────┘
```

---

## 9. License
Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
