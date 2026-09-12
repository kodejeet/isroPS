"""Unit and integration tests for the rebuilt Lunar Registration Dashboard UI and services.

SIH 2026 Problem Statement 26166.
"""

import json
import os
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.components.dataset_loader import (
    get_available_ch2_products,
    get_available_lroc_references,
    load_windowed_pair,
    calculate_footprint_iou,
)
from app.components.metadata_service import (
    get_ch2_metadata,
    get_lroc_metadata,
    get_combined_planetary_metadata,
)
from app.components.transform_service import (
    decompose_homography,
    generate_json_export,
    generate_pdf_report,
    make_checkerboard_image,
    make_false_color_overlay,
)


def test_dataset_loader_enumeration():
    """Verify CH-2 products and LROC references are discovered without leaking % labels."""
    ohrc_prods = get_available_ch2_products("OHRC")
    assert len(ohrc_prods) >= 1
    assert any("ch2_ohr" in p for p in ohrc_prods)

    tmc2_prods = get_available_ch2_products("TMC-2")
    assert len(tmc2_prods) >= 1
    assert any("ch2_tmc" in p for p in tmc2_prods)

    ohrc_refs = get_available_lroc_references("OHRC")
    assert len(ohrc_refs) >= 1
    assert "NAC_POLE_SOUTH_CM_AVG_P848S0337.tiff" in [r["filename"] for r in ohrc_refs]

    tmc2_refs = get_available_lroc_references("TMC-2")
    assert len(tmc2_refs) == 11
    # Crucial prompt requirement: filenames ONLY, no % folders in filenames
    for r in tmc2_refs:
        assert "%" not in r["filename"], f"Found % in filename: {r['filename']}"
        assert r["filename"].endswith((".IMG", ".tif", ".tiff"))


def test_footprint_iou_logic():
    """Verify IoU calculations for normal and failure cases."""
    # Failure case
    fail_iou = calculate_footprint_iou("TMC-2", "ch2_tmc_ncn", "M1225104036LC.IMG", "/path/to/0%/M1225104036LC.IMG")
    assert fail_iou < 0.05

    # 100% case
    success_iou = calculate_footprint_iou("TMC-2", "ch2_tmc_ncn", "M1347345441RC.IMG", "/path/to/100%/M1347345441RC.IMG")
    assert success_iou > 0.15


def test_windowed_pair_loading():
    """Verify rasterio windowed reads return normalized (512, 512, 1) uint8 ImageData."""
    refs = get_available_lroc_references("TMC-2")
    ref_map = {r["filename"]: r["filepath"] for r in refs}

    src, ref, meta = load_windowed_pair(
        instrument="TMC-2",
        ch2_product_id="ch2_tmc_ncn_20260813T0627378557_d_img_d18",
        lroc_filename="M1347345441RC.IMG",
        lroc_filepath=ref_map["M1347345441RC.IMG"],
        target_crop_size=512,
    )

    assert src.array.shape == (512, 512, 1)
    assert ref.array.shape == (512, 512, 1)
    assert src.array.dtype == np.uint8
    assert ref.array.dtype == np.uint8
    assert meta["footprint_iou"] > 0.15


def test_transform_decomposition():
    """Verify homography decomposition accurately computes shift, rotation, scale, shear."""
    H = np.array([
        [0.9939, -0.0010, 0.8208],
        [-0.0028, 0.9972, 0.6236],
        [-1.07e-5, -2.62e-6, 1.0]
    ])
    decomp = decompose_homography(H)
    assert abs(decomp["shift_x"] - 0.8208) < 1e-3
    assert abs(decomp["shift_y"] - 0.6236) < 1e-3
    assert abs(decomp["rotation_deg"]) < 1.0
    assert abs(decomp["scale_avg"] - 1.0) < 0.05


def test_export_services():
    """Verify PDF and JSON generation return non-empty valid artifacts with planetary telemetry."""
    import re

    run_data = {
        "instrument": "TMC-2",
        "ch2_product_id": "ch2_tmc_ncn_20260813T0627378557_d_img_d18",
        "lroc_filename": "M1347345441RC.IMG",
        "inlier_matches": 49,
        "total_matches": 50,
        "inlier_ratio": 0.98,
        "rmse_pixels": 0.668,
        "coverage": 68.75,
        "processing_time_seconds": 0.22,
        "footprint_iou": 0.842,
        "decomposition": {"shift_x": 0.82, "shift_y": 0.62, "shift_total": 1.03, "rotation_deg": -0.16, "scale_x": 0.99, "scale_y": 1.0, "scale_avg": 0.99, "shear": 0.0, "perspective": 1e-5},
    }
    json_export = generate_json_export(run_data)
    parsed = json.loads(json_export)
    assert parsed["inlier_matches"] == 49
    assert "planetary_metadata" in parsed
    assert "ch2" in parsed["planetary_metadata"]
    assert "lroc" in parsed["planetary_metadata"]

    dummy_img = np.zeros((512, 512), dtype=np.uint8)
    pdf_bytes = generate_pdf_report(run_data, {"moving": dummy_img, "reference": dummy_img, "registered": dummy_img, "difference": dummy_img})
    assert len(pdf_bytes) > 5000
    assert pdf_bytes.startswith(b"%PDF")

    # Verify 4-page publication dossier
    count_match = re.search(rb"/Type\s*/Pages.*?/Count\s+(\d+)", pdf_bytes, re.DOTALL)
    assert count_match is not None, "Could not find /Count in PDF"
    assert int(count_match.group(1)) == 4, f"Expected 4 pages in PDF dossier, found {count_match.group(1)}"


def test_planetary_metadata_service():
    """Verify planetary telemetry extraction for both TMC-2 and OHRC missions."""
    tmc_meta = get_ch2_metadata("TMC-2", "ch2_tmc_ncn_20260813T0627378557_d_img_d18")
    assert tmc_meta["instrument"] == "TMC-2"
    assert tmc_meta["pixel_resolution_m_px"] == 5.48
    assert abs(tmc_meta["solar_incidence_deg"] - 39.06) < 0.1

    ohrc_meta = get_ch2_metadata("OHRC", "ch2_ohr_ncp_20260103T1005177268_d_img_d18")
    assert ohrc_meta["instrument"] == "OHRC"
    assert ohrc_meta["pixel_resolution_m_px"] == 0.25
    assert ohrc_meta["orbit_number"] == 28372

    lroc_meta = get_lroc_metadata("M1347345441RC.IMG", "")
    assert abs(lroc_meta["solar_incidence_deg"] - 36.94) < 0.1
    assert lroc_meta["orbit_number"] == 49487

    combined = get_combined_planetary_metadata(
        instrument="TMC-2",
        ch2_product_id="ch2_tmc_ncn_20260813T0627378557_d_img_d18",
        lroc_filename="M1347345441RC.IMG",
        lroc_filepath="",
        footprint_iou=0.842,
    )
    cross = combined["cross_sensor"]
    assert abs(cross["delta_solar_incidence_deg"] - 2.12) < 0.05
    assert "Excellent" in cross["solar_illumination_compatibility"]


def test_visual_verification_helpers():
    """Verify 8x8 checkerboard mosaic and false-color multi-band anaglyph generation."""
    img1 = np.full((256, 256), 40, dtype=np.uint8)
    img2 = np.full((256, 256), 220, dtype=np.uint8)

    cb = make_checkerboard_image(img1, img2, num_cells=8)
    assert cb.shape == (256, 256)
    assert cb.dtype == np.uint8
    # Top-left cell (0, 0) should be img1 (40), (0, 1) cell should be img2 (220)
    assert cb[10, 10] == 40
    assert cb[10, 50] == 220

    fc = make_false_color_overlay(img1, img2)
    assert fc.shape == (256, 256, 3)
    assert fc.dtype == np.uint8


def test_streamlit_apptest_flow():
    """Verify Streamlit app initial run, failure preset, and dual warning banners."""
    at = AppTest.from_file("/home/kode/gh-projs/sih2k26/isroPS/app/app.py", default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    assert at.sidebar.radio[0].value == "TMC-2"

    # Click Known failure case button
    at.sidebar.button[0].click().run()
    assert len(at.exception) == 0
    assert at.sidebar.radio[0].value == "TMC-2"
    assert at.sidebar.selectbox[1].value == "M1225104036LC.IMG"

    # Check that warning banners were produced
    warning_stacks = [m.value for m in at.markdown if '<div class="warning-stack">' in m.value]
    assert len(warning_stacks) == 1
    assert "Footprint IoU 0.000" in warning_stacks[0]
    assert "Low inlier ratio" in warning_stacks[0]


def test_compositor_html_sbs_structure():
    """Verify side-by-side view contains 1:1 square aspect ratio, 3 cards with names, and green dots on 3rd image only."""
    from app.components.compositor import build_compositor_html

    dummy = np.zeros((512, 512), dtype=np.uint8)
    tiepoints = {
        "moving": [[100.0, 150.0], [200.0, 250.0]],
        "ref": [[102.0, 151.0], [201.0, 249.0]],
        "inlier_mask": [True, False],
        "residuals_px": [0.5, 3.2],
    }
    html = build_compositor_html(
        moving_arr=dummy,
        reference_arr=dummy,
        registered_arr=dummy,
        tiepoints=tiepoints,
        instrument_label="CH-2 OHRC",
        reference_label="LROC NAC (pole_south.tiff)",
        canvas_size=560,
    )

    # 1:1 Aspect ratio in CSS
    assert "aspect-ratio: 1 / 1" in html

    # 3 Separate canvases for the 3 side-by-side cards
    assert 'id="sbsCanvas1"' in html
    assert 'id="sbsCanvas2"' in html
    assert 'id="sbsCanvas3"' in html

    # Names under each image in order
    assert "1. Moving" in html
    assert "CH-2 OHRC" in html
    assert "2. Reference" in html
    assert "LROC NAC (pole_south.tiff)" in html
    assert "3. Mapped" in html
    assert "Registered Output" in html

    # Green inlier dots and point-to-point match line logic on sbsCanvas3
    assert '#3FD68C' in html
    assert 'drawMatchLine' in html
    assert 'btnVectors' in html
    assert 'showVectors' in html

    # Main stage swipe tiepoints rendering intact
    assert "renderTiepoints()" in html


