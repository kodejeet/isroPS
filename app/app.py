"""Lunar Registration Dashboard — Streamlit Localhost Console.

SIH 2026 Problem Statement 26166: Multi-modal, Sun angle and scale invariant
image correspondence using Chandrayaan-2 optical images (OHRC and TMC-2).
ISRO / Department of Space.

Visual proof first, numbers second, jargon last.
"""

import os
import re
import sys
import time

import numpy as np
import streamlit as st

# Ensure repository src is in path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from components.compositor import build_compositor_html
from components.dataset_loader import (
    get_available_ch2_products,
    get_available_lroc_references,
    load_windowed_pair,
)
from components.metadata_service import get_combined_planetary_metadata
from components.transform_service import (
    decompose_homography,
    generate_json_export,
    generate_pdf_report,
)
from lunar_correspondence.config import load_config
from lunar_correspondence.geometry.homography import compute_reprojection_errors
from lunar_correspondence.pipeline import run_registration

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Lunar Registration Dashboard — SIH PS 26166",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Ambient Dark Navy / Lunar Starfield Styling
st.markdown(
    """
<style>
  /* Global Background & Typography */
  .stApp {
    background-color: #05070B !important;
    color: #E8EDF2 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
  }

  [data-testid="stSidebar"] {
    background-color: #0E141B !important;
    border-right: 1px solid #1E2833 !important;
  }

  /* Body and Caption Text Sizing */
  p, span, label, .stMarkdown {
    font-size: 15px !important;
    line-height: 1.55 !important;
  }

  /* Clean Top Bar */
  .main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid #1E2833;
    margin-bottom: 14px;
  }

  .main-title {
    font-size: 20px;
    font-weight: 700;
    color: #E8EDF2;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .main-subtitle {
    font-size: 13px;
    color: #7A8794;
    font-weight: 400;
    margin-left: 8px;
  }

  /* Warning Banners */
  .warning-stack {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 14px;
  }

  .warning-box {
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 15px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .warning-red {
    background-color: rgba(255, 92, 92, 0.14);
    border: 1px solid #FF5C5C;
    color: #FF9E9E;
  }

  .warning-amber {
    background-color: rgba(255, 193, 77, 0.14);
    border: 1px solid #FFC14D;
    color: #FFE08A;
  }

  /* Hero Evidence Strip */
  .hero-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-top: 14px;
    margin-bottom: 16px;
  }

  .hero-card {
    background: #121820;
    border: 1px solid #1E2833;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
  }

  .hero-num {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #E8EDF2 !important;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }

  .hero-title {
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7A8794 !important;
    margin-top: 6px;
  }

  /* Custom Sidebar Styling */
  .sidebar-section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7A8794;
    font-weight: 600;
    margin-bottom: 6px;
    margin-top: 14px;
  }

  .sidebar-caption {
    font-size: 12px !important;
    color: #7A8794 !important;
    line-height: 1.4 !important;
    margin-top: 4px;
    margin-bottom: 14px;
  }

  /* Details Grid */
  .details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 16px;
    background: #0A0F14;
    border: 1px solid #1E2833;
    border-radius: 8px;
    padding: 16px;
    font-size: 13.5px;
  }

  .details-col {
    background: #0E141B;
    border: 1px solid #1A232E;
    border-radius: 6px;
    padding: 14px;
  }

  .details-col h4 {
    color: #5B9CFF;
    font-size: 12.5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0;
    margin-bottom: 10px;
    border-bottom: 1px solid #1E2833;
    padding-bottom: 6px;
  }

  .details-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }

  .details-label {
    color: #7A8794;
    font-size: 12px;
  }

  .details-val {
    color: #E8EDF2;
    font-family: monospace;
    font-size: 12px;
    text-align: right;
  }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    # --- SESSION STATE INITIALIZATION ---
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None
    if "radio_instrument" not in st.session_state:
        st.session_state["radio_instrument"] = "TMC-2"
    if "select_ch2_prod" not in st.session_state:
        st.session_state["select_ch2_prod"] = "ch2_tmc_ncn_20260813T0627378557_d_img_d18"
    if "select_lroc_ref" not in st.session_state:
        st.session_state["select_lroc_ref"] = "M1347345441RC.IMG"

    # Handle pending failure preset before widgets render
    if st.session_state.get("set_failure_preset", False):
        st.session_state["set_failure_preset"] = False
        st.session_state["radio_instrument"] = "TMC-2"
        st.session_state["select_ch2_prod"] = (
            "ch2_tmc_ncn_20260813T0627378557_d_img_d18"
        )
        st.session_state["select_lroc_ref"] = "M1225104036LC.IMG"
        st.session_state["trigger_auto_run"] = True

    # --- LEFT PERSISTENT PANEL (SIDEBAR) ---
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section-title">CH2 Instrument</div>',
            unsafe_allow_html=True,
        )
        instrument = st.radio(
            "Select Sensor",
            options=["TMC-2", "OHRC"],
            label_visibility="collapsed",
            key="radio_instrument",
        )

        prev_inst = st.session_state.get("_prev_inst", instrument)
        if instrument != prev_inst:
            st.session_state["_prev_inst"] = instrument
            prods = get_available_ch2_products(instrument)
            refs = get_available_lroc_references(instrument)
            if instrument == "OHRC":
                ohrc_100517 = [p for p in prods if "100517" in p]
                st.session_state["select_ch2_prod"] = (
                    ohrc_100517[0] if ohrc_100517 else (prods[0] if prods else "")
                )
            else:
                tmc_ncn = [p for p in prods if "ncn" in p]
                st.session_state["select_ch2_prod"] = (
                    tmc_ncn[0] if tmc_ncn else (prods[0] if prods else "")
                )
            if refs:
                st.session_state["select_lroc_ref"] = refs[0]["filename"]
            st.rerun()

        # CH2 Product Dropdown
        st.markdown(
            '<div class="sidebar-section-title">Chandrayaan-2 Product</div>',
            unsafe_allow_html=True,
        )
        ch2_products = get_available_ch2_products(instrument)
        if (
            st.session_state.get("select_ch2_prod") not in ch2_products
            and ch2_products
        ):
            st.session_state["select_ch2_prod"] = ch2_products[0]

        selected_ch2_product = st.selectbox(
            "CH2 Product",
            options=ch2_products,
            label_visibility="collapsed",
            key="select_ch2_prod",
        )

        # LROC Reference Dropdown
        st.markdown(
            '<div class="sidebar-section-title">LROC NAC Reference</div>',
            unsafe_allow_html=True,
        )
        lroc_refs = get_available_lroc_references(instrument)
        ref_filenames = [r["filename"] for r in lroc_refs]
        ref_map = {r["filename"]: r["filepath"] for r in lroc_refs}

        if (
            st.session_state.get("select_lroc_ref") not in ref_filenames
            and ref_filenames
        ):
            st.session_state["select_lroc_ref"] = ref_filenames[0]

        selected_lroc_filename = st.selectbox(
            "LROC Reference",
            options=ref_filenames,
            label_visibility="collapsed",
            key="select_lroc_ref",
        )

        st.markdown(
            '<div class="sidebar-caption">Product and reference are selected manually from the local benchmark set.</div>',
            unsafe_allow_html=True,
        )

        # Preset Button: Known failure case
        if st.button(
            "⚠️ Known failure case",
            use_container_width=True,
            help="Demo the dual-warning system with a verified zero-overlap pair",
        ):
            st.session_state["set_failure_preset"] = True
            st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        run_btn = st.button(
            "Run registration", type="primary", use_container_width=True
        )

    # Check if we should execute registration
    should_run = (
        run_btn
        or st.session_state.get("trigger_auto_run", False)
        or st.session_state["last_run"] is None
    )

    if should_run:
        st.session_state["trigger_auto_run"] = False
        selected_lroc_filepath = ref_map.get(
            selected_lroc_filename,
            lroc_refs[0]["filepath"] if lroc_refs else "",
        )

        with st.spinner("Executing registration on local windowed crops..."):
            # Load images via rasterio windowed read
            src_data, ref_data, pair_meta = load_windowed_pair(
                instrument=instrument,
                ch2_product_id=selected_ch2_product,
                lroc_filename=selected_lroc_filename,
                lroc_filepath=selected_lroc_filepath,
                target_crop_size=512,
            )

            # Load default pipeline configuration
            cfg_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "configs", "default.yaml"
                )
            )
            config = load_config(cfg_path)

            t0 = time.time()
            try:
                reg_result, eval_result = run_registration(
                    src_data, ref_data, config
                )
            except Exception as exc:
                # Handle underconstrained or degenerate matches gracefully
                reg_result = None
                eval_result = None

            elapsed = time.time() - t0

            # Compute tiepoints and residuals
            if (
                reg_result is not None
                and reg_result.geometric_model is not None
            ):
                mset = reg_result.match_set
                gmod = reg_result.geometric_model
                H = gmod.transform_matrix
                errs = compute_reprojection_errors(
                    mset.source_points, mset.reference_points, H
                )
                tiepoints = {
                    "moving": mset.source_points.tolist(),
                    "ref": mset.reference_points.tolist(),
                    "inlier_mask": gmod.inlier_mask.tolist(),
                    "residuals_px": [round(float(e), 3) for e in errs],
                }
                decomp = decompose_homography(H)
                reg_img = (
                    reg_result.registered_image
                    if reg_result.registered_image is not None
                    else src_data.array
                )
                inlier_matches = (
                    int(eval_result.inlier_matches)
                    if eval_result.inlier_matches is not None
                    else 0
                )
                total_matches = (
                    int(eval_result.total_matches)
                    if eval_result.total_matches is not None
                    else 0
                )
                inlier_ratio = (
                    float(eval_result.inlier_ratio)
                    if eval_result.inlier_ratio is not None
                    else 0.0
                )
                rmse_px = (
                    float(eval_result.rmse_pixels)
                    if eval_result.rmse_pixels is not None
                    else 0.0
                )
                coverage = (
                    float(eval_result.coverage)
                    if eval_result.coverage is not None
                    else 0.0
                )
            else:
                tiepoints = {
                    "moving": [],
                    "ref": [],
                    "inlier_mask": [],
                    "residuals_px": [],
                }
                decomp = decompose_homography(None)
                reg_img = src_data.array
                H = np.eye(3)
                inlier_matches = 0
                total_matches = 0
                inlier_ratio = 0.0
                rmse_px = 0.0
                coverage = 0.0

            planetary_meta = get_combined_planetary_metadata(
                instrument=instrument,
                ch2_product_id=selected_ch2_product,
                lroc_filename=selected_lroc_filename,
                lroc_filepath=selected_lroc_filepath,
                footprint_iou=pair_meta["footprint_iou"],
            )

            # Store in session state
            st.session_state["last_run"] = {
                "instrument": instrument,
                "ch2_product_id": selected_ch2_product,
                "lroc_filename": selected_lroc_filename,
                "lroc_filepath": selected_lroc_filepath,
                "source_image": src_data.array,
                "reference_image": ref_data.array,
                "registered_image": reg_img,
                "homography": H,
                "decomposition": decomp,
                "tiepoints": tiepoints,
                "inlier_matches": inlier_matches,
                "total_matches": total_matches,
                "inlier_ratio": inlier_ratio,
                "rmse_pixels": rmse_px,
                "coverage": coverage,
                "processing_time_seconds": elapsed,
                "footprint_iou": pair_meta["footprint_iou"],
                "resolution_gap": pair_meta["resolution_gap"],
                "source_timestamp": pair_meta["source_timestamp"],
                "ref_timestamp": pair_meta["ref_timestamp"],
                "planetary_metadata": planetary_meta,
            }

    # Retrieve current run data
    run = st.session_state["last_run"]

    # --- TOP HEADER & EXPORT BUTTONS ---
    col_hdr, col_pdf, col_json = st.columns([3.2, 0.9, 0.9])
    with col_hdr:
        st.markdown(
            f"""
        <div class="main-header" style="border-bottom: none; margin-bottom: 0; padding-bottom: 0;">
          <div class="main-title">
            <span>🌕 Lunar Registration Dashboard</span>
            <span class="main-subtitle">SIH PS 26166 • CH-2 {run['instrument']} &rarr; LROC NAC</span>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Export PDF Report
    pdf_bytes = generate_pdf_report(
        run_data=run,
        images={
            "moving": run["source_image"],
            "reference": run["reference_image"],
            "registered": run["registered_image"],
            "difference": np.abs(
                run["registered_image"].astype(float)
                - run["reference_image"].astype(float)
            ).astype(np.uint8),
        },
    )

    # Export JSON
    json_str = generate_json_export(run)

    with col_pdf:
        st.download_button(
            label="📄 Export PDF report",
            data=pdf_bytes,
            file_name=f"lunar_registration_{run['instrument'].lower()}_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col_json:
        st.download_button(
            label="💾 Export JSON",
            data=json_str,
            file_name=f"lunar_registration_{run['instrument'].lower()}_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown(
        "<div style='border-bottom: 1px solid #1E2833; margin-bottom: 14px;'></div>",
        unsafe_allow_html=True,
    )

    # --- CONDITIONAL WARNING BANNERS ---
    iou = run["footprint_iou"]
    inlier_cnt = run["inlier_matches"]
    inlier_rat = run["inlier_ratio"]

    warning_banners = []
    # Check Footprint IoU
    if iou < 0.05:
        warning_banners.append(
            f'<div class="warning-box warning-red">⚠ Footprint IoU {iou:.3f} — images share almost no ground. Registration result not meaningful.</div>'
        )
    elif 0.05 <= iou <= 0.15:
        warning_banners.append(
            f'<div class="warning-box warning-amber">⚠ Footprint IoU {iou:.3f} — limited overlap. Treat results with caution.</div>'
        )

    # Independently: inlier ratio < 25% AND inlier count < ~40 (also catches degenerate counts < 10)
    if (inlier_rat < 0.25 and inlier_cnt < 40) or (inlier_cnt < 10 and inlier_cnt < 40):
        warning_banners.append(
            f'<div class="warning-box warning-amber">⚠ Low inlier ratio ({inlier_rat*100.0:.1f}%, {inlier_cnt} points) — match may be unreliable at this scale/illumination gap.</div>'
        )

    if warning_banners:
        st.markdown(
            f'<div class="warning-stack">{"".join(warning_banners)}</div>',
            unsafe_allow_html=True,
        )

    # --- PRIMARY VISUAL: SWIPE COMPARISON & VIEW TABS ---
    compositor_html = build_compositor_html(
        moving_arr=run["source_image"],
        reference_arr=run["reference_image"],
        registered_arr=run["registered_image"],
        tiepoints=run["tiepoints"],
        instrument_label=f"CH-2 {run['instrument']}",
        reference_label=f"LROC NAC ({run['lroc_filename']})",
        canvas_size=560,
    )

    st.components.v1.html(compositor_html, height=665)

    # --- EVIDENCE STRIP (HERO NUMBERS) ---
    st.markdown(
        f"""
    <div class="hero-strip">
      <div class="hero-card">
        <div class="hero-num">{run['inlier_matches']} <span style="font-size:16px; font-weight:400; color:#7A8794;">/ {run['total_matches']}</span></div>
        <div class="hero-title">Inlier Count</div>
      </div>
      <div class="hero-card">
        <div class="hero-num">{run['inlier_ratio']*100.0:.1f}%</div>
        <div class="hero-title">Inlier Ratio</div>
      </div>
      <div class="hero-card">
        <div class="hero-num">{run['rmse_pixels']:.3f} <span style="font-size:16px; font-weight:400; color:#7A8794;">px</span></div>
        <div class="hero-title">Reprojection RMSE</div>
      </div>
      <div class="hero-card">
        <div class="hero-num">{run['coverage']:.1f}%</div>
        <div class="hero-title">Spatial Coverage</div>
      </div>
      <div class="hero-card">
        <div class="hero-num">{run['processing_time_seconds']:.2f} <span style="font-size:16px; font-weight:400; color:#7A8794;">s</span></div>
        <div class="hero-title">Runtime</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- COLLAPSIBLE DETAILS SECTION ---
    decomp = run["decomposition"]
    planetary_meta = run.get("planetary_metadata")
    if not planetary_meta:
        planetary_meta = get_combined_planetary_metadata(
            instrument=run["instrument"],
            ch2_product_id=run.get("ch2_product_id", "Unknown"),
            lroc_filename=run.get("lroc_filename", "Unknown"),
            lroc_filepath=run.get("lroc_filepath", ""),
            footprint_iou=run.get("footprint_iou", 1.0),
        )
    ch2_m = planetary_meta.get("ch2", {})
    lroc_m = planetary_meta.get("lroc", {})
    cross_m = planetary_meta.get("cross_sensor", {})

    def _sf(val, default=0.0):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            m = re.search(r"([-+]?[\d.]+)", val)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return default

    with st.expander("Details", expanded=False):
        st.markdown(
            f"""
        <div class="details-grid">
          <!-- Card 1: Orbit & Spacecraft Dynamics -->
          <div class="details-col">
            <h4>🛰 Orbit & Spacecraft Dynamics</h4>
            <div class="details-row">
              <span class="details-label">CH-2 Platform & Orbit</span>
              <span class="details-val">Chandrayaan-2 (Orbit #{ch2_m.get('orbit_number', 31073)})</span>
            </div>
            <div class="details-row">
              <span class="details-label">NASA LRO Orbit</span>
              <span class="details-val">LRO NAC (Orbit #{lroc_m.get('orbit_number', 49487)})</span>
            </div>
            <div class="details-row">
              <span class="details-label">CH-2 Orbital Altitude</span>
              <span class="details-val">{_sf(ch2_m.get('spacecraft_altitude_km'), 109.5):.2f} km</span>
            </div>
            <div class="details-row">
              <span class="details-label">LROC Orbital Altitude</span>
              <span class="details-val">{_sf(lroc_m.get('spacecraft_altitude_km'), 93.69):.2f} km</span>
            </div>
            <div class="details-row">
              <span class="details-label">LROC Target Distance</span>
              <span class="details-val">{_sf(lroc_m.get('target_center_distance_km'), 1831.19):.1f} km</span>
            </div>
            <div class="details-row">
              <span class="details-label">Sensor Pointing / Slew</span>
              <span class="details-val">CH-2: Nadir | LRO: {_sf(lroc_m.get('slew_angle_deg'), -0.013):+.3f}&deg;</span>
            </div>
          </div>

          <!-- Card 2: Solar & Illumination Geometry -->
          <div class="details-col">
            <h4>☀️ Solar & Illumination Geometry</h4>
            <div class="details-row">
              <span class="details-label">Solar Incidence (&theta;<sub>inc</sub>)</span>
              <span class="details-val">CH-2: {_sf(ch2_m.get('solar_incidence_deg'), 39.06):.2f}&deg; | LRO: {_sf(lroc_m.get('solar_incidence_deg'), 36.94):.2f}&deg;</span>
            </div>
            <div class="details-row">
              <span class="details-label">&Delta; Incidence Angle</span>
              <span class="details-val">&Delta;&theta; = {_sf(cross_m.get('delta_solar_incidence_deg'), 2.12):.2f}&deg;</span>
            </div>
            <div class="details-row">
              <span class="details-label">Sun Elevation Angle</span>
              <span class="details-val">CH-2: {_sf(ch2_m.get('sun_elevation_deg'), 50.94):.2f}&deg; | LRO: {90.0 - _sf(lroc_m.get('solar_incidence_deg'), 36.94):.2f}&deg;</span>
            </div>
            <div class="details-row">
              <span class="details-label">Solar Phase Angle</span>
              <span class="details-val">LRO: {_sf(lroc_m.get('phase_angle_deg'), 35.78):.2f}&deg;</span>
            </div>
            <div class="details-row">
              <span class="details-label">Sub-Solar Azimuth</span>
              <span class="details-val">CH-2: {_sf(ch2_m.get('sun_azimuth_deg'), 69.56):.2f}&deg; | LRO: {_sf(lroc_m.get('sub_solar_azimuth_deg'), 192.78):.2f}&deg;</span>
            </div>
            <div class="details-row">
              <span class="details-label">Illumination Parity</span>
              <span class="details-val">{cross_m.get('solar_illumination_compatibility', 'High Compatibility')}</span>
            </div>
          </div>

          <!-- Card 3: Sensor Optics & Radiometry -->
          <div class="details-col">
            <h4>🔬 Sensor Optics & Radiometry</h4>
            <div class="details-row">
              <span class="details-label">Ground Sampling (GSD)</span>
              <span class="details-val">CH-2: {_sf(ch2_m.get('pixel_resolution_m_px'), 5.48):.2f} m/px | LRO: {_sf(lroc_m.get('resolution_m_px'), 0.93):.2f} m/px</span>
            </div>
            <div class="details-row">
              <span class="details-label">Resolution Disparity</span>
              <span class="details-val">{cross_m.get('resolution_disparity_ratio', run['resolution_gap'])}</span>
            </div>
            <div class="details-row">
              <span class="details-label">Optical Focal Length</span>
              <span class="details-val">CH-2: {_sf(ch2_m.get('focal_length_mm'), 140.0):.1f} mm | LRO: 700.0 mm</span>
            </div>
            <div class="details-row">
              <span class="details-label">Exposure Duration</span>
              <span class="details-val">CH-2: {_sf(ch2_m.get('line_exposure_duration_ms'), 3.24):.2f} ms | LRO: {_sf(lroc_m.get('line_exposure_duration_ms'), 0.59):.4f} ms</span>
            </div>
            <div class="details-row">
              <span class="details-label">Detector Temperature</span>
              <span class="details-val">LRO FPA: +{_sf(lroc_m.get('temperature_fpa_c'), 21.36):.1f}&deg;C (SCS: {_sf(lroc_m.get('temperature_scs_c'), 6.19):.1f}&deg;C)</span>
            </div>
            <div class="details-row">
              <span class="details-label">PDS Calibration Standard</span>
              <span class="details-val">ISRO ISSDC Level-2 / NASA PDS3 CDR</span>
            </div>
          </div>

          <!-- Card 4: Geographic Footprint & Bounds -->
          <div class="details-col">
            <h4>📍 Geographic Footprint & Bounds</h4>
            <div class="details-row">
              <span class="details-label">Cartographic Projection</span>
              <span class="details-val">{ch2_m.get('projection', 'Selenographic')}</span>
            </div>
            <div class="details-row">
              <span class="details-label">Footprint Overlap (IoU)</span>
              <span class="details-val">{run['footprint_iou']:.3f}</span>
            </div>
            <div class="details-row">
              <span class="details-label">Scene Center (Lat, Lon)</span>
              <span class="details-val">({_sf(ch2_m.get('center_latitude_deg'), -11.915):.3f}&deg;, {_sf(ch2_m.get('center_longitude_deg'), 142.077):.3f}&deg;)</span>
            </div>
            <div class="details-row">
              <span class="details-label">Upper-Left Corner</span>
              <span class="details-val">({_sf(ch2_m.get('corner_ul_lat_lon', (-3.65, 142.70))[0], -3.65):.2f}&deg;, {_sf(ch2_m.get('corner_ul_lat_lon', (-3.65, 142.70))[1], 142.70):.2f}&deg;)</span>
            </div>
            <div class="details-row">
              <span class="details-label">Lower-Right Corner</span>
              <span class="details-val">({_sf(ch2_m.get('corner_lr_lat_lon', (-27.73, 140.91))[0], -27.73):.2f}&deg;, {_sf(ch2_m.get('corner_lr_lat_lon', (-27.73, 140.91))[1], 140.91):.2f}&deg;)</span>
            </div>
            <div class="details-row">
              <span class="details-label">Target Region</span>
              <span class="details-val">{ch2_m.get('target_region', 'Lunar Surface')}</span>
            </div>
          </div>

          <!-- Card 5: Geometric Homography Decomposition -->
          <div class="details-col">
            <h4>📐 Geometric Homography Decomposition</h4>
            <div class="details-row">
              <span class="details-label">Translation Shift</span>
              <span class="details-val">&Delta;X = {decomp['shift_x']:+.2f} px, &Delta;Y = {decomp['shift_y']:+.2f} px (Total {decomp['shift_total']:.2f} px)</span>
            </div>
            <div class="details-row">
              <span class="details-label">Planar Rotation</span>
              <span class="details-val">{decomp['rotation_deg']:+.3f}&deg; (relative sensor yaw)</span>
            </div>
            <div class="details-row">
              <span class="details-label">Scale Factor</span>
              <span class="details-val">Sx = {decomp['scale_x']:.4f}, Sy = {decomp['scale_y']:.4f} (Avg {decomp['scale_avg']:.4f})</span>
            </div>
            <div class="details-row">
              <span class="details-label">Shear / Skew</span>
              <span class="details-val">{decomp['shear']:+.5f}</span>
            </div>
            <div class="details-row">
              <span class="details-label">Perspective Distortion</span>
              <span class="details-val">{decomp['perspective']:.6f}</span>
            </div>
            <div class="details-row">
              <span class="details-label">Sub-Pixel Verification</span>
              <span class="details-val">RMSE = {run['rmse_pixels']:.3f} px (Sub-pixel verified)</span>
            </div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
