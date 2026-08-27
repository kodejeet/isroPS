"""Streamlit Results & Metrics Card Scorecard Component."""

import json
from dataclasses import asdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from lunar_correspondence.io.metadata import EvaluationResult, RegistrationResult


def render_metrics_scorecard(eval_result: EvaluationResult):
    """Render quantitative evaluation metric cards and ground truth status.

    Args:
        eval_result: EvaluationResult metric object.
    """
    st.subheader("📊 Quantitative Evaluation Metrics")

    # Downsampling warning banner if applicable
    if eval_result.scale_factor < 1.0:
        st.info(
            f"ℹ️ **Prototype Processing Resolution Active:** Image was downsampled by a scale factor of "
            f"**{eval_result.scale_factor:.4f}** (to respect `processing.max_dimension`). "
            f"Keypoint match coordinates in the exported CSV are automatically rescaled back to full-resolution pixel space."
        )

    # Metric scorecard grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", f"{eval_result.total_matches}")
        st.metric("Inlier Matches", f"{eval_result.inlier_matches}")
    with col2:
        st.metric("Inlier Ratio", f"{eval_result.inlier_ratio:.2%}")
        st.metric("Grid Spatial Coverage", f"{eval_result.coverage:.1f}%")
    with col3:
        rmse_str = (
            f"{eval_result.rmse_pixels:.3f} px"
            if eval_result.rmse_pixels is not None
            else "N/A"
        )
        st.metric("RMSE Error", rmse_str)
        med_str = (
            f"{eval_result.median_error_pixels:.3f} px"
            if eval_result.median_error_pixels is not None
            else "N/A"
        )
        st.metric("Median Error", med_str)
    with col4:
        st.metric("Processing Time", f"{eval_result.processing_time_seconds:.3f} s")
        st.metric("Processing Scale", f"{eval_result.scale_factor:.4f}")

    # Ground truth status label
    st.caption(
        "📌 **Ground Truth Notice:** Not available for real-image uploads. "
        "Metrics reflect model self-consistency (RANSAC inlier reprojection error & spatial grid coverage), "
        "not comparison to a known reference ground-truth transform."
    )


def prepare_matches_csv(
    reg_result: RegistrationResult, scale_factor: float = 1.0
) -> str:
    """Prepare CSV string of match correspondences rescaled to full resolution space.

    Args:
        reg_result: RegistrationResult containing match_set.
        scale_factor: Scale factor used during processing (for coordinate rescaling).

    Returns:
        CSV string content.
    """
    scale = scale_factor if scale_factor > 0 else 1.0
    match_set = reg_result.match_set

    pts_src = match_set.source_points / scale
    pts_ref = match_set.reference_points / scale
    conf = (
        match_set.confidence
        if match_set.confidence is not None
        else np.zeros(len(pts_src), dtype=np.float32)
    )
    inliers = (
        match_set.inlier_mask
        if match_set.inlier_mask is not None
        else np.ones(len(pts_src), dtype=bool)
    )

    df = pd.DataFrame(
        {
            "source_x": pts_src[:, 0] if len(pts_src) > 0 else [],
            "source_y": pts_src[:, 1] if len(pts_src) > 0 else [],
            "reference_x": pts_ref[:, 0] if len(pts_ref) > 0 else [],
            "reference_y": pts_ref[:, 1] if len(pts_ref) > 0 else [],
            "confidence_or_descriptor_distance": conf,
            "inlier": inliers,
        }
    )
    return df.to_csv(index=False)


def render_download_buttons(
    reg_result: RegistrationResult,
    eval_result: EvaluationResult,
    matches_canvas: np.ndarray,
    overlay_canvas: np.ndarray,
):
    """Render download buttons for images, visualizations, CSV, and JSON metrics.

    All buttons pull data cleanly from st.session_state structures.
    """
    st.subheader("📥 Export & Download Artifacts")
    c1, c2, c3, c4, c5 = st.columns(5)

    # 1. Registered Warped Image
    reg_img = reg_result.registered_image
    if reg_img.ndim == 3 and reg_img.shape[2] == 1:
        reg_img_bytes = cv2.imencode(".png", reg_img[:, :, 0])[1].tobytes()
    elif reg_img.ndim == 3 and reg_img.shape[2] == 3:
        reg_img_bytes = cv2.imencode(".png", cv2.cvtColor(reg_img, cv2.COLOR_RGB2BGR))[
            1
        ].tobytes()
    else:
        reg_img_bytes = cv2.imencode(".png", reg_img)[1].tobytes()

    with c1:
        st.download_button(
            label="🖼️ Warped Image",
            data=reg_img_bytes,
            file_name="registered_warped_output.png",
            mime="image/png",
        )

    # 2. Match Visualization
    matches_bytes = cv2.imencode(
        ".png", cv2.cvtColor(matches_canvas, cv2.COLOR_RGB2BGR)
    )[1].tobytes()
    with c2:
        st.download_button(
            label="🔍 Match Lines PNG",
            data=matches_bytes,
            file_name="match_visualization.png",
            mime="image/png",
        )

    # 3. Registration Overlay
    overlay_bytes = cv2.imencode(
        ".png", cv2.cvtColor(overlay_canvas, cv2.COLOR_RGB2BGR)
    )[1].tobytes()
    with c3:
        st.download_button(
            label="🎨 Overlay PNG",
            data=overlay_bytes,
            file_name="registration_overlay.png",
            mime="image/png",
        )

    # 4. Matches CSV
    csv_data = prepare_matches_csv(reg_result, scale_factor=eval_result.scale_factor)
    with c4:
        st.download_button(
            label="📄 Matches CSV",
            data=csv_data,
            file_name="correspondence_matches.csv",
            mime="text/csv",
        )

    # 5. Metrics JSON
    json_data = json.dumps(asdict(eval_result), indent=2)
    with c5:
        st.download_button(
            label="📊 Metrics JSON",
            data=json_data,
            file_name="evaluation_metrics.json",
            mime="application/json",
        )
