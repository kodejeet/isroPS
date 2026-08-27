"""Streamlit Visualization Plots Component."""

import numpy as np
import streamlit as st

from lunar_correspondence.io.metadata import ImageData, RegistrationResult
from lunar_correspondence.visualization.matches import draw_match_lines
from lunar_correspondence.visualization.registration import plot_registration_overlay


def render_visualizations(
    source_data: ImageData,
    ref_data: ImageData,
    reg_result: RegistrationResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Render interactive tabs for match lines, false-color overlay, and registered image.

    Args:
        source_data: Source ImageData.
        ref_data: Reference ImageData.
        reg_result: Computed RegistrationResult.

    Returns:
        Tuple of (matches_canvas, overlay_canvas) RGB numpy arrays.
    """
    st.subheader("🖼️ Registration & Correspondence Visualizations")

    # Generate canvases using existing visualization engine
    matches_canvas = draw_match_lines(
        source_data.array,
        ref_data.array,
        reg_result.match_set,
        max_matches_to_show=120,
    )

    overlay_canvas = plot_registration_overlay(
        ref_data.array,
        reg_result.registered_image,
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 Feature Matches (Inliers Green / Outliers Red)",
            "🎨 False-Color Overlay (Red=Source, Green=Ref)",
            "🖼️ Warped Registered Source Image",
        ]
    )

    with tab1:
        st.image(
            matches_canvas,
            caption="Feature Correspondences (Left: Source / Right: Reference). Green lines = RANSAC inliers, Red lines = outliers.",
            use_container_width=True,
        )

    with tab2:
        st.image(
            overlay_canvas,
            caption="False Color Overlay. Well-aligned terrain features merge into gray/white; misalignments appear as red/green color fringes.",
            use_container_width=True,
        )

    with tab3:
        # Prepare display copy of warped image
        warped = reg_result.registered_image
        if warped.ndim == 3 and warped.shape[2] == 1:
            warped_disp = warped[:, :, 0]
        else:
            warped_disp = warped

        st.image(
            warped_disp,
            caption="Warped Source Image aligned into Reference Canvas geometry.",
            use_container_width=True,
        )

    return matches_canvas, overlay_canvas
