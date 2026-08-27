"""Streamlit Web Application Entry Point.

SIH 2026 Problem Statement 26166: Lunar Multimodal Image Correspondence Prototype.
Registers uploaded lunar images via the single orchestration function run_registration().
Preserves all results in st.session_state across Streamlit reruns and download interactions.
"""

import os
import sys
import time

import streamlit as st

# Add src to Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from components.plots import render_visualizations
from components.results import render_download_buttons, render_metrics_scorecard
from components.upload import get_display_preview, load_uploaded_image
from lunar_correspondence.config import load_config
from lunar_correspondence.pipeline import run_registration


def main():
    st.set_page_config(
        page_title="Lunar Image Registration — SIH26166",
        page_icon="🌕",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Page Header & Title
    st.title("🌕 Lunar Multimodal Image Correspondence — SIH 2026 PS 26166")
    st.caption(
        "*Register two overlapping lunar images and evaluate feature correspondence. "
        "Current prototype: SIFT + robust geometric registration.*"
    )
    st.divider()

    # Load defaults from configs/default.yaml
    default_config_path = os.path.abspath("configs/default.yaml")
    base_config = load_config(default_config_path)

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Registration Parameters")

    # Fixed method disclosures
    st.sidebar.text_input("Feature Extractor", value="SIFT (Baseline)", disabled=True)
    st.sidebar.text_input(
        "Descriptor Matcher", value="Lowe's Ratio Test (k=2)", disabled=True
    )

    # User tunable settings
    model_type = st.sidebar.selectbox(
        "Geometric Model",
        options=["homography", "affine"],
        index=0 if base_config["geometry"]["model_type"] == "homography" else 1,
        help="Homography (8-DOF projective) or Affine (6-DOF transformation).",
    )

    n_features = st.sidebar.slider(
        "Max Features (nfeatures)",
        min_value=500,
        max_value=10000,
        value=int(base_config["feature_extraction"]["sift"]["nfeatures"]),
        step=500,
    )

    ratio_thresh = st.sidebar.slider(
        "Lowe Ratio Threshold",
        min_value=0.50,
        max_value=0.95,
        value=float(base_config["matching"]["descriptor"]["ratio_test_threshold"]),
        step=0.05,
        help="Tighter values filter out ambiguous match pairs.",
    )

    ransac_thresh = st.sidebar.slider(
        "RANSAC Reproj Threshold (px)",
        min_value=0.5,
        max_value=10.0,
        value=float(base_config["geometry"]["ransac"]["reproj_threshold"]),
        step=0.5,
        help="Maximum distance error in pixels for RANSAC inliers.",
    )

    grid_rows = st.sidebar.slider(
        "Grid Coverage Rows",
        min_value=2,
        max_value=10,
        value=int(base_config["evaluation"]["grid_rows"]),
    )

    grid_cols = st.sidebar.slider(
        "Grid Coverage Cols",
        min_value=2,
        max_value=10,
        value=int(base_config["evaluation"]["grid_cols"]),
    )

    max_dimension = st.sidebar.number_input(
        "Processing Max Dimension (px)",
        min_value=1024,
        max_value=8192,
        value=int(base_config.get("processing", {}).get("max_dimension", 4096)),
        step=512,
        help="Images larger than this dimension will be downsampled for prototype processing.",
    )

    # Build dynamically updated configuration dictionary
    active_config = load_config(default_config_path)
    active_config["geometry"]["model_type"] = model_type
    active_config["feature_extraction"]["sift"]["nfeatures"] = n_features
    active_config["matching"]["descriptor"]["ratio_test_threshold"] = ratio_thresh
    active_config["geometry"]["ransac"]["reproj_threshold"] = ransac_thresh
    active_config["evaluation"]["grid_rows"] = grid_rows
    active_config["evaluation"]["grid_cols"] = grid_cols
    active_config["processing"]["max_dimension"] = max_dimension

    # --- IMAGE UPLOAD & PREVIEW SECTION ---
    st.subheader("1. Select Lunar Images & Metadata Hints")
    col_src, col_ref = st.columns(2)

    INSTRUMENT_OPTIONS = [
        "Auto",
        "OHRC",
        "TMC-2",
        "IIRS",
        "LRO_NAC",
        "SELENE",
        "UNKNOWN",
    ]

    with col_src:
        st.markdown("### Source / Moving Image")
        file_src = st.file_uploader(
            "Upload Source Image (PNG, JPG, TIFF, GeoTIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="uploader_source",
        )
        hint_src = st.selectbox(
            "Metadata hint (Source Sensor)",
            options=INSTRUMENT_OPTIONS,
            index=1,  # Default to OHRC
            key="hint_source",
        )

    with col_ref:
        st.markdown("### Reference / Fixed Image")
        file_ref = st.file_uploader(
            "Upload Reference Image (PNG, JPG, TIFF, GeoTIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="uploader_ref",
        )
        hint_ref = st.selectbox(
            "Metadata hint (Reference Sensor)",
            options=INSTRUMENT_OPTIONS,
            index=2,  # Default to TMC-2
            key="hint_ref",
        )

    # --- IIRS HONESTY GUARD WARNING ---
    if hint_src == "IIRS" or hint_ref == "IIRS":
        st.warning(
            "⚠️ **IIRS Hyperspectral Note:** IIRS is hyperspectral (~256 bands). "
            "This prototype's loader does not yet parse real IIRS hyperspectral cubes — "
            "if you upload a standard image file here, it will be processed as a generic multi-band image, "
            "not real IIRS spectral data. This tag is for bookkeeping only until real IIRS support lands."
        )

    # --- LOAD & PREVIEW IMAGES ---
    source_data = None
    ref_data = None

    if file_src is not None and file_ref is not None:
        try:
            source_data = load_uploaded_image(file_src, instrument_hint=hint_src)
            ref_data = load_uploaded_image(file_ref, instrument_hint=hint_ref)
        except ValueError as ve:
            st.error(f"❌ **File Load Error:** {ve!s}")
            return
        except Exception as e:
            st.error(f"❌ **Unexpected File Error:** {e!s}")
            return

        # Display side-by-side previews
        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            preview_src = get_display_preview(source_data)
            st.image(
                preview_src,
                caption=f"Source: {source_data.path} | Shape: {source_data.array.shape} | Hint: {source_data.metadata.instrument}",
                use_container_width=True,
            )
        with col_prev2:
            preview_ref = get_display_preview(ref_data)
            st.image(
                preview_ref,
                caption=f"Reference: {ref_data.path} | Shape: {ref_data.array.shape} | Hint: {ref_data.metadata.instrument}",
                use_container_width=True,
            )

        # --- REGISTRATION TRIGGER BUTTON ---
        st.divider()
        if st.button("🚀 REGISTER IMAGES", type="primary", use_container_width=True):
            try:
                start_t = time.time()
                with st.spinner(
                    "Executing SIFT baseline feature extraction, descriptor matching, and RANSAC homography..."
                ):
                    reg_res, eval_res = run_registration(
                        source=source_data,
                        reference=ref_data,
                        config=active_config,
                    )
                elapsed = time.time() - start_t

                # Store in session state to survive reruns and download clicks
                st.session_state["registration_result"] = reg_res
                st.session_state["evaluation_result"] = eval_res
                st.session_state["source_data"] = source_data
                st.session_state["ref_data"] = ref_data

                st.success(f"✅ Registration completed in {elapsed:.3f} seconds!")
            except Exception as e:
                err_msg = str(e)
                if (
                    "fewer than 4" in err_msg.lower()
                    or "not enough" in err_msg.lower()
                    or "none" in err_msg.lower()
                ):
                    st.error(
                        "⚠️ **Registration Failed:** Fewer than 4 geometrically valid correspondences were found. "
                        "Try uploading images with greater spatial overlap, or adjust the Lowe Ratio / RANSAC thresholds in the sidebar."
                    )
                else:
                    st.error(f"⚠️ **Registration Pipeline Error:** {err_msg}")

    # --- SESSION STATE RENDER BLOCK (CRITICAL FIX FOR STREAMLIT RERUNS & DOWNLOADS) ---
    if (
        "registration_result" in st.session_state
        and "evaluation_result" in st.session_state
    ):
        st.divider()
        reg_res = st.session_state["registration_result"]
        eval_res = st.session_state["evaluation_result"]
        stored_src = st.session_state["source_data"]
        stored_ref = st.session_state["ref_data"]

        # Render metric scorecard
        render_metrics_scorecard(eval_res)
        st.divider()

        # Render visualizations tabs
        matches_canvas, overlay_canvas = render_visualizations(
            source_data=stored_src,
            ref_data=stored_ref,
            reg_result=reg_res,
        )
        st.divider()

        # Render download buttons (reads cleanly from session state!)
        render_download_buttons(
            reg_result=reg_res,
            eval_result=eval_res,
            matches_canvas=matches_canvas,
            overlay_canvas=overlay_canvas,
        )


if __name__ == "__main__":
    main()
