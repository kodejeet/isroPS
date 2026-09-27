"""Compact technical ANSI streaming terminal logger for CHANDRAMAP.

Follows strict telemetry formatting for orbital and planetary correspondence:
- Cyan: major stages
- Blue: neutral metadata/info
- Green: successful operations/results
- Yellow: warnings/uncertain validation
- Red: failures/rejections
- Default/white: numerical details
"""

import os
import sys
from typing import Any

import numpy as np


class TerminalColors:
    """ANSI Escape sequences for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    BOLD_CYAN = "\033[1;36m"
    BLUE = "\033[34m"
    LIGHT_BLUE = "\033[94m"
    GREEN = "\033[32m"
    BOLD_GREEN = "\033[1;32m"
    YELLOW = "\033[33m"
    BOLD_YELLOW = "\033[1;33m"
    RED = "\033[31m"
    BOLD_RED = "\033[1;31m"
    WHITE = "\033[37m"
    DIM = "\033[2m"


class TerminalLogger:
    """Technical CLI streaming logger for CHANDRAMAP image registration."""

    def __init__(self, enabled: bool = True, force_color: bool = True):
        self.enabled = enabled
        self.use_color = force_color or (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )
        if os.environ.get("NO_COLOR"):
            self.use_color = False

    def _c(self, color: str, text: str) -> str:
        if not self.use_color:
            return text
        return f"{color}{text}{TerminalColors.RESET}"

    def stage(self, name: str) -> None:
        """Log a major pipeline stage in Cyan."""
        if not self.enabled:
            return
        tag = self._c(TerminalColors.BOLD_CYAN, "[STAGE]")
        stage_name = self._c(TerminalColors.CYAN, name)
        sys.stdout.write(f"\n{tag} {stage_name}\n")
        sys.stdout.flush()

    def info(self, label: str, value: str = "") -> None:
        """Log neutral metadata or information in Blue / White."""
        if not self.enabled:
            return
        lbl = self._c(TerminalColors.LIGHT_BLUE, f"  {label:<22s}")
        val = self._c(TerminalColors.WHITE, str(value))
        sys.stdout.write(f"{lbl} {val}\n")
        sys.stdout.flush()

    def success(self, message: str) -> None:
        """Log a successful step in Green."""
        if not self.enabled:
            return
        tag = self._c(TerminalColors.BOLD_GREEN, "[OK]")
        sys.stdout.write(f"  {tag} {message}\n")
        sys.stdout.flush()

    def warning(self, message: str) -> None:
        """Log a warning / uncertain validation in Yellow."""
        if not self.enabled:
            return
        tag = self._c(TerminalColors.BOLD_YELLOW, "[WARNING]")
        msg = self._c(TerminalColors.YELLOW, message)
        sys.stdout.write(f"  {tag} {msg}\n")
        sys.stdout.flush()

    def error(self, message: str) -> None:
        """Log an error / failure in Red."""
        if not self.enabled:
            return
        tag = self._c(TerminalColors.BOLD_RED, "[FAIL]")
        msg = self._c(TerminalColors.RED, message)
        sys.stdout.write(f"  {tag} {msg}\n")
        sys.stdout.flush()

    def log_loading(
        self,
        src_desc: str,
        src_shape: tuple,
        src_dtype: str,
        ref_desc: str,
        ref_shape: tuple,
        ref_dtype: str,
    ) -> None:
        self.stage("loading")
        self.info("SOURCE", f"{src_desc} | dims={src_shape} dtype={src_dtype}")
        self.info("REFERENCE", f"{ref_desc} | dims={ref_shape} dtype={ref_dtype}")

    def log_metadata(
        self,
        profile_name: str,
        src_inst: str,
        ref_inst: str,
        src_gsd: float | None = None,
        ref_gsd: float | None = None,
        sun_elevation: float | None = None,
        sun_azimuth: float | None = None,
        solar_incidence: float | None = None,
        is_night_pass: bool = False,
        scale_strategy: str | None = None,
    ) -> None:
        self.stage("metadata/profile")
        self.info("Sensor Profile", profile_name)
        ratio_str = ""
        ratio = 1.0
        if src_gsd and ref_gsd:
            ratio = max(src_gsd, ref_gsd) / max(1e-4, min(src_gsd, ref_gsd))
            ratio_str = f" (disparity={ratio:.1f}x)"
        self.info(
            "Spatial GSD",
            f"source={src_gsd or 'N/A'} m/px, ref={ref_gsd or 'N/A'} m/px{ratio_str}",
        )
        if scale_strategy:
            self.info("Scale Strategy", scale_strategy)
        elif src_gsd and ref_gsd and ratio >= 1.5:
            self.info(
                "Scale Strategy",
                f"SIFT multi-octave DoG pyramid (disparity {ratio:.1f}x bridged across octaves 2-3)",
            )

        if sun_elevation is not None or solar_incidence is not None:
            el_str = f"el={sun_elevation:.2f}°" if sun_elevation is not None else ""
            inc_str = f"inc={solar_incidence:.2f}°" if solar_incidence is not None else ""
            az_str = f"az={sun_azimuth:.2f}°" if sun_azimuth is not None else ""
            angles = " ".join([p for p in [el_str, inc_str, az_str] if p])
            self.info("Solar Geometry", angles)

        if is_night_pass or (sun_elevation is not None and sun_elevation < 0.0):
            self.error("Quality gate triggered: INSUFFICIENT_ILLUMINATION (sun elevation < 0.0°)")
        else:
            self.success("Quality gate: nominal illumination verified")

    def log_preprocessing(
        self,
        src_prep: str,
        ref_prep: str,
        src_minmax: tuple[float, float],
        ref_minmax: tuple[float, float],
        is_float_radiance: bool = False,
        negatives_masked: int = 0,
    ) -> None:
        self.stage("preprocessing")
        rad_tag = " [float32 radiance preserved]" if is_float_radiance else ""
        self.info("Source Prep", f"{src_prep} range=[{src_minmax[0]:.2f}, {src_minmax[1]:.2f}]{rad_tag}")
        if negatives_masked > 0:
            self.info("Negative Masking", f"{negatives_masked} anomalous pixels masked")
        self.info("Reference Prep", f"{ref_prep} range=[{ref_minmax[0]:.2f}, {ref_minmax[1]:.2f}]")

    def log_feature_extraction(
        self,
        method: str,
        keypoints_src: int,
        keypoints_ref: int,
        elapsed_sec: float,
        cap_src: int | None = None,
        cap_ref: int | None = None,
        scale_space_info: str | None = None,
    ) -> None:
        self.stage("feature extraction")
        self.info("Method", method)
        src_cap_tag = " [configured cap]" if cap_src and keypoints_src >= cap_src else ""
        ref_cap_tag = " [configured cap]" if cap_ref and keypoints_ref >= cap_ref else ""
        self.info(
            "Keypoints Extracted",
            f"source={keypoints_src:,d}{src_cap_tag} | reference={keypoints_ref:,d}{ref_cap_tag}",
        )
        if scale_space_info:
            self.info("Scale Normalization", scale_space_info)
        self.info("Extraction Runtime", f"{elapsed_sec:.3f} s")

    def log_matching(
        self,
        matcher_name: str,
        raw_matches: int,
        elapsed_sec: float,
        filtered_matches: int | None = None,
        filter_method: str | None = None,
        ratio_thresh: float | None = None,
    ) -> None:
        self.stage("matching")
        self.info("Matcher", matcher_name)
        ratio_note = f" (Lowe's ratio test threshold={ratio_thresh:.2f})" if ratio_thresh else ""
        self.info("Putative Matches", f"{raw_matches:,d}{ratio_note}")
        if filtered_matches is not None and filtered_matches != raw_matches:
            f_desc = filter_method or "spatial grid binning"
            self.info("Spatial Filter", f"{raw_matches:,d} -> {filtered_matches:,d} ({f_desc})")
            self.info("Filtered Pool", f"{filtered_matches:,d} correspondences passed to RANSAC")
        self.info("Matching Runtime", f"{elapsed_sec:.3f} s")

    def log_geometry(
        self,
        model_type: str,
        raw_matches: int,
        inliers: int,
        inlier_ratio: float,
        reproj_thresh: float,
        block_info: str | None = None,
    ) -> None:
        self.stage("geometric estimation")
        self.info("Geometric Model", f"{model_type} (threshold={reproj_thresh:.1f} px)")
        self.info("RANSAC Input", f"{raw_matches:,d} correspondences")
        if block_info:
            self.info("Pushbroom Blocks", block_info)
        ratio_pct = inlier_ratio * 100.0 if inlier_ratio <= 1.0 else inlier_ratio
        val_str = f"{inliers:,d} / {raw_matches:,d} ({ratio_pct:.1f}%)"
        if inliers >= 12:
            self.success(f"RANSAC consensus verified: inliers={val_str}")
        elif inliers >= 4:
            self.warning(f"RANSAC weak consensus: inliers={val_str}")
        else:
            self.error(f"RANSAC underconstrained / degenerate: inliers={val_str}")

    def log_refinement(
        self,
        enabled: bool,
        win_size: tuple[int, int] | None = None,
        pre_rmse: float | None = None,
        post_rmse: float | None = None,
    ) -> None:
        self.stage("refinement")
        if not enabled or pre_rmse is None or post_rmse is None:
            self.info(
                "Sub-pixel Tuning",
                "bypassed (baseline mode; benchmark with --subpixel or geometry.subpixel_refinement.enabled=true)",
            )
            return
        delta = post_rmse - pre_rmse
        win_str = f"win={win_size}" if win_size else ""
        self.info(
            "Sub-pixel Precision",
            f"{win_str} pre_rmse={pre_rmse:.4f} px -> post_rmse={post_rmse:.4f} px (Δ={delta:+.4f} px)",
        )

    def log_evaluation(
        self,
        rmse_px: float | None,
        rmse_m: float | None,
        median_px: float | None,
        median_m: float | None,
        p90_px: float | None,
        coverage_pct: float,
        elapsed_sec: float,
        reference_gsd_m: float | None = None,
        p90_m: float | None = None,
    ) -> None:
        self.stage("evaluation")
        if reference_gsd_m is not None and rmse_px is not None and rmse_m is not None:
            self.info(
                "Reference Frame RMSE",
                f"{rmse_px:.3f} px × {reference_gsd_m:.2f} m/px = {rmse_m:.2f} m (in-sample reprojection)",
            )
        else:
            px_s = f"{rmse_px:.3f} px" if rmse_px is not None else "N/A"
            m_s = f" ({rmse_m:.2f} m)" if rmse_m is not None else ""
            self.info("Reprojection RMSE", f"{px_s}{m_s}")

        if reference_gsd_m is not None and median_px is not None and median_m is not None:
            self.info(
                "Median Residual (Ref)",
                f"{median_px:.3f} px × {reference_gsd_m:.2f} m/px = {median_m:.2f} m",
            )
        else:
            med_px_s = f"{median_px:.3f} px" if median_px is not None else "N/A"
            med_m_s = f" ({median_m:.2f} m)" if median_m is not None else ""
            self.info("Median Residual", f"{med_px_s}{med_m_s}")

        if p90_px is not None:
            if reference_gsd_m is not None and p90_m is not None:
                self.info(
                    "P90 Residual (Ref)",
                    f"{p90_px:.3f} px × {reference_gsd_m:.2f} m/px = {p90_m:.2f} m",
                )
            else:
                self.info("P90 Residual", f"{p90_px:.3f} px")

        self.info("Spatial Coverage", f"{coverage_pct:.1f}%")
        self.info("Total Pipeline Time", f"{elapsed_sec:.3f} s")

        # Scientific caveat on spatial clustering & in-sample fit
        if coverage_pct < 25.0:
            self.warning(
                f"Spatially clustered fit (coverage={coverage_pct:.1f}%). In-sample residual "
                "reflects local reprojection agreement, NOT true lunar geodetic/cartographic accuracy."
            )
        elif rmse_px is not None and rmse_px < 0.05:
            self.warning(
                "Near-zero residual on sparse inliers suggests overparameterized local fit. "
                "Verify spatial distribution across terrain before scientific ingestion."
            )
        else:
            self.info(
                "Consensus Scope",
                "Residuals reflect in-sample tiepoint consensus, NOT independent ground geodetic validation without GCPs.",
            )

    def log_final_result(
        self,
        status: str,
        inliers: int,
        rmse_px: float | None,
        rmse_m: float | None,
        coverage_pct: float,
        model_type: str,
        reference_gsd_m: float | None = None,
    ) -> None:
        self.stage("final result")
        px_s = f"{rmse_px:.3f} px" if rmse_px is not None else "N/A"
        ref_tag = " in ref frame" if reference_gsd_m is not None else ""
        m_s = f" ({rmse_m:.2f} m{ref_tag})" if rmse_m is not None else ""
        summary = (
            f"inliers={inliers} | RMSE={px_s}{m_s} | "
            f"coverage={coverage_pct:.1f}% | model={model_type}"
        )
        if status.upper() in ["SUCCESS", "OK"]:
            tag = self._c(TerminalColors.BOLD_GREEN, "[RESULT] SUCCESS (in-sample consensus):")
            msg = self._c(TerminalColors.WHITE, f" {summary}")
        else:
            tag = self._c(TerminalColors.BOLD_RED, f"[RESULT] {status.upper()}:")
            msg = self._c(TerminalColors.RED, f" {summary}")
        sys.stdout.write(f"\n{tag}{msg}\n\n")
        sys.stdout.flush()
