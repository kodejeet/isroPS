"""Comprehensive test suite for SIH 2026 official datasets and CHANDRAMAP architecture."""

import os
import numpy as np
import pytest

from lunar_correspondence.geometry.block_geometry import (
    BlockGeometryModel,
    estimate_block_geometry,
    project_points_block_geometry,
)
from lunar_correspondence.io.metadata import MatchSet
from lunar_correspondence.io.sih_dataset import (
    SIHDatasetPair,
    discover_sih_pairs,
    load_sih_image_pair,
)
from lunar_correspondence.io.xml_parser import ISACMetadata, parse_isac_xml
from lunar_correspondence.planetary.footprints import estimate_geospatial_roi
from lunar_correspondence.preprocessing.normalization import create_feature_image
from lunar_correspondence.profiles.sensor_profile import (
    PROFILES,
    detect_sensor_profile,
    get_sensor_profile,
)

BASE_SIH_DIR = "/home/kode/gh-projs/sih2k26/data_for_sih_2026"


class TestISACXMLParser:
    """Tests for ISAC Chandrayaan-2 XML metadata extraction."""

    def test_parse_ohrc_nominal(self):
        xml_path = os.path.join(
            BASE_SIH_DIR, "ohrc", "OHRXXD18CHO2470502NNNN25067152549847_V2_1_02.xml"
        )
        if not os.path.exists(xml_path):
            pytest.skip("SIH dataset not mounted in environment")

        meta = parse_isac_xml(xml_path)
        assert meta.instrument == "OHRC"
        assert meta.imaging_orbit_number == 24698
        assert meta.sun_elevation_deg is not None
        assert abs(meta.sun_elevation_deg - 3.088558) < 1e-4
        assert meta.is_night_pass is False
        assert len(meta.quality_flags) == 0
        assert len(meta.corners_geo) == 4
        assert len(meta.corners_projected) == 4
        assert "topleft" in meta.corners_geo
        assert "bottomright" in meta.corners_projected

    def test_parse_ohrc_night_pass_rejection(self):
        xml_path = os.path.join(
            BASE_SIH_DIR, "ohrc", "OHRXXD18CHO2736702NNNN25285183733061_V1_0_00.xml"
        )
        if not os.path.exists(xml_path):
            pytest.skip("SIH dataset not mounted in environment")

        meta = parse_isac_xml(xml_path)
        assert meta.instrument == "OHRC"
        assert meta.sun_elevation_deg is not None
        assert meta.sun_elevation_deg < 0.0
        assert meta.is_night_pass is True
        assert "INSUFFICIENT_ILLUMINATION" in meta.quality_flags

    def test_parse_iirs_qube(self):
        xml_path = os.path.join(
            BASE_SIH_DIR, "IIRS", "IIRXXD18CHO2686502NNNN25244140531312_V2_1.xml"
        )
        if not os.path.exists(xml_path):
            pytest.skip("SIH dataset not mounted in environment")

        meta = parse_isac_xml(xml_path)
        assert meta.instrument == "IIRS"
        assert meta.imaging_orbit_number == 26864
        assert meta.raw_qube_height == 7783
        assert meta.raw_qube_width == 250
        assert meta.detector_temp_k is not None
        assert abs(meta.detector_temp_k - 88.97) < 1e-2


class TestSIHDatasetDiscovery:
    """Tests for official SIH 2026 dataset pair discovery and loading."""

    def test_discover_pairs(self):
        if not os.path.exists(BASE_SIH_DIR):
            pytest.skip("SIH dataset not mounted in environment")

        pairs = discover_sih_pairs(BASE_SIH_DIR)
        if len(pairs) == 0:
            pytest.skip("SIH dataset files not accessible in sandbox")
        assert len(pairs) >= 6
        ohrc_pairs = [p for p in pairs if p.instrument == "OHRC"]
        iirs_pairs = [p for p in pairs if p.instrument == "IIRS"]

        assert len(ohrc_pairs) == 4
        assert len(iirs_pairs) == 2

        # Verify OHRC 5m resolution
        for op in ohrc_pairs:
            assert op.source_gsd_m == 5.0
            assert op.reference_gsd_m == 5.0
            assert op.source_dtype == "uint8"

        # Verify IIRS float32 radiance
        for ip in iirs_pairs:
            assert ip.source_dtype == "float32"
            assert ip.reference_dtype == "float32"
            assert ip.source_shape[1] in [104, 250]

    def test_load_sih_image_pair_preserves_float32(self):
        if not os.path.exists(BASE_SIH_DIR):
            pytest.skip("SIH dataset not mounted in environment")

        pairs = discover_sih_pairs(BASE_SIH_DIR)
        if len(pairs) == 0:
            pytest.skip("SIH dataset files not accessible in sandbox")
        iirs_pair = next(p for p in pairs if p.instrument == "IIRS")
        src_data, ref_data = load_sih_image_pair(iirs_pair)

        assert src_data.array.dtype == np.float32
        assert ref_data.array.dtype == np.float32
        assert src_data.metadata.resolution_m_per_px is not None
        assert ref_data.metadata.resolution_m_per_px is not None


class TestRadiometricNormalization:
    """Tests for physical radiance preservation and feature image generation."""

    def test_create_feature_image_preserves_raw_array(self):
        # Create synthetic radiance array with negative baseline noise
        raw = np.array([[-70.5, 0.0, 50.0], [200.0, 800.0, 1712.0]], dtype=np.float32)
        raw_copy = raw.copy()

        feat_img, mask = create_feature_image(
            raw, method="percentile", p_low=1.0, p_high=99.0, mask_negatives=True
        )

        # Raw array must be 100% untouched
        np.testing.assert_array_equal(raw, raw_copy)
        assert raw.dtype == np.float32

        # Feature image must be uint8 in [0, 255]
        assert feat_img.dtype == np.uint8
        assert feat_img.shape == raw.shape
        assert mask.dtype == bool
        assert mask[0, 0] is np.False_  # Negative value masked out
        assert mask[1, 2] is np.True_

    def test_clahe_feature_image(self):
        raw = np.linspace(10, 500, 100, dtype=np.float32).reshape(10, 10)
        feat_img, mask = create_feature_image(raw, apply_clahe=True, clahe_grid_size=(4, 4))
        assert feat_img.dtype == np.uint8
        assert feat_img.shape == (10, 10)


class TestGeospatialROI:
    """Tests for geospatial footprint prior ROI calculation."""

    def test_estimate_roi_valid(self):
        xml_path = os.path.join(
            BASE_SIH_DIR, "IIRS", "IIRXXD18CHO2686502NNNN25244140531312_V2_1.xml"
        )
        ref_path = os.path.join(
            BASE_SIH_DIR, "IIRS", "IIRXXD18CHO2686502NNNN25244140531312_V2_1_reference.tif"
        )
        if not os.path.exists(xml_path) or not os.path.exists(ref_path):
            pytest.skip("SIH dataset not mounted in environment")

        meta = parse_isac_xml(xml_path)
        roi = estimate_geospatial_roi(meta, ref_path, margin_px=64)

        assert roi is not None
        min_y, min_x, max_y, max_x = roi
        assert 0 <= min_y < max_y <= 7681
        assert 0 <= min_x < max_x <= 960


class TestSensorProfiles:
    """Tests for SensorProfile registry and deduction."""

    def test_profiles_exist(self):
        assert "OHRC_SIH_5M" in PROFILES
        assert "TMC2" in PROFILES
        assert "IIRS_EQUATORIAL_WAC" in PROFILES
        assert "IIRS_SOUTH_POLE_WAC" in PROFILES

    def test_detect_profiles(self):
        prof_ohrc = detect_sensor_profile("OHRC")
        assert prof_ohrc.name == "OHRC_SIH_5M"
        assert prof_ohrc.approx_source_gsd_m == 5.0

        prof_tmc = detect_sensor_profile("TMC-2")
        assert prof_tmc.name == "TMC2"

        prof_iirs_eq = detect_sensor_profile("IIRS", gsd=94.0, is_polar=False)
        assert prof_iirs_eq.name == "IIRS_EQUATORIAL_WAC"

        prof_iirs_polar = detect_sensor_profile("IIRS", gsd=83.0, is_polar=True)
        assert prof_iirs_polar.name == "IIRS_SOUTH_POLE_WAC"
        assert prof_iirs_polar.scale_policy in ["native", "pyramid_to_reference"]


class TestBlockGeometry:
    """Tests for piecewise along-track block geometry modeling."""

    def test_continuous_block_geometry(self):
        # Generate synthetic points along a long strip (0 to 3000 y)
        np.random.seed(42)
        n = 100
        x_src = np.random.uniform(10, 240, n)
        y_src = np.random.uniform(0, 3000, n)
        src = np.column_stack([x_src, y_src]).astype(np.float32)

        # Apply consistent affine transform: rot 5 deg + shift (50, 100)
        theta = np.radians(5.0)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        dst = (src @ R.T) + np.array([50.0, 100.0], dtype=np.float32)

        match_set = MatchSet(source_points=src, reference_points=dst)
        model = estimate_block_geometry(match_set, block_length=1500, block_overlap=300)

        assert model.total_inliers >= 80
        assert model.is_continuous is True
        assert len(model.continuity_warnings) == 0

        # Project points and check reprojection accuracy
        proj = project_points_block_geometry(src, model)
        err = np.linalg.norm(proj - dst, axis=1)
        assert np.mean(err) < 1.0  # Sub-pixel reconstruction
