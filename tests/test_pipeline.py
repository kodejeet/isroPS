"""End-to-end integration smoke tests for registration pipeline."""

import os

import cv2
import numpy as np

from lunar_correspondence.config import load_config
from lunar_correspondence.io.metadata import ImageData, ImageMetadata
from lunar_correspondence.pipeline import RegistrationPipeline
from scripts.run_baseline import generate_synthetic_lunar_pair


def test_pipeline_end_to_end_on_synthetic_pair(tmp_path):
    # 1. Generate synthetic pair
    synth_dir = str(tmp_path / "synthetic")
    src_path, ref_path, _ = generate_synthetic_lunar_pair(synth_dir, seed=42)

    # 2. Load images into ImageData
    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

    src_data = ImageData(
        array=src_img,
        path=src_path,
        metadata=ImageMetadata(instrument="OHRC", source_path=src_path),
    )
    ref_data = ImageData(
        array=ref_img,
        path=ref_path,
        metadata=ImageMetadata(instrument="TMC-2", source_path=ref_path),
    )

    # 3. Load config and execute pipeline
    config_path = os.path.abspath("configs/default.yaml")
    config = load_config(config_path)
    pipeline = RegistrationPipeline(config)

    reg_res, eval_res = pipeline.run(src_data, ref_data)

    # 4. Assertions
    assert reg_res.registered_image is not None
    assert reg_res.registered_image.shape == ref_data.array.shape
    assert eval_res.total_matches > 0
    assert eval_res.inlier_matches > 0
    assert eval_res.inlier_ratio > 0.1
    assert eval_res.rmse_pixels is not None
    assert eval_res.coverage > 0.0


def test_pipeline_self_match_end_to_end(tmp_path):
    """Registering an image to itself must succeed with zero error and valid metrics."""
    synth_dir = str(tmp_path / "self_match")
    src_path, _, _ = generate_synthetic_lunar_pair(synth_dir, seed=42)

    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    src_data = ImageData(
        array=src_img,
        path=src_path,
        metadata=ImageMetadata(instrument="OHRC", source_path=src_path),
    )

    config = load_config(os.path.abspath("configs/default.yaml"))
    pipeline = RegistrationPipeline(config)

    reg_res, eval_res = pipeline.run(src_data, src_data)

    assert reg_res.registered_image is not None
    assert eval_res.inlier_matches > 0
    assert eval_res.rmse_pixels is not None
    assert eval_res.median_error_pixels is not None
    assert eval_res.rmse_pixels < 0.5
    assert eval_res.median_error_pixels < 0.5


def test_pipeline_spatial_selection_configurable(tmp_path):
    """Compare candidate matches vs selected matches when spatial selection is enabled vs disabled."""
    synth_dir = str(tmp_path / "spatial_cfg")
    src_path, ref_path, _ = generate_synthetic_lunar_pair(synth_dir, seed=42)

    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

    src_data = ImageData(
        array=src_img,
        path=src_path,
        metadata=ImageMetadata(instrument="OHRC", source_path=src_path),
    )
    ref_data = ImageData(
        array=ref_img,
        path=ref_path,
        metadata=ImageMetadata(instrument="TMC-2", source_path=ref_path),
    )

    # 1. Disabled
    cfg_off = load_config(os.path.abspath("configs/default.yaml"))
    cfg_off["matching"]["spatial_selection"] = {"enabled": False}
    pipe_off = RegistrationPipeline(cfg_off)
    _, eval_off = pipe_off.run(src_data, ref_data)

    # 2. Enabled with top_k=2
    cfg_on = load_config(os.path.abspath("configs/default.yaml"))
    cfg_on["matching"]["spatial_selection"] = {
        "enabled": True,
        "grid_rows": 8,
        "grid_cols": 8,
        "top_k": 2,
    }
    pipe_on = RegistrationPipeline(cfg_on)
    _, eval_on = pipe_on.run(src_data, ref_data)

    # With top_k=2 over 8x8 grid (max 128 matches), total_matches should be constrained
    assert eval_on.total_matches <= (8 * 8 * 2)
    assert eval_on.total_matches <= eval_off.total_matches
    assert eval_on.inlier_matches > 0


def test_pipeline_clahe_enabled(tmp_path):
    """Pipeline runs end-to-end with CLAHE contrast enhancement enabled."""
    synth_dir = str(tmp_path / "clahe_test")
    src_path, ref_path, _ = generate_synthetic_lunar_pair(synth_dir, seed=42)

    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

    src_data = ImageData(
        array=src_img, path=src_path, metadata=ImageMetadata(instrument="TEST")
    )
    ref_data = ImageData(
        array=ref_img, path=ref_path, metadata=ImageMetadata(instrument="TEST")
    )

    cfg = load_config(os.path.abspath("configs/default.yaml"))
    cfg["preprocessing"] = {
        "clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid_size": [8, 8]}
    }
    pipe = RegistrationPipeline(cfg)
    reg_res, eval_res = pipe.run(src_data, ref_data)
    assert reg_res.registered_image is not None
    assert eval_res.inlier_matches > 0


def test_pipeline_tiling_enabled(tmp_path):
    """Pipeline extracts features via spatial tiling when enabled."""
    synth_dir = str(tmp_path / "tiling_test")
    src_path, ref_path, _ = generate_synthetic_lunar_pair(synth_dir, seed=42)

    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

    src_data = ImageData(
        array=src_img, path=src_path, metadata=ImageMetadata(instrument="TEST")
    )
    ref_data = ImageData(
        array=ref_img, path=ref_path, metadata=ImageMetadata(instrument="TEST")
    )

    cfg = load_config(os.path.abspath("configs/default.yaml"))
    cfg["processing"] = {
        "tiling": {"enabled": True, "tile_size": [256, 256], "overlap": 32}
    }
    pipe = RegistrationPipeline(cfg)
    reg_res, eval_res = pipe.run(src_data, ref_data)
    assert reg_res.registered_image is not None
    assert eval_res.inlier_matches > 0


def test_pipeline_fusion_enabled(tmp_path):
    """Pipeline executes multi-matcher fusion (SIFT + RIFT) when enabled."""
    synth_dir = str(tmp_path / "fusion_test")
    src_path, ref_path, _ = generate_synthetic_lunar_pair(synth_dir, seed=42)

    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

    src_data = ImageData(
        array=src_img, path=src_path, metadata=ImageMetadata(instrument="TEST")
    )
    ref_data = ImageData(
        array=ref_img, path=ref_path, metadata=ImageMetadata(instrument="TEST")
    )

    cfg = load_config(os.path.abspath("configs/default.yaml"))
    cfg["matching"]["fusion"] = {"enabled": True}
    cfg["feature_extraction"]["rift"] = {"npt": 500}
    pipe = RegistrationPipeline(cfg)
    reg_res, eval_res = pipe.run(src_data, ref_data)
    assert reg_res.registered_image is not None
    assert eval_res.total_matches > 0


def test_pipeline_degeneracy_detection():
    """EvaluationResult correctly flags underconstrained/degenerate models (< 6 unique inliers)."""
    from lunar_correspondence.evaluation.metrics import evaluate_registration
    from lunar_correspondence.io.metadata import GeometricModel, MatchSet

    # 4 identical/clustered points
    src_pts = np.array([[10, 10], [11, 10], [10, 11], [50, 50]], dtype=np.float32)
    ref_pts = np.array([[20, 20], [21, 20], [20, 21], [60, 60]], dtype=np.float32)
    mset = MatchSet(source_points=src_pts, reference_points=ref_pts)
    gmodel = GeometricModel(
        model_type="homography",
        transform_matrix=np.eye(3, dtype=np.float32),
        inlier_mask=np.array([True, True, True, True]),
        reprojection_errors=np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
    )

    eval_res = evaluate_registration(mset, gmodel, reference_shape=(512, 512))
    assert eval_res.is_degenerate is True
    assert eval_res.quality_warning is not None
    assert "Underconstrained model" in eval_res.quality_warning
