"""Unit and integration tests for sub-pixel keypoint refinement."""

import os

import cv2
import numpy as np

from lunar_correspondence.config import load_config
from lunar_correspondence.geometry.refinement import refine_subpixel
from lunar_correspondence.io.metadata import ImageData, ImageMetadata
from lunar_correspondence.pipeline import RegistrationPipeline
from scripts.run_baseline import generate_synthetic_lunar_pair


def test_refine_subpixel_empty():
    """Empty keypoint array should return empty array."""
    img = np.zeros((100, 100), dtype=np.uint8)
    kps = np.zeros((0, 2), dtype=np.float32)
    refined = refine_subpixel(img, kps)
    assert refined.shape == (0, 2)


def test_refine_subpixel_boundary_margin():
    """Points near or on image borders should be preserved safely without crashes."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Put points right at image edges (0, 0), (99, 99)
    kps = np.array([[0.0, 0.0], [99.0, 99.0], [50.0, 50.0]], dtype=np.float32)
    refined = refine_subpixel(img, kps, win_size=(5, 5))
    assert refined.shape == (3, 2)
    assert np.all(np.isfinite(refined))
    # Border points should remain untouched
    np.testing.assert_allclose(refined[0], [0.0, 0.0])
    np.testing.assert_allclose(refined[1], [99.0, 99.0])


def test_refine_subpixel_synthetic_corner():
    """Test refinement on a synthetic high-contrast corner."""
    # Create an image with a sharp corner at (50, 50)
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50:, 50:] = 255
    img[:50, :50] = 255

    # Perturb initial point slightly near corner
    init_pt = np.array([[50.3, 49.7]], dtype=np.float32)
    refined = refine_subpixel(img, init_pt, win_size=(5, 5))

    assert refined.shape == (1, 2)
    assert np.all(np.isfinite(refined))
    assert 48.0 < refined[0, 0] < 52.0
    assert 48.0 < refined[0, 1] < 52.0


def test_subpixel_refinement_pipeline_integration(tmp_path):
    """Synthetic integration test: compare pipeline without and with refinement.

    Validates that:
    - Pipeline runs end-to-end with subpixel refinement enabled.
    - Pre-refinement RMSE and post-refinement RMSE are both reported honestly.
    - Output coordinates and images remain valid.
    """
    synth_dir = str(tmp_path / "synthetic_refinement")
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

    # 1. Run without subpixel refinement
    config_base = load_config(os.path.abspath("configs/default.yaml"))
    config_base["geometry"]["subpixel_refinement"]["enabled"] = False
    pipeline_base = RegistrationPipeline(config_base)
    _reg_base, eval_base = pipeline_base.run(src_data, ref_data)

    assert eval_base.rmse_pixels is not None
    assert eval_base.pre_refinement_rmse_pixels is not None
    assert eval_base.post_refinement_rmse_pixels is None

    # 2. Run with subpixel refinement enabled
    config_refined = load_config(os.path.abspath("configs/default.yaml"))
    config_refined["geometry"]["subpixel_refinement"]["enabled"] = True
    config_refined["geometry"]["subpixel_refinement"]["win_size"] = 5
    pipeline_refined = RegistrationPipeline(config_refined)
    reg_refined, eval_refined = pipeline_refined.run(src_data, ref_data)

    assert eval_refined.pre_refinement_rmse_pixels is not None
    assert eval_refined.post_refinement_rmse_pixels is not None
    assert np.isfinite(eval_refined.post_refinement_rmse_pixels)
    assert reg_refined.registered_image.shape == ref_data.array.shape
    assert eval_refined.inlier_matches > 0
