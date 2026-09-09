"""Integration test for real lunar image registration using real example crops."""

import os

import pytest

from lunar_correspondence.config import load_config
from lunar_correspondence.io.image_loader import load_image
from lunar_correspondence.pipeline import RegistrationPipeline, run_registration


def test_real_pair_registration(real_data_dir):
    """Attempts a real registration using files in LUNAR_REAL_DATA_DIR.
    Expects source.tif and reference.tif in that directory.
    """
    source_path = os.path.join(real_data_dir, "source.tif")
    reference_path = os.path.join(real_data_dir, "reference.tif")
    assert os.path.exists(source_path), f"expected {source_path}"
    assert os.path.exists(reference_path), f"expected {reference_path}"

    config = load_config("configs/default.yaml")
    source = load_image(source_path)
    reference = load_image(reference_path)

    _reg_res, eval_result = run_registration(source, reference, config)
    assert eval_result.total_matches > 0


def test_committed_real_crops_sift_and_rift():
    """Test feature extraction and matching on committed real OHRC example crops."""
    c1_path = os.path.abspath("data/examples/real/ohrc_20260103T100517_crop.tif")
    c2_path = os.path.abspath("data/examples/real/ohrc_20260103T041022_crop.tif")

    if not os.path.exists(c1_path) or not os.path.exists(c2_path):
        pytest.skip("Committed real OHRC crops missing — skipping test")

    src = load_image(c1_path)
    ref = load_image(c2_path)

    # 1. SIFT pipeline run
    sift_config = {
        "pipeline": {"random_seed": 42},
        "feature_extraction": {"method": "sift"},
        "matching": {"method": "descriptor"},
        "geometry": {"model_type": "homography"},
        "evaluation": {"grid_rows": 4, "grid_cols": 4},
    }
    sift_pipe = RegistrationPipeline(sift_config)
    _reg_sift, eval_sift = sift_pipe.run(src, ref)
    assert eval_sift.total_matches > 0
    assert eval_sift.inlier_matches > 0

    # 2. RIFT2 pipeline run
    rift_config = {
        "pipeline": {"random_seed": 42},
        "feature_extraction": {"method": "rift", "rift": {"npt": 2000}},
        "matching": {"method": "rift", "rift": {"ratio_test_threshold": 0.90}},
        "geometry": {"model_type": "homography"},
        "evaluation": {"grid_rows": 4, "grid_cols": 4},
    }
    rift_pipe = RegistrationPipeline(rift_config)
    _reg_rift, eval_rift = rift_pipe.run(src, ref)
    assert eval_rift.total_matches > 0
    assert eval_rift.inlier_matches > 0
