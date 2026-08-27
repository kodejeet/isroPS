"""Integration test for real lunar image registration using LUNAR_REAL_DATA_DIR."""

import os
from lunar_correspondence.config import load_config
from lunar_correspondence.io.image_loader import load_image
from lunar_correspondence.pipeline import run_registration


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

    result, eval_result = run_registration(source, reference, config)
    assert eval_result.total_matches > 0
