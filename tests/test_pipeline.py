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
