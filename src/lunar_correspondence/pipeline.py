"""Main Registration Pipeline orchestrator module."""

import random
import time
from typing import Any

import cv2
import numpy as np

from lunar_correspondence.evaluation.metrics import evaluate_registration
from lunar_correspondence.features.learned_features import LearnedFeatureExtractor
from lunar_correspondence.features.rift_features import RIFTFeatureExtractor
from lunar_correspondence.features.sift_features import SIFTFeatureExtractor
from lunar_correspondence.geometry.ransac import estimate_geometric_model
from lunar_correspondence.geometry.transforms import warp_image
from lunar_correspondence.io.metadata import (
    EvaluationResult,
    ImageData,
    RegistrationResult,
)
from lunar_correspondence.matching.descriptor_matcher import DescriptorMatcher
from lunar_correspondence.matching.lightglue_matcher import LightGlueMatcher
from lunar_correspondence.matching.rift_matcher import RIFTMatcher


class RegistrationPipeline:
    """Orchestrates end-to-end multi-modal lunar image correspondence registration."""

    def __init__(self, config: dict[str, Any]):
        """Initialize pipeline with configuration parameters.

        Args:
            config: Full pipeline configuration dictionary.
        """
        self.config = config
        pipe_cfg = config.get("pipeline", {})
        self.random_seed = pipe_cfg.get("random_seed", 42)

        # Instantiate Feature Extractor based on config
        feat_cfg = config.get("feature_extraction", {})
        feat_method = feat_cfg.get("method", "sift").lower()
        if feat_method == "sift":
            self.feature_extractor = SIFTFeatureExtractor(feat_cfg.get("sift", {}))
        elif feat_method == "rift":
            self.feature_extractor = RIFTFeatureExtractor(feat_cfg.get("rift", {}))
        elif feat_method in ["learned", "lightglue"]:
            self.feature_extractor = LearnedFeatureExtractor(
                feat_cfg.get("learned", {})
            )
        else:
            raise ValueError(f"Unsupported feature extraction method: {feat_method}")

        # Instantiate Matcher based on config
        match_cfg = config.get("matching", {})
        match_method = match_cfg.get("method", "descriptor").lower()
        if match_method == "descriptor":
            self.matcher = DescriptorMatcher(match_cfg.get("descriptor", {}))
        elif match_method == "rift":
            self.matcher = RIFTMatcher(match_cfg.get("rift", {}))
        elif match_method == "lightglue":
            self.matcher = LightGlueMatcher(match_cfg.get("lightglue", {}))
        else:
            raise ValueError(f"Unsupported matching method: {match_method}")

    def run(
        self, source_image: ImageData, reference_image: ImageData
    ) -> tuple[RegistrationResult, EvaluationResult]:
        """Execute end-to-end registration pipeline on source and reference images.

        Args:
            source_image: ImageData for source image to warp.
            reference_image: ImageData for reference image baseline.

        Returns:
            Tuple of (RegistrationResult, EvaluationResult).
        """
        start_time = time.time()

        # Seed random number generators for reproducibility
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            cv2.setRNGSeed(self.random_seed)

        # 1. Feature Extraction
        features_src = self.feature_extractor.extract(source_image)
        features_ref = self.feature_extractor.extract(reference_image)

        # 2. Feature Matching
        match_set = self.matcher.match(features_src, features_ref)

        # 3. Geometric RANSAC Estimation
        geo_cfg = self.config.get("geometry", {})
        ransac_cfg = geo_cfg.get("ransac", {})
        model_type = geo_cfg.get("model_type", "homography")

        geometric_model = estimate_geometric_model(
            match_set=match_set,
            model_type=model_type,
            reproj_threshold=ransac_cfg.get("reproj_threshold", 3.0),
            max_iters=ransac_cfg.get("max_iters", 2000),
            confidence=ransac_cfg.get("confidence", 0.99),
            random_seed=self.random_seed,
        )

        # Update match_set inlier_mask
        match_set.inlier_mask = geometric_model.inlier_mask

        # 4. Warp Source Image to Reference Canvas
        ref_shape = (reference_image.height, reference_image.width)
        registered_img = warp_image(
            image_array=source_image.array,
            geometric_model=geometric_model,
            output_shape=ref_shape,
        )

        reg_result = RegistrationResult(
            registered_image=registered_img,
            geometric_model=geometric_model,
            match_set=match_set,
        )

        # 5. Compute Quantitative Metrics
        elapsed_time = time.time() - start_time
        eval_cfg = self.config.get("evaluation", {})

        eval_result = evaluate_registration(
            match_set=match_set,
            geometric_model=geometric_model,
            reference_shape=ref_shape,
            grid_rows=eval_cfg.get("grid_rows", 4),
            grid_cols=eval_cfg.get("grid_cols", 4),
            processing_time_seconds=elapsed_time,
            random_seed=self.random_seed,
        )

        return reg_result, eval_result
