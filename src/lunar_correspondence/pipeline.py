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
from lunar_correspondence.geometry.homography import compute_reprojection_errors
from lunar_correspondence.geometry.ransac import estimate_geometric_model
from lunar_correspondence.geometry.refinement import refine_subpixel
from lunar_correspondence.geometry.transforms import warp_image
from lunar_correspondence.io.metadata import (
    EvaluationResult,
    ImageData,
    RegistrationResult,
)
from lunar_correspondence.matching.descriptor_matcher import DescriptorMatcher
from lunar_correspondence.matching.lightglue_matcher import LightGlueMatcher
from lunar_correspondence.matching.rift_matcher import RIFTMatcher
from lunar_correspondence.matching.spatial_selection import select_spatial_matches


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
        raw_match_set = self.matcher.match(features_src, features_ref)

        # 2b. Spatial Match Selection (prior to RANSAC)
        match_cfg = self.config.get("matching", {})
        spatial_cfg = match_cfg.get("spatial_selection", {})
        if spatial_cfg.get("enabled", False):
            src_shape = (source_image.height, source_image.width)
            match_set = select_spatial_matches(
                match_set=raw_match_set,
                image_shape=src_shape,
                grid_rows=spatial_cfg.get("grid_rows", 8),
                grid_cols=spatial_cfg.get("grid_cols", 8),
                top_k=spatial_cfg.get("top_k", 4),
            )
        else:
            match_set = raw_match_set

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

        # 3b. Sub-pixel Refinement (optional/configurable, post-RANSAC on inliers)
        subpix_cfg = geo_cfg.get("subpixel_refinement", {})
        pre_refinement_rmse: float | None = None
        post_refinement_rmse: float | None = None

        inlier_count = (
            int(np.sum(geometric_model.inlier_mask))
            if geometric_model.inlier_mask is not None
            else 0
        )
        if inlier_count > 0 and geometric_model.reprojection_errors is not None:
            inlier_errs = geometric_model.reprojection_errors[
                geometric_model.inlier_mask
            ]
            pre_refinement_rmse = (
                float(np.sqrt(np.mean(inlier_errs**2)))
                if len(inlier_errs) > 0
                else None
            )

        if subpix_cfg.get("enabled", False) and inlier_count > 0:
            win_size_val = subpix_cfg.get("win_size", 5)
            win_size = (
                (win_size_val, win_size_val)
                if isinstance(win_size_val, int)
                else tuple(win_size_val)
            )
            zero_zone_val = subpix_cfg.get("zero_zone", -1)
            zero_zone = (
                (zero_zone_val, zero_zone_val)
                if isinstance(zero_zone_val, int)
                else tuple(zero_zone_val)
            )

            inliers_src = match_set.source_points[geometric_model.inlier_mask]
            inliers_ref = match_set.reference_points[geometric_model.inlier_mask]

            refined_src = refine_subpixel(
                image_array=source_image.array,
                keypoints_xy=inliers_src,
                win_size=win_size,
                zero_zone=zero_zone,
            )
            refined_ref = refine_subpixel(
                image_array=reference_image.array,
                keypoints_xy=inliers_ref,
                win_size=win_size,
                zero_zone=zero_zone,
            )

            # Update match set inlier coordinates
            match_set.source_points[geometric_model.inlier_mask] = refined_src
            match_set.reference_points[geometric_model.inlier_mask] = refined_ref

            # Recompute reprojection errors on refined inlier points under current model
            post_errors = compute_reprojection_errors(
                refined_src, refined_ref, geometric_model.transform_matrix
            )
            post_refinement_rmse = (
                float(np.sqrt(np.mean(post_errors**2)))
                if len(post_errors) > 0
                else None
            )

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
            pre_refinement_rmse_pixels=pre_refinement_rmse,
            post_refinement_rmse_pixels=post_refinement_rmse,
        )

        return reg_result, eval_result


def run_registration(
    source: ImageData,
    reference: ImageData,
    config: dict[str, Any],
) -> tuple[RegistrationResult, EvaluationResult]:
    """Single entry point function for registering two ImageData objects.

    All callers (run_baseline.py, scripts/register.py, app.py) MUST call this function.
    Handles optional downsampling for large images if processing.max_dimension is set in config.
    """
    max_dim = config.get("processing", {}).get("max_dimension", None)

    source_to_process = source
    ref_to_process = reference
    scale_factor = 1.0

    if max_dim is not None:
        max_src = max(source.height, source.width)
        max_ref = max(reference.height, reference.width)
        max_sz = max(max_src, max_ref)
        if max_sz > max_dim:
            scale_factor = float(max_dim) / float(max_sz)
            new_src_w = round(source.width * scale_factor)
            new_src_h = round(source.height * scale_factor)
            new_ref_w = round(reference.width * scale_factor)
            new_ref_h = round(reference.height * scale_factor)

            src_arr_ds = cv2.resize(source.array, (new_src_w, new_src_h))
            if src_arr_ds.ndim == 2:
                src_arr_ds = src_arr_ds[:, :, np.newaxis]
            ref_arr_ds = cv2.resize(reference.array, (new_ref_w, new_ref_h))
            if ref_arr_ds.ndim == 2:
                ref_arr_ds = ref_arr_ds[:, :, np.newaxis]

            source_to_process = ImageData(
                array=src_arr_ds, path=source.path, metadata=source.metadata
            )
            ref_to_process = ImageData(
                array=ref_arr_ds, path=reference.path, metadata=reference.metadata
            )

    pipeline = RegistrationPipeline(config)
    reg_result, eval_result = pipeline.run(source_to_process, ref_to_process)
    eval_result.scale_factor = scale_factor
    return reg_result, eval_result
