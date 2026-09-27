"""Main Registration Pipeline orchestrator module."""

import os
import random
import time
from typing import Any

import cv2
import numpy as np

from lunar_correspondence.evaluation.metrics import evaluate_registration
from lunar_correspondence.features.learned_features import LearnedFeatureExtractor
from lunar_correspondence.features.rift_features import RIFTFeatureExtractor
from lunar_correspondence.features.sift_features import SIFTFeatureExtractor
from lunar_correspondence.geometry.block_geometry import estimate_block_geometry
from lunar_correspondence.geometry.homography import compute_reprojection_errors
from lunar_correspondence.geometry.ransac import estimate_geometric_model
from lunar_correspondence.geometry.refinement import refine_subpixel
from lunar_correspondence.geometry.transforms import warp_image
from lunar_correspondence.io.metadata import (
    EvaluationResult,
    FeatureSet,
    GeometricModel,
    ImageData,
    RegistrationResult,
)
from lunar_correspondence.io.terminal_logger import TerminalLogger
from lunar_correspondence.matching.descriptor_matcher import DescriptorMatcher
from lunar_correspondence.matching.fusion import fuse_match_sets
from lunar_correspondence.matching.lightglue_matcher import LightGlueMatcher
from lunar_correspondence.matching.rift_matcher import RIFTMatcher
from lunar_correspondence.matching.spatial_selection import select_spatial_matches
from lunar_correspondence.preprocessing.enhancement import apply_clahe
from lunar_correspondence.preprocessing.normalization import create_feature_image
from lunar_correspondence.preprocessing.tiling import generate_tiles
from lunar_correspondence.profiles.sensor_profile import SensorProfile, get_sensor_profile


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

        # Technical CLI Streaming Logger
        term_log = config.get("terminal_logging", True) and not config.get("quiet", False)
        self.logger = TerminalLogger(enabled=term_log)

        # Resolve optional SensorProfile
        prof_obj = None
        if "profile" in config:
            p_val = config["profile"]
            if isinstance(p_val, str):
                try:
                    prof_obj = get_sensor_profile(p_val)
                except Exception:
                    pass
            elif isinstance(p_val, SensorProfile):
                prof_obj = p_val
        self.profile = prof_obj

        # Preprocessing: CLAHE
        prep_cfg = config.get("preprocessing", {})
        clahe_cfg = prep_cfg.get("clahe", {})
        default_clahe = self.profile.radiometry.apply_clahe if self.profile else False
        self.clahe_enabled = clahe_cfg.get("enabled", default_clahe)
        default_clip = self.profile.radiometry.clahe_clip_limit if self.profile else 2.0
        self.clahe_clip_limit = float(clahe_cfg.get("clip_limit", default_clip))
        default_grid = self.profile.radiometry.clahe_grid_size if self.profile else (8, 8)
        grid_size = clahe_cfg.get("tile_grid_size", list(default_grid))
        self.clahe_grid_size = (
            tuple(grid_size) if isinstance(grid_size, (list, tuple)) else (8, 8)
        )

        # Processing / Preprocessing: Tiling
        tiling_cfg = config.get("processing", {}).get("tiling", {})
        if not tiling_cfg:
            tiling_cfg = prep_cfg.get("tiling", {})
        self.tiling_enabled = tiling_cfg.get("enabled", False)
        tile_sz = tiling_cfg.get("tile_size", [256, 256])
        self.tile_size = (
            tuple(tile_sz) if isinstance(tile_sz, (list, tuple)) else (256, 256)
        )
        self.tile_overlap = int(tiling_cfg.get("overlap", 32))

        # Matching: Fusion
        feat_cfg = config.get("feature_extraction", {})
        default_feat = self.profile.feature_method if self.profile else "sift"
        feat_method = feat_cfg.get("method", default_feat).lower()
        match_cfg = config.get("matching", {})
        default_match = "fusion" if feat_method == "fusion" else ("rift" if feat_method == "rift" else "descriptor")
        match_method = match_cfg.get("method", default_match).lower()
        fusion_cfg = config.get("matching", {}).get("fusion", {})
        if not fusion_cfg:
            fusion_cfg = config.get("fusion", {})
        self.fusion_enabled = (
            fusion_cfg.get("enabled", False)
            or match_method == "fusion"
            or feat_method == "fusion"
        )
        if self.fusion_enabled:
            sift_feat_cfg = feat_cfg.get("sift", {})
            self.sift_extractor = SIFTFeatureExtractor(sift_feat_cfg)
            self.sift_matcher = DescriptorMatcher(match_cfg.get("descriptor", {}))

            rift_feat_cfg = feat_cfg.get("rift", {})
            self.rift_extractor = RIFTFeatureExtractor(rift_feat_cfg)
            self.rift_matcher = RIFTMatcher(match_cfg.get("rift", {}))
            self.feature_extractor = self.sift_extractor
            self.matcher = self.sift_matcher
        else:
            # Instantiate Feature Extractor based on config
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
            if match_method == "descriptor":
                self.matcher = DescriptorMatcher(match_cfg.get("descriptor", {}))
            elif match_method == "rift":
                self.matcher = RIFTMatcher(match_cfg.get("rift", {}))
            elif match_method == "lightglue":
                self.matcher = LightGlueMatcher(match_cfg.get("lightglue", {}))
            else:
                raise ValueError(f"Unsupported matching method: {match_method}")

    def _extract_features(self, extractor: Any, image: ImageData) -> FeatureSet:
        """Extract features, with optional grid tiling if enabled."""
        if not self.tiling_enabled:
            return extractor.extract(image)

        tiles = generate_tiles(
            image.array, tile_size=self.tile_size, overlap=self.tile_overlap
        )
        all_kps = []
        all_descs = []
        for (min_y, min_x, max_y, max_x), tile_crop in tiles:
            tile_data = ImageData(
                array=tile_crop, path=image.path, metadata=image.metadata
            )
            fset = extractor.extract(tile_data)
            if len(fset.keypoints) > 0:
                offset_kps = fset.keypoints.copy()
                offset_kps[:, 0] += min_x
                offset_kps[:, 1] += min_y
                all_kps.append(offset_kps)
                if fset.descriptors is not None:
                    all_descs.append(fset.descriptors)

        if not all_kps:
            return FeatureSet(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=None,
                method=getattr(extractor, "method_name", "tiled"),
            )

        merged_kps = np.vstack(all_kps).astype(np.float32)
        merged_descs = np.vstack(all_descs) if all_descs else None
        return FeatureSet(
            keypoints=merged_kps,
            descriptors=merged_descs,
            method=getattr(extractor, "method_name", "tiled"),
        )

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

        # Stage 1: loading
        self.logger.log_loading(
            src_desc=os.path.basename(source_image.path) if source_image.path else "source_array",
            src_shape=source_image.array.shape,
            src_dtype=str(source_image.array.dtype),
            ref_desc=os.path.basename(reference_image.path) if reference_image.path else "reference_array",
            ref_shape=reference_image.array.shape,
            ref_dtype=str(reference_image.array.dtype),
        )

        # Stage 2: metadata/profile
        prof_name = self.profile.name if self.profile else "DEFAULT"
        src_meta = source_image.metadata
        ref_meta = reference_image.metadata
        src_inst = getattr(src_meta, "instrument", "UNKNOWN") if src_meta else "UNKNOWN"
        ref_inst = getattr(ref_meta, "instrument", "UNKNOWN") if ref_meta else "UNKNOWN"
        src_gsd = getattr(src_meta, "resolution_m_per_px", None) if src_meta else (self.profile.approx_source_gsd_m if self.profile else None)
        ref_gsd = getattr(ref_meta, "resolution_m_per_px", None) if ref_meta else (self.profile.approx_reference_gsd_m if self.profile else None)
        sun_el = getattr(src_meta, "sun_elevation_deg", None) if src_meta else None
        sun_az = getattr(src_meta, "sun_azimuth_deg", None) if src_meta else None
        sol_inc = getattr(src_meta, "solar_incidence_deg", None) if src_meta else None
        is_night = getattr(src_meta, "is_night_pass", False) if src_meta else False

        scale_strat = None
        if src_gsd and ref_gsd:
            ratio = max(src_gsd, ref_gsd) / max(1e-4, min(src_gsd, ref_gsd))
            if ratio >= 1.5:
                scale_strat = (
                    f"SIFT multi-octave DoG pyramid (disparity {ratio:.1f}x bridged across octaves 2-3)"
                )

        self.logger.log_metadata(
            profile_name=prof_name,
            src_inst=src_inst,
            ref_inst=ref_inst,
            src_gsd=src_gsd,
            ref_gsd=ref_gsd,
            sun_elevation=sun_el,
            sun_azimuth=sun_az,
            solar_incidence=sol_inc,
            is_night_pass=is_night,
            scale_strategy=scale_strat,
        )

        # 0. Preprocessing & Radiometry
        norm_cfg = self.config.get("preprocessing", {}).get("normalization", {})
        is_float_source = np.issubdtype(source_image.array.dtype, np.floating)
        is_float_ref = np.issubdtype(reference_image.array.dtype, np.floating)
        prof_rad = self.profile.radiometry if self.profile else None
        use_norm = (
            norm_cfg.get("enabled", False)
            or is_float_source
            or is_float_ref
            or (prof_rad is not None and prof_rad.method == "percentile")
        )

        if use_norm:
            method = norm_cfg.get(
                "method",
                prof_rad.method if prof_rad else ("percentile" if is_float_source else "minmax"),
            )
            p_low = norm_cfg.get("percentile_low", prof_rad.p_low if prof_rad else 0.5)
            p_high = norm_cfg.get("percentile_high", prof_rad.p_high if prof_rad else 99.5)
            mask_neg = norm_cfg.get(
                "mask_negatives", prof_rad.mask_negatives if prof_rad else is_float_source
            )
            apply_c = norm_cfg.get("apply_clahe", self.clahe_enabled)

            src_feat_img, _ = create_feature_image(
                source_image.array,
                method=method,
                p_low=p_low,
                p_high=p_high,
                mask_negatives=mask_neg,
                apply_clahe=apply_c,
                clahe_clip_limit=self.clahe_clip_limit,
                clahe_grid_size=self.clahe_grid_size,
            )
            ref_feat_img, _ = create_feature_image(
                reference_image.array,
                method=method if is_float_ref else "minmax",
                p_low=p_low,
                p_high=p_high,
                mask_negatives=False,
                apply_clahe=apply_c,
                clahe_clip_limit=self.clahe_clip_limit,
                clahe_grid_size=self.clahe_grid_size,
            )
            src_to_process = ImageData(
                array=src_feat_img[:, :, np.newaxis] if src_feat_img.ndim == 2 else src_feat_img,
                path=source_image.path,
                metadata=source_image.metadata,
            )
            ref_to_process = ImageData(
                array=ref_feat_img[:, :, np.newaxis] if ref_feat_img.ndim == 2 else ref_feat_img,
                path=reference_image.path,
                metadata=reference_image.metadata,
            )
        elif self.clahe_enabled:
            src_clahe = apply_clahe(
                source_image.array,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_grid_size,
            )
            ref_clahe = apply_clahe(
                reference_image.array,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_grid_size,
            )
            src_to_process = ImageData(
                array=src_clahe, path=source_image.path, metadata=source_image.metadata
            )
            ref_to_process = ImageData(
                array=ref_clahe, path=reference_image.path, metadata=reference_image.metadata
            )
        else:
            src_to_process = source_image
            ref_to_process = reference_image

        # Stage 3: preprocessing
        src_min = float(np.nanmin(source_image.array))
        src_max = float(np.nanmax(source_image.array))
        ref_min = float(np.nanmin(reference_image.array))
        ref_max = float(np.nanmax(reference_image.array))
        neg_count = int(np.sum(source_image.array < 0)) if is_float_source else 0
        src_prep_desc = "Percentile norm + CLAHE" if use_norm else ("CLAHE" if self.clahe_enabled else "Pass-through")
        ref_prep_desc = "MinMax norm + CLAHE" if use_norm else ("CLAHE" if self.clahe_enabled else "Pass-through")

        self.logger.log_preprocessing(
            src_prep=src_prep_desc,
            ref_prep=ref_prep_desc,
            src_minmax=(src_min, src_max),
            ref_minmax=(ref_min, ref_max),
            is_float_radiance=is_float_source,
            negatives_masked=neg_count,
        )

        # 1 & 2. Feature Extraction & Matching
        t_feat_start = time.time()
        if self.fusion_enabled:
            sift_src = self._extract_features(self.sift_extractor, src_to_process)
            sift_ref = self._extract_features(self.sift_extractor, ref_to_process)
            rift_src = self._extract_features(self.rift_extractor, src_to_process)
            rift_ref = self._extract_features(self.rift_extractor, ref_to_process)
            kps_src = len(sift_src.keypoints) + len(rift_src.keypoints)
            kps_ref = len(sift_ref.keypoints) + len(rift_ref.keypoints)
            feat_method_name = "SIFT + RIFT (Fusion)"
        else:
            features_src = self._extract_features(self.feature_extractor, src_to_process)
            features_ref = self._extract_features(self.feature_extractor, ref_to_process)
            kps_src = len(features_src.keypoints)
            kps_ref = len(features_ref.keypoints)
            feat_method_name = getattr(self.feature_extractor, "method_name", "SIFT")
        t_feat_elapsed = time.time() - t_feat_start

        feat_cfg = self.config.get("feature_extraction", {})
        sift_cfg = feat_cfg.get("sift", {})
        cap_val = sift_cfg.get("nfeatures", 5000)
        n_oct = sift_cfg.get("nOctaveLayers", 4)
        pyramid_info = f"{n_oct} octave layers (multi-scale DoG covering 1.0x to 8.0x scale space)"

        self.logger.log_feature_extraction(
            method=feat_method_name,
            keypoints_src=kps_src,
            keypoints_ref=kps_ref,
            elapsed_sec=t_feat_elapsed,
            cap_src=cap_val,
            cap_ref=cap_val,
            scale_space_info=pyramid_info,
        )

        t_match_start = time.time()
        if self.fusion_enabled:
            sift_matches = self.sift_matcher.match(sift_src, sift_ref)
            rift_matches = self.rift_matcher.match(rift_src, rift_ref)
            raw_match_set = fuse_match_sets([sift_matches, rift_matches])
            matcher_name = "Cross-Modal Dual-Matcher Fusion"
        else:
            raw_match_set = self.matcher.match(features_src, features_ref)
            matcher_name = type(self.matcher).__name__
        t_match_elapsed = time.time() - t_match_start

        # 2b. Spatial Match Selection (prior to RANSAC)
        match_cfg = self.config.get("matching", {})
        spatial_cfg = match_cfg.get("spatial_selection", {})
        filter_desc = None
        if spatial_cfg.get("enabled", False):
            src_shape = (source_image.height, source_image.width)
            grid_r = spatial_cfg.get("grid_rows", 8)
            grid_c = spatial_cfg.get("grid_cols", 8)
            top_k = spatial_cfg.get("top_k", 8)
            match_set = select_spatial_matches(
                match_set=raw_match_set,
                image_shape=src_shape,
                grid_rows=grid_r,
                grid_cols=grid_c,
                top_k=top_k,
            )
            filter_desc = f"{grid_r}x{grid_c} grid binning, top_k={top_k}"
        else:
            match_set = raw_match_set

        ratio_thresh = match_cfg.get("descriptor", {}).get("ratio_test_threshold", 0.82)
        self.logger.log_matching(
            matcher_name=matcher_name,
            raw_matches=len(raw_match_set.source_points),
            elapsed_sec=t_match_elapsed,
            filtered_matches=len(match_set.source_points),
            filter_method=filter_desc,
            ratio_thresh=ratio_thresh,
        )

        # 3. Geometric RANSAC Estimation
        geo_cfg = self.config.get("geometry", {})
        ransac_cfg = geo_cfg.get("ransac", {})
        default_model = self.profile.geometry.primary_model if self.profile else "homography"
        model_type = geo_cfg.get("model_type", default_model)
        default_thresh = self.profile.geometry.reproj_threshold if self.profile else 3.0
        reproj_thresh = float(ransac_cfg.get("reproj_threshold", default_thresh))

        if model_type in ["block_affine", "block_homography"]:
            local_model = "affine" if model_type == "block_affine" else "homography"
            def_len = self.profile.geometry.block_length_px if self.profile else 1500
            def_ov = self.profile.geometry.block_overlap_px if self.profile else 300
            block_len = int(geo_cfg.get("block_length", def_len))
            block_ov = int(geo_cfg.get("block_overlap", def_ov))

            block_model = estimate_block_geometry(
                match_set=match_set,
                block_length=block_len,
                block_overlap=block_ov,
                local_model=local_model,
                reproj_threshold=reproj_thresh,
                max_iters=int(ransac_cfg.get("max_iters", 2000)),
            )

            # Fit composite transformation matrix for downstream components
            inlier_indices = np.where(block_model.inlier_mask)[0]
            min_pts = 4 if local_model == "homography" else 3

            if len(inlier_indices) >= min_pts:
                inl_src = match_set.source_points[block_model.inlier_mask]
                inl_dst = match_set.reference_points[block_model.inlier_mask]
                if local_model == "homography":
                    comp_H, _ = cv2.findHomography(
                        inl_src, inl_dst, cv2.RANSAC, reproj_thresh
                    )
                else:
                    comp_M, _ = (
                        cv2.estimateAffinePartial2D(inl_src, inl_dst)
                        if len(inl_src) < 4
                        else cv2.estimateAffine2D(inl_src, inl_dst)
                    )
                    comp_H = (
                        np.vstack([comp_M, [0, 0, 1]])
                        if comp_M is not None
                        else np.eye(3, dtype=np.float32)
                    )
                if comp_H is None:
                    comp_H = np.eye(3, dtype=np.float32)
            else:
                comp_H = np.eye(3, dtype=np.float32)

            geometric_model = GeometricModel(
                transform_matrix=comp_H.astype(np.float32),
                model_type=model_type,
                inlier_mask=block_model.inlier_mask,
                reprojection_errors=block_model.reprojection_errors,
            )
            geometric_model.block_model = block_model
        else:
            geometric_model = estimate_geometric_model(
                match_set=match_set,
                model_type=model_type,
                reproj_threshold=reproj_thresh,
                max_iters=ransac_cfg.get("max_iters", 2000),
                confidence=ransac_cfg.get("confidence", 0.99),
                random_seed=self.random_seed,
            )

        # Update match_set inlier_mask
        match_set.inlier_mask = geometric_model.inlier_mask

        # Stage 6: geometric estimation
        inl_count = int(np.sum(geometric_model.inlier_mask)) if geometric_model.inlier_mask is not None else 0
        raw_count = len(match_set.source_points)
        inl_ratio = float(inl_count / max(1, raw_count))
        blk_info = None
        if model_type in ["block_affine", "block_homography"] and hasattr(geometric_model, "block_model"):
            bm = geometric_model.block_model
            blk_info = f"{len(bm.blocks)} blocks (length={bm.block_length}px, overlap={bm.block_overlap}px)"

        self.logger.log_geometry(
            model_type=model_type,
            raw_matches=raw_count,
            inliers=inl_count,
            inlier_ratio=inl_ratio,
            reproj_thresh=reproj_thresh,
            block_info=blk_info,
        )

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
            valid_errs = inlier_errs[~np.isnan(inlier_errs)]
            pre_refinement_rmse = (
                float(np.sqrt(np.mean(valid_errs**2)))
                if len(valid_errs) > 0
                else None
            )

        win_size = None
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
            if geometric_model.reprojection_errors is not None:
                geometric_model.reprojection_errors[geometric_model.inlier_mask] = post_errors

        # Stage 7: refinement
        subpix_on = subpix_cfg.get("enabled", False) and inlier_count > 0
        self.logger.log_refinement(
            enabled=subpix_on,
            win_size=win_size if subpix_on else None,
            pre_rmse=pre_refinement_rmse,
            post_rmse=post_refinement_rmse,
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

        ref_gsd = None
        if reference_image.metadata and reference_image.metadata.resolution_m_per_px:
            ref_gsd = reference_image.metadata.resolution_m_per_px
        elif self.config.get("reference_gsd_m"):
            ref_gsd = float(self.config["reference_gsd_m"])
        elif self.profile:
            ref_gsd = self.profile.approx_reference_gsd_m

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
            reference_gsd_m=ref_gsd,
        )

        # Stage 8: evaluation
        self.logger.log_evaluation(
            rmse_px=eval_result.rmse_pixels,
            rmse_m=eval_result.rmse_meters,
            median_px=eval_result.median_error_pixels,
            median_m=eval_result.median_error_meters,
            p90_px=eval_result.p90_error_pixels,
            coverage_pct=eval_result.coverage,
            elapsed_sec=elapsed_time,
            reference_gsd_m=ref_gsd,
            p90_m=eval_result.p90_error_meters,
        )

        # Stage 9: final result
        status_str = "SUCCESS" if (eval_result.inlier_matches >= 12 and (eval_result.rmse_pixels is None or eval_result.rmse_pixels < 5.0)) else ("WARNING" if eval_result.inlier_matches >= 4 else "FAILED")
        self.logger.log_final_result(
            status=status_str,
            inliers=eval_result.inlier_matches,
            rmse_px=eval_result.rmse_pixels,
            rmse_m=eval_result.rmse_meters,
            coverage_pct=eval_result.coverage,
            model_type=geometric_model.model_type,
            reference_gsd_m=ref_gsd,
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
