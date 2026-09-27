"""Controlled Empirical Registration Ablation Runner for CHANDRAMAP / SIH 2026.

Executes the ablation matrix on both official IIRS pairs:
1. Equatorial Mare (Orbit 26864, ~94m GSD)
2. South Polar Highland (Orbit 15193, ~83m GSD vs 200m WAC)

And verifies the 4 official OHRC 5m pairs (including night-pass rejection).

Matrix axes:
- Features: SIFT, RIFT2, Fused (SIFT + RIFT2)
- Preprocessing: Percentile Stretch vs Percentile Stretch + CLAHE
- Scale: Native Scale vs Common Physical Scale (Pyramid)
- Geometry: Global Affine, Global Homography, Block Affine
"""

import copy
import json
import os
import sys
import time
from typing import Any

import cv2
import numpy as np
import rasterio

# Add root and src to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from lunar_correspondence.features.rift_features import RIFTFeatureExtractor
from lunar_correspondence.features.sift_features import SIFTFeatureExtractor
from lunar_correspondence.geometry.homography import compute_reprojection_errors
from lunar_correspondence.geometry.ransac import estimate_geometric_model
from lunar_correspondence.io.metadata import FeatureSet, ImageData, ImageMetadata, MatchSet
from lunar_correspondence.io.sih_dataset import SIHDatasetPair, discover_sih_pairs, load_sih_image_pair
from lunar_correspondence.matching.descriptor_matcher import DescriptorMatcher
from lunar_correspondence.matching.fusion import fuse_match_sets
from lunar_correspondence.matching.rift_matcher import RIFTMatcher
from lunar_correspondence.planetary.footprints import estimate_geospatial_roi
from lunar_correspondence.preprocessing.normalization import create_feature_image


def evaluate_geometry(
    match_set: MatchSet,
    geo_model: str,
    reproj_thresh: float = 3.5,
    block_length: int = 1500,
    block_overlap: int = 300,
) -> dict[str, Any]:
    """Fit geometric model and compute quantitative residual statistics."""
    n_raw = len(match_set.source_points)
    if n_raw < 4:
        return {
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse_px": None,
            "median_err_px": None,
            "model_valid": False,
        }

    if geo_model == "global_homography":
        g_model = estimate_geometric_model(
            match_set,
            model_type="homography",
            reproj_threshold=reproj_thresh,
            max_iters=2000,
        )
        inliers = int(np.sum(g_model.inlier_mask)) if g_model.inlier_mask is not None else 0
        ratio = (inliers / n_raw * 100.0) if n_raw > 0 else 0.0
        if inliers > 0 and g_model.reprojection_errors is not None:
            errs = g_model.reprojection_errors[g_model.inlier_mask]
            rmse = float(np.sqrt(np.mean(errs**2)))
            med = float(np.median(errs))
        else:
            rmse, med = None, None
        return {
            "inliers": inliers,
            "inlier_ratio": ratio,
            "rmse_px": rmse,
            "median_err_px": med,
            "model_valid": inliers >= 4,
        }

    elif geo_model == "global_affine":
        g_model = estimate_geometric_model(
            match_set,
            model_type="affine",
            reproj_threshold=reproj_thresh,
            max_iters=2000,
        )
        inliers = int(np.sum(g_model.inlier_mask)) if g_model.inlier_mask is not None else 0
        ratio = (inliers / n_raw * 100.0) if n_raw > 0 else 0.0
        if inliers > 0 and g_model.reprojection_errors is not None:
            errs = g_model.reprojection_errors[g_model.inlier_mask]
            rmse = float(np.sqrt(np.mean(errs**2)))
            med = float(np.median(errs))
        else:
            rmse, med = None, None
        return {
            "inliers": inliers,
            "inlier_ratio": ratio,
            "rmse_px": rmse,
            "median_err_px": med,
            "model_valid": inliers >= 3,
        }

    else:  # block_affine
        src_pts = match_set.source_points
        dst_pts = match_set.reference_points
        max_y = float(np.max(src_pts[:, 1])) if len(src_pts) > 0 else 0.0

        step = max(200, block_length - block_overlap)
        y_starts = list(range(0, int(max_y) + 1, step)) if max_y > 0 else [0]

        total_inliers = 0
        block_errs = []
        valid_blocks = 0

        for y0 in y_starts:
            y1 = y0 + block_length
            b_mask = (src_pts[:, 1] >= y0) & (src_pts[:, 1] < y1)
            b_src = src_pts[b_mask]
            b_dst = dst_pts[b_mask]

            if len(b_src) >= 4:
                b_match = MatchSet(source_points=b_src, reference_points=b_dst)
                m = estimate_geometric_model(
                    b_match, model_type="affine", reproj_threshold=reproj_thresh
                )
                b_inl = int(np.sum(m.inlier_mask)) if m.inlier_mask is not None else 0
                if b_inl >= 3:
                    valid_blocks += 1
                    total_inliers += b_inl
                    errs = m.reprojection_errors[m.inlier_mask]
                    block_errs.extend(errs.tolist())

        ratio = (total_inliers / n_raw * 100.0) if n_raw > 0 else 0.0
        if block_errs:
            rmse = float(np.sqrt(np.mean(np.array(block_errs) ** 2)))
            med = float(np.median(np.array(block_errs)))
        else:
            rmse, med = None, None

        return {
            "inliers": total_inliers,
            "inlier_ratio": ratio,
            "rmse_px": rmse,
            "median_err_px": med,
            "valid_blocks": valid_blocks,
            "total_blocks": len(y_starts),
            "model_valid": valid_blocks > 0,
        }


def run_ablation() -> dict[str, Any]:
    """Run full ablation matrix across SIH pairs and print findings."""
    print("=" * 80, flush=True)
    print("CHANDRAMAP ABLATION MATRIX RUNNER — OFFICIAL SIH 2026 BENCHMARKS", flush=True)
    print("=" * 80, flush=True)

    pairs = discover_sih_pairs()
    if not pairs:
        print("ERROR: No SIH dataset pairs discovered!", flush=True)
        return {}

    # Initialize extractors with high-efficiency presets
    sift_extractor = SIFTFeatureExtractor({"nfeatures": 2500, "contrastThreshold": 0.012})
    sift_matcher = DescriptorMatcher({"ratio_test_threshold": 0.82})
    rift_extractor = RIFTFeatureExtractor({
        "npt": 1500,
        "patch_size": 64,
        "minWaveLength": 6,
        "nscale": 4,
        "norient": 6,
    })
    rift_matcher = RIFTMatcher({"ratio_test_threshold": 0.90})

    results_record = []

    # 1. Run IIRS Pairs
    iirs_pairs = [p for p in pairs if p.instrument == "IIRS"]
    for pair in iirs_pairs:
        print(f"\n=======================================================", flush=True)
        print(f"EVALUATING IIRS PAIR: {pair.pair_id}", flush=True)
        print(f"=======================================================", flush=True)
        src_img, ref_img = load_sih_image_pair(pair)
        s_arr = src_img.array
        r_arr = ref_img.array

        # For fast ablation, use representative sub-strip if image is huge (> 4000 lines)
        # e.g. first 3500 lines provides over 300km along track
        max_eval_lines = 3500
        if s_arr.shape[0] > max_eval_lines:
            s_arr_eval = s_arr[:max_eval_lines, :]
            print(f"  Using representative along-track slice: {s_arr_eval.shape[:2]} of {s_arr.shape[:2]}", flush=True)
        else:
            s_arr_eval = s_arr

        # Compute geospatial ROI crop on reference if possible
        ref_roi = estimate_geospatial_roi(pair.metadata, pair.reference_path, margin_px=64)
        if ref_roi is not None:
            min_y, min_x, max_y, max_x = ref_roi
            # Clamp to max_eval_lines proportionally
            if s_arr.shape[0] > max_eval_lines:
                max_y = min(max_y, min_y + max_eval_lines + 200)
            # Limit width to 1500 to keep FFT operations sub-5 seconds
            max_x = min(max_x, min_x + 1500)
            r_crop = r_arr[min_y:max_y, min_x:max_x]
            offset_y, offset_x = min_y, min_x
            print(f"  Geospatial ROI applied: ({min_y}, {min_x}) -> ({max_y}, {max_x}) [shape={r_crop.shape[:2]}]", flush=True)
        else:
            r_crop = r_arr[:max_eval_lines, : min(1500, r_arr.shape[1])] if r_arr.shape[0] > max_eval_lines else r_arr
            offset_y, offset_x = 0, 0
            print(f"  Reference crop used: shape={r_crop.shape[:2]}", flush=True)


        scale_modes = ["native"]
        if pair.source_gsd_m < pair.reference_gsd_m * 0.7:
            scale_modes.append("common_physical_scale")

        prep_modes = ["percentile", "percentile_clahe"]
        geo_modes = ["global_affine", "global_homography", "block_affine"]

        for scale_m in scale_modes:
            if scale_m == "common_physical_scale":
                factor = pair.source_gsd_m / pair.reference_gsd_m
                new_w = max(16, int(s_arr_eval.shape[1] * factor))
                new_h = max(16, int(s_arr_eval.shape[0] * factor))
                s_scaled = cv2.resize(s_arr_eval, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"\n--- Scale Mode: {scale_m} (Resized source to {s_scaled.shape[:2]}, factor={factor:.3f}) ---", flush=True)
            else:
                s_scaled = s_arr_eval
                factor = 1.0
                print(f"\n--- Scale Mode: {scale_m} (Native {s_scaled.shape[:2]}) ---", flush=True)

            for prep_m in prep_modes:
                apply_c = (prep_m == "percentile_clahe")
                print(f"  Preprocessing: {prep_m}...", flush=True)
                src_feat_img, _ = create_feature_image(
                    s_scaled,
                    method="percentile",
                    p_low=0.5,
                    p_high=99.5,
                    mask_negatives=True,
                    apply_clahe=apply_c,
                    clahe_clip_limit=3.0,
                    clahe_grid_size=(16, 16),
                )
                ref_feat_img, _ = create_feature_image(
                    r_crop,
                    method="percentile",
                    p_low=0.5,
                    p_high=99.5,
                    mask_negatives=False,
                    apply_clahe=apply_c,
                    clahe_clip_limit=3.0,
                    clahe_grid_size=(16, 16),
                )

                src_data = ImageData(src_feat_img[:, :, np.newaxis], "s", ImageMetadata("SRC"))
                ref_data = ImageData(ref_feat_img[:, :, np.newaxis], "r", ImageMetadata("REF"))

                # 1. Extract SIFT
                t0 = time.time()
                f_src_s = sift_extractor.extract(src_data)
                f_ref_s = sift_extractor.extract(ref_data)
                m_sift = sift_matcher.match(f_src_s, f_ref_s)
                t_sift = time.time() - t0

                # 2. Extract RIFT2
                t0 = time.time()
                f_src_r = rift_extractor.extract(src_data)
                f_ref_r = rift_extractor.extract(ref_data)
                m_rift = rift_matcher.match(f_src_r, f_ref_r)
                t_rift = time.time() - t0

                # 3. Fuse
                m_fused = fuse_match_sets([m_sift, m_rift])

                feature_cases = [
                    ("SIFT", len(f_src_s.keypoints), len(f_ref_s.keypoints), m_sift, t_sift),
                    ("RIFT2", len(f_src_r.keypoints), len(f_ref_r.keypoints), m_rift, t_rift),
                    ("Fused", len(f_src_s.keypoints) + len(f_src_r.keypoints), len(f_ref_s.keypoints) + len(f_ref_r.keypoints), m_fused, t_sift + t_rift),
                ]

                for feat_name, k_src, k_ref, match_obj, feat_time in feature_cases:
                    # Offset reference points back to global ROI coordinates
                    if offset_x > 0 or offset_y > 0:
                        adj_match = MatchSet(
                            source_points=match_obj.source_points.copy(),
                            reference_points=match_obj.reference_points.copy()
                            + np.array([offset_x, offset_y], dtype=np.float32),
                            confidence=match_obj.confidence,
                        )
                    else:
                        adj_match = match_obj

                    for geo_m in geo_modes:
                        geo_res = evaluate_geometry(adj_match, geo_m)
                        rmse_m = (
                            geo_res["rmse_px"] * pair.reference_gsd_m
                            if geo_res["rmse_px"] is not None
                            else None
                        )

                        record = {
                            "pair_id": pair.pair_id,
                            "instrument": "IIRS",
                            "scale_mode": scale_m,
                            "preprocessing": prep_m,
                            "feature_method": feat_name,
                            "geometry_model": geo_m,
                            "keypoints_src": k_src,
                            "keypoints_ref": k_ref,
                            "raw_matches": len(match_obj.source_points),
                            "inliers": geo_res["inliers"],
                            "inlier_ratio_pct": round(geo_res["inlier_ratio"], 2),
                            "rmse_px": round(geo_res["rmse_px"], 3) if geo_res["rmse_px"] else None,
                            "rmse_meters": round(rmse_m, 2) if rmse_m else None,
                            "median_err_px": round(geo_res["median_err_px"], 3) if geo_res["median_err_px"] else None,
                            "runtime_sec": round(feat_time, 2),
                        }
                        results_record.append(record)

                        print(
                            f"    [{feat_name:5s}] + [{geo_m:16s}] -> Inliers: {geo_res['inliers']:3d}/{len(match_obj.source_points):3d} "
                            f"({geo_res['inlier_ratio']:5.1f}%) | RMSE: {str(record['rmse_px']):6s}px ({str(record['rmse_meters']):6s}m)",
                            flush=True,
                        )

    # 2. Run OHRC Pairs (Verifying optical path and night-pass rejection)
    print("\n" + "=" * 80, flush=True)
    print("VERIFYING OHRC 5M BENCHMARK PAIRS", flush=True)
    print("=" * 80, flush=True)
    ohrc_pairs = [p for p in pairs if p.instrument == "OHRC"]
    for pair in ohrc_pairs:
        print(f"\nEvaluating OHRC Pair: {pair.pair_id}", flush=True)
        if pair.is_night_pass:
            print(f"  --> REJECTED / SCREENED: {pair.quality_flags} (Sun Elevation < 0)", flush=True)
            record = {
                "pair_id": pair.pair_id,
                "instrument": "OHRC",
                "scale_mode": "native",
                "preprocessing": "clahe",
                "feature_method": "fused",
                "geometry_model": "homography",
                "keypoints_src": 0,
                "keypoints_ref": 0,
                "raw_matches": 0,
                "inliers": 0,
                "inlier_ratio_pct": 0.0,
                "rmse_px": None,
                "rmse_meters": None,
                "status": "REJECTED_NIGHT_PASS",
            }
            results_record.append(record)
            continue

        src_img, ref_img = load_sih_image_pair(pair)
        # For high-efficiency optical verification, evaluate 1500x600 source vs 1500x1500 ref
        s_eval = src_img.array[:1500, :]
        r_eval = ref_img.array[:1500, : min(1500, ref_img.array.shape[1])]

        src_feat, _ = create_feature_image(s_eval, method="minmax", apply_clahe=True)
        ref_feat, _ = create_feature_image(r_eval, method="minmax", apply_clahe=True)

        s_data = ImageData(src_feat[:, :, np.newaxis], "s", ImageMetadata("SRC"))
        r_data = ImageData(ref_feat[:, :, np.newaxis], "r", ImageMetadata("REF"))

        t0 = time.time()
        f_s_s = sift_extractor.extract(s_data)
        f_r_s = sift_extractor.extract(r_data)
        matches = sift_matcher.match(f_s_s, f_r_s)
        elapsed = time.time() - t0

        geo_res = evaluate_geometry(matches, "global_homography")
        rmse_m = geo_res["rmse_px"] * pair.reference_gsd_m if geo_res["rmse_px"] else None

        record = {
            "pair_id": pair.pair_id,
            "instrument": "OHRC",
            "scale_mode": "native",
            "preprocessing": "clahe",
            "feature_method": "SIFT",
            "geometry_model": "global_homography",
            "keypoints_src": len(f_s_s.keypoints),
            "keypoints_ref": len(f_r_s.keypoints),
            "raw_matches": len(matches.source_points),
            "inliers": geo_res["inliers"],
            "inlier_ratio_pct": round(geo_res["inlier_ratio"], 2),
            "rmse_px": round(geo_res["rmse_px"], 3) if geo_res["rmse_px"] else None,
            "rmse_meters": round(rmse_m, 2) if rmse_m else None,
            "runtime_sec": round(elapsed, 2),
            "status": "NOMINAL",
        }
        results_record.append(record)
        print(f"  Matches: {len(matches.source_points)} | Inliers: {geo_res['inliers']} ({geo_res['inlier_ratio']:.1f}%) | RMSE: {geo_res['rmse_px']} px ({rmse_m} m) [runtime: {elapsed:.2f}s]", flush=True)

    # Save to JSON
    os.makedirs(os.path.join(ROOT_DIR, "outputs"), exist_ok=True)
    out_path = os.path.join(ROOT_DIR, "outputs", "iirs_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results_record, f, indent=2)

    print(f"\nAblation results successfully saved to: {out_path}", flush=True)
    return {"results": results_record}



if __name__ == "__main__":
    run_ablation()
