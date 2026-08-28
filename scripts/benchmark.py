"""Saturday Quantitative Benchmark Runner for Lunar Image Registration Pipeline.

Compares three progressive pipeline configurations on synthetic lunar imagery:
- Stage A: Baseline (SIFT -> Lowe Ratio -> RANSAC)
- Stage B: Spatial Selection (SIFT -> Lowe Ratio -> Spatial 8x8 top-k=4 -> RANSAC)
- Stage C: Spatial + Subpixel (SIFT -> Lowe Ratio -> Spatial 8x8 -> RANSAC -> cornerSubPix)

Demonstrates quantitative impact on match count, inlier ratio, RMSE, median error,
and grid spatial coverage percentage under scientific honesty.
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Any

import cv2
import numpy as np

# Add root and src to python path for standalone execution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from lunar_correspondence.config import load_config
from lunar_correspondence.io.metadata import ImageData, ImageMetadata
from lunar_correspondence.pipeline import RegistrationPipeline
from scripts.run_baseline import generate_synthetic_lunar_pair


def run_benchmark(
    src_data: ImageData,
    ref_data: ImageData,
    base_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute the three benchmark stages on the provided image pair."""
    stages = [
        {
            "stage_id": "Stage A: Baseline",
            "description": "SIFT -> Lowe Ratio -> RANSAC",
            "spatial_enabled": False,
            "subpixel_enabled": False,
        },
        {
            "stage_id": "Stage B: Spatial Selection",
            "description": "SIFT -> Lowe Ratio -> Spatial (8x8, k=4) -> RANSAC",
            "spatial_enabled": True,
            "subpixel_enabled": False,
        },
        {
            "stage_id": "Stage C: Spatial + Subpixel",
            "description": "SIFT -> Lowe -> Spatial (8x8, k=4) -> RANSAC -> Subpixel",
            "spatial_enabled": True,
            "subpixel_enabled": True,
        },
    ]

    records = []

    # First, run Feature Extraction & Matching alone to get raw candidate count
    pipe_temp = RegistrationPipeline(base_config)
    feat_src = pipe_temp.feature_extractor.extract(src_data)
    feat_ref = pipe_temp.feature_extractor.extract(ref_data)
    raw_matches = pipe_temp.matcher.match(feat_src, feat_ref)
    candidate_matches_count = len(raw_matches.source_points)

    for stg in stages:
        cfg = copy.deepcopy(base_config)
        cfg["matching"]["spatial_selection"] = {
            "enabled": stg["spatial_enabled"],
            "grid_rows": 8,
            "grid_cols": 8,
            "top_k": 4,
        }
        cfg["geometry"]["subpixel_refinement"] = {
            "enabled": stg["subpixel_enabled"],
            "win_size": 5,
            "zero_zone": -1,
        }

        pipeline = RegistrationPipeline(cfg)
        t0 = time.perf_counter()
        _reg_res, eval_res = pipeline.run(src_data, ref_data)
        elapsed_sec = time.perf_counter() - t0

        selected_matches_count = eval_res.total_matches
        inliers_count = eval_res.inlier_matches
        inlier_ratio = eval_res.inlier_ratio
        pre_rmse = eval_res.pre_refinement_rmse_pixels
        post_rmse = eval_res.post_refinement_rmse_pixels
        median_err = eval_res.median_error_pixels
        coverage = eval_res.coverage

        records.append(
            {
                "Stage": stg["stage_id"],
                "Description": stg["description"],
                "Candidate Matches": candidate_matches_count,
                "Selected Matches": selected_matches_count,
                "RANSAC Inliers": inliers_count,
                "Inlier Ratio": inlier_ratio,
                "Pre-Refinement RMSE (px)": pre_rmse,
                "Post-Refinement RMSE (px)": post_rmse,
                "Median Error (px)": median_err,
                "Grid Coverage (%)": coverage,
                "Time (s)": elapsed_sec,
            }
        )

    return records


def format_table(records: list[dict[str, Any]]) -> str:
    """Format benchmark records into a GitHub-style markdown table."""
    headers = [
        "Stage",
        "Candidate",
        "Selected",
        "Inliers",
        "Inlier Ratio",
        "Pre-RMSE (px)",
        "Post-RMSE (px)",
        "Median Err (px)",
        "Coverage (%)",
        "Time (s)",
    ]

    rows = []
    for r in records:
        pre_s = f"{r['Pre-Refinement RMSE (px)']:.4f}" if r['Pre-Refinement RMSE (px)'] is not None else "N/A"
        post_s = f"{r['Post-Refinement RMSE (px)']:.4f}" if r['Post-Refinement RMSE (px)'] is not None else "N/A"
        med_s = f"{r['Median Error (px)']:.4f}" if r['Median Error (px)'] is not None else "N/A"
        rows.append(
            [
                r["Stage"],
                str(r["Candidate Matches"]),
                str(r["Selected Matches"]),
                str(r["RANSAC Inliers"]),
                f"{r['Inlier Ratio']:.2%}",
                pre_s,
                post_s,
                med_s,
                f"{r['Grid Coverage (%)']:.1f}%",
                f"{r['Time (s)']:.3f}",
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + data_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run quantitative registration benchmark."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to pipeline configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic dataset generation",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to output results JSON file",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("      SIH 2026 PS 26166 — SATURDAY QUANTITATIVE BENCHMARK       ")
    print("=" * 70)

    # 1. Generate standard synthetic pair
    synth_dir = "./data/examples"
    print(f"[*] Preparing synthetic lunar benchmark pair (seed={args.seed})...")
    src_path, ref_path, _ = generate_synthetic_lunar_pair(synth_dir, seed=args.seed)

    src_array = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    ref_array = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

    src_data = ImageData(
        array=src_array,
        path=src_path,
        metadata=ImageMetadata(instrument="OHRC", source_path=src_path),
    )
    ref_data = ImageData(
        array=ref_array,
        path=ref_path,
        metadata=ImageMetadata(instrument="TMC-2", source_path=ref_path),
    )

    # 2. Load configuration
    config = load_config(args.config)

    # 3. Execute benchmark
    print("[*] Running comparative stages A, B, C...")
    records = run_benchmark(src_data, ref_data, config)

    # 4. Display results
    print("\n" + format_table(records) + "\n")

    # 5. Output JSON if requested
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"[*] Benchmark metrics saved to: {os.path.abspath(args.output_json)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
