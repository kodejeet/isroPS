"""CLI driver for registering real lunar image pairs.

Uses the single orchestration function run_registration() from pipeline.py.
Outputs results to timestamped per-run subdirectories.
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from lunar_correspondence.config import load_config
from lunar_correspondence.io.image_loader import load_image
from lunar_correspondence.io.writers import save_metrics_json, save_registered_image
from lunar_correspondence.pipeline import run_registration
from lunar_correspondence.visualization.matches import draw_match_lines
from lunar_correspondence.visualization.registration import plot_registration_overlay


def main():
    parser = argparse.ArgumentParser(
        description="Register real lunar image pairs using SIFT baseline pipeline."
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Path to source (moving) image"
    )
    parser.add_argument(
        "--reference", type=str, required=True, help="Path to reference (fixed) image"
    )
    parser.add_argument(
        "--source-instrument",
        type=str,
        default="UNKNOWN",
        help="Metadata hint for source sensor (OHRC, TMC-2, IIRS, LRO_NAC, SELENE)",
    )
    parser.add_argument(
        "--reference-instrument",
        type=str,
        default="UNKNOWN",
        help="Metadata hint for reference sensor (OHRC, TMC-2, IIRS, LRO_NAC, SELENE)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Base output directory"
    )
    args = parser.parse_args()

    # Check for PDS4 raw product extension attempt
    for path in [args.source, args.reference]:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".xml", ".lbl", ".img", ".qub"]:
            print(f"\n[!] Error: File '{path}' appears to be a raw PDS4 product.")
            print("    Direct PDS4 ingestion is not supported in this prototype.")
            print(
                "    Please convert to GeoTIFF using GDAL: gdal_translate <input.xml> <output.tif>"
            )
            print("    See data/README.md for details.")
            sys.exit(1)

    print("==========================================================")
    print("        LUNAR REAL-IMAGE REGISTRATION PIPELINE             ")
    print("==========================================================")

    config = load_config(args.config)
    print(
        f"[*] Loaded configuration: {config.get('pipeline', {}).get('name', 'sift_baseline')}"
    )
    print(f"[*] Loading Source:    {args.source} [{args.source_instrument}]")
    print(f"[*] Loading Reference: {args.reference} [{args.reference_instrument}]")

    source_data = load_image(args.source, instrument=args.source_instrument)
    ref_data = load_image(args.reference, instrument=args.reference_instrument)

    print(f"    - Source shape:    {source_data.array.shape}")
    print(f"    - Reference shape: {ref_data.array.shape}")

    print("[*] Running registration via run_registration()...")
    reg_result, eval_result = run_registration(source_data, ref_data, config)

    # Per-run timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    src_stem = os.path.splitext(os.path.basename(args.source))[0]
    ref_stem = os.path.splitext(os.path.basename(args.reference))[0]
    run_output_dir = os.path.join(args.output_dir, f"{timestamp}_{src_stem}_{ref_stem}")
    os.makedirs(run_output_dir, exist_ok=True)

    reg_out_path = os.path.join(run_output_dir, "registered_output.png")
    matches_viz_path = os.path.join(run_output_dir, "match_visualization.png")
    overlay_viz_path = os.path.join(run_output_dir, "registration_overlay.png")
    metrics_json_path = os.path.join(run_output_dir, "metrics.json")
    csv_matches_path = os.path.join(run_output_dir, "matches.csv")

    save_registered_image(reg_result, reg_out_path)
    draw_match_lines(
        source_data.array,
        ref_data.array,
        reg_result.match_set,
        output_path=matches_viz_path,
    )
    plot_registration_overlay(
        ref_data.array, reg_result.registered_image, output_path=overlay_viz_path
    )
    save_metrics_json(eval_result, metrics_json_path)

    # Rescale coordinates for CSV output if downsampling occurred
    scale = eval_result.scale_factor if eval_result.scale_factor > 0 else 1.0
    match_set = reg_result.match_set
    src_pts = match_set.source_points / scale
    ref_pts = match_set.reference_points / scale
    conf = (
        match_set.confidence
        if match_set.confidence is not None
        else [0.0] * len(src_pts)
    )
    inliers = (
        match_set.inlier_mask
        if match_set.inlier_mask is not None
        else [True] * len(src_pts)
    )

    df_matches = pd.DataFrame(
        {
            "source_x": src_pts[:, 0],
            "source_y": src_pts[:, 1],
            "reference_x": ref_pts[:, 0],
            "reference_y": ref_pts[:, 1],
            "confidence_or_descriptor_distance": conf,
            "inlier": inliers,
        }
    )
    df_matches.to_csv(csv_matches_path, index=False)

    print("\n------------------- EVALUATION RESULTS -------------------")
    print(f" Total Matches:           {eval_result.total_matches}")
    print(f" Inlier Matches:          {eval_result.inlier_matches}")
    print(f" Inlier Ratio:            {eval_result.inlier_ratio:.4f}")
    print(
        f" RMSE (pixels):           {eval_result.rmse_pixels:.4f}"
        if eval_result.rmse_pixels
        else " RMSE: N/A"
    )
    print(
        f" Median Error (pixels):   {eval_result.median_error_pixels:.4f}"
        if eval_result.median_error_pixels
        else " Median Error: N/A"
    )
    print(f" Grid Coverage (%):       {eval_result.coverage:.2f}%")
    print(f" Scale Factor:            {eval_result.scale_factor}")
    print(f" Processing Time:         {eval_result.processing_time_seconds:.3f} s")
    print("----------------------------------------------------------")
    print(f"[*] Results saved to: {os.path.abspath(run_output_dir)}")
    print("Done.")


if __name__ == "__main__":
    main()
