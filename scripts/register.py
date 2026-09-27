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
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from app.components.dataset_loader import load_windowed_pair
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
    parser.add_argument(
        "--subpixel",
        action="store_true",
        help="Enable sub-pixel corner refinement on inlier tiepoints for benchmarking",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.subpixel:
        if "geometry" not in config:
            config["geometry"] = {}
        if "subpixel_refinement" not in config["geometry"]:
            config["geometry"]["subpixel_refinement"] = {}
        config["geometry"]["subpixel_refinement"]["enabled"] = True

    source_exists = os.path.isfile(args.source)
    ref_exists = os.path.isfile(args.reference)

    if (
        source_exists
        and ref_exists
        and not (args.source.lower().endswith(".img") or args.reference.lower().endswith(".img"))
    ):
        source_data = load_image(args.source, instrument=args.source_instrument)
        ref_data = load_image(args.reference, instrument=args.reference_instrument)
    else:
        inst = args.source_instrument
        if inst == "UNKNOWN":
            s_low = args.source.lower()
            if "tmc" in s_low:
                inst = "TMC-2"
            elif "ohrc" in s_low:
                inst = "OHRC"
            elif "iirs" in s_low:
                inst = "IIRS"
            else:
                inst = "TMC-2"

        ref_file = args.reference if ref_exists else ""
        source_data, ref_data, pair_meta = load_windowed_pair(
            instrument=inst,
            ch2_product_id=args.source,
            lroc_filename=os.path.basename(args.reference),
            lroc_filepath=ref_file,
            target_crop_size=512,
        )
        if "profile" not in config:
            if inst == "TMC-2":
                config["profile"] = "TMC2"
            elif inst == "OHRC":
                config["profile"] = "OHRC_SIH_5M"
            elif inst == "IIRS":
                config["profile"] = (
                    "IIRS_SOUTH_POLE_WAC"
                    if "15194" in args.source or "South" in args.source
                    else "IIRS_EQUATORIAL_WAC"
                )

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

    print(f"[*] Artifacts successfully written to: {os.path.abspath(run_output_dir)}")
    print(f"    - Registered image:   {os.path.basename(reg_out_path)}")
    print(f"    - Match lines plot:   {os.path.basename(matches_viz_path)}")
    print(f"    - Blend overlay:      {os.path.basename(overlay_viz_path)}")
    print(f"    - Evaluation metrics: {os.path.basename(metrics_json_path)}")
    print(f"    - Inlier tiepoints:   {os.path.basename(csv_matches_path)}\n")


if __name__ == "__main__":
    main()
