"""End-to-end baseline registration script on synthetic image pair.

Generates a synthetic lunar image pair (source and transformed reference),
executes the SIFT baseline pipeline, computes metrics, and outputs visualizations.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Add src to python path for standalone execution
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from lunar_correspondence.config import load_config
from lunar_correspondence.io.metadata import ImageData, ImageMetadata
from lunar_correspondence.io.writers import save_metrics_json, save_registered_image
from lunar_correspondence.pipeline import RegistrationPipeline
from lunar_correspondence.visualization.matches import draw_match_lines
from lunar_correspondence.visualization.registration import plot_registration_overlay


def generate_synthetic_lunar_pair(
    output_dir: str = "./data/examples", seed: int = 42
) -> tuple[str, str, np.ndarray]:
    """Generate synthetic lunar crater image pair (source and transformed reference).

    Returns:
        (source_path, reference_path, ground_truth_H)
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    h, w = 512, 512
    # 1. Base lunar surface noise background
    surface = np.random.normal(120, 15, (h, w)).astype(np.float32)

    # 2. Draw synthetic craters (circle rims with shadows)
    num_craters = 25
    for _ in range(num_craters):
        cx = np.random.randint(40, w - 40)
        cy = np.random.randint(40, h - 40)
        r = np.random.randint(10, 45)

        # Draw crater rim and interior shadow gradient
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Crater interior (dark shadow)
        shadow_mask = dist <= (r * 0.7)
        surface[shadow_mask] -= 45.0

        # Crater rim (bright peak)
        rim_mask = (dist > (r * 0.7)) & (dist <= r)
        surface[rim_mask] += 60.0

    src_img = np.clip(surface, 0, 255).astype(np.uint8)

    # 3. Apply known geometric transform to create reference image
    # Rotation (12 deg), Scale (1.08), Shift (15, -10), mild perspective
    angle = np.radians(12)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    scale = 1.08
    tx, ty = 18.0, -12.0

    # Transform matrix relative to image center
    cx, cy = w / 2.0, h / 2.0
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    R = np.array(
        [
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a, scale * cos_a, ty],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
    H_gt = T2 @ R @ T1
    H_gt[2, 0] += 0.00008  # mild perspective shear

    ref_img = cv2.warpPerspective(src_img, H_gt, (w, h))

    # Add mild illumination gradient to reference image (simulating sun angle variation)
    X, Y = np.meshgrid(np.linspace(0.8, 1.2, w), np.linspace(0.7, 1.1, h))
    ref_img = np.clip(ref_img.astype(np.float32) * X * Y, 0, 255).astype(np.uint8)

    # Save to disk
    src_path = os.path.join(output_dir, "synthetic_source.png")
    ref_path = os.path.join(output_dir, "synthetic_reference.png")

    cv2.imwrite(src_path, src_img)
    cv2.imwrite(ref_path, ref_img)

    return src_path, ref_path, H_gt


def main():
    parser = argparse.ArgumentParser(
        description="Run SIFT baseline lunar image registration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--override-config", type=str, default=None, help="Path to config override file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )
    args = parser.parse_args()

    print("==========================================================")
    print("      SIH 2026 PS 26166 - LUNAR IMAGE CORRESPONDENCE       ")
    print("==========================================================")

    # Load configuration with deep merging
    config = load_config(args.config, args.override_config)
    print(f"[*] Loaded configuration: pipeline='{config['pipeline']['name']}'")

    # Generate synthetic image pair
    print("[*] Generating synthetic lunar demo pair in ./data/examples/...")
    src_path, ref_path, GT_H = generate_synthetic_lunar_pair(
        "./data/examples", config["pipeline"]["random_seed"]
    )
    print(f"    - Source Image:    {src_path}")
    print(f"    - Reference Image: {ref_path}")

    # Load ImageData structures
    src_array = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    ref_array = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

    src_meta = ImageMetadata(instrument="OHRC", source_path=src_path)
    ref_meta = ImageMetadata(instrument="TMC-2", source_path=ref_path)

    src_data = ImageData(array=src_array, path=src_path, metadata=src_meta)
    ref_data = ImageData(array=ref_array, path=ref_path, metadata=ref_meta)

    # Instantiate and execute pipeline
    pipeline = RegistrationPipeline(config)
    print("[*] Executing registration pipeline...")
    reg_result, eval_result = pipeline.run(src_data, ref_data)

    # Save output artifacts
    os.makedirs(args.output_dir, exist_ok=True)

    reg_out_path = os.path.join(args.output_dir, "registered_output.png")
    matches_viz_path = os.path.join(args.output_dir, "match_visualization.png")
    overlay_viz_path = os.path.join(args.output_dir, "registration_overlay.png")
    metrics_json_path = os.path.join(args.output_dir, "metrics.json")

    save_registered_image(reg_result, reg_out_path)
    draw_match_lines(
        src_data.array,
        ref_data.array,
        reg_result.match_set,
        output_path=matches_viz_path,
    )
    plot_registration_overlay(
        ref_data.array, reg_result.registered_image, output_path=overlay_viz_path
    )
    save_metrics_json(eval_result, metrics_json_path)

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
    print(f" Processing Time:         {eval_result.processing_time_seconds:.3f} s")
    print("----------------------------------------------------------")
    print(f"[*] Results saved to directory: {os.path.abspath(args.output_dir)}")
    print("Done.")


if __name__ == "__main__":
    main()
