"""Diagnostic plotting utilities for residual reprojection error distributions."""

import os

import matplotlib.pyplot as plt

from lunar_correspondence.io.metadata import GeometricModel


def plot_error_histogram(
    geometric_model: GeometricModel, output_path: str | None = None
) -> None:
    """Plot histogram of RANSAC inlier reprojection error distances.

    Args:
        geometric_model: Estimated GeometricModel with reprojection_errors and inlier_mask.
        output_path: Optional output path to save histogram plot.
    """
    if (
        geometric_model.reprojection_errors is None
        or len(geometric_model.reprojection_errors) == 0
    ):
        return

    inliers = geometric_model.inlier_mask
    errors = geometric_model.reprojection_errors[inliers]

    if len(errors) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title("RANSAC Inlier Reprojection Error Distribution")
    plt.xlabel("Reprojection Error (pixels)")
    plt.ylabel("Inlier Match Count")
    plt.grid(True, linestyle="--", alpha=0.5)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
