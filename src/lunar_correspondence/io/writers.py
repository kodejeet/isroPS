"""File output writers for images and JSON metrics results."""

import json
import os
from dataclasses import asdict

import cv2

from lunar_correspondence.io.metadata import EvaluationResult, RegistrationResult


def save_registered_image(result: RegistrationResult, output_path: str) -> str:
    """Save warped registered image to file.

    Args:
        result: RegistrationResult containing warped registered_image.
        output_path: Path to output image file.

    Returns:
        Absolute path to written output file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img = result.registered_image

    if img.ndim == 3 and img.shape[2] == 3:
        # Convert RGB to BGR for OpenCV imwrite
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_bgr)
    else:
        cv2.imwrite(output_path, img)

    return os.path.abspath(output_path)


def save_metrics_json(eval_result: EvaluationResult, output_path: str) -> str:
    """Save EvaluationResult metrics to formatted JSON file.

    Args:
        eval_result: Computed EvaluationResult instance.
        output_path: Path to target JSON file.

    Returns:
        Absolute path to written JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    data = asdict(eval_result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return os.path.abspath(output_path)
