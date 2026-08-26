"""I/O package for metadata, formats, image loading, and outputs."""

from lunar_correspondence.io.formats import ImageFormat, detect_image_format
from lunar_correspondence.io.image_loader import load_image
from lunar_correspondence.io.metadata import (
    EvaluationResult,
    FeatureSet,
    GeometricModel,
    ImageData,
    ImageMetadata,
    MatchSet,
    RegistrationResult,
)
from lunar_correspondence.io.writers import save_metrics_json, save_registered_image

__all__ = [
    "EvaluationResult",
    "FeatureSet",
    "GeometricModel",
    "ImageData",
    "ImageFormat",
    "ImageMetadata",
    "MatchSet",
    "RegistrationResult",
    "detect_image_format",
    "load_image",
    "save_metrics_json",
    "save_registered_image",
]
