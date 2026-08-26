"""Lunar Image Correspondence Package (SIH Problem Statement 26166)."""

from lunar_correspondence.config import deep_merge, load_config
from lunar_correspondence.io import (
    EvaluationResult,
    FeatureSet,
    GeometricModel,
    ImageData,
    ImageMetadata,
    MatchSet,
    RegistrationResult,
    load_image,
)
from lunar_correspondence.pipeline import RegistrationPipeline

__version__ = "0.1.0"

__all__ = [
    "EvaluationResult",
    "FeatureSet",
    "GeometricModel",
    "ImageData",
    "ImageMetadata",
    "MatchSet",
    "RegistrationPipeline",
    "RegistrationResult",
    "deep_merge",
    "load_config",
    "load_image",
]
