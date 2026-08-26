"""Configuration loading and deep-merging utilities."""

import os
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into base configuration dictionary.

    Nested dictionaries are merged without losing un-overridden keys in base.
    """
    merged = base.copy()
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(base_path: str, override_path: str | None = None) -> dict[str, Any]:
    """Load base YAML configuration file, optionally deep-merging override YAML file.

    Args:
        base_path: Path to default YAML configuration (e.g. configs/default.yaml).
        override_path: Optional path to override YAML (e.g. configs/cpu.yaml).

    Returns:
        Merged configuration dictionary.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base config file not found: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if override_path and os.path.exists(override_path):
        with open(override_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        config = deep_merge(config, overrides)

    return config
