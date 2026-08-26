"""LightGlue learned matcher stub (Pending P2)."""

from typing import Any

from lunar_correspondence.io.metadata import FeatureSet, MatchSet
from lunar_correspondence.matching.base import Matcher


class LightGlueMatcher(Matcher):
    """LightGlue learned matcher stub (Pending P2)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def match(
        self, features_src: FeatureSet, features_ref: FeatureSet, **kwargs
    ) -> MatchSet:
        raise NotImplementedError(
            "LightGlue matcher is pending P2 development. "
            "For Day-1 execution, please use descriptor matcher (matching.method: 'descriptor')."
        )
