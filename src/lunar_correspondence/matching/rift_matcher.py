"""RIFT feature matcher stub (Pending P1)."""

from typing import Any

from lunar_correspondence.io.metadata import FeatureSet, MatchSet
from lunar_correspondence.matching.base import Matcher


class RIFTMatcher(Matcher):
    """RIFT matcher adapter stub (Pending P1)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def match(
        self, features_src: FeatureSet, features_ref: FeatureSet, **kwargs
    ) -> MatchSet:
        raise NotImplementedError(
            "RIFT feature matcher is pending P1 development. "
            "For Day-1 execution, please configure matching.method: 'descriptor'."
        )
