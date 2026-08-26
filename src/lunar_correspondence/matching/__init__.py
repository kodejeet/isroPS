"""Feature matching package."""

from lunar_correspondence.matching.base import Matcher
from lunar_correspondence.matching.descriptor_matcher import DescriptorMatcher
from lunar_correspondence.matching.fusion import fuse_match_sets
from lunar_correspondence.matching.lightglue_matcher import LightGlueMatcher
from lunar_correspondence.matching.rift_matcher import RIFTMatcher

__all__ = [
    "DescriptorMatcher",
    "LightGlueMatcher",
    "Matcher",
    "RIFTMatcher",
    "fuse_match_sets",
]
