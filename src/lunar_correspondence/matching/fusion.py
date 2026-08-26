"""Multi-matcher score fusion module.

Combines match confidences across multiple matcher methods using configurable weights.
"""

import numpy as np

from lunar_correspondence.io.metadata import MatchSet


def fuse_match_sets(
    match_sets: list[MatchSet], weights: list[float] | None = None
) -> MatchSet:
    """Fuse multiple MatchSet predictions into a unified weighted MatchSet stub.

    Args:
        match_sets: List of MatchSet instances from different matchers.
        weights: Optional list of confidence weights for each matcher (read from config).

    Returns:
        Fused MatchSet.
    """
    if not match_sets:
        return MatchSet(
            source_points=np.zeros((0, 2), dtype=np.float32),
            reference_points=np.zeros((0, 2), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
        )

    if weights is None:
        weights = [1.0 / len(match_sets)] * len(match_sets)

    # Simple concatenation for stub baseline
    src_pts = (
        np.vstack([m.source_points for m in match_sets if len(m.source_points) > 0])
        if match_sets
        else np.zeros((0, 2))
    )
    ref_pts = (
        np.vstack(
            [m.reference_points for m in match_sets if len(m.reference_points) > 0]
        )
        if match_sets
        else np.zeros((0, 2))
    )

    return MatchSet(
        source_points=src_pts,
        reference_points=ref_pts,
        confidence=None,
        inlier_mask=None,
    )
