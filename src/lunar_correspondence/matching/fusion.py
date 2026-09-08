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
    valid_src = [m.source_points for m in match_sets if len(m.source_points) > 0]
    valid_ref = [m.reference_points for m in match_sets if len(m.reference_points) > 0]

    src_pts = (
        np.vstack(valid_src).astype(np.float32)
        if valid_src
        else np.zeros((0, 2), dtype=np.float32)
    )
    ref_pts = (
        np.vstack(valid_ref).astype(np.float32)
        if valid_ref
        else np.zeros((0, 2), dtype=np.float32)
    )

    return MatchSet(
        source_points=src_pts,
        reference_points=ref_pts,
        confidence=None,
        inlier_mask=None,
    )
