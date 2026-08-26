"""Spatial footprint overlap detection utilities."""


def calculate_footprint_intersection(
    bounds1: tuple[float, float, float, float],
    bounds2: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Compute bounding box intersection between two image geographical footprints.

    Args:
        bounds1: (min_lon1, min_lat1, max_lon1, max_lat1)
        bounds2: (min_lon2, min_lat2, max_lon2, max_lat2)

    Returns:
        Intersection (min_lon, min_lat, max_lon, max_lat) or None if no overlap.
    """
    min_lon = max(bounds1[0], bounds2[0])
    min_lat = max(bounds1[1], bounds2[1])
    max_lon = min(bounds1[2], bounds2[2])
    max_lat = min(bounds1[3], bounds2[3])

    if min_lon < max_lon and min_lat < max_lat:
        return (min_lon, min_lat, max_lon, max_lat)
    return None
