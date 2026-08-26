"""Planetary geographic coordinate conversion utilities.

Gloss:
- Geographic Coordinates: Lunar latitude/longitude coordinates (Moon-centered planetocentric or planetographic).
"""


def pixel_to_geographic(
    x: float,
    y: float,
    geographic_bounds: tuple[float, float, float, float] | None,
    image_shape: tuple[int, int],
) -> tuple[float, float] | None:
    """Convert pixel (x, y) coordinates to approximate lunar latitude/longitude.

    Args:
        x: Pixel column coordinate.
        y: Pixel row coordinate.
        geographic_bounds: Optional (min_lon, min_lat, max_lon, max_lat).
        image_shape: (height, width).

    Returns:
        (longitude, latitude) tuple or None if bounds unavailable.
    """
    if geographic_bounds is None:
        return None

    min_lon, min_lat, max_lon, max_lat = geographic_bounds
    h, w = image_shape

    lon = min_lon + (x / w) * (max_lon - min_lon)
    lat = max_lat - (y / h) * (max_lat - min_lat)

    return float(lon), float(lat)
