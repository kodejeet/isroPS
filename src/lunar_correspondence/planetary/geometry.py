"""Orbital camera geometry modeling stub."""


def compute_sun_vector(
    sun_azimuth_deg: float, sun_elevation_deg: float
) -> dict[str, float]:
    """Compute 3D unit direction vector pointing toward the Sun from local terrain."""
    import math

    az_rad = math.radians(sun_azimuth_deg)
    el_rad = math.radians(sun_elevation_deg)

    x = math.cos(el_rad) * math.sin(az_rad)
    y = math.cos(el_rad) * math.cos(az_rad)
    z = math.sin(el_rad)

    return {"x": x, "y": y, "z": z}
