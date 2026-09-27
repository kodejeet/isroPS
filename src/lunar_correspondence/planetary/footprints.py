"""Spatial footprint overlap detection utilities."""

from typing import Any
import numpy as np


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


def estimate_geospatial_roi(
    metadata: Any,
    reference_path: str,
    margin_px: int = 64,
) -> tuple[int, int, int, int] | None:
    """Estimate pixel bounding box (min_y, min_x, max_y, max_x) in reference raster.

    Uses XML corner coordinates (projected easting/northing or geographic lon/lat)
    and the reference GeoTIFF inverse affine geotransform.

    Args:
        metadata: ISACMetadata or object containing corners_geo / corners_projected.
        reference_path: Path to georeferenced reference GeoTIFF file.
        margin_px: Pixel margin padding around the computed footprint.

    Returns:
        (min_y, min_x, max_y, max_x) slice coordinates in reference raster, or None.
    """
    import os
    import rasterio

    if not os.path.exists(reference_path):
        return None

    try:
        with rasterio.open(reference_path) as ref_src:
            if not ref_src.transform or ref_src.transform.is_identity:
                return None

            inv_transform = ~ref_src.transform
            h, w = ref_src.height, ref_src.width
            pixel_scale = abs(ref_src.transform[0])
            is_meter_coords = (pixel_scale >= 1.0) or (ref_src.crs and ref_src.crs.is_projected)

            corners: dict[str, tuple[float, float]] = {}
            if getattr(metadata, "corners_projected", None) and len(metadata.corners_projected) >= 3 and is_meter_coords:
                corners = metadata.corners_projected
            elif getattr(metadata, "corners_geo", None) and len(metadata.corners_geo) >= 3:
                corners = metadata.corners_geo


            if not corners:
                return None

            cols: list[float] = []
            rows: list[float] = []
            for _pos, (x_coord, y_coord) in corners.items():
                c, r = inv_transform * (x_coord, y_coord)
                cols.append(c)
                rows.append(r)

            min_c = max(0, int(np.floor(min(cols))) - margin_px)
            max_c = min(w, int(np.ceil(max(cols))) + margin_px)
            min_r = max(0, int(np.floor(min(rows))) - margin_px)
            max_r = min(h, int(np.ceil(max(rows))) + margin_px)

            if max_c > min_c and max_r > min_r:
                return (min_r, min_c, max_r, max_c)
            return None
    except Exception:
        return None

