"""Official SIH 2026 dataset discovery and ingestion module.

Discovers, verifies, and loads official ISRO benchmark pairs:
1. OHRC 5m source + 5m LRO reference GeoTIFFs + ISAC XML
2. IIRS unrectified pushbroom radiance + LRO WAC reference GeoTIFFs + ISAC XML
"""

from dataclasses import dataclass, field
import glob
import os
import rasterio
import numpy as np

from lunar_correspondence.io.metadata import ImageData, ImageMetadata
from lunar_correspondence.io.xml_parser import ISACMetadata, parse_isac_xml


@dataclass
class SIHDatasetPair:
    """Represents a verified source-reference-metadata pair from SIH 2026 benchmarks."""

    pair_id: str
    instrument: str  # "OHRC" or "IIRS"
    source_path: str
    reference_path: str
    xml_path: str | None
    metadata: ISACMetadata | None
    source_gsd_m: float
    reference_gsd_m: float
    source_shape: tuple[int, int]  # (height, width)
    reference_shape: tuple[int, int]  # (height, width)
    source_dtype: str
    reference_dtype: str
    reference_crs: str
    is_night_pass: bool = False
    quality_flags: list[str] = field(default_factory=list)


def discover_sih_pairs(
    base_dir: str = "/home/kode/gh-projs/sih2k26/data_for_sih_2026",
) -> list[SIHDatasetPair]:
    """Scan base directory for all official OHRC and IIRS dataset pairs.

    Args:
        base_dir: Root directory containing 'ohrc' and 'IIRS' subfolders.

    Returns:
        List of discovered and verified SIHDatasetPair instances.
    """
    pairs: list[SIHDatasetPair] = []
    if not os.path.exists(base_dir):
        return pairs

    # 1. Discover OHRC pairs (*_source_at_5m.tif)
    ohrc_sources = glob.glob(
        os.path.join(base_dir, "**", "*_source_at_5m.tif"), recursive=True
    )
    for src_path in sorted(ohrc_sources):
        ref_path = src_path.replace("_source_at_5m.tif", "_reference_at_5m.tif")
        if not os.path.exists(ref_path):
            continue

        base_prefix = src_path.replace("_source_at_5m.tif", "")
        xml_candidates = glob.glob(f"{base_prefix}*.xml")
        xml_path = xml_candidates[0] if xml_candidates else None

        metadata = parse_isac_xml(xml_path) if xml_path and os.path.exists(xml_path) else None

        # Inspect rasters with rasterio
        with rasterio.open(src_path) as s_src, rasterio.open(ref_path) as r_src:
            s_shape = (s_src.height, s_src.width)
            r_shape = (r_src.height, r_src.width)
            s_dtype = str(s_src.dtypes[0])
            r_dtype = str(r_src.dtypes[0])
            ref_crs = str(r_src.crs) if r_src.crs else "Unknown"
            ref_res = float(r_src.transform[0]) if r_src.transform else 5.0

        pair_id = os.path.basename(base_prefix).rstrip("_")
        quality_flags = list(metadata.quality_flags) if metadata else []
        is_night = metadata.is_night_pass if metadata else False

        pairs.append(
            SIHDatasetPair(
                pair_id=pair_id,
                instrument="OHRC",
                source_path=src_path,
                reference_path=ref_path,
                xml_path=xml_path,
                metadata=metadata,
                source_gsd_m=5.0,  # Official downsampled 5m benchmark
                reference_gsd_m=ref_res,
                source_shape=s_shape,
                reference_shape=r_shape,
                source_dtype=s_dtype,
                reference_dtype=r_dtype,
                reference_crs=ref_crs,
                is_night_pass=is_night,
                quality_flags=quality_flags,
            )
        )

    # 2. Discover IIRS pairs (*_source.tif)
    iirs_sources = [
        p
        for p in glob.glob(os.path.join(base_dir, "**", "*_source.tif"), recursive=True)
        if "at_5m" not in p
    ]
    for src_path in sorted(iirs_sources):
        ref_path = src_path.replace("_source.tif", "_reference.tif")
        if not os.path.exists(ref_path):
            continue

        base_prefix = src_path.replace("_source.tif", "")
        xml_candidates = glob.glob(f"{base_prefix}*.xml")
        xml_path = xml_candidates[0] if xml_candidates else None

        metadata = parse_isac_xml(xml_path) if xml_path and os.path.exists(xml_path) else None

        with rasterio.open(src_path) as s_src, rasterio.open(ref_path) as r_src:
            s_shape = (s_src.height, s_src.width)
            r_shape = (r_src.height, r_src.width)
            s_dtype = str(s_src.dtypes[0])
            r_dtype = str(r_src.dtypes[0])
            ref_crs = str(r_src.crs) if r_src.crs else "Unknown"
            
            # Determine GSD
            ref_res = abs(float(r_src.transform[0])) if r_src.transform else 100.0
            # If in degrees (e.g. 0.0003125 deg), convert to approximate meters
            if ref_res < 0.01:
                # Selenographic degrees: 1 deg ~ 30.323 km at equator
                ref_res_m = ref_res * (1737400.0 * np.pi / 180.0)
            else:
                ref_res_m = ref_res

        source_gsd = metadata.resolution_m_px if (metadata and metadata.resolution_m_px) else 80.0
        pair_id = os.path.basename(base_prefix).rstrip("_")
        quality_flags = list(metadata.quality_flags) if metadata else []

        pairs.append(
            SIHDatasetPair(
                pair_id=pair_id,
                instrument="IIRS",
                source_path=src_path,
                reference_path=ref_path,
                xml_path=xml_path,
                metadata=metadata,
                source_gsd_m=float(source_gsd),
                reference_gsd_m=float(ref_res_m),
                source_shape=s_shape,
                reference_shape=r_shape,
                source_dtype=s_dtype,
                reference_dtype=r_dtype,
                reference_crs=ref_crs,
                is_night_pass=metadata.is_night_pass if metadata else False,
                quality_flags=quality_flags,
            )
        )

    return pairs


def load_sih_image_pair(pair: SIHDatasetPair) -> tuple[ImageData, ImageData]:
    """Load source and reference rasters, preserving native dtypes and physical values.

    Args:
        pair: Verified SIHDatasetPair.

    Returns:
        Tuple of (source_image_data, reference_image_data).
    """
    with rasterio.open(pair.source_path) as s_src:
        s_arr = s_src.read()
        s_arr = np.moveaxis(s_arr, 0, -1)  # (H, W, C)

    with rasterio.open(pair.reference_path) as r_src:
        r_arr = r_src.read()
        r_arr = np.moveaxis(r_arr, 0, -1)

    s_bounds = None
    if pair.metadata and "topleft" in pair.metadata.corners_geo:
        c = pair.metadata.corners_geo
        s_bounds = (
            min(c["topleft"][0], c["bottomleft"][0]),
            min(c["bottomleft"][1], c["bottomright"][1]),
            max(c["topright"][0], c["bottomright"][0]),
            max(c["topleft"][1], c["topright"][1]),
        )

    s_meta = ImageMetadata(
        instrument=pair.instrument,
        acquisition_time=None,
        resolution_m_per_px=pair.source_gsd_m,
        sun_azimuth_deg=pair.metadata.sun_azimuth_deg if pair.metadata else None,
        sun_elevation_deg=pair.metadata.sun_elevation_deg if pair.metadata else None,
        geographic_bounds=s_bounds,
        projection=pair.metadata.projection if pair.metadata else None,
        source_path=pair.source_path,
    )

    r_meta = ImageMetadata(
        instrument="LRO_WAC" if pair.instrument == "IIRS" else "LRO_NAC",
        acquisition_time=None,
        resolution_m_per_px=pair.reference_gsd_m,
        sun_azimuth_deg=None,
        sun_elevation_deg=None,
        geographic_bounds=None,
        projection=pair.reference_crs,
        source_path=pair.reference_path,
    )

    return (
        ImageData(array=s_arr, path=pair.source_path, metadata=s_meta),
        ImageData(array=r_arr, path=pair.reference_path, metadata=r_meta),
    )
