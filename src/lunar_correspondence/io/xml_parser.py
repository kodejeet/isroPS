"""ISAC Chandrayaan-2 XML metadata parser for OHRC and IIRS products.

Parses official ISRO ISDA / ISAC XML metadata schema, extracting:
- Orbital geometry and spacecraft attitude (orbit number, altitude, roll, pitch, yaw)
- Solar illumination (elevation, azimuth, incidence)
- Sensor dimensions and detector health diagnostics
- Geographic (lon/lat) and projected (Easting/Northing) corner coordinates
- Automatic scene quality screening (e.g. night-pass detection)
"""

from dataclasses import dataclass, field
import os
import xml.etree.ElementTree as ET


@dataclass
class ISACMetadata:
    """Parsed metadata from Chandrayaan-2 ISAC product XML."""

    job_id: str
    product_id: str
    instrument: str  # "OHRC", "IIRS", etc.
    imaging_orbit_number: int | None = None
    spacecraft_altitude_km: float | None = None
    resolution_m_px: float | None = None
    sun_elevation_deg: float | None = None
    sun_azimuth_deg: float | None = None
    solar_incidence_deg: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    projection: str = "Unknown"
    reference_used: str = "System"
    corners_geo: dict[str, tuple[float, float]] = field(default_factory=dict)
    corners_projected: dict[str, tuple[float, float]] = field(default_factory=dict)
    raw_qube_height: int | None = None
    raw_qube_width: int | None = None
    line_loss_pct: float | None = None
    detector_temp_k: float | None = None
    dewar_temp_c: float | None = None
    is_night_pass: bool = False
    quality_flags: list[str] = field(default_factory=list)


def _safe_float(text: str | None) -> float | None:
    if text is None or not text.strip():
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def _safe_int(text: str | None) -> int | None:
    if text is None or not text.strip():
        return None
    try:
        return int(float(text.strip()))
    except ValueError:
        return None


def parse_isac_xml(xml_path: str) -> ISACMetadata:
    """Parse an ISAC XML file and return a structured ISACMetadata object.

    Args:
        xml_path: Path to the ISAC XML file.

    Returns:
        Populated ISACMetadata object.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML metadata file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def get_text(tag: str) -> str | None:
        elem = root.find(f".//{tag}")
        return elem.text.strip() if elem is not None and elem.text else None

    job_id = get_text("job_id") or os.path.basename(xml_path)
    product_id = get_text("product_id") or get_text("level0_dataset") or job_id

    # Determine instrument
    instrument = "UNKNOWN"
    base_upper = os.path.basename(xml_path).upper()
    if "OHR" in base_upper or root.find(".//ohr") is not None:
        instrument = "OHRC"
    elif "IIR" in base_upper or root.find(".//raw_qube_image_height") is not None:
        instrument = "IIRS"

    orbit_num = _safe_int(get_text("imaging_orbit_number")) or _safe_int(get_text("dumping_orbit_number"))
    altitude_km = _safe_float(get_text("spacecraft_altitude_in_km"))
    resolution_m = _safe_float(get_text("Resolution_in_meter")) or _safe_float(get_text("spatial_resolution"))

    sun_elev = _safe_float(get_text("Sun_elevation_in_degree"))
    sun_azim = _safe_float(get_text("Sun_azimuth_in_degree"))
    solar_inc = _safe_float(get_text("Solar_incidence_angle_in_degree"))

    roll = _safe_float(get_text("Roll_in_degree"))
    pitch = _safe_float(get_text("Pitch_in_degree"))
    yaw = _safe_float(get_text("Yaw_in_degree"))

    projection = get_text("projection") or "Unknown"
    reference_used = get_text("ReferenceUsed") or "System"

    # IIRS specific diagnostics
    qube_height = _safe_int(get_text("raw_qube_image_height"))
    qube_width = _safe_int(get_text("raw_qube_image_width"))
    line_loss = _safe_float(get_text("line_loss_percentage"))
    det_temp = _safe_float(get_text("detector_temperature_kelvin"))
    dewar_temp = _safe_float(get_text("dewar_vwt_temperature"))

    # Extract corners
    corners_geo: dict[str, tuple[float, float]] = {}
    corners_projected: dict[str, tuple[float, float]] = {}

    corner_positions = ["topleft", "topright", "bottomleft", "bottomright"]
    for pos in corner_positions:
        lat = _safe_float(get_text(f"{pos}_latitude"))
        lon = _safe_float(get_text(f"{pos}_longitude"))
        if lat is not None and lon is not None:
            corners_geo[pos] = (lon, lat)

        northing = _safe_float(get_text(f"{pos}_latitude_en"))
        easting = _safe_float(get_text(f"{pos}_longitude_en"))
        if easting is not None and northing is not None:
            corners_projected[pos] = (easting, northing)

    # Illumination and quality screening
    quality_flags: list[str] = []
    is_night = False
    if sun_elev is not None and sun_elev < 0.0:
        is_night = True
        quality_flags.append("INSUFFICIENT_ILLUMINATION")

    if line_loss is not None and line_loss > 5.0:
        quality_flags.append("HIGH_LINE_LOSS")

    if det_temp is not None and det_temp > 120.0:
        quality_flags.append("ELEVATED_DETECTOR_TEMPERATURE")

    return ISACMetadata(
        job_id=job_id,
        product_id=product_id,
        instrument=instrument,
        imaging_orbit_number=orbit_num,
        spacecraft_altitude_km=altitude_km,
        resolution_m_px=resolution_m,
        sun_elevation_deg=sun_elev,
        sun_azimuth_deg=sun_azim,
        solar_incidence_deg=solar_inc,
        roll_deg=roll,
        pitch_deg=pitch,
        yaw_deg=yaw,
        projection=projection,
        reference_used=reference_used,
        corners_geo=corners_geo,
        corners_projected=corners_projected,
        raw_qube_height=qube_height,
        raw_qube_width=qube_width,
        line_loss_pct=line_loss,
        detector_temp_k=det_temp,
        dewar_temp_c=dewar_temp,
        is_night_pass=is_night,
        quality_flags=quality_flags,
    )
