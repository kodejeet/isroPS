# Real Lunar Example Data Crops

This directory contains small, representative 512x512 GeoTIFF crops extracted from real Chandrayaan-2 OHRC PDS4 data products for offline testing and real-data validation.

---

## Products Included

### 1. `ohrc_20260103T100517_crop.tif` (Source Image)
- **Source Mission:** Chandrayaan-2 Orbiter
- **Instrument:** Orbiter High Resolution Camera (OHRC)
- **Product Logical Identifier:** `urn:isro:isda:ch2_cho.ohr:data_calibrated:ch2_ohr_ncp_20260103t1005176450_d_img_d18`
- **Original Source File:** `ch2_ohr_ncp_20260103T1005176450_d_img_d18.img`
- **Acquisition Timestamp:** 2026-01-03T10:05:17.6450Z (Orbit 28372)
- **Ground Sampling Distance:** 0.25 m/pixel
- **Sun Azimuth:** 337.07°
- **Sun Elevation:** 5.52° (Solar Incidence: 84.48°)
- **Crop Bounds:** Lines 30000:30512, Samples 4000:4512 (512×512 uint8)
- **Footprint Region:** Lunar South Pole (Lat -85.28° to -84.52°, Lon 22.79° to 27.75°)

### 2. `ohrc_20260103T041022_crop.tif` (Reference Image)
- **Source Mission:** Chandrayaan-2 Orbiter
- **Instrument:** Orbiter High Resolution Camera (OHRC)
- **Product Logical Identifier:** `urn:isro:isda:ch2_cho.ohr:data_calibrated:ch2_ohr_ncp_20260103t0410224157_d_img_d18`
- **Original Source File:** `ch2_ohr_ncp_20260103T0410224157_d_img_d18.img`
- **Acquisition Timestamp:** 2026-01-03T04:10:22.4157Z (Orbit 28369, ~6 hours prior)
- **Ground Sampling Distance:** 0.25 m/pixel
- **Sun Azimuth:** 336.77°
- **Sun Elevation:** 5.54°
- **Crop Bounds:** Lines 30000:30512, Samples 4000:4512 (512×512 uint8)
- **Footprint Region:** Lunar South Pole (Lat -85.45° to -84.68°, Lon 23.57° to 28.31°)

---

## Why This Pair Was Selected

This multi-orbit pair represents an authentic planetary imaging correspondence problem:
- **Identical Terrain:** Both images cover cratered terrain near the Lunar South Pole.
- **Multi-Orbit View & Sun Angle Shift:** Acquired ~6 hours apart across different orbits (Orbit 28369 vs Orbit 28372), resulting in subtle orbital track drift, perspective shift, and illumination variations.
- **Extreme Shadow Conditions:** Low solar elevation (~5.5°) creates long shadows and extreme contrast, testing phase congruency vs. standard intensity gradient feature extraction.

---

## Conversion & Processing Method

1. **Raw PDS4 Reading:** Zero-memory mmap (`np.memmap`) based on XML label metadata (`UnsignedByte`, shape 101075×12000, offset 0).
2. **Crop Extraction:** Slice of lines 30000:30512 and samples 4000:4512 selected for high crater density and high feature variance.
3. **Format:** Single-channel uint8 GeoTIFF/TIFF written via `tifffile` (262 KB per file, total < 600 KB).

---

## Provenance & License

- **Data Origin:** Indian Space Research Organisation (ISRO) / ISSDC PRADAN Data Portal.
- **License:** Public scientific data released under ISRO Chandrayaan-2 open data policy.
