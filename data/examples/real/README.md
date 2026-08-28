# Real Lunar Example Data Crops

This directory contains small, representative 512x512 GeoTIFF crops extracted from real Chandrayaan-2 OHRC and TMC-2 PDS4 data products for offline testing and multi-pair real-data validation.

---

## Real Image Crops Included (512×512 uint8 GeoTIFF)

### 1. `ohrc_20260103T100517_crop.tif` (OHRC Orbit 28372)
- **Mission / Instrument:** Chandrayaan-2 Orbiter / OHRC (0.25 m/pixel)
- **Logical Identifier:** `urn:isro:isda:ch2_cho.ohr:data_calibrated:ch2_ohr_ncp_20260103t1005176450_d_img_d18`
- **Acquisition Timestamp:** 2026-01-03T10:05:17.6450Z
- **Sun Azimuth / Elevation:** 337.07° / 5.52° (Solar Incidence: 84.48°)
- **Crop Bounds:** Lines 30,000:30,512, Samples 4,000:4,512
- **Footprint:** Lunar South Pole (Lat -85.28° to -84.52°, Lon 22.79° to 27.75°)

### 2. `ohrc_20260103T041022_crop.tif` (OHRC Orbit 28369)
- **Mission / Instrument:** Chandrayaan-2 Orbiter / OHRC (0.25 m/pixel)
- **Logical Identifier:** `urn:isro:isda:ch2_cho.ohr:data_calibrated:ch2_ohr_ncp_20260103t0410224157_d_img_d18`
- **Acquisition Timestamp:** 2026-01-03T04:10:22.4157Z (~6 hours prior)
- **Sun Azimuth / Elevation:** 336.77° / 5.54°
- **Crop Bounds:** Lines 30,000:30,512, Samples 4,000:4,512
- **Footprint:** Lunar South Pole (Lat -85.45° to -84.68°, Lon 23.57° to 28.31°)

### 3. `ohrc_20260103T120356_crop.tif` (OHRC Orbit 28373)
- **Mission / Instrument:** Chandrayaan-2 Orbiter / OHRC (0.25 m/pixel)
- **Logical Identifier:** `urn:isro:isda:ch2_cho.ohr:data_calibrated:ch2_ohr_ncp_20260103t1203563771_d_img_d18`
- **Acquisition Timestamp:** 2026-01-03T12:03:56.3771Z (~2 hours later)
- **Sun Azimuth / Elevation:** 341.86° / 6.58°
- **Crop Bounds:** Lines 30,000:30,512, Samples 4,000:4,512
- **Footprint:** Lunar South Pole (Lat -85.28° to -84.53°, Lon 22.79° to 27.94°)

### 4. `tmc2_20260813_ncn_crop.tif` (TMC-2 Orbit 31073 Nadir)
- **Mission / Instrument:** Chandrayaan-2 Orbiter / TMC-2 Nadir View (5.48 m/pixel)
- **Logical Identifier:** `urn:isro:isda:ch2_cho.tmc:data_calibrated:ch2_tmc_ncn_20260813t0627378557_d_img_d18`
- **Acquisition Timestamp:** 2026-08-13T06:27:37.8557Z
- **Sun Azimuth / Elevation:** 69.56° / 50.94°
- **Crop Bounds:** Lines 50,000:50,512, Samples 1,500:2,012
- **Footprint:** Equatorial Region (Lat -27.76° to -3.62°, Lon 140.91° to 142.70°)

### 5. `tmc2_20260813_nra_crop.tif` (TMC-2 Orbit 31073 Aft)
- **Mission / Instrument:** Chandrayaan-2 Orbiter / TMC-2 Aft View (5.48 m/pixel, +25° Look Angle)
- **Logical Identifier:** `urn:isro:isda:ch2_cho.tmc:data_raw:ch2_tmc_nra_20260813t0627378526_d_img_d18`
- **Acquisition Timestamp:** 2026-08-13T06:27:37.8526Z
- **Sun Azimuth / Elevation:** 69.56° / 50.94°
- **Crop Bounds:** Lines 50,000:50,512, Samples 1,500:2,012
- **Footprint:** Equatorial Region (Lat -29.46° to -5.36°, Lon 140.78° to 142.69°)

---

## Evaluation Benchmark Pairs

1. **`pair1_ohrc_track_shift`**: `ohrc_20260103T100517_crop.tif` vs `ohrc_20260103T041022_crop.tif` (OHRC South Pole multi-orbit, ~6h gap, track shift)
2. **`pair2_ohrc_high_overlap`**: `ohrc_20260103T100517_crop.tif` vs `ohrc_20260103T120356_crop.tif` (OHRC South Pole multi-orbit, ~2h gap, >95% geographic overlap)
3. **`pair3_tmc2_stereo_nadir_aft`**: `tmc2_20260813_ncn_crop.tif` vs `tmc2_20260813_nra_crop.tif` (TMC-2 Equatorial stereo multi-view, 25° perspective look angle difference)

---

## Provenance & License

- **Data Origin:** Indian Space Research Organisation (ISRO) / ISSDC PRADAN Data Portal.
- **License:** Public scientific data released under ISRO Chandrayaan-2 open data policy.
