# Lunar Imaging Basics: Beginner Domain Reference

This document provides plain-English definitions and explanations for planetary science and remote-sensing terminology used throughout the codebase.

---

## Instruments & Data Sources

### OHRC (Orbiter High Resolution Camera)
- *Gloss:* Chandrayaan-2's ultra-high-resolution optical camera (~25 cm/pixel resolution).
- OHRC captures extremely sharp, narrow-footprint (*small geographical coverage area*) images of the lunar surface. It is primarily used for identifying landing hazards and inspecting fine crater morphology. Due to its very fine spatial resolution, OHRC images show intricate shadow details that change radically with slight movements of the Sun.

### TMC-2 (Terrain Mapping Camera 2)
- *Gloss:* Chandrayaan-2's stereo panchromatic camera (~5 m/pixel resolution).
- TMC-2 captures wide-swath (*broad geographical coverage strip*) 3D stereo imagery in the visible spectrum. By capturing surface patches from fore, aft, and nadir (*straight down*) angles, TMC-2 data enables the generation of digital surface elevation models of the Moon.

### IIRS (Imaging Infra-Red Spectrometer)
- *Gloss:* Chandrayaan-2's hyperspectral sensor (~80 m/pixel resolution, ~256 spectral channels).
- Unlike standard 3-channel RGB cameras, IIRS measures reflected solar radiation across hundreds of narrow infrared wavelength bands (~0.8 to 5.0 µm). It is used to measure lunar mineral composition and water-ice signatures. Before feature extraction, multi-band hyperspectral cubes (*3D arrays of spatial + spectral data*) must be collapsed or filtered to single-band representative images.

### LRO NAC (Lunar Reconnaissance Orbiter - Narrow Angle Camera)
- *Gloss:* NASA's high-resolution lunar reference camera (~0.5 to 2.0 m/pixel resolution).
- Carried aboard NASA's Lunar Reconnaissance Orbiter (LRO), LRO NAC has mapped vast regions of the Moon. It serves as one of the primary high-quality reference baselines against which Chandrayaan-2 imagery is co-registered.

### SELENE (Kaguya Terrain Camera)
- *Gloss:* JAXA's (Japanese Space Agency) lunar orbiter camera (~10 m/pixel resolution).
- SELENE (Kaguya) orbited the Moon from 2007 to 2009, collecting global optical imagery and altimetry. SELENE imagery serves as an additional cross-mission reference dataset for validating multi-modal registration algorithms.

---

## Geometric & Illumination Physics

### Sun Incidence / Elevation & Azimuth Angle
- *Gloss:* The position of the Sun relative to the lunar terrain surface during camera exposure.
- Sun Elevation (*angle of the sun above the local horizon*) and Sun Azimuth (*compass heading toward the sun*) determine shadow lengths and orientations. On the airless Moon, shadows are dark and razor-sharp; low sun elevation angles create extremely long shadows that obscure or invert crater appearance.

### Viewpoint Variation
- *Gloss:* Geometric distortion caused by differences in orbiter position and camera pointing angles.
- When an orbiter takes photos of the same location from different altitude points or tilt angles, the resulting images exhibit translation (*shifting*), rotation, scale changes, and perspective shear (*trapezoidal stretching*).

### Scale & Resolution (GSD)
- *Gloss:* Ground Sampling Distance (GSD) - the physical size on the lunar surface represented by one image pixel.
- Because different instruments orbit at varying altitudes and possess different optics, GSD varies dramatically (e.g. 0.25 m/px for OHRC vs 5 m/px for TMC-2). Scale-invariant matching requires algorithms capable of identifying the same crater feature whether it spans 300 pixels or 15 pixels.

---

## Data Structures, Correction & Orbit Geometry

### DEM (Digital Elevation Model)
- *Gloss:* A 2D grid where each pixel value represents the physical terrain height (elevation) above a reference lunar sphere.
- DEMs provide the 3D surface shape needed to unwarp terrain distortion caused by hills, crater rims, and valleys.

### Orthorectification
- *Gloss:* The process of mathematically reprojecting a raw perspective satellite photo onto a flat map grid using a terrain DEM.
- Orthorectification removes camera tilt distortion and topographic relief displacement (*mountains leaning away from camera sensor*), resulting in a uniform planimetric image (*map-accurate picture*).

### SPICE Kernels
- *Gloss:* NASA's standardized navigation data format tracking space mission trajectories, orientation, and planet orientation over time.
- Managed by NASA's NAIF (Navigation and Ancillary Information Facility), SPICE kernels provide precise spacecraft position, velocity, instrument orientation, and target body shape matrices required for accurate geographic coordinate projection.

### PDS4 (Planetary Data System v4)
- *Gloss:* NASA/ISRO's standardized archive file format combining raw/calibrated binary imagery with XML metadata headers.
- PDS4 products consist of a explicit XML label file (`.xml`) describing image dimensions, bit depth, coordinate systems, and instrument settings, paired with a binary data file (`.IMG` or `.DAT`).
