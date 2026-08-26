# Data Directory Organization

This directory structure separates data products according to their processing level and source.

## Directory Overview

- `data/raw/`: Store raw, uncalibrated instrument products downloaded from archives (e.g. raw `.IMG` or `.xml` PDS4 products). *Gitignored.*
- `data/calibrated/`: Store radiometrically and sensor-calibrated products (e.g. Chandrayaan-2 OHRC/TMC-2/IIRS calibrated PDS4 data). Preferred for experiments over raw products. *Gitignored.*
- `data/reference/`: Store reference lunar datasets such as NASA LROC NAC optical images or JAXA SELENE (Kaguya) images. *Gitignored.*
- `data/processed/`: Intermediate output, tiles, sub-sampled patches, or orthorectified outputs created by the pipeline. *Gitignored.*
- `data/examples/`: Small, synthetic demonstration image pairs used for testing and baseline execution without external data dependencies. **Committed to repository.**

## Acquiring Real Lunar Imagery

### 1. ISRO PRADAN Archive (Chandrayaan-2)
- **Portal:** [PRADAN Archive](https://pradan.issdc.gov.in/ch2/) (Requires free registration)
- **Map Browser:** [Chandrayaan-2 Map Browser](https://chmapbrowse.issdc.gov.in/)
  - Select South Pole projection.
  - Enable footprint layers (e.g. `CH2_OHR_Calibrated_Product`).
  - Click overlapping footprints to inspect metadata and download calibrated PDS4 products.

### 2. NASA LROC Reference Imagery
- **LROC QuickMap:** [QuickMap Browser](https://quickmap.lroc.im-ldi.com)
- **LROC Downloads:** [LROC Downloads Portal](https://lroc.im-ldi.com/images/downloads/)
  - *Note:* LROC web services migrated from `asu.edu` to `im-ldi.com`.

### 3. JAXA SELENE (Kaguya)
- Reference imagery can also be downloaded from the DARTS JAXA portal for cross-mission multimodal matching.
