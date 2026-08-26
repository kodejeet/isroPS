# Research Notes & Future Roadmap

This document captures domain research, algorithm selection rationale, and pending architectural transitions for SIH Problem Statement 26166.

---

## 1. Why RIFT is the Core Next Step (P1 Plan)

### The Failure Mode of SIFT under Solar Illumination Changes
Standard feature extractors (SIFT, SURF, ORB) rely on local intensity gradients. When solar azimuth flips by 180°, shadow boundaries invert: dark crater interiors become bright illuminated rims, and vice versa. As a result, the intensity gradient vectors rotate by 180°, causing SIFT descriptor distance matching to fail completely.

### How RIFT Solves Illumination Invariance
**RIFT (Rotation and Illumination Invariant Feature Transform)** replaces intensity gradients with **Phase Congruency**:
1. *Phase Congruency Map:* Uses Log-Gabor filter banks to compute frequency-domain phase alignment. Phase congruency measures feature boundaries (edges/corners) independent of image illumination and contrast changes.
2. *MIM (Maximum Index Map):* Constructs an illumination-invariant map based on the log-Gabor filter response index yielding maximum energy at each pixel.
3. *Log-Gabor Grid Histogram:* Constructs feature descriptors over the MIM map.

Because MIM and Phase Congruency maps remain stable under extreme solar elevation/azimuth variations, RIFT achieves robust correspondence across multimodal (OHRC vs TMC-2 vs LROC) and solar-variant image pairs.

---

## 2. LightGlue & Learned Feature Adaptors (P2 Plan)

### Sparse Learned Matchers
Traditional nearest-neighbor descriptor matching (Lowe's ratio test) treats each keypoint in isolation. **LightGlue** (and SuperPoint) uses graph neural networks (GNNs) with self- and cross-attention mechanisms to reason over spatial relationships across the entire keypoint cloud.

### Integration Strategy
In P2, LightGlue will be wrapped as an implementation of `matching.base.Matcher`. By accepting `**kwargs` in signature, `LightGlueMatcher` can be swapped into `RegistrationPipeline` via configuration (`matching.method: "lightglue"`) without changing existing registration logic.

---

## 3. Dataset Risks & Mitigation

- **Risk:** Delay in official ISRO PRADAN dataset links.
- **Mitigation:** Pipeline architecture is completely format-agnostic. Interface contracts operate on standard `ImageData` arrays. Synthetic pair generator (`data/examples/`) ensures full unit and end-to-end testing proceeds independently of dataset release timing.
