# Research Notes & Future Roadmap

This document captures domain research, algorithm selection rationale, and pending architectural transitions for SIH Problem Statement 26166.

---

## 1. RIFT2 Integration (Implemented)

### The Failure Mode of SIFT under Solar Illumination Changes
Standard feature extractors (SIFT, SURF, ORB) rely on local intensity gradients. When solar azimuth flips by 180°, shadow boundaries invert: dark crater interiors become bright illuminated rims, and vice versa. As a result, the intensity gradient vectors rotate by 180°, causing SIFT descriptor distance matching to fail completely.

### How RIFT/RIFT2 Solves Illumination Invariance
**RIFT (Rotation and Illumination Invariant Feature Transform)** replaces intensity gradients with **Phase Congruency**:
1. *Phase Congruency Map:* Uses Log-Gabor filter banks to compute frequency-domain phase alignment. Phase congruency measures feature boundaries (edges/corners) independent of image illumination and contrast changes.
2. *MIM (Maximum Index Map):* Constructs an illumination-invariant map based on the log-Gabor filter response index yielding maximum energy at each pixel.
3. *MIM Grid Histogram:* Constructs feature descriptors over the MIM map using spatial grid histograms.

**RIFT2** is the rotation-invariant successor in the same algorithmic lineage. It extends the original RIFT with an improved rotation-invariance technique, speeding up the feature extraction while maintaining multimodal robustness.

### Original Paper Citation (RIFT)
Li, Jiayuan, Qingwu Hu, and Mingyao Ai. "RIFT: Multi-modal image matching based on radiation-variation insensitive feature transform." *IEEE Transactions on Image Processing* 29 (2020): 3296–3310.

### RIFT2 Paper Citation
Li, Jiayuan, Qingwu Hu, and Mingyao Ai. "RIFT2: Speeding-up RIFT with A New Rotation-Invariance Technique" (2023). arXiv:2303.00319.

### Implementation Details
- The integrated implementation is **RIFT2** (not the original RIFT).
- RIFT2 is the rotation-invariant successor in the same algorithmic lineage as RIFT.
- The implementation uses the **phase-congruency / MIM lineage** from the RIFT family.
- The project **adapts an existing upstream Python implementation** rather than reproducing the paper from scratch.
- Upstream repository: https://github.com/canyagmur/RIFT2-multimodal-matching-rotation-python
- The upstream implementation itself adapts phase congruency code from PhasePack (https://github.com/alimuldal/phasepack).
- MATLAB original by the paper authors: https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation

### Current Status
- **RIFT2 adapter integrated, synthetic-tested, and real-data validated.**
- Feature extractor (`RIFTFeatureExtractor`) and matcher (`RIFTMatcher`) implemented as thin adapters over the vendored RIFT2 core.
- Selectable via `feature_extraction.method: rift` and `matching.method: rift` in configuration.
- Produces valid `FeatureSet` and `MatchSet` compatible with the entire pipeline (spatial selection, RANSAC, subpixel refinement, evaluation, visualization).
- CPU-only; no GPU dependencies.
- Full test suite passes (architecture, extraction, matching, end-to-end pipeline, and real OHRC crop integration).

### Real-Data Benchmark Results (Chandrayaan-2 OHRC Multi-Orbit Pair)

Validated on authentic 512×512 Lunar South Pole crops from Chandrayaan-2 OHRC Orbit 28372 (2026-01-03T10:05:17Z) and Orbit 28369 (2026-01-03T04:10:22Z, ~6 hours prior):

| Metric | SIFT Baseline | RIFT2 |
|---|---|---|
| Keypoints (src / ref) | 378 / 652 | 3965 / 4166 |
| Candidate Matches | 5 | 5 |
| Spatially Selected Matches (8×8 grid, top-4) | 5 | 5 |
| RANSAC Inliers | 5 | 4 |
| Inlier Ratio | 1.0000 | 0.8000 |
| Pre-refinement RMSE (px) | 0.000025 | 0.000039 |
| Post-refinement RMSE (px) | 0.000000 | 0.000044 |
| Median Error (px) | 0.000031 | 0.000024 |
| Grid Coverage (%) | 25.00% | 18.75% |
| Processing Time (s) | 0.16s | 11.97s |

*On larger 1024×1024 crops under track shift, RIFT2 established 38 candidate matches and 5 RANSAC inliers vs. SIFT's 18 candidate matches and 4 RANSAC inliers, demonstrating higher candidate density in low-elevation shadow regions.*

### Limitations
- RIFT2 feature extraction is computationally more expensive than SIFT due to phase congruency computation (FFT-based Log-Gabor filter banks taking ~12s per 512×512 pair on CPU).
- The upstream repository (`canyagmur/RIFT2-multimodal-matching-rotation-python`) does not include an explicit license file.
- Single real pair validation (OHRC ↔ OHRC multi-orbit); broader multi-pair benchmarking (e.g., OHRC ↔ LRO NAC) scheduled for Sunday fusion track.

### Next Steps
- Implement Sunday match fusion (`matching/fusion.py`) combining SIFT and RIFT2 match sets into a single unified correspondence set.
- Evaluate fusion performance on real lunar imagery.

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
