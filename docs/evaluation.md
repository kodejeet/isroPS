# Evaluation Metrics Guide

Quantitative evaluation of lunar image registration requires looking beyond a single numerical score. This document defines the metrics calculated by `evaluation.metrics` and explains how to interpret them.

---

## Calculated Metrics

### 1. Total Matches (`total_matches`)
- **Definition:** Total count of raw feature correspondences established between descriptor sets prior to geometric RANSAC filtering.
- **Interpretation:** High total matches indicate both images contain rich texture, but without RANSAC filtering, many may be false positives (*spurious matches caused by repetitive terrain or shadows*).

### 2. Inlier Matches (`inlier_matches`)
- **Definition:** The number of feature correspondences consistent with the estimated geometric homography/affine transformation matrix under a RANSAC error threshold (e.g. reprojection error < 3.0 pixels).
- **Interpretation:** Inliers represent true physical terrain correspondences. At least 15–20 high-quality inliers are required to reliably constrain a 3x3 homography matrix.

### 3. Inlier Ratio (`inlier_ratio`)
- **Definition:** `inlier_matches / total_matches` (expressed as a float between 0.0 and 1.0).
- **Interpretation:** Measures matcher precision. An inlier ratio > 0.40 indicates high feature match quality. Ratios below 0.15 suggest high noise or severe illumination breakdown.

### 4. RMSE (`rmse_pixels`)
- **Definition:** Root Mean Square Error of inlier point reprojections:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| H(x_i^s) - x_i^r \|^2}$$
- **Interpretation:** Measures reprojection fit in pixels. Sub-pixel alignment achieves $\text{RMSE} < 1.0 \text{ pixel}$.
- **CRITICAL CAVEAT:** **RMSE alone is misleading.** A transformation fit to only 3 or 4 lucky clustered matches can yield an RMSE near 0.0, despite being completely wrong across the rest of the image. Always evaluate RMSE in conjunction with **Inlier Ratio** and **Grid Coverage**.

### 5. Grid-Cell Spatial Coverage (`coverage`)
- **Definition:** Percentage of cells in an $N \times M$ grid across the reference image that contain at least one inlier match.
- **Interpretation:** Evaluates the spatial uniformity of matches across the image layout. A score of 80–100% indicates matches span the whole scene, preventing localized warping distortion.

### 6. Median Error (`median_error_pixels`)
- **Definition:** Median reprojection error among inlier matches. Robust against residual outlier noise.
