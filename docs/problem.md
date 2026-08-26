# Problem Overview: SIH 2026 Problem Statement 26166

## Core Information
- **Problem Statement ID:** SIH26166
- **Track:** Software
- **Theme:** Space Technology
- **Sponsoring Organization:** Indian Space Research Organisation (ISRO), Department of Space
- **Title:** Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS)

---

## Plain-English Explanation

### What is Image Correspondence?
Image correspondence (or image registration - *aligning two pictures so matching features overlap exactly*) is the process of taking two pictures of the exact same location on the Moon and finding matching pixels between them. Once matching pixels are identified, one image can be geometrically warped to align perfectly on top of the other.

### Why is Lunar Image Alignment Hard?
Matching pictures of the Moon taken by spacecraft orbiters (*satellite cameras orbiting the Moon*) is significantly harder than matching standard Earth photos because of three main physical challenges:

1. **Illumination Variation (Sun Angle Changes)**
   - *Gloss:* Sun Azimuth (*compass direction of the sun*) and Sun Elevation (*angle of the sun above the horizon*) dictate how shadows fall.
   - Because the Moon has no atmosphere to scatter light, lunar shadows are stark black and sharp. When the Sun angle changes between orbits, shadows move, invert (a crater shadow flips sides), or disappear entirely. Standard feature matchers (like SIFT) that rely on bright-to-dark gradient patterns fail because gradient directions flip when the shadow flips.

2. **Viewpoint Variation (Camera Angle & Orientation)**
   - *Gloss:* Viewpoint variation refers to changes in spacecraft position and camera angle between orbital passes.
   - Images captured from different orbital passes exhibit horizontal/vertical shifts, rotations, scaling differences, and perspective distortion (*trapezoidal stretching*).

3. **Scale Variation (Resolution Differences across Cameras)**
   - *Gloss:* Ground Sampling Distance (GSD - *how many meters on the lunar surface are represented by a single image pixel*) varies dramatically across sensors.
   - An Orbiter High Resolution Camera (OHRC) image might have a GSD of ~25 cm/pixel, whereas a Terrain Mapping Camera (TMC-2) image might have ~5 m/pixel. A 15-meter crater is 60 pixels wide in OHRC, but only 3 pixels wide in TMC-2.

---

## Technical Deliverables Expected by ISRO

1. **Generic Correspondence Engine:** A software tool capable of finding corresponding feature points across multimodal lunar images with **sub-pixel accuracy** (*aligning pixel matches to fractions of a pixel*).
2. **Uniform Match Distribution:** Feature matches must be evenly spread across the full image area rather than tightly clustered in a single high-contrast corner.
3. **Registration & Metrics Output:**
   - Geometrically registered output image.
   - Pairwise point correspondences (match coordinates).
   - Evaluation report containing Root Mean Square Error (RMSE - *average pixel misalignment distance*), total matches, inlier matches (*valid correct matches*), inlier ratio, and spatial coverage percentage.
