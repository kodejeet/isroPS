"""RIFT2 feature detection and description core algorithm.

Vendored and adapted from the upstream Python RIFT2 implementation by canyagmur:
https://github.com/canyagmur/RIFT2-multimodal-matching-rotation-python

Original paper:
    Li, Jiayuan, Qingwu Hu, and Mingyao Ai. "RIFT2: Speeding-up RIFT with
    A New Rotation-Invariance Technique" (2023). arXiv:2303.00319

MATLAB original: https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation

Modifications from upstream:
- Removed joblib parallelism dependency; uses serial processing.
- Removed yaml config-file loading; config is passed as a dict.
- Removed print statements; operates silently.
- Fixed indentation issues present in upstream source.
- Adapted imports to use vendored phase congruency module.
- Added type hints and docstrings for project consistency.
"""

import cv2
import numpy as np

from .phasecong import phasecong


class RIFT2Core:
    """Core RIFT2 feature detector and descriptor.

    This class implements the RIFT2 algorithm: phase-congruency-based keypoint
    detection, orientation assignment via gradient histograms, and MIM
    (Maximum Index Map) histogram descriptors.
    """

    def __init__(self, config: dict | None = None):
        self.default_config = {
            "nscale": 4,
            "norient": 6,
            "npt": 5000,
            "minWaveLength": 3,
            "mult": 1.6,
            "sigmaOnf": 0.75,
            "g": 3,
            "k": 1,
            "patch_size": 96,
            "no": 6,
            "nbin": 6,
            "is_ori": 1,
            "ori_peak_ratio": 0.8,
        }
        self.config = {**self.default_config, **(config or {})}

    def feature_detection(self, im: np.ndarray):
        """Detect keypoints using phase congruency and FAST detector.

        Args:
            im: 2D grayscale image array (H, W), uint8 or float.

        Returns:
            Tuple of (kpts, m, eo) where kpts is (2, N) array of (x, y)
            keypoint coordinates, m is the phase congruency map, and eo
            is the filter response tensor.
        """
        config = self.config
        M, _, _, _, _, eo, _ = phasecong(
            im,
            nscale=config["nscale"],
            norient=config["norient"],
            minWaveLength=config["minWaveLength"],
            mult=config["mult"],
            sigmaOnf=config["sigmaOnf"],
            g=config["g"],
            k=config["k"],
        )
        a = np.max(M)
        b = np.min(M)
        if a - b > 0:
            m_norm = (M - b) / (a - b)
        else:
            m_norm = np.zeros_like(M)

        m_image = (m_norm * 255).astype(np.uint8)
        eo = np.transpose(eo, (1, 0, 2, 3))

        fast = cv2.FastFeatureDetector_create(threshold=1, nonmaxSuppression=True)
        keypoints = fast.detect(m_image, None)

        if len(keypoints) == 0:
            return np.zeros((2, 0), dtype=np.float64), m_norm, eo

        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
        keypoints = keypoints[: config["npt"]]
        kpts = np.array([kp.pt for kp in keypoints]).T  # (2, N): row0=x, row1=y

        return kpts, m_norm, eo

    def compute_orientation(self, key: np.ndarray, im: np.ndarray) -> np.ndarray:
        """Assign dominant orientation(s) to each keypoint.

        Args:
            key: (2, N) keypoint array from feature_detection.
            im: Phase congruency map (normalized float).

        Returns:
            (3, M) array where rows are [x, y, angle_degrees].
            M >= N because keypoints may have multiple dominant orientations.
        """
        config = self.config
        if key.shape[1] == 0:
            return np.zeros((3, 0), dtype=np.float64)

        if config["is_ori"] == 1:
            n = 24
            ORI_PEAK_RATIO = config["ori_peak_ratio"]
            h_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)

            gradient_x = cv2.filter2D(im, -1, h_kernel, borderType=cv2.BORDER_REPLICATE)
            gradient_y = cv2.filter2D(
                im, -1, h_kernel.T, borderType=cv2.BORDER_REPLICATE
            )
            gradientImg = np.sqrt(gradient_x**2 + gradient_y**2)
            temp_angle = np.degrees(np.arctan2(gradient_y, gradient_x))
            temp_angle[temp_angle < 0] += 360
            gradientAng = temp_angle
        else:
            n = 24
            ORI_PEAK_RATIO = config["ori_peak_ratio"]
            gradientImg = None
            gradientAng = None

        feat_index = 0
        kpts = np.zeros((3, key.shape[1] * 6))
        for k_idx in range(key.shape[1]):
            x = int(round(key[0, k_idx]))
            y = int(round(key[1, k_idx]))
            r = int(round(config["patch_size"]))

            x1 = max(1, x - r // 2)
            y1 = max(1, y - r // 2)
            x2 = min(x + r // 2, im.shape[1] - 1)
            y2 = min(y + r // 2, im.shape[0] - 1)

            if y2 - y1 != r or x2 - x1 != r:
                continue

            if config["is_ori"] == 1:
                angle = self._orientation(
                    x, y, gradientImg, gradientAng, r, n, ORI_PEAK_RATIO
                )
                for i in range(len(angle)):
                    if feat_index < kpts.shape[1]:
                        kpts[:, feat_index] = [x, y, angle[i]]
                        feat_index += 1
            else:
                if feat_index < kpts.shape[1]:
                    kpts[:, feat_index] = [x, y, 0]
                    feat_index += 1

        kpts = kpts[:, :feat_index]
        # Remove zero-coordinate keypoints
        valid = kpts[0, :] != 0
        kpts = kpts[:, valid]

        return kpts

    def feature_description(
        self,
        img_hw: tuple[int, int],
        eo: np.ndarray,
        kpts: np.ndarray,
    ) -> np.ndarray:
        """Compute MIM histogram descriptors for each oriented keypoint.

        Args:
            img_hw: (height, width) of the original image.
            eo: Filter response tensor from feature_detection.
            kpts: (3, M) keypoints with orientation from compute_orientation.

        Returns:
            (D, M) descriptor array where D = no * no * nbin.
        """
        config = self.config
        if kpts.shape[1] == 0:
            desc_dim = config["no"] * config["no"] * config["nbin"]
            return np.zeros((desc_dim, 0), dtype=np.float64)

        n = kpts.shape[1]
        yim, xim = img_hw
        CS = np.zeros((yim, xim, config["no"]))

        for j in range(config["no"]):
            for i in range(4):
                CS[:, :, j] += np.abs(eo[i][j])

        MIM = np.argmax(CS, axis=2)

        desc_dim = config["no"] * config["no"] * config["nbin"]
        des = np.zeros((desc_dim, n))

        for k_idx in range(n):
            x = kpts[0, k_idx]
            y = kpts[1, k_idx]
            r = config["patch_size"]
            ang = kpts[2, k_idx]

            patch = self._extract_patches(MIM, x, y, round(r / 2), ang)
            patch = cv2.resize(
                patch, (r + 1, r + 1), interpolation=cv2.INTER_LINEAR
            )
            h, _ = np.histogram(patch, bins=np.arange(1, config["no"] + 2))
            idx = np.argmax(h)
            patch_rot = patch - idx + 1
            patch_rot[patch_rot < 0] += config["no"]

            ys, xs = patch_rot.shape[:2]
            if patch_rot.ndim == 3:
                patch_rot = patch_rot[:, :, 0]

            histo = np.zeros((config["no"], config["no"], config["nbin"]))

            for j in range(config["no"]):
                for i in range(config["no"]):
                    clip = patch_rot[
                        round(j * ys / config["no"]) : round(
                            (j + 1) * ys / config["no"]
                        ),
                        round(i * xs / config["no"]) : round(
                            (i + 1) * xs / config["no"]
                        ),
                    ]
                    histo[j, i, :] = np.histogram(
                        clip, bins=np.arange(1, config["nbin"] + 2)
                    )[0]

            histo_flat = histo.flatten()
            norm = np.linalg.norm(histo_flat)
            if norm != 0:
                histo_flat = histo_flat / norm

            des[:, k_idx] = histo_flat

        return des

    def detect_and_describe(self, img_gray: np.ndarray):
        """Run full RIFT2 pipeline on a single grayscale image.

        Args:
            img_gray: 2D grayscale image (H, W), uint8.

        Returns:
            Tuple of (keypoints_xy, descriptors) where:
            - keypoints_xy: (M, 2) array of (x, y) coordinates.
            - descriptors: (M, D) array of float32 descriptors.
        """
        key, m, eo = self.feature_detection(img_gray)

        if key.shape[1] == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(
                (0, self.config["no"] * self.config["no"] * self.config["nbin"]),
                dtype=np.float32,
            )

        kpts = self.compute_orientation(key, m)

        if kpts.shape[1] == 0:
            desc_dim = self.config["no"] * self.config["no"] * self.config["nbin"]
            return np.zeros((0, 2), dtype=np.float32), np.zeros(
                (0, desc_dim), dtype=np.float32
            )

        des = self.feature_description(img_gray.shape, eo, kpts)

        # Transpose to (M, 2) and (M, D)
        keypoints_xy = kpts[:2, :].T.astype(np.float32)  # (M, 2) in (x, y)
        descriptors = des.T.astype(np.float32)  # (M, D)

        return keypoints_xy, descriptors

    # ---- Private helper methods ----

    def _extract_patches(self, img, x, y, s, t):
        """Extract a rotated patch from the MIM map using bilinear interpolation."""
        img = img.astype(np.float32)
        h, w = img.shape[:2]
        m = img.shape[2] if img.ndim == 3 else 1

        x = np.clip(np.round(x).astype(int), 0, w - 1)
        y = np.clip(np.round(y).astype(int), 0, h - 1)

        s = int(round(s))
        t = np.deg2rad(t)

        patchsize = s * 2 + 1
        xg, yg = np.meshgrid(np.arange(-s, s + 1), np.arange(-s, s + 1))
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        xygrot = R @ np.vstack([xg.ravel(), yg.ravel()])
        xygrot[0, :] += x
        xygrot[1, :] += y

        xr = xygrot[0, :]
        yr = xygrot[1, :]
        xf = np.floor(xr).astype(int)
        yf = np.floor(yr).astype(int)
        xp = xr - xf
        yp = yr - yf

        patch = np.zeros((patchsize, patchsize, m))

        valid_mask = (xf >= 0) & (xf <= w - 2) & (yf >= 0) & (yf <= h - 2)
        xf_v = xf[valid_mask]
        yf_v = yf[valid_mask]
        xp_v = xp[valid_mask]
        yp_v = yp[valid_mask]

        if len(xf_v) == 0:
            return patch if m > 1 else patch[:, :, 0]

        ind1 = np.ravel_multi_index((yf_v, xf_v), (h, w))
        ind2 = np.ravel_multi_index((yf_v, xf_v + 1), (h, w))
        ind3 = np.ravel_multi_index((yf_v + 1, xf_v), (h, w))
        ind4 = np.ravel_multi_index((yf_v + 1, xf_v + 1), (h, w))

        for ch in range(m):
            imgch = img[:, :, ch] if m > 1 else img
            ivec = (1 - yp_v) * (
                xp_v * imgch.ravel()[ind2] + (1 - xp_v) * imgch.ravel()[ind1]
            ) + yp_v * (
                xp_v * imgch.ravel()[ind4] + (1 - xp_v) * imgch.ravel()[ind3]
            )
            temp = np.zeros(patchsize * patchsize)
            temp[valid_mask] = ivec
            patch[:, :, ch] = temp.reshape(patchsize, patchsize)

        if m == 1:
            return patch[:, :, 0]
        return patch

    def _orientation(self, x, y, gradientImg, gradientAng, patch_size, n, ORI_PEAK_RATIO):
        """Compute dominant orientation(s) for a single keypoint."""
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(patch_size + 1), int(patch_size + 1))
        )
        Sa = se.astype(np.uint8)
        hist, max_value = self._calculate_orientation_hist(
            x, y, patch_size / 2, gradientImg, gradientAng, n, Sa
        )

        mag_thr = max_value * ORI_PEAK_RATIO
        ANG = []
        for k in range(n):
            k1 = n - 1 if k == 0 else k - 1
            k2 = 0 if k == n - 1 else k + 1
            if hist[k] > hist[k1] and hist[k] > hist[k2] and hist[k] > mag_thr:
                denom = hist[k1] + hist[k2] - 2 * hist[k]
                if abs(denom) > 1e-10:
                    bin_val = k - 1 + 0.5 * (hist[k1] - hist[k2]) / denom
                else:
                    bin_val = float(k)
                if bin_val < 0:
                    bin_val = n + bin_val
                elif bin_val >= n:
                    bin_val = bin_val - n
                angle = (360 / n) * bin_val
                ANG.append(angle)
        return ANG

    def _calculate_orientation_hist(self, x, y, radius, gradient, angle, n, Sa):
        """Calculate orientation histogram for a keypoint."""
        sigma = radius / 3

        radius_x_left = int(x - radius)
        radius_x_right = int(x + radius)
        radius_y_up = int(y - radius)
        radius_y_down = int(y + radius)

        radius_x_left = max(0, radius_x_left)
        radius_x_right = min(gradient.shape[1], radius_x_right + 1)
        radius_y_up = max(0, radius_y_up)
        radius_y_down = min(gradient.shape[0], radius_y_down + 1)

        sub_gradient = gradient[radius_y_up:radius_y_down, radius_x_left:radius_x_right]
        sub_angle = angle[radius_y_up:radius_y_down, radius_x_left:radius_x_right]

        X = np.arange(-(x - radius_x_left), (radius_x_right - x))
        Y = np.arange(-(y - radius_y_up), (radius_y_down - y))
        XX, YY = np.meshgrid(X, Y)

        gaussian_weight = np.exp(-(XX**2 + YY**2) / (2 * sigma**2))
        W1 = sub_gradient * gaussian_weight
        W = np.double(Sa[: W1.shape[0], : W1.shape[1]]) * np.double(W1)

        bin_arr = np.round(sub_angle * n / 360).astype(int)
        bin_arr[bin_arr >= n] -= n
        bin_arr[bin_arr < 0] += n

        temp_hist = np.zeros(n)
        for i in range(n):
            wM = W[bin_arr == i]
            if wM.size > 0:
                temp_hist[i] = np.sum(wM)

        hist = np.zeros(n)
        hist[0] = (
            (temp_hist[n - 2] + temp_hist[2]) / 16
            + 4 * (temp_hist[n - 1] + temp_hist[1]) / 16
            + temp_hist[0] * 6 / 16
        )
        hist[1] = (
            (temp_hist[n - 1] + temp_hist[3]) / 16
            + 4 * (temp_hist[0] + temp_hist[2]) / 16
            + temp_hist[1] * 6 / 16
        )
        hist[2 : n - 2] = (
            (temp_hist[0 : n - 4] + temp_hist[4:n]) / 16
            + 4 * (temp_hist[1 : n - 3] + temp_hist[3 : n - 1]) / 16
            + temp_hist[2 : n - 2] * 6 / 16
        )
        hist[n - 2] = (
            (temp_hist[n - 4] + temp_hist[0]) / 16
            + 4 * (temp_hist[n - 3] + temp_hist[n - 1]) / 16
            + temp_hist[n - 2] * 6 / 16
        )
        hist[n - 1] = (
            (temp_hist[n - 3] + temp_hist[1]) / 16
            + 4 * (temp_hist[n - 2] + temp_hist[0]) / 16
            + temp_hist[n - 1] * 6 / 16
        )

        max_value = np.max(hist)
        return hist, max_value
