"""Phase congruency computation for RIFT2 feature detection.

Vendored from PhasePack by Ali Shervin Muldal (https://github.com/alimuldal/phasepack),
as adapted by canyagmur in the RIFT2 Python implementation
(https://github.com/canyagmur/RIFT2-multimodal-matching-rotation-python).

References:
    Peter Kovesi, "Image Features From Phase Congruency". Videre: A Journal of
    Computer Vision Research. MIT Press. Volume 1, Number 3, Summer 1999.

    Peter Kovesi, "Phase Congruency Detects Corners and Edges". Proceedings
    DICTA 2003, Sydney Dec 10-12.

Modifications from upstream:
- Uses scipy.fft instead of pyfftw/scipy.fftpack.
- Minor style adjustments for project linting compatibility.
"""

import numpy as np
from scipy.fft import ifftshift

from .tools import fft2, ifft2
from .tools import lowpassfilter as _lowpassfilter
from .tools import rayleighmode as _rayleighmode


def phasecong(
    img,
    nscale=5,
    norient=6,
    minWaveLength=3,
    mult=2.1,
    sigmaOnf=0.55,
    k=2.0,
    cutOff=0.5,
    g=10.0,
    noiseMethod=-1,
):
    """Compute phase congruency on an image.

    This is a contrast-invariant edge and corner detector.

    Args:
        img: Input 2D grayscale image.
        nscale: Number of wavelet scales (3–6 recommended).
        norient: Number of filter orientations.
        minWaveLength: Wavelength of smallest scale filter.
        mult: Scaling factor between successive filters.
        sigmaOnf: Ratio of std dev of Gaussian describing log Gabor filter.
        k: Noise compensation factor.
        cutOff: Fractional measure of frequency spread below which PC values
            get penalized.
        g: Controls sharpness of sigmoid weighting function.
        noiseMethod: Method for noise statistics (-1: median, -2: mode,
            >=0: fixed threshold).

    Returns:
        Tuple of (M, m, ori, ft, PC, EO, T) where:
        - M: Maximum moment of phase congruency covariance (edge strength).
        - m: Minimum moment (corner strength).
        - ori: Orientation image in integer degrees (0–180).
        - ft: Local weighted mean phase angle.
        - PC: List of phase congruency images per orientation.
        - EO: List of sublists of complex convolution results [orient][scale].
        - T: Calculated noise threshold.
    """
    if img.dtype not in [np.float32, np.float64]:
        img = np.float64(img)
        imgdtype = "float64"
    else:
        imgdtype = str(img.dtype)

    if img.ndim == 3:
        img = img.mean(2)

    rows, cols = img.shape

    epsilon = 1e-4
    IM = fft2(img)

    EO = []
    PC = []

    zeromat = np.zeros((rows, cols), dtype=imgdtype)

    covx2 = zeromat.copy()
    covy2 = zeromat.copy()
    covxy = zeromat.copy()

    EnergyV = np.zeros((rows, cols, 3), dtype=imgdtype)
    pcSum = zeromat.copy()

    if cols % 2:
        xvals = np.arange(-(cols - 1) / 2.0, ((cols - 1) / 2.0) + 1) / float(
            cols - 1
        )
    else:
        xvals = np.arange(-cols / 2.0, cols / 2.0) / float(cols)

    if rows % 2:
        yvals = np.arange(-(rows - 1) / 2.0, ((rows - 1) / 2.0) + 1) / float(
            rows - 1
        )
    else:
        yvals = np.arange(-rows / 2.0, rows / 2.0) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)

    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(-y, x)

    radius = ifftshift(radius)
    theta = ifftshift(theta)

    radius[0, 0] = 1.0

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    del x, y, theta

    lp = _lowpassfilter((rows, cols), 0.45, 15)

    logGaborDenom = 2.0 * np.log(sigmaOnf) ** 2.0
    logGabor = []

    for ss in range(nscale):
        wavelength = minWaveLength * mult**ss
        fo = 1.0 / wavelength
        logRadOverFo = np.log(radius / fo)
        tmp = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        tmp = tmp * lp
        tmp[0, 0] = 0.0
        logGabor.append(tmp)

    for oo in range(norient):
        angl = oo * (np.pi / norient)

        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        np.clip(dtheta * norient / 2.0, a_min=0, a_max=np.pi, out=dtheta)
        spread = (np.cos(dtheta) + 1.0) / 2.0

        sumE_ThisOrient = zeromat.copy()
        sumO_ThisOrient = zeromat.copy()
        sumAn_ThisOrient = zeromat.copy()
        Energy = zeromat.copy()

        EOscale = []

        for ss in range(nscale):
            filt = logGabor[ss] * spread
            thisEO = ifft2(IM * filt)
            An = np.abs(thisEO)
            sumAn_ThisOrient += An
            sumE_ThisOrient += np.real(thisEO)
            sumO_ThisOrient += np.imag(thisEO)

            if ss == 0:
                if noiseMethod == -1:
                    tau = np.median(sumAn_ThisOrient.ravel()) / np.sqrt(np.log(4))
                elif noiseMethod == -2:
                    tau = _rayleighmode(sumAn_ThisOrient.ravel())
                maxAn = An
            else:
                maxAn = np.maximum(maxAn, An)

            EOscale.append(thisEO)

        EnergyV[:, :, 0] += sumE_ThisOrient
        EnergyV[:, :, 1] += np.cos(angl) * sumO_ThisOrient
        EnergyV[:, :, 2] += np.sin(angl) * sumO_ThisOrient

        XEnergy = (
            np.sqrt(
                sumE_ThisOrient * sumE_ThisOrient
                + sumO_ThisOrient * sumO_ThisOrient
            )
            + epsilon
        )
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        for ss in range(nscale):
            E = np.real(EOscale[ss])
            O = np.imag(EOscale[ss])
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        if noiseMethod >= 0:
            T = noiseMethod
        else:
            totalTau = tau * (1.0 - (1.0 / mult) ** nscale) / (1.0 - (1.0 / mult))
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.0)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.0)
            T = np.maximum(EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon)

        Energy = np.maximum(Energy - T, 0)

        width = (sumAn_ThisOrient / (maxAn + epsilon) - 1.0) / (nscale - 1)
        weight = 1.0 / (1.0 + np.exp(g * (cutOff - width)))

        # Guard against division by zero on blank/uniform images
        with np.errstate(invalid="ignore", divide="ignore"):
            thisPC = np.where(
                sumAn_ThisOrient > epsilon,
                weight * Energy / sumAn_ThisOrient,
                0.0,
            )
        pcSum += thisPC

        covx = thisPC * np.cos(angl)
        covy = thisPC * np.sin(angl)
        covx2 += covx * covx
        covy2 += covy * covy
        covxy += covx * covy

        PC.append(thisPC)
        EO.append(EOscale)

    covx2 /= norient / 2.0
    covy2 /= norient / 2.0
    covxy *= 4.0 / norient
    denom = (
        np.sqrt(covxy * covxy + (covx2 - covy2) * (covx2 - covy2)) + epsilon
    )

    M = (covx2 + covy2 + denom) / 2.0
    m = (covx2 + covy2 - denom) / 2.0

    ori = np.arctan2(EnergyV[:, :, 2], EnergyV[:, :, 1])
    ori = np.round((ori % np.pi) * 180.0 / np.pi)

    OddV = np.sqrt(
        EnergyV[:, :, 1] * EnergyV[:, :, 1]
        + EnergyV[:, :, 2] * EnergyV[:, :, 2]
    )
    ft = np.arctan2(EnergyV[:, :, 0], OddV)

    return M, m, ori, ft, PC, EO, T
