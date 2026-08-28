"""Low-level FFT and filter utilities for phase congruency computation.

Vendored from PhasePack by Ali Shervin Muldal (https://github.com/alimuldal/phasepack).
Adapted for use in the RIFT2 adapter within the SIH26166 pipeline.

Modifications from upstream:
- Removed pyfftw optional dependency; uses scipy.fft directly.
- Stripped unused perfft2 function.
- Minor style adjustments for project linting compatibility.
"""

import numpy as np
from scipy.fft import fft2, ifft2  # noqa: F401 — re-exported
from scipy.fft import ifftshift


def lowpassfilter(size, cutoff, n):
    """Construct a low-pass Butterworth filter.

    f = 1 / (1 + (w/cutoff)^2n)

    Args:
        size: (rows, cols) tuple specifying filter dimensions.
        cutoff: Cutoff frequency of the filter, 0–0.5.
        n: Order of the filter (integer >= 1).

    Returns:
        Filter array with frequency origin at the corners.
    """
    if cutoff < 0.0 or cutoff > 0.5:
        raise ValueError("cutoff must be between 0 and 0.5")
    if n % 1:
        raise ValueError("n must be an integer >= 1")
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size

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

    return ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2.0 * n)))


def rayleighmode(data, nbins=50):
    """Compute mode of data assumed to come from a Rayleigh distribution.

    Args:
        data: Input data array.
        nbins: Number of histogram bins.

    Returns:
        Estimated mode value.
    """
    n, edges = np.histogram(data, nbins)
    ind = np.argmax(n)
    return (edges[ind] + edges[ind + 1]) / 2.0
