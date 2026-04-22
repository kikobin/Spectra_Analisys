"""
demo_data.py

Helpers to generate small, synthetic FITS spectra for demos and docs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


def make_dummy_fits(
    path: str | Path,
    *,
    seed: int = 42,
    n_points: int = 1000,
    w_min_um: float = 1.0,
    w_max_um: float = 12.0,
) -> Path:
    """
    Create a small synthetic 1D spectrum FITS (BinTableHDU) with columns:
    WAVELENGTH, FLUX, ERROR (in microns / arbitrary flux units).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    w = np.linspace(w_min_um, w_max_um, n_points)

    # Smooth, "blackbody-ish" continuum proxy.
    continuum = 100 * (1 / w**2) * np.exp(-1 / w)

    # Add one obvious absorption feature near 1.4um (H2O-like).
    flux = continuum.copy()
    mask = (w > 1.35) & (w < 1.45)
    flux[mask] *= 0.80

    # Add heteroscedastic noise.
    noise = rng.normal(0, 0.05 * np.maximum(flux, 1e-12), size=len(w))
    flux = flux + noise
    err = 0.05 * np.maximum(flux, 1e-12)

    t = Table([w, flux, err], names=("WAVELENGTH", "FLUX", "ERROR"))

    hdu = fits.BinTableHDU(t)
    hdu.writeto(out_path, overwrite=True)
    return out_path

