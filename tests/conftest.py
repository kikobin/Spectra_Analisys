"""
conftest.py — shared pytest fixtures for the Spectra pipeline test suite.
"""
import sys
import os
import numpy as np
import pytest

# Ensure the project root is importable (for CI environments without pip install -e .)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Synthetic spectrum fixtures ───────────────────────────────────────────────

@pytest.fixture
def wavelength_um():
    """1–5 μm wavelength grid at R ≈ 300 (NIRSpec PRISM-like)."""
    return np.linspace(1.0, 5.0, 400)


@pytest.fixture
def blackbody_flux(wavelength_um):
    """Simple Rayleigh-Jeans proxy flux (T = 1200 K) without absorption."""
    w_m = wavelength_um * 1e-6
    return 1.0 / (w_m ** 4 * (np.exp(0.014388 / (w_m * 1200)) - 1))


@pytest.fixture
def flat_residual(wavelength_um):
    """Flat residual (no features) with small Gaussian noise."""
    rng = np.random.default_rng(42)
    return np.ones_like(wavelength_um) + rng.normal(0, 0.02, len(wavelength_um))


@pytest.fixture
def h2o_residual(wavelength_um, flat_residual):
    """Residual with a synthetic H2O absorption dip at 1.38 μm (J-band)."""
    res = flat_residual.copy()
    mask = (wavelength_um >= 1.35) & (wavelength_um <= 1.45)
    res[mask] -= 0.25   # 25 % absorption depth
    return res


@pytest.fixture
def noise_err(blackbody_flux):
    """Per-pixel flux error at SNR ≈ 20 (median)."""
    return blackbody_flux / 20.0
