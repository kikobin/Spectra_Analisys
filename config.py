"""
config.py

Central configuration for the Spectra pipeline.
All band definitions, detection thresholds, and quality settings live here.
Changing a value here propagates to every module that imports it.
"""

# ── Detection thresholds ──────────────────────────────────────────────────────
SNR_VALID_BAND = 2.0     # minimum per-band SNR to count a band as "valid detection"
SNR_ANNOTATE   = 2.0     # minimum SNR to draw a label annotation on plots

# ── Depth prior bonus boundaries ─────────────────────────────────────────────
DEPTH_BONUS_WEAK   = 0.05   # depth ≥ this earns a +0.03 prior bonus
DEPTH_BONUS_MED    = 0.10   # depth ≥ this earns a +0.07 prior bonus
DEPTH_BONUS_STRONG = 0.20   # depth ≥ this earns a +0.10 prior bonus

# ── Data-quality ceilings ─────────────────────────────────────────────────────
QUALITY_USABLE       = 0.40  # below this → data flagged as unusable
QUALITY_CAP_POOR     = 0.25  # quality < this → confidence capped at 0.25
QUALITY_CAP_MODERATE = 0.50  # quality < this → confidence capped at 0.50

# ── Confidence-status thresholds ─────────────────────────────────────────────
CONF_STRONG   = 0.80
CONF_LIKELY   = 0.60
CONF_MARGINAL = 0.40
CONF_WEAK     = 0.20

# ── Molecular band definitions ────────────────────────────────────────────────
# Format: 'Molecule': [{'band': (start_μm, end_μm), 'side': (la, lb, ra, rb)}, ...]
# 'band' is the core absorption window.
# 'side' defines left (la→lb) and right (ra→rb) continuum sideband windows
#   used for the linear-interpolation baseline estimate.
MOLECULE_BANDS = {

    # ── Water vapour ──────────────────────────────────────────────────────────
    # Dominant absorber in substellar atmospheres across 1–12 μm.
    'H2O': [
        {'band': (1.35, 1.45), 'side': (1.28, 1.35, 1.45, 1.52)},   # J-band wing
        {'band': (1.80, 2.00), 'side': (1.68, 1.80, 2.00, 2.12)},   # H-band wing
        {'band': (2.55, 2.85), 'side': (2.42, 2.55, 2.85, 2.97)},   # K-band wing
        {'band': (2.95, 3.20), 'side': (2.82, 2.95, 3.20, 3.33)},   # L-band short
        {'band': (3.85, 4.20), 'side': (3.68, 3.85, 4.20, 4.37)},   # L-band long
        {'band': (5.80, 6.60), 'side': (5.45, 5.80, 6.60, 6.95)},   # MIRI MRS
        {'band': (10.0, 12.0), 'side': (9.00, 10.0, 12.0, 13.0)},   # MIRI 10 μm
    ],

    # ── Molecular oxygen ──────────────────────────────────────────────────────
    # Rare in substellar objects; O2 A-band (0.76 μm) outside most NIRSpec configs.
    'O2': [
        {'band': (1.24, 1.30), 'side': (1.19, 1.24, 1.30, 1.35)},   # 1.27 μm band
        {'band': (0.75, 0.77), 'side': (0.73, 0.75, 0.77, 0.79)},   # A-band (NIRSpec G140)
    ],

    # ── Ozone ─────────────────────────────────────────────────────────────────
    # 9.6 μm Hartley band; can be confused with silicate dust features.
    'O3': [
        {'band': (9.30, 9.90), 'side': (8.90, 9.30, 9.90, 10.3)},
    ],

    # ── Methane ───────────────────────────────────────────────────────────────
    # Dominant carbon carrier for T < 1400 K (T/Y dwarfs).
    'CH4': [
        {'band': (1.63, 1.80), 'side': (1.55, 1.63, 1.80, 1.90)},   # H-band overtone
        {'band': (2.20, 2.45), 'side': (2.10, 2.20, 2.45, 2.58)},   # K-band
        {'band': (3.25, 3.60), 'side': (3.10, 3.25, 3.60, 3.75)},   # L-band fundamental
    ],

    # ── Carbon monoxide ───────────────────────────────────────────────────────
    # Expected in warm atmospheres (T > 1200 K); anti-correlated with CH4.
    'CO': [
        {'band': (4.60, 4.80), 'side': (4.44, 4.60, 4.80, 4.96)},   # 1-0 fundamental
        {'band': (2.29, 2.38), 'side': (2.22, 2.29, 2.38, 2.46)},   # 2-0 overtone (K-band)
    ],

    # ── Carbon dioxide ────────────────────────────────────────────────────────
    # ν3 band at 4.3 μm detectable even at trace mixing ratios.
    'CO2': [
        {'band': (4.22, 4.38), 'side': (4.06, 4.22, 4.38, 4.54)},
    ],
}
