import numpy as np
import logging
from typing import Dict, Any, Optional


def assess_quality(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: Optional[np.ndarray] = None,
    snr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Assesses spectrum quality on four independent axes, then combines them.

    Axes:
        1. SNR score       — based on median per-pixel S/N or NMAD estimate
        2. Coverage score  — fraction of the 1–14 μm standard range covered
        3. Gap score       — penalizes NaN / missing data
        4. Smoothness      — detects excessive pixel-to-pixel noise spikes

    Final quality_score = product of all four sub-scores, clipped to [0, 1].

    Returns:
        {quality_score, usable, snr_estimate, notes}
    """
    default_result = {
        "quality_score": 0.5,
        "usable": True,
        "snr_estimate": 0.0,
        "notes": ["Quality assessment failed — defaulting to usable"],
    }

    try:
        notes = []

        # ── Basic validation ────────────────────────────────────────────────
        wavelength = np.asarray(wavelength, dtype=float)
        flux = np.asarray(flux, dtype=float)

        if wavelength.size == 0 or flux.size == 0:
            return {"quality_score": 0.0, "usable": False,
                    "snr_estimate": 0.0, "notes": ["Empty data"]}
        if wavelength.size != flux.size:
            return {"quality_score": 0.0, "usable": False,
                    "snr_estimate": 0.0, "notes": ["Shape mismatch"]}

        # ── 1. Gap / NaN score ──────────────────────────────────────────────
        is_finite = np.isfinite(flux) & np.isfinite(wavelength)
        valid_frac = float(np.sum(is_finite)) / len(flux)

        if valid_frac < 0.40:
            return {"quality_score": 0.0, "usable": False,
                    "snr_estimate": 0.0,
                    "notes": [f"Too many NaNs ({100*(1-valid_frac):.1f}%)"]}
        if valid_frac < 0.80:
            notes.append(f"Significant gaps ({100*(1-valid_frac):.1f}% NaN)")

        gap_score = float(np.clip(valid_frac, 0.0, 1.0))

        valid_flux = flux[is_finite]
        valid_wave = wavelength[is_finite]

        # ── 2. SNR score ────────────────────────────────────────────────────
        calc_snr = 0.0
        if snr is not None:
            calc_snr = float(snr)
        elif error is not None:
            err_arr = np.asarray(error, dtype=float)
            valid_err_mask = is_finite & np.isfinite(err_arr) & (err_arr > 0) & (flux > 0)
            if np.any(valid_err_mask):
                px_snr = flux[valid_err_mask] / err_arr[valid_err_mask]
                # Clip hot-pixel outliers: discard px_snr > 5× median
                med_px = np.nanmedian(px_snr)
                clipped = px_snr[px_snr < 5.0 * med_px] if med_px > 0 else px_snr
                calc_snr = float(np.nanmedian(clipped)) if len(clipped) > 0 else med_px
            else:
                notes.append("Error array all-zero or invalid")
        else:
            # NMAD-based SNR proxy: signal / scatter (no error array)
            med_f = np.nanmedian(valid_flux)
            nmad_f = 1.4826 * np.nanmedian(np.abs(valid_flux - med_f))
            calc_snr = float(med_f / nmad_f) if nmad_f > 0 else 0.0
            notes.append("No error array — SNR estimated from flux NMAD")

        calc_snr = max(0.0, calc_snr)
        if calc_snr < 3.0:
            notes.append(f"Low SNR ({calc_snr:.1f})")
        elif calc_snr > 50.0:
            notes.append(f"Very high SNR ({calc_snr:.1f})")

        # Saturates at SNR = 15 (realistic upper bound for exoplanet atmospheres)
        snr_score = float(np.clip(calc_snr / 15.0, 0.0, 1.0))

        # ── 3. Wavelength coverage score ────────────────────────────────────
        # Reference: 1–14 μm encompasses all H2O, O2, O3 features of interest
        REF_MIN, REF_MAX = 1.0, 14.0
        obs_min = float(np.nanmin(valid_wave))
        obs_max = float(np.nanmax(valid_wave))
        covered = max(0.0, min(obs_max, REF_MAX) - max(obs_min, REF_MIN))
        coverage_score = float(np.clip(covered / (REF_MAX - REF_MIN), 0.0, 1.0))
        if coverage_score < 0.3:
            notes.append(f"Limited wavelength coverage ({obs_min:.1f}–{obs_max:.1f} μm)")

        # ── 4. Spectral smoothness score ────────────────────────────────────
        # Rolling NMAD normalised by rolling median.
        # High roughness → low score.
        smoothness_score = 1.0
        if len(valid_flux) >= 20:
            win = max(5, len(valid_flux) // 20)
            local_rough = []
            for i in range(0, len(valid_flux) - win, max(1, win // 2)):
                chunk = valid_flux[i : i + win]
                med_c = np.nanmedian(chunk)
                if med_c > 0:
                    nmad_c = 1.4826 * np.nanmedian(np.abs(chunk - med_c))
                    local_rough.append(nmad_c / med_c)
            if local_rough:
                roughness = float(np.nanmedian(local_rough))
                # roughness > 0.30 → spectrum is very noisy
                smoothness_score = float(np.clip(1.0 - roughness / 0.30, 0.1, 1.0))
                if roughness > 0.20:
                    notes.append(f"Noisy spectrum (roughness index={roughness:.3f})")

        # ── Combine ─────────────────────────────────────────────────────────
        # Geometric mean gives balanced penalty across all axes
        quality_score = float(
            (snr_score * gap_score * coverage_score * smoothness_score) ** 0.5
        )
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        return {
            "quality_score": round(quality_score, 3),
            "usable": quality_score > 0.05,
            "snr_estimate": round(calc_snr, 2),
            "notes": notes,
        }

    except Exception as exc:
        logging.error(f"Quality assessment failed: {exc}", exc_info=True)
        return default_result


if __name__ == "__main__":
    print("Quality module verification")

    def _test(name, w, f, e=None, snr=None):
        r = assess_quality(w, f, e, snr)
        print(f"[{name:12s}] score={r['quality_score']:.3f}  "
              f"snr={r['snr_estimate']:.1f}  usable={r['usable']}  {r['notes']}")

    rng = np.random.default_rng(42)
    w = np.linspace(1, 5, 200)

    _test("High SNR",   w, np.ones(200) + rng.normal(0, 0.01, 200),
          snr=100)
    _test("Low SNR",    w, np.ones(200) + rng.normal(0, 0.4, 200),
          snr=2.5)
    _test("With errs",  w, np.ones(200) + rng.normal(0, 0.05, 200),
          e=np.full(200, 0.05))
    f_gap = np.ones(200) + rng.normal(0, 0.02, 200)
    f_gap[:80] = np.nan
    _test("50% gaps",   w, f_gap)
    _test("Empty",      np.array([]), np.array([]))
