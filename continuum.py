#!/usr/bin/env python3
"""
continuum.py

Module for robust continuum fitting (BlackBody) to spectra.
Part of the FITS -> Report pipeline.

Author: Antigravity Agent
"""

import numpy as np
import argparse
import sys
import warnings

from astropy.modeling import models, fitting
from astropy import units as u

try:
    import io_fits
except ImportError:
    io_fits = None

# Physical constant h·c / k_B in m·K
_HC_OVER_K = 0.014387769  # NIST 2018

def _estimate_T_twocolor(w_um, flux):
    """
    Estimates blackbody temperature using two-color flux ratio (Wien approximation).

    Uses the 75th-percentile flux in two spectral windows (near blue and red ends)
    to avoid bias from molecular absorption bands. More robust than peak-position
    Wien's law when the spectrum is dominated by absorption features.

    T = (hc/k) * (1/λ2 - 1/λ1) / ln[ F1/F2 * (λ1/λ2)^5 ]
    where λ1 < λ2 are the two reference wavelengths.

    Returns T in Kelvin, or None if the estimate is unreliable.
    """
    finite_mask = np.isfinite(flux) & (flux > 0) & np.isfinite(w_um) & (w_um > 0)
    if np.sum(finite_mask) < 10:
        return None

    w_f = w_um[finite_mask]
    w_span = w_f.max() - w_f.min()
    if w_span < 0.3:
        return None  # Too narrow — no leverage for color ratio

    # Window 1: 5–25% from blue end  (avoid edge artefacts)
    # Window 2: 75–95% from blue end
    w_min = w_f.min()
    m1 = (w_um >= w_min + 0.05 * w_span) & (w_um <= w_min + 0.25 * w_span) & finite_mask
    m2 = (w_um >= w_min + 0.75 * w_span) & (w_um <= w_min + 0.95 * w_span) & finite_mask

    if np.sum(m1) < 3 or np.sum(m2) < 3:
        return None

    # 75th percentile: biased away from absorption troughs
    f1 = float(np.percentile(flux[m1], 75))
    f2 = float(np.percentile(flux[m2], 75))
    lam1 = float(np.nanmedian(w_um[m1])) * 1e-6  # m
    lam2 = float(np.nanmedian(w_um[m2])) * 1e-6  # m

    if f1 <= 0 or f2 <= 0 or lam1 <= 0 or lam2 <= 0 or lam1 >= lam2:
        return None

    try:
        log_arg = (f1 / f2) * (lam1 / lam2) ** 5
        if log_arg <= 0:
            return None
        T_est = _HC_OVER_K * (1.0 / lam2 - 1.0 / lam1) / np.log(log_arg)
    except Exception:
        return None

    return float(T_est) if 50.0 < T_est < 1e5 else None


def fit_blackbody(w_um, flux, err=None, clip_sigma=3.0, max_iter=10):
    """
    Fits a BlackBody model to the data with robust asymmetric sigma-clipping.
    
    Since we are dealing with absorption spectra, we want the continuum to follow the 
    "upper envelope" of the data. Points significantly BELOW the model (absorption)
    should be rejected more aggressively than points ABOVE the model (noise).

    Args:
        w_um (array): Wavelength in microns.
        flux (array): Flux (arbitrary units).
        err (array, optional): Flux errors.
        clip_sigma (float): Sigma threshold for clipping.
        max_iter (int): Max iterations for clipping loop.
        
    Returns:
        dict: {
            'T_K': Temperature (K),
            'scale': Scale factor,
            'continuum': Fitted continuum array,
            'residual': flux / continuum,
            'delta': continuum - flux,
            'fit_ok': bool,
            'notes': list of strings
        }
    """
    # 1. Preparation
    w = np.array(w_um, dtype=float)
    f = np.array(flux, dtype=float)
    notes = []
    
    # Basic mask
    mask = np.isfinite(w) & np.isfinite(f) & (w > 0)
    if err is not None:
        e = np.array(err, dtype=float)
        mask &= np.isfinite(e) & (e > 0)
    else:
        e = np.ones_like(f)
        
    w_valid = w[mask]
    f_valid = f[mask]
    e_valid = e[mask]
    
    if len(w_valid) < 10:
        return _fallback_result(w, f, "Not enough valid points (<10)", success=False)

    # 2. Normalization
    # Fit in normalized space to avoid numerical issues
    f_norm_factor = np.nanmedian(f_valid)
    if f_norm_factor <= 0:
        f_norm_factor = np.nanmax(f_valid)
    if f_norm_factor <= 0:
        return _fallback_polynomial(w, f, ["All flux <= 0, cannot fit BB"])

    f_norm = f_valid / f_norm_factor
    e_norm = e_valid / f_norm_factor
    w_fit = w_valid # microns
    
    # 3. Initial Guess — two-color ratio (robust vs. absorption-dominated spectra)
    T_guess = _estimate_T_twocolor(w_valid, f_valid)
    if T_guess is not None:
        T_guess = np.clip(T_guess, 100, 50000)
        notes.append(f"T_guess (two-color): {T_guess:.0f} K")
    else:
        # Fallback: Wien's law from peak of flux
        peak_idx = np.argmax(f_valid)
        peak_w = w_valid[peak_idx]
        T_guess = np.clip(2898.0 / peak_w, 100, 50000) if peak_w > 0 else 3000.0
        notes.append(f"T_guess (Wien peak): {T_guess:.0f} K")
    
    # Model: Scale * BB(T)
    # Note: astropy BlackBody returns units. We will strip them for fitting stability.
    bb_init = models.BlackBody(temperature=T_guess * u.K, scale=1.0 * u.dimensionless_unscaled)
    
    # Estimate scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_guess = bb_init(w_fit * u.um).value
        
    scale_guess = np.median(f_norm) / np.median(model_guess) if np.median(model_guess) > 0 else 1.0
    bb_init.scale = scale_guess * u.dimensionless_unscaled

    # Bounds
    bb_init.temperature.bounds = (10, 50000)
    bb_init.scale.bounds = (1e-30, 1e20)
    
    fitter = fitting.LevMarLSQFitter()
    
    # 4. Iterative Asymmetric Sigma Clipping
    current_mask = np.ones(len(w_fit), dtype=bool)
    best_model = None
    
    for i in range(max_iter):
        # Subset
        it_w = w_fit[current_mask]
        it_f = f_norm[current_mask]
        it_e = e_norm[current_mask]
        
        if len(it_w) < 5:
            notes.append("Clipping removed too many points.")
            break
            
        # Weights = 1/sigma
        it_weights = 1.0 / it_e
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fitted_model = fitter(bb_init, it_w * u.um, it_f, weights=it_weights, maxiter=500)
        except Exception as ex:
            notes.append(f"Fit crashed iter {i}: {ex}")
            break
            
        # Calculate residuals for ALL points to update mask
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_vals = fitted_model(w_fit * u.um).value
            
        resid = f_norm - model_vals # Observed - Model
        
        # Calculate stats strictly on the CURRENT subset
        resid_subset = resid[current_mask]
        # NMAD: robust σ estimate, unaffected by deep absorption lines in the subset
        med_r = np.nanmedian(resid_subset)
        sigma = 1.4826 * np.nanmedian(np.abs(resid_subset - med_r))
        if sigma <= 0:
            sigma = np.nanstd(resid_subset)  # fallback if all residuals identical

        if sigma == 0:
            best_model = fitted_model
            break
            
        # ASYMMETRIC CLIPPING:
        # We want to reject deep absorption lines (flux << model => resid << 0).
        # We are more tolerant of positive excursions (noise or emission lines, though usually rare in this context).
        # Criteria: Keep if resid > -3sigma AND resid < +3sigma?
        # Actually, for "upper envelope":
        # - Reject if data is significantly BELOW model (resid < -sigma_lower)
        # - Reject if data is significantly ABOVE model (resid > sigma_upper)
        # Usually absorption lines are many sigma deep.
        
        lower_threshold = -1.5 * sigma # Strict on absorption (exclude dips)
        upper_threshold = 3.5 * sigma  # Loose on noise
        
        new_mask = (resid > lower_threshold) & (resid < upper_threshold)
        
        # Convergence check
        if np.array_equal(new_mask, current_mask):
            best_model = fitted_model
            break
            
        current_mask = new_mask
        bb_init = fitted_model # Update guess
        
    if best_model is None:
        # If loop failed, use last available or fallback
         if 'fitted_model' in locals():
             best_model = fitted_model
         else:
             return _fallback_polynomial(w, f, notes + ["Loop failed, no model."])

    # 5. Final Evaluation & Sanity Check
    T_final = best_model.temperature.value
    scale_val = best_model.scale.value
    
    # Recover physical scale
    scale_final = scale_val * f_norm_factor
    
    # Generate full continuum
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Re-create model with physical scale
        # Careful: the originally fitted 'scale' was for f_norm.
        # So we just multiply the OUTPUT of the fitted model by f_norm_factor.
        final_continuum = best_model(w * u.um).value * f_norm_factor
    
    # Check for unphysical results
    if T_final < 100 or T_final > 50000:
        notes.append(f"Unphysical T={T_final:.1f}K. Using fallback.")
        return _fallback_polynomial(w, f, notes, err=err)

    return _pack_result(w, f, final_continuum, T_final, scale_final, True, notes, err=err)

def _fallback_polynomial(w, f, notes, err=None):
    """Fallback: cubic polynomial fit in log-log space."""
    notes.append("Using polynomial fallback.")

    mask = (f > 0) & (w > 0) & np.isfinite(f) & np.isfinite(w)
    if np.sum(mask) < 5:
        return _fallback_result(w, f, notes + ["Not enough points for poly."], success=False)

    w_valid = w[mask]
    f_valid = f[mask]

    x = np.log10(w_valid)
    y = np.log10(f_valid)

    try:
        p = np.polyfit(x, y, 3)
        poly = np.poly1d(p)

        cont = np.full_like(w, np.nan)
        pos = (w > 0)
        cont[pos] = 10**poly(np.log10(w[pos]))
        
        return _pack_result(w, f, cont, -1.0, 1.0, False, notes, err=err)
    except Exception as e:
        return _fallback_result(w, f, notes + [f"Polyfit failed: {e}"], success=False)

def _pack_result(w, f, continuum, T, scale, ok_flag, notes, err=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        residual = f / continuum
        bad_cont = (continuum <= 0) | np.isnan(continuum)
        residual[bad_cont] = np.nan
        delta = continuum - f

    # Reduced chi-squared (2 free parameters: T and scale)
    chi2_reduced = np.nan
    if err is not None:
        e = np.asarray(err, dtype=float)
        valid = np.isfinite(f) & np.isfinite(continuum) & np.isfinite(e) & (e > 0)
        n_dof = int(np.sum(valid)) - 2
        if n_dof > 0:
            chi2_reduced = float(
                np.sum(((f[valid] - continuum[valid]) / e[valid]) ** 2) / n_dof
            )

    return {
        'T_K': T,
        'scale': scale,
        'continuum': continuum,
        'residual': residual,
        'delta': delta,
        'fit_ok': ok_flag,
        'chi2_reduced': chi2_reduced,
        'notes': notes
    }

def _fallback_result(w, f, notes, success=False):
    if isinstance(notes, str): notes = [notes]
    cont = np.full_like(f, np.nan)
    return {
        'T_K': np.nan,
        'scale': np.nan,
        'continuum': cont,
        'residual': np.full_like(f, np.nan),
        'delta': np.full_like(f, np.nan),
        'fit_ok': success,
        'chi2_reduced': np.nan,
        'notes': notes
    }

def plot_result(w, f, res):
    """Simple debug plot."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(w, f, 'k-', alpha=0.5, label='Data')
    ax1.plot(w, res['continuum'], 'r-', lw=2, label=f"Fit (T={res['T_K']:.1f}K)")
    ax1.set_ylabel('Flux')
    ax1.legend()
    ax1.set_title(f"Continuum Fit: {'OK' if res['fit_ok'] else 'FAIL'}")
    
    if res['notes']:
        ax1.text(0.05, 0.9, "\n".join(res['notes']), transform=ax1.transAxes, color='blue', fontsize=8)
        
    ax2.plot(w, res['residual'], 'b-', alpha=0.8)
    ax2.axhline(1.0, color='gray', ls='--')
    ax2.set_ylabel('Residual (Flux/Cont)')
    ax2.set_xlabel('Wavelength [um]')
    ax2.set_ylim(0, 2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="FITS file")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    if io_fits is None:
        print("Error: io_fits module missing.")
        sys.exit(1)
        
    try:
        data = io_fits.read_spectrum(args.filename)
        res = fit_blackbody(data['wavelength_um'], data['flux'], err=data['err'])
        
        print(f"Fit OK: {res['fit_ok']}")
        print(f"Temp: {res['T_K']:.1f} K")
        if res['notes']:
            print("Notes:", res['notes'])
            
        if args.plot:
            plot_result(data['wavelength_um'], data['flux'], res)
            
    except Exception as e:
        print(f"Error: {e}")
