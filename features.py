#!/usr/bin/env python3
"""
features.py

Module for measuring spectral absorption features.
Analyzes residual spectra (Flux/Continuum) for H2O, O2, O3.

Author: Antigravity Agent
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.special import erfc
from scipy.stats import t as t_dist


try:
    import io_fits
    import continuum
    from ml.detection_confidence import ConfidenceAssessor
    from config import MOLECULE_BANDS, SNR_VALID_BAND
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from config import MOLECULE_BANDS, SNR_VALID_BAND
    except ImportError:
        SNR_VALID_BAND = 2.0
        MOLECULE_BANDS = {}

def _sideband_contamination(side, mol_name):
    """
    Returns a list of molecules whose absorption core overlaps with this band's sidebands.
    Such overlap biases the baseline estimate (absorption in sideband looks like continuum),
    causing the measured depth to be underestimated.
    """
    la, lb, ra, rb = side
    flags = []
    for other_mol, other_bands in MOLECULE_BANDS.items():
        if other_mol == mol_name:
            continue
        for b_def in other_bands:
            bs, be = b_def['band']
            left_ov  = max(la, bs) < min(lb, be)
            right_ov = max(ra, bs) < min(rb, be)
            if left_ov or right_ov:
                sides = '+'.join((['L'] if left_ov else []) + (['R'] if right_ov else []))
                flags.append(f"{other_mol}@{bs:.2f}-{be:.2f}({sides})")
    return flags


def measure_band(w_um, residual, band=None, side=None, err=None, mol_name=None):
    """
    Measures band properties: Depth, EQW, SNR.

    Args:
        w_um (array): wavelengths (um).
        residual (array): normalized flux (flux/continuum).
        band (tuple): (start, end) of absorption band in μm.
        side (tuple): (la, lb, ra, rb) sideband windows for baseline.
        err (array, optional): flux errors — improves noise estimate.

    Returns:
        dict: { 'depth', 'depth_err', 'eqw', 'snr', 'covered',
                'baseline', 'noise_std', 'n_band' }
    """
    if band is None or side is None:
        return {'covered': False, 'note': 'No definition'}

    # 1. Coverage Check
    # Check if band overlaps with data
    band_mask = (w_um >= band[0]) & (w_um <= band[1])
    n_band = np.sum(band_mask)
    
    if n_band == 0:
        return {
            'covered': False, 'depth': 0, 'eqw': 0, 'snr': 0, 
            'n_band': 0, 'baseline': 1.0, 'noise_std': 0
        }
    
    # 2. Sidebands — linear interpolation baseline
    la, lb, ra, rb = side
    left_mask = (w_um >= la) & (w_um <= lb) & np.isfinite(residual)
    right_mask = (w_um >= ra) & (w_um <= rb) & np.isfinite(residual)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left + n_right < 2:
        return {
            'covered': False, 'note': 'No sideband coverage',
            'depth': 0, 'eqw': 0, 'snr': 0, 'n_band': n_band
        }

    # Noise from combined sidebands — NMAD (robust vs. sideband outliers)
    all_side_vals = np.concatenate([
        residual[left_mask] if n_left > 0 else np.array([]),
        residual[right_mask] if n_right > 0 else np.array([])
    ])
    med_side = np.nanmedian(all_side_vals)
    noise_std = 1.4826 * np.nanmedian(np.abs(all_side_vals - med_side))
    if noise_std <= 1e-12:
        noise_std = np.nanstd(all_side_vals)  # fallback if all identical
    if noise_std <= 1e-12:
        noise_std = 1e-12

    # Cross-check with formal pipeline errors when available
    # Take max(NMAD, formal_err) — conservative: catches pipeline underestimates
    if err is not None:
        err_arr = np.asarray(err, dtype=float)
        side_errs = np.concatenate([
            err_arr[left_mask] if n_left > 0 else np.array([]),
            err_arr[right_mask] if n_right > 0 else np.array([])
        ])
        valid_errs = side_errs[np.isfinite(side_errs) & (side_errs > 0)]
        if len(valid_errs) > 0:
            formal_noise = float(np.nanmedian(valid_errs))
            noise_std = max(noise_std, formal_noise)

    # 3. Band Measurements — wavelength-dependent (sloped) baseline
    w_band = w_um[band_mask]
    res_band = residual[band_mask]

    if n_left >= 2 and n_right >= 2:
        # Linear interpolation between left and right sideband medians
        w_L = np.nanmedian(w_um[left_mask])
        w_R = np.nanmedian(w_um[right_mask])
        v_L = np.nanmedian(residual[left_mask])
        v_R = np.nanmedian(residual[right_mask])
        baseline_arr = np.interp(w_band, [w_L, w_R], [v_L, v_R])
    else:
        # Only one side available — constant baseline
        baseline_arr = np.full(len(w_band), np.nanmedian(all_side_vals))

    baseline = float(np.nanmean(baseline_arr))

    # Absorption array (positive = absorption dip)
    absorption = baseline_arr - res_band

    # Robust depth: 95th percentile of absorption (avoids single noisy pixels)
    if len(absorption) > 5:
        depth = float(np.percentile(absorption, 95))
    else:
        depth = float(np.max(absorption)) if len(absorption) > 0 else 0.0
    depth = max(depth, 0.0)

    # EQW using sloped baseline
    safe_bl = np.where(np.abs(baseline_arr) > 1e-10, baseline_arr, 1e-10)
    integrand = absorption / safe_bl
    eqw = 0.0
    if n_band > 1:
        eqw = float(trapezoid(integrand, w_band))

    # ── SNR (two methods, take the more sensitive) ───────────────────────────
    # 1. Peak-depth SNR: classical approach
    snr_peak = depth / noise_std if noise_std > 0 else 0.0

    # 2. Matched-filter (integrated) SNR:
    #    SNR_mf = mean_absorption × √n_band / noise_std
    #    Gains √n_band sensitivity over peak SNR for broad, low-res bands.
    #    Equivalent to a top-hat matched filter across the whole band.
    mean_absorption = float(np.nanmean(absorption)) if len(absorption) > 0 else 0.0
    snr_integrated = max(0.0, mean_absorption * np.sqrt(n_band) / noise_std) \
        if noise_std > 0 else 0.0

    # Use the more sensitive of the two
    snr = max(snr_peak, snr_integrated)

    # Formal depth uncertainty: σ_depth = noise / √n_band  (standard error)
    depth_err = noise_std / np.sqrt(max(1, n_band))

    # ── Spectral resolution correction ──────────────────────────────────────
    # At low resolution (PRISM/CLEAR, R ≈ 100-300) broad molecular features
    # are diluted: the observed depth understates the true depth.
    # Correction factor: f = W_band / (W_band + δλ_pixel)
    # where δλ_pixel = median pixel size, a proxy for the instrument FWHM.
    R_eff = 0.0
    dilution_factor = 1.0
    if n_band >= 2:
        dw = float(np.median(np.diff(w_band)))   # median pixel size (μm)
        w_center = float(np.nanmean(w_band))
        band_width = band[1] - band[0]
        if dw > 0 and band_width > 0:
            R_eff = w_center / dw
            dilution_factor = band_width / (band_width + dw)
            if dilution_factor > 0.05:           # avoid div-by-near-zero
                depth     = depth     / dilution_factor
                depth_err = depth_err / dilution_factor

    # ── False Alarm Probability ──────────────────────────────────────────────
    # Use t-distribution for small samples (n_band < 30); Gaussian tail otherwise.
    # The t-distribution has heavier tails → more conservative FAP for few pixels.
    if snr > 0:
        if n_band < 30:
            fap = float(t_dist.sf(snr, df=max(1, n_band - 1)))
        else:
            fap = float(0.5 * erfc(snr / np.sqrt(2)))
    else:
        fap = 1.0

    # ── Sideband contamination check ────────────────────────────────────────
    # Warn when another molecule's absorption core overlaps this band's sidebands,
    # which biases the linear baseline and causes depth underestimation.
    contamination = _sideband_contamination(side, mol_name) if mol_name else []

    return {
        'covered': True,
        'depth': depth,
        'depth_err': depth_err,
        'eqw': eqw,
        'snr': snr,
        'snr_peak': snr_peak,
        'snr_integrated': snr_integrated,
        'fap': fap,
        'baseline': baseline,
        'noise_std': noise_std,
        'n_band': n_band,
        'R_eff': round(R_eff, 1),
        'dilution_factor': round(dilution_factor, 4),
        'contamination': contamination,
        'band_mask': band_mask
    }

def detect_molecules(w_um, residual, err=None, object_params=None):
    """
    Detects molecules using ConfidenceAssessor.

    Args:
        w_um (array): wavelengths (μm).
        residual (array): normalized flux (flux/continuum).
        err (array, optional): flux errors — passed to measure_band for noise estimate.
        object_params (dict, optional): physical context {'temperature', 'type'}.

    Returns:
        dict: Report per molecule with confidence assessment.
    """
    if object_params is None:
        object_params = {}

    report = {}
    assessor = ConfidenceAssessor()

    for mol, bands in MOLECULE_BANDS.items():
        mol_res = {
            'bands': [],
            'max_snr': 0.0,
            'max_depth': 0.0,
            'spectral_coverage': False
        }

        valid_bands = 0

        for b_def in bands:
            m = measure_band(w_um, residual, band=b_def['band'],
                             side=b_def['side'], err=err, mol_name=mol)

            info = m.copy()
            info['band_range'] = b_def['band']
            info.pop('band_mask', None)

            mol_res['bands'].append(info)

            if m['covered']:
                mol_res['spectral_coverage'] = True
                if m['snr'] > SNR_VALID_BAND:
                    valid_bands += 1
                if m['snr'] > mol_res['max_snr']:
                    mol_res['max_snr'] = m['snr']
                if m['depth'] > mol_res['max_depth']:
                    mol_res['max_depth'] = m['depth']

        # Combined SNR: quadrature sum over all bands with snr > 2
        # sqrt(Σ snr_i²) is the matched-filter significance for independent bands
        covered_snrs = [b['snr'] for b in mol_res['bands']
                        if b.get('covered') and b['snr'] > SNR_VALID_BAND]
        combined_snr = float(np.sqrt(np.sum(np.array(covered_snrs) ** 2))) \
            if covered_snrs else 0.0

        # Molecule-level FAP from combined SNR (joint significance across bands).
        # Use t-distribution when fewer than 30 contributing bands.
        n_contributing = len(covered_snrs)
        mol_res['combined_snr'] = combined_snr
        if combined_snr > 0:
            if n_contributing < 30:
                mol_res['fap'] = float(t_dist.sf(combined_snr, df=max(1, n_contributing - 1)))
            else:
                mol_res['fap'] = float(0.5 * erfc(combined_snr / np.sqrt(2)))
        else:
            mol_res['fap'] = 1.0

        # Assess Confidence
        assessment = assessor.assess(
            molecule_name=mol,
            snr=mol_res['max_snr'],
            num_bands=valid_bands,
            depth=mol_res['max_depth'],
            spectral_coverage=mol_res['spectral_coverage'],
            object_params=object_params,
            combined_snr=combined_snr
        )
        
        # Merge assessment into result
        mol_res.update(assessment)
        
        # Legacy boolean for compatibility (optional)
        mol_res['detected'] = (assessment['status'] in ["STRONG", "LIKELY", "MARGINAL"])
            
        report[mol] = mol_res
        
    return report

def plot_features_debug(w_um, residual, report, filename=""):
    """Debug plot for feature detection."""
    plt.figure(figsize=(12, 6))
    plt.plot(w_um, residual, 'k-', lw=0.8, alpha=0.8, label="Residual")
    plt.axhline(1.0, color='gray', ls='--')
    
    colors = {'H2O': 'b', 'O2': 'g', 'O3': 'r'}
    
    for mol, res in report.items():
        c = colors.get(mol, 'm')
        # Highlight if at least Marginal
        if res.get('detected', False):
            for b in res['bands']:
                if b['covered']:
                    start, end = b['band_range']
                    # Highlight only usually positive bands? Or all?
                    plt.axvspan(start, end, color=c, alpha=0.1)
                    if b['snr'] >= 1.5:
                         plt.text((start+end)/2, 0.9, f"{b['snr']:.1f}", 
                                  color=c, ha='center', fontsize=8)
    
    plt.title(f"Feature Analysis: {filename}")
    plt.ylim(0, 1.2)
    plt.xlabel("Wavelength [um]")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input FITS")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Only show Strong/Likely detections")
    args = parser.parse_args()
    
    if 'io_fits' not in globals() or 'continuum' not in globals():
        print("Error: modules missing")
        sys.exit(1)
        
    try:
        # Load
        data = io_fits.read_spectrum(args.file)
        w = data['wavelength_um']
        f = data['flux']
        
        # Continuum
        c_res = continuum.fit_blackbody(w, f, err=data['err'])
        if not c_res['fit_ok']:
            print("Warning: Continuum fit failed. Results may be garbage.")
            
        residual = c_res['residual']
        
        # Prepare Context
        # Infer Type/Temp from continuum
        T_est = c_res.get('T_K', 1500)
        obj_params = {
            'temperature': T_est,
            'type': 'Y' if T_est < 500 else ('T' if T_est < 1400 else 'L')
        }
        
        # Features
        report = detect_molecules(w, residual, object_params=obj_params)
        
        # Print
        print(f"Analysis: {args.file}")
        print(f"Object Context: T={T_est:.0f}K (Type ~{obj_params['type']})")
        print("="*60)
        
        for mol, info in report.items():
            # If strict mode, skip weak things
            if args.strict and info['status'] not in ["STRONG", "LIKELY"]:
                continue
                
            status = info['status']
            conf = info['confidence']
            expl = info['explanation']
            
            print(f"{mol}:")
            print(f"  Status:     {status}")
            print(f"  Confidence: {conf:.2f}")
            print(f"  Max SNR:    {info['max_snr']:.2f}")
            print(f"  Note:       {expl}")
            
            if info['spectral_coverage']:
                print("  Bands:")
                for b in info['bands']:
                    if b['covered']:
                        # Don't clutter with low SNR bands unless they are relevant
                        if b['snr'] > 1.0 or info['max_snr'] < 2.0:
                            print(f"    {b['band_range']}: Depth={b['depth']:.3f}, SNR={b['snr']:.1f}")
            else:
                print("  (No Spectral Coverage)")
            print("-" * 30)
        
        if args.plot:
            plot_features_debug(w, residual, report, args.file)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
