#!/usr/bin/env python3
"""
features.py

Module for measuring spectral absorption features.
Analyzes residual spectra (Flux/Continuum) for H2O, O2, O3.

Author: Antigravity Agent
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


try:
    import io_fits
    import continuum
    from ml.detection_confidence import ConfidenceAssessor
except ImportError:
    pass

# Band Definitions
# Format: 'Molecule': [ {'band': (start, end), 'side': (left_start, left_end, right_start, right_end)}, ... ]
# Adding more precise windows if needed.
MOLECULE_BANDS = {
    'H2O': [
        {'band': (1.35, 1.45), 'side': (1.30, 1.35, 1.45, 1.50)},
        {'band': (1.80, 2.00), 'side': (1.70, 1.80, 2.00, 2.10)},
        {'band': (2.55, 2.85), 'side': (2.45, 2.55, 2.85, 2.95)},
        {'band': (2.95, 3.20), 'side': (2.85, 2.95, 3.20, 3.30)},
        {'band': (3.85, 4.20), 'side': (3.70, 3.85, 4.20, 4.35)},
        {'band': (5.80, 6.60), 'side': (5.50, 5.80, 6.60, 6.90)},
        {'band': (10.0, 12.0), 'side': (9.00, 10.0, 12.0, 13.0)},
    ],
    'O2': [
        {'band': (1.24, 1.30), 'side': (1.20, 1.24, 1.30, 1.34)},
        {'band': (0.75, 0.77), 'side': (0.74, 0.75, 0.77, 0.78)}, # A-band
    ],
    'O3': [
        {'band': (9.30, 9.90), 'side': (9.00, 9.30, 9.90, 10.2)},
    ],
}

def measure_band(w_um, residual, band=None, side=None):
    """
    Measures band properties: Depth, EQW, SNR.

    Args:
        w_um (array): wavelengths (um).
        residual (array): normalized flux (flux/continuum).
        band (tuple): (start, end) of absorption band.
        side (tuple): (la, lb, ra, rb) sidebands for baseline.

    Returns:
        dict: { 'depth', 'eqw', 'snr', 'covered', 'baseline', 'noise_std', 'n_band' }
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
    
    # 2. Sidebands (Baseline & Noise)
    # Side: (la, lb) and (ra, rb)
    la, lb, ra, rb = side
    side_mask = ((w_um >= la) & (w_um <= lb)) | ((w_um >= ra) & (w_um <= rb))
    n_side = np.sum(side_mask)
    
    res_side = residual[side_mask]
    
    if n_side < 2:
        # Not enough sideband data. 
        # If we have band data but no sidebands, we can't reliably estimate baseline.
        # However, for normalized spectra, baseline ~ 1.0.
        baseline = 1.0
        # Estimate noise from band itself? Dangerous if deep absorption.
        # Fallback to global noise estimate? Or just return covered=False (Strict mode).
        # Let's try to be helpful: assume baseline 1, noise = 0.01 (1%) dummy?
        # Better: Mark as unreliable.
        return {
            'covered': False, 'note': 'No sideband coverage',
            'depth': 0, 'eqw': 0, 'snr': 0, 'n_band': n_band
        }

    # Robust baseline (median of sidebands)
    baseline = np.nanmedian(res_side)
    
    # Noise calculation (std in sidebands)
    # We assume sidebands are flat-ish.
    noise_std = np.nanstd(res_side)
    if noise_std <= 1e-12:
        noise_std = 1e-12 # Prevent div/0

    # 3. Band Measurements
    res_band = residual[band_mask]
    w_band = w_um[band_mask]
    
    # Depth: baseline - min(res_band)? 
    # For low-res, single pixel might be noise. Use median for robustness?
    # Requirement: "Если спектр низкого разрешения — искать не 'min', а медиану/интеграл".
    # Let's use percentile for depth to be robust against single pixels.
    # Depth = baseline - minimum (max absorption).
    # Robust Minimum: 5th percentile?
    if n_band > 5:
        val_at_depth = np.percentile(res_band, 5) # Close to bottom
    else:
        val_at_depth = np.min(res_band) # Few points, take min
        
    depth = baseline - val_at_depth
    
    # EQW
    # Integral of (1 - flux/cont) dlambda? Or (baseline - residual) dlambda?
    # EQW ~ Integral( (Ic - I)/Ic ) dlam. Ic ~ baseline. I/Ic = residual.
    # So Integrand = 1 - (residual/baseline).
    # Normalized to baseline.
    integrand = 1.0 - (res_band / baseline)
    eqw = 0.0
    if n_band > 1:
        eqw = trapezoid(integrand, w_band)
        
    # 4. SNR
    # signal = Depth
    # noise = noise_std
    snr = depth / noise_std if noise_std > 0 else 0
    
    return {
        'covered': True,
        'depth': depth,
        'eqw': eqw,
        'snr': snr,
        'baseline': baseline,
        'noise_std': noise_std,
        'n_band': n_band,
        'band_mask': band_mask
    }

def detect_molecules(w_um, residual, err=None, object_params=None):
    """
    Detects molecules using ConfidenceAssessor.
    
    Returns:
        dict: Report per molecule.
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
            m = measure_band(w_um, residual, band=b_def['band'], side=b_def['side'])
            
            # Store info
            info = m.copy()
            info['band_range'] = b_def['band']
            info.pop('band_mask', None) # Too heavy for report
            
            mol_res['bands'].append(info)
            
            if m['covered']:
                mol_res['spectral_coverage'] = True
                # Only count bands with meaningful signal towards multi-band bonus?
                # Or just any covered band? Assessor expects "detected bands".
                # Let's count bands with +SNR as "detected" for the count, 
                # but pass specific metrics to assessor.
                if m['snr'] > 1.0: 
                    valid_bands += 1
                
                if m['snr'] > mol_res['max_snr']:
                    mol_res['max_snr'] = m['snr']
                if m['depth'] > mol_res['max_depth']:
                    mol_res['max_depth'] = m['depth']
        
        # Assess Confidence
        assessment = assessor.assess(
            molecule_name=mol,
            snr=mol_res['max_snr'],
            num_bands=valid_bands,
            depth=mol_res['max_depth'],
            spectral_coverage=mol_res['spectral_coverage'],
            object_params=object_params
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
            label = f"{mol} ({res['status']})"
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
        report = detect_molecules(w, residual, err=data['err'], object_params=obj_params)
        
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
