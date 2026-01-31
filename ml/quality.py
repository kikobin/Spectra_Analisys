import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union

def assess_quality(
    wavelength: np.ndarray,
    normalized_flux: np.ndarray,
    error: Optional[np.ndarray] = None,
    snr: Optional[float] = None
) -> Dict[str, Any]:
    """
    Assesses the quality of a spectrum for analysis.
    
    Args:
        wavelength: 1D array of wavelengths.
        normalized_flux: 1D array of normalized flux.
        error: 1D array of flux errors (optional).
        snr: Pre-calculated Signal-to-Noise Ratio (optional).

    Returns:
        Dict with:
            - quality_score: float (0.0 - 1.0)
            - usable: bool
            - notes: list of strings
    """
    # FAIL-SAFE: Default return in case of any internal error
    default_result = {
        "quality_score": 0.0,
        "usable": True,  # Per requirements: return usable=True on error
        "notes": ["ML Error: check logs"]
    }

    try:
        notes = []
        
        # 1. Basic Validation
        if len(wavelength) == 0 or len(normalized_flux) == 0:
            return {"quality_score": 0.0, "usable": False, "notes": ["Empty data"]}
        
        if len(wavelength) != len(normalized_flux):
             return {"quality_score": 0.0, "usable": False, "notes": ["Shape mismatch"]}

        # 2. Check for Gaps / NaNs
        is_finite = np.isfinite(normalized_flux)
        valid_fraction = np.sum(is_finite) / len(normalized_flux)
        
        if valid_fraction < 0.5:
            return {
                "quality_score": 0.0, 
                "usable": False, 
                "notes": [f"Too many NaNs ({100*(1-valid_fraction):.1f}%)"]
            }
        
        if valid_fraction < 0.9:
            notes.append(f"Gaps detected ({100*(1-valid_fraction):.1f}% NaN)")

        # Filter valid data for stats
        valid_flux = normalized_flux[is_finite]
        
        # 3. SNR Calculation
        calculated_snr = 0.0
        if snr is not None:
            calculated_snr = snr
        elif error is not None:
            # Avoid division by zero
            valid_mask = is_finite & (error > 0)
            if np.any(valid_mask):
                # Median SNR is robust
                snrs = normalized_flux[valid_mask] / error[valid_mask]
                calculated_snr = np.nanmedian(snrs)
            else:
                notes.append("Error array invalid/zero")
        else:
            # Estimate SNR from signal scatter if no error provided
            # (Rough estimate: 1/std of residuals from smoothed signal - too complex for now?)
            # Just assume moderate/low if missing
            calculated_snr = 5.0 
            notes.append("No SNR/Error info provided, assuming low-mod")

        if calculated_snr < 3.0:
            notes.append(f"Low SNR ({calculated_snr:.1f})")
        elif calculated_snr > 50.0:
            notes.append(f"High SNR ({calculated_snr:.1f})")

        # 4. Continuum Stability
        # Check standard deviation of normalized flux. 
        # Ideally, normalized flux ~ 1.0. Large deviations > +/- 0.5 imply bad normalization or huge features/noise.
        std_dev = np.std(valid_flux)
        if std_dev > 0.5:
            notes.append(f"Unstable continuum (std={std_dev:.2f})")
            
        # 5. Quality Score Calculation
        # Base on SNR
        # SNR 100 -> 1.0
        # SNR 10 -> 0.5
        # SNR 0 -> 0.0
        # Simple log scale or linear? Let's use linear saturation for simplicity.
        score_snr = min(calculated_snr / 20.0, 1.0) # Reach 1.0 at SNR=20
        
        # Penalties
        score = score_snr
        if valid_fraction < 0.9:
            score *= 0.8
        if std_dev > 0.5:
            score *= 0.5
            
        return {
            "quality_score": round(max(0.0, min(score, 1.0)), 3),
            "usable": True,
            "notes": notes
        }

    except Exception as e:
        logging.error(f"ML Quality Assessment Failed: {e}", exc_info=True)
        return default_result

if __name__ == "__main__":
    # Internal Verification
    print("Running Quality Module Verification...")
    
    # helper
    def test(name, w, f, e=None, snr=None):
        res = assess_quality(w, f, e, snr)
        print(f"[{name}] Score: {res['quality_score']}, Usable: {res['usable']}, Notes: {res['notes']}")

    # 1. Perfect Data
    w = np.linspace(1, 5, 100)
    f = np.ones(100) + np.random.normal(0, 0.01, 100) # SNR ~ 100
    test("High SNR", w, f, snr=100)

    # 2. Low SNR
    f_noisy = np.ones(100) + np.random.normal(0, 0.5, 100) # SNR ~ 2
    test("Low SNR", w, f_noisy, snr=2)

    # 3. Gaps
    f_gaps = f.copy()
    f_gaps[0:40] = np.nan
    test("Gaps", w, f_gaps, snr=20)

    # 4. Total Fail (Empty)
    test("Empty", np.array([]), np.array([]))

    # 5. Exception Test (Input mismatch)
    test("Mismatch Force", np.array([1]), np.array([1, 2])) 
    
    print("Verification Complete.")
