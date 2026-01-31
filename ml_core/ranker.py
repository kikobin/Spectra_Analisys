import numpy as np
import os
from astropy.io import fits

class SpectrumRanker:
    """
    Selects the best spectrum from a collection based on technical quality metrics.
    Does NOT use physical interpretation (molecules), only signal statistics.
    """
    def __init__(self):
        # Weighting factors for the score
        self.weights = {
            'snr': 0.5,       # Signal-to-Noise is king
            'coverage': 0.3,  # More wavelengths = better
            'nans': -0.3,     # Penalize bad pixels / gaps
            'noise': -0.2,    # Penalize high raw noise floor
            'smoothness': -0.1 # Penalize jagged data (often artifacts)
        }

    def _extract_features(self, filepath):
        """
        Extracts technical metrics from a single FITS file.
        Returns dict of raw features or None if load fails.
        """
        try:
            with fits.open(filepath) as hdul:
                # Basic validation
                if len(hdul) < 2:
                    return None
                    
                data = hdul[1].data
                
                # Check for standard column names (JWST x1d)
                # Usually 'WAVELENGTH', 'FLUX', 'ERROR' or 'FLUX_ERROR'
                cols = [c.name.upper() for c in data.columns]
                
                # Resolving column names
                w_col = next((c for c in cols if 'WAVE' in c), None)
                f_col = next((c for c in cols if 'FLUX' in c and 'ERR' not in c), None)
                e_col = next((c for c in cols if 'ERR' in c), None)
                q_col = next((c for c in cols if 'DQ' in c or 'QUAL' in c), None)

                if not w_col or not f_col:
                    return None

                wave = data[w_col]
                flux = data[f_col]
                err = data[e_col] if e_col else np.ones_like(flux) * np.nan
                
                # 1. NaN Count (Bad pixels)
                n_nans = np.sum(np.isnan(flux))
                
                # Mask for valid stats
                valid_mask = ~np.isnan(flux) & (flux > 0)
                if e_col:
                    valid_mask &= (err > 0)
                
                n_valid = np.sum(valid_mask)
                if n_valid == 0:
                    return None
                
                v_flux = flux[valid_mask]
                v_err = err[valid_mask]
                v_wave = wave[valid_mask]

                # 2. Median SNR
                # Avoid divide by zero
                snr_arr = v_flux / v_err
                median_snr = np.median(snr_arr) if len(snr_arr) > 0 else 0
                
                # 3. Wavelength Coverage
                # Approximation: max - min (ignoring small internal gaps for now)
                coverage = v_wave.max() - v_wave.min()
                
                # 4. Noise Level (Median Error)
                median_noise = np.median(v_err)
                
                # 5. Smoothness (Roughness)
                # Std dev of the first difference of flux (measures "jaggedness")
                # Normalize by median flux to make it scale-invariant
                diff = np.diff(v_flux)
                roughness = np.std(diff) / (np.median(v_flux) + 1e-9)

                return {
                    'snr': median_snr,
                    'coverage': coverage,
                    'nans': n_nans / len(flux), # Fraction
                    'noise': median_noise,
                    'smoothness': roughness,
                    'valid_pixels': n_valid
                }
                
        except Exception as e:
            print(f"DEBUG: Failed to extract features from {os.path.basename(filepath)}: {e}")
            return None

    def rank(self, file_list):
        """
        Ranks FITS files by quality score.
        Input: list of filepaths.
        Output: list of {"file": path, "score": float, "features": dict}
        """
        ranked_results = []
        features_list = []
        
        # 1. Extract Features
        for f in file_list:
            feat = self._extract_features(f)
            if feat:
                feat['file'] = f
                features_list.append(feat)
        
        if not features_list:
            return []

        # 2. Normalize Features (Min-Max scaling for score calculation)
        # We need a reference to normalize against the batch
        keys = ['snr', 'coverage', 'nans', 'noise', 'smoothness']
        max_vals = {k: max(item[k] for item in features_list) for k in keys}
        min_vals = {k: min(item[k] for item in features_list) for k in keys}
        
        # Avoid division by zero
        for k in keys:
            if max_vals[k] == min_vals[k]:
                max_vals[k] += 1e-9

        # 3. Compute Score
        for item in features_list:
            score = 0.0
            
            # SNR (Maximize)
            norm_snr = (item['snr'] - min_vals['snr']) / (max_vals['snr'] - min_vals['snr'])
            score += self.weights['snr'] * norm_snr
            
            # Coverage (Maximize)
            norm_cov = (item['coverage'] - min_vals['coverage']) / (max_vals['coverage'] - min_vals['coverage'])
            score += self.weights['coverage'] * norm_cov
            
            # NaNs (Minimize)
            norm_nans = (item['nans'] - min_vals['nans']) / (max_vals['nans'] - min_vals['nans'])
            score += self.weights['nans'] * norm_nans
            
            # Noise (Minimize)
            norm_noise = (item['noise'] - min_vals['noise']) / (max_vals['noise'] - min_vals['noise'])
            score += self.weights['noise'] * norm_noise
            
            # Smoothness (Minimize)
            norm_smooth = (item['smoothness'] - min_vals['smoothness']) / (max_vals['smoothness'] - min_vals['smoothness'])
            score += self.weights['smoothness'] * norm_smooth
            
            # Clip score to 0-1 range (approx) via sigmoid or just raw?
            # Let's keep raw relative score for sorting, then normalize list to 0-1
            ranked_results.append({
                'file': item['file'],
                'raw_score': score,
                'features': item
            })

        # Final Sort
        ranked_results.sort(key=lambda x: x['raw_score'], reverse=True)
        
        # Format Output
        final_output = []
        if ranked_results:
             # Normalize scores 0.0 - 1.0 for user display
             best_score = ranked_results[0]['raw_score']
             worst_score = ranked_results[-1]['raw_score']
             denom = best_score - worst_score
             if denom == 0: denom = 1
             
             for res in ranked_results:
                 # Re-scale to 0-1 range based on batch min/max
                 final_score = (res['raw_score'] - worst_score) / denom
                 # Heuristic baseline: pure average files shouldn't be 0
                 # Let's just output the tuple structure needed by interface
                 
                 final_output.append(
                     (res['file'], round(final_score, 2))
                 )

        return final_output

    def get_best(self, file_list):
        """
        Returns full dict for the best file.
        """
        ranked = self.rank(file_list)
        if not ranked:
            return None
        return {
            "best_file": ranked[0][0],
            "score": ranked[0][1],
            "explanation": "Highest weighted score based on SNR, Coverage, and Data Quality."
        }
