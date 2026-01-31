
import argparse
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
import warnings

def normalize_column_name(col_name: str) -> str:
    """Normalize column name for comparisons (uppercase, stripped)."""
    return col_name.strip().upper()

def find_spectral_columns(data_columns, candidates):
    """Find the first matching column name from candidates in the data columns."""
    data_cols_normalized = {normalize_column_name(c): c for c in data_columns}
    for candidate in candidates:
        cand_norm = normalize_column_name(candidate)
        if cand_norm in data_cols_normalized:
            return data_cols_normalized[cand_norm]
    return None

def read_fits_wavelength_unit(hdu, col_name):
    """Attempt to retrieve units for the wavelength column from FITS header."""
    # Find the column index (1-based for TUNITn)
    col_idx = None
    for i, name in enumerate(hdu.columns.names):
        if name == col_name:
            col_idx = i + 1
            break
    
    if col_idx is None:
        return None

    unit_key = f'TUNIT{col_idx}'
    if unit_key in hdu.header:
        unit_str = hdu.header[unit_key]
        try:
            return u.Unit(unit_str)
        except ValueError:
            return None
    return None

def read_spectrum(filename: str, hdu_index=None, mask_dq=False):
    """
    Reads a 1D spectrum from a FITS file with enhanced robustness.
    
    Args:
        filename (str): Path to the FITS file.
        hdu_index (int, optional): Force specific HDU index. If None, auto-search.
        mask_dq (bool): If True, filter out points where DQ > 0.
        
    Returns:
        dict: Containing 'wavelength_um', 'flux', 'err', 'dq', 'meta', 'source_hdu', 'columns_used'.
    """
    
    WAVELENGTH_CANDIDATES = ['WAVELENGTH', 'LAMBDA', 'WAVE', 'LOGLAM', 'WAVELENGTH_UM', 'MICRONS']
    FLUX_CANDIDATES = ['FLUX', 'FNU', 'FLAM', 'SCI', 'FLUX_DENSITY', 'INTENSITY']
    ERR_CANDIDATES = ['FLUX_ERROR', 'ERROR', 'ERR', 'SIGMA', 'UNC', 'UNCERTAINTY', 'FLUX_ERR']
    DQ_CANDIDATES = ['DQ', 'QUALITY', 'MASK', 'FLAGS', 'BITMASK']

    try:
        with fits.open(filename) as hdul:
            target_hdu = None
            found_cols = {}
            found_hdu_idx = -1

            # Determine HDUs to search
            if hdu_index is not None:
                search_list = [(hdu_index, hdul[hdu_index])]
            else:
                # Search all BinTableHDUs
                search_list = []
                for idx, hdu in enumerate(hdul):
                    if isinstance(hdu, fits.BinTableHDU):
                        search_list.append((idx, hdu))
            
            # Search logic
            for idx, hdu in search_list:
                col_names = hdu.columns.names
                wave_col = find_spectral_columns(col_names, WAVELENGTH_CANDIDATES)
                flux_col = find_spectral_columns(col_names, FLUX_CANDIDATES)

                if wave_col and flux_col:
                    target_hdu = hdu
                    found_hdu_idx = idx
                    found_cols['wavelength'] = wave_col
                    found_cols['flux'] = flux_col
                    # Optional columns
                    found_cols['err'] = find_spectral_columns(col_names, ERR_CANDIDATES)
                    found_cols['dq'] = find_spectral_columns(col_names, DQ_CANDIDATES)
                    
                    # If we found a valid one and we are in auto-mode, pick it.
                    # Heuristic: Prefer HDU=1 if multiple match? Or just first found?
                    # Usually EXT=1 is science. Let's take the first reasonable one.
                    break
            
            if target_hdu is None:
                msg = f"No spectral table found in {filename}."
                if hdu_index is not None:
                    msg += f" (Checked HDU {hdu_index})"
                raise ValueError(msg)

            # Load data from HDU
            data = target_hdu.data
            wave_raw = data[found_cols['wavelength']].flatten()
            flux_raw = data[found_cols['flux']].flatten()
            
            err_raw = None
            if found_cols.get('err'):
                err_raw = data[found_cols['err']].flatten()
            
            dq_raw = None
            if found_cols.get('dq'):
                dq_raw = data[found_cols['dq']].flatten()
            else:
                dq_raw = np.zeros_like(wave_raw, dtype=int)

            # Unit conversion for wavelength
            wave_unit = read_fits_wavelength_unit(target_hdu, found_cols['wavelength'])
            wave_um = wave_raw.copy()
            
            if wave_unit is not None:
                try:
                    # Convert to micrometers
                    if wave_unit != u.um:
                         # Use equivalencies for spectral (e.g. Hz -> um) if needed, 
                         # but mostly we expect length units.
                         wave_um = (wave_raw * wave_unit).to(u.um, equivalencies=u.spectral()).value
                except (u.UnitConversionError, ValueError):
                    # Fallback: try to guess from range? 
                    # If mean > 1000, likely Angstroms.
                    if np.mean(wave_raw) > 3000:
                         warnings.warn("Unit conversion failed, but values look like Angstroms. Dividing by 10000.")
                         wave_um = wave_raw / 10000.0
                    else:
                         warnings.warn(f"Could not convert {wave_unit} to um. Using raw values.")
            else:
                # No unit. Heuristic check.
                # JWST usually microns. 
                # If values are huge, maybe Angstroms.
                if np.nanmedian(wave_raw) > 5000:
                    warnings.warn("No units found, but values look like Angstroms. Assuming Angstroms -> um.")
                    wave_um = wave_raw / 10000.0

            # Metadata extraction
            meta = {
                'filename': filename,
                'hdu_idx': found_hdu_idx,
                'header': dict(target_hdu.header)
            }
            # Extract basic instrument/target info if available
            # Check both Primary and Science Heaaders
            combined_header = hdul[0].header.copy()
            combined_header.update(target_hdu.header)
            
            for key in ['INSTRUME', 'TELESCOP', 'TARGNAME', 'OBJECT', 'GRATING', 'FILTER', 'EXP_TYPE']:
                if key in combined_header:
                    meta[key.lower()] = combined_header[key]

            # Initial cleanup: Remove NaNs and Infs from Wave and Flux
            valid_mask = np.isfinite(wave_um) & np.isfinite(flux_raw) & (wave_um > 0)
            
            if err_raw is not None:
                # Fill NaN errors with Infinity (ignore point) or huge number?
                # Better to mask them.
                valid_mask &= np.isfinite(err_raw)
                # Ensure non-negative error?
                # valid_mask &= (err_raw >= 0) # Some pipelines might have -1 for bad pixels.
            
            # DQ Masking
            if mask_dq and dq_raw is not None:
                valid_mask &= (dq_raw == 0)
            
            # Apply mask
            wave_clean = wave_um[valid_mask]
            flux_clean = flux_raw[valid_mask]
            err_clean = err_raw[valid_mask] if err_raw is not None else None
            dq_clean = dq_raw[valid_mask] if dq_raw is not None else None

            # Sort by wavelength
            sort_idx = np.argsort(wave_clean)
            wave_sorted = wave_clean[sort_idx]
            flux_sorted = flux_clean[sort_idx]
            err_sorted = err_clean[sort_idx] if err_clean is not None else None
            dq_sorted = dq_clean[sort_idx] if dq_clean is not None else None

            # Final check for valid data
            if len(wave_sorted) == 0:
                warnings.warn(f"No valid data points remaining in {filename} after cleaning/masking.")

            return {
                "wavelength_um": wave_sorted,
                "flux": flux_sorted,
                "err": err_sorted,
                "dq": dq_sorted,
                "meta": meta,
                "source_hdu": found_hdu_idx,
                "columns_used": found_cols
            }

    except Exception as e:
        raise RuntimeError(f"Failed to read {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Read 1D FITS spectrum and print statistics.")
    parser.add_argument("file", help="Path to the FITS file")
    parser.add_argument("--hdu", type=int, default=None, help="Specify HDU index to read")
    parser.add_argument("--mask-dq", action="store_true", help="Filter out data points with DQ > 0")
    
    args = parser.parse_args()

    try:
        data = read_spectrum(args.file, hdu_index=args.hdu, mask_dq=args.mask_dq)
        
        meta = data['meta']
        cols = data['columns_used']
        wave = data['wavelength_um']
        flux = data['flux']
        dq = data['dq']

        print(f"File: {meta['filename']}")
        print(f"Source HDU Index: {data['source_hdu']}")
        print(f"Instrument: {meta.get('instrume', 'N/A')}")
        print(f"Target: {meta.get('targname', meta.get('object', 'N/A'))}")
        print(f"Columns Found: {cols}")
        
        if len(wave) > 0:
            print(f"Wavelength Range: {wave.min():.4f} - {wave.max():.4f} um")
            print(f"Points: {len(wave)}")
            print(f"Mean Flux: {np.mean(flux):.4e}")
            
            if dq is not None:
                bad_pixels = np.count_nonzero(dq)
                print(f"DQ flagged points (>0): {bad_pixels} ({bad_pixels/len(dq):.1%})")
                if args.mask_dq:
                    print(" (Note: DQ masked points are excluded from Stats)")
            
            if data['err'] is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    snr = flux / data['err']
                    print(f"Median SNR: {np.nanmedian(snr):.2f}")
        else:
            print("No valid data points found.")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def write_spectrum(filename, wavelength, flux, err=None, header_dict=None):
    """
    Writes a 1D spectrum to a FITS file.
    
    Args:
        filename (str): Output filename.
        wavelength (array): Wavelength array (um).
        flux (array): Flux array.
        err (array, optional): Error array.
        header_dict (dict, optional): Dictionary of header keywords.
    """
    # Create Columns
    col_defs = []
    col_defs.append(fits.Column(name='WAVELENGTH', format='D', unit='um', array=wavelength))
    col_defs.append(fits.Column(name='FLUX', format='D', unit='Jy', array=flux))
    
    if err is not None:
        col_defs.append(fits.Column(name='ERROR', format='D', unit='Jy', array=err))
        
    hdu = fits.BinTableHDU.from_columns(col_defs)
    
    # Update Header
    if header_dict:
        for k, v in header_dict.items():
            # Truncate strings if needed to avoid FITS errors, simplistic approach
            if isinstance(v, str) and len(v) > 68:
                v = v[:68]
            try:
                hdu.header[k] = v
            except Exception:
                pass # Ignore invalid keywords
                
    # Create Primary HDU
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu, hdu])
    
    try:
        hdul.writeto(filename, overwrite=True)
    except Exception as e:
        raise IOError(f"Failed to write FITS file {filename}: {e}")

if __name__ == "__main__":
    main()
