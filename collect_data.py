#!/usr/bin/env python3
"""
collect_data.py

A script to search, filter, and download JWST 1D spectral data (x1d.fits) from MAST.
Specifically targets NIRSpec (nrs1/nrs2) and MIRI (LRS/MRS) products for biosignature analysis.

Usage:
    python collect_data.py --target "SIMP J013656.5+093347" --download
    python collect_data.py --ra 24.2354 --dec 9.5631 --radius 0.02 --download
"""

import argparse
import sys
import os
import shutil
import csv
from pathlib import Path

# Try importing astroquery; handle if missing
try:
    from astroquery.mast import Observations
    from astropy.table import Table, unique
except ImportError:
    print("Error: This script requires 'astroquery' and 'astropy'.")
    print("Please install them via: pip install astroquery astropy")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Collect JWST 1D spectra (x1d) from MAST.")
    
    # Target selection group
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--target", type=str, help="Target name to resolve (e.g. 'SIMP J013656.5+093347')")
    group.add_argument("--coords", nargs=2, type=float, metavar=('RA', 'DEC'), help="Coordinates in degrees (RA DEC)")
    
    # Alternative to group for RA/DEC
    parser.add_argument("--ra", type=float, help="RA in degrees")
    parser.add_argument("--dec", type=float, help="Dec in degrees")
    
    parser.add_argument("--radius", type=float, default=0.02, help="Search radius in degrees (default: 0.02)")
    parser.add_argument("--outdir", type=str, default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("--index", type=str, default="data/index.csv", help="Path to index file (default: data/index.csv)")
    
    # Filtering Flags
    parser.add_argument("--include-x1dints", action="store_true", help="Include x1dints files (integrations)")
    parser.add_argument("--include-mirimage", action="store_true", help="Include mirimage files (usually imaging)")
    
    # Instrument selection
    parser.add_argument("--only", type=str, choices=['nirspec', 'miri'], help="Restrict to specific instrument (overrides default)")
    parser.add_argument("--want", type=str, default="nirspec,miri", help="Comma-separated list of instruments (default: nirspec,miri)")

    parser.add_argument("--max-results", type=int, default=50, help="Max results for query (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Perform search and filtering without downloading")
    parser.add_argument("--download", action="store_true", help="Enable downloading (default is list only)")
    parser.add_argument("--clobber", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()
    
    # Consolidate RA/DEC
    if args.ra is not None and args.dec is not None:
        args.coords = [args.ra, args.dec]
    
    # Ensure loc
    if not args.target and not args.coords:
         parser.error("You must provide --target or both --ra and --dec (or --coords).")

    # Handle --only override
    if args.only:
        args.want = args.only

    return args

def search_obs(args):
    """Search MAST for JWST observations."""
    obs_table = None
    if args.verbose:
        print(f"Searching MAST for target={args.target} or coords={args.coords}, radius={args.radius} deg...")

    try:
        if args.target:
            obs_table = Observations.query_object(args.target, radius=f"{args.radius} deg")
        else:
            ra, dec = args.coords
            obs_table = Observations.query_region(f"{ra} {dec}", radius=f"{args.radius} deg")
    except Exception as e:
        print(f"Error querying MAST: {e}")
        return None

    if obs_table is None or len(obs_table) == 0:
        print("No observations found.")
        return None

    # Filter by obs_collection == "JWST"
    if 'obs_collection' in obs_table.colnames:
        obs_table = obs_table[obs_table['obs_collection'] == 'JWST']
    
    if len(obs_table) == 0:
        print("No JWST observations found.")
        return None

    if args.verbose:
        print(f"Found {len(obs_table)} JWST observations.")
    
    return obs_table

def filter_products(products, args):
    """
    Filter products based on strict criteria and specific flags.
    """
    wanted_insts = [w.strip().lower() for w in args.want.split(',')]
    
    filtered_indices = []
    
    if args.verbose:
        print(f"Filtering {len(products)} products...")

    # For deduplication later
    candidates = []

    for i, row in enumerate(products):
        filename = row['productFilename']
        fname_lower = filename.lower()
        
        # 1. Basic Extension Check
        if not fname_lower.endswith('.fits'):
            continue
            
        # 2. Must contain x1d
        if 'x1d' not in fname_lower:
            continue
            
        # 3. Exclude 'x1dints' unless requested
        if not args.include_x1dints and 'x1dints' in fname_lower:
            continue
            
        # 4. Exclude 'mirimage' unless requested
        # 'mirimage' often appears in filename
        if not args.include_mirimage and 'mirimage' in fname_lower:
            continue

        # 5. Instrument & Mode Logic
        
        is_nirspec = False
        is_miri = False
        
        # NIRSpec Logic
        # Accept: *_nrs1_x1d.fits, *_nrs2_x1d.fits, *_nirspec_*_x1d.fits
        if 'nirspec' in fname_lower or 'nrs' in fname_lower:
             # Basic identification
             pass

        if 'nrs1_x1d' in fname_lower or 'nrs2_x1d' in fname_lower:
            is_nirspec = True
        elif 'nirspec' in fname_lower and 'x1d' in fname_lower:
            # Fallback for generic naming if it exists, check for nrs presence?
            # User said: *_nirspec_*_x1d.fits is OK.
            is_nirspec = True
            
        # MIRI Logic
        # Accept: *_miri_*_x1d.fits IF LRS/MRS (slitlessprism, mrs, p750l)
        # AND NOT mirimage (already handled above)
        if 'miri' in fname_lower or 'mir' in fname_lower:
            # Check for allowed modes 
            allowed_modes = ['slitlessprism', 'mrs', 'p750l'] 
            # p750l is LRS, mrs is MRS. slitlessprism is LRS.
            # check if any allowed mode string is in filename
            if any(mode in fname_lower for mode in allowed_modes):
                is_miri = True
            
            # If specifically asked to include miri and it's x1d, maybe accept if strict check fails?
            # User requirement: "принимать *_miri_*_x1d.fits но только если это LRS/MRS"
            # So if it doesn't have these tags, we reject.
        
        # Final decision based on unwanted instruments
        keep = False
        category = "unknown"
        
        if is_nirspec and 'nirspec' in wanted_insts:
            keep = True
            category = "NIRSpec"
        elif is_miri and 'miri' in wanted_insts:
            keep = True
            category = "MIRI"
            
        if keep:
            candidates.append({
                'index': i,
                'filename': filename,
                'size': row['size'],
                'category': category
            })

    # Deduplication
    # Group by filename
    unique_map = {}
    for cand in candidates:
        fn = cand['filename']
        if fn not in unique_map:
            unique_map[fn] = cand
        else:
            # Conflict: Prefer larger size (more data?) or just first.
            # User: "выбрать ту, у которой size больше (или первую)"
            if cand['size'] > unique_map[fn]['size']:
                unique_map[fn] = cand
    
    # Collect indices
    final_indices = [v['index'] for v in unique_map.values()]
    filtered_products = products[final_indices]
    
    # Stats for Dry Run / Reporting
    nirspec_count = sum(1 for v in unique_map.values() if v['category'] == 'NIRSpec')
    miri_count = sum(1 for v in unique_map.values() if v['category'] == 'MIRI')
    total_size = sum(v['size'] for v in unique_map.values())
    
    print(f"Selected {len(filtered_products)} unique files:")
    print(f"  NIRSpec x1d: {nirspec_count}")
    print(f"  MIRI x1d:    {miri_count}")
    print(f"  Total Size:  {total_size/1024/1024:.2f} MB")
    
    return filtered_products

def build_index(manifest, outdir, index_path, verbose=False):
    """Build index.csv from downloaded manifest."""
    # ... (Same logic as before, just ensuring robustness) ...
    # We will just traverse the directories as before.
    if verbose:
        print(f"Building index at {index_path}...")
    
    header = ['filename', 'local_path', 'instrument', 'subinstrument', 'target', 'obs_id', 'wmin', 'wmax', 'note']
    
    with open(index_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for inst in ['nirspec', 'miri']:
            d = Path(outdir) / inst
            if not d.exists():
                continue
                
            for path in d.glob('*.fits'):
                fname = path.name
                
                # Defaults
                target = "Unknown"
                obs_id = "Unknown"
                wmin = 0.0
                wmax = 0.0
                subinst = "Unknown"
                instrument = "NIRSpec" if inst == "nirspec" else "MIRI"
                note = "O3 coverage: yes" if instrument == "MIRI" else "no coverage for O3"

                try:
                    from astropy.io import fits
                    with fits.open(path) as hdul:
                        hdr = hdul[0].header
                        target = hdr.get('TARGNAME', target)
                        obs_id = hdr.get('OBS_ID', obs_id)
                        if instrument == "NIRSpec":
                            subinst = hdr.get('DETECTOR', 'Unknown')
                        else:
                            subinst = hdr.get('EXP_TYPE', hdr.get('SUBARRAY', 'Unknown'))

                        if len(hdul) > 1:
                            data = hdul[1].data
                            if 'WAVELENGTH' in data.columns.names:
                                wav = data['WAVELENGTH']
                                wmin = min(wav)
                                wmax = max(wav)
                except Exception:
                    pass

                writer.writerow([fname, str(path), instrument, subinst, target, obs_id, wmin, wmax, note])
                
    print(f"Index created at {index_path}")

def organize_files(manifest, outdir, verbose=False):
    """Organize files into subdirectories."""
    nirspec_dir = Path(outdir) / 'nirspec'
    miri_dir = Path(outdir) / 'miri'
    nirspec_dir.mkdir(parents=True, exist_ok=True)
    miri_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for row in manifest:
        if row['Status'] != 'COMPLETE':
            continue
        src = Path(row['Local Path'])
        fname = src.name.lower()
        
        dest_folder = None
        if 'nrs1' in fname or 'nrs2' in fname or 'nirspec' in fname:
            dest_folder = nirspec_dir
        elif 'miri' in fname or 'mir' in fname:
            dest_folder = miri_dir
            
        if dest_folder:
            dest = dest_folder / src.name
            if verbose:
                print(f"Moving {src} -> {dest}")
            shutil.move(str(src), str(dest))
            moved_count += 1
            
    print(f"Organized {moved_count} files into {outdir}")

def main():
    args = parse_args()
    
    # 1. Search
    obs_table = search_obs(args)
    if obs_table is None:
        return

    # 2. Get Products
    if args.verbose:
        print("Retrieving product list...")
    products = Observations.get_product_list(obs_table)
    
    # 3. Filter & Deduplicate
    filtered_products = filter_products(products, args)
    
    if len(filtered_products) == 0:
        print("No suitable products found.")
        return

    # 4. Dry Run / Download
    if args.dry_run or not args.download:
        print("\n[Dry Run / List Only] Files to be downloaded:")
        for row in filtered_products:
            print(f" - {row['productFilename']} ({row['size']/1024/1024:.2f} MB)")
        print("\nUse --download to fetch them.")
        return

    # 5. Download
    print(f"\nDownloading {len(filtered_products)} files to {args.outdir}...")
    manifest = Observations.download_products(filtered_products, download_dir=args.outdir, mrp_only=False)
    
    # 6. Organize & Index
    organize_files(manifest, args.outdir, args.verbose)
    build_index(manifest, args.outdir, args.index, args.verbose)

if __name__ == "__main__":
    main()
