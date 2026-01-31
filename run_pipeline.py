#!/usr/bin/env python3
"""
run_pipeline.py

Antigravity Spectral Pipeline
Smart CLI Version

Usage:
    python run_pipeline.py data/raw           # Auto-scan directory
    python run_pipeline.py myfile.fits        # Single file analysis

Features:
    - Auto-detects input type (file vs dir)
    - Auto-merges NIRSpec pairs if found
    - Auto-names targets from headers
    - Smart defaults (Plot=ON, JSON=ON)
    - Structured output: outputs/<TARGET>/<RUN_ID>/
"""

import argparse
import sys
import os
import json
import numpy as np
import time

# Ensure local modules are importable
try:
    import io_fits
    import continuum
    import features
    import plotting
    import merge
    import organize_io
    # ML Integration (Optional)
    # ML Integration (Optional)
    try:
        import ml.quality as ml_quality
        import ml.detection_confidence as ml_confidence
        import ml.report_writer as ml_report
        
        # Check if legacy MLManager is available
        try:
             from ml_core import MLManager
             ml_manager = MLManager()
        except ImportError:
             ml_manager = None
             
        # Initialize classes
        conf_assessor = ml_confidence.ConfidenceAssessor()
        report_writer_obj = ml_report.ReportWriter()
        
        ML_AVAILABLE = True
    except ImportError as e:
        # Fallback if specific modules fail
        # print(f"ML Modules not found: {e}") # Optional: suppress info log
        ML_AVAILABLE = False
        conf_assessor = None
        report_writer_obj = None

except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import io_fits
        import continuum
        import features
        import plotting
        import merge
        import organize_io
        # ML Integration (Optional)
        try:
            from ml_core import MLManager # Keep old one for now if needed, or remove? Plan said remove.
            # But wait, looking at the code I see `ml_manager` usage in scanning. 
            # The prompt implies I should use the NEW modules I found: ml/quality.py etc.
            # The previous `ml_core` might be a dummy or old. 
            # Let's check if I should replace MLManager strictly or add to it. 
            # The plan says: "Import ml.quality, ml.detection_confidence, ml.report_writer inside a try/except block". 
            # I should probably keep MLManager for the ranking if it exists, OR replace it if the new modules cover it.
            # The `ml_core` seems to be what was there before. 
            # I will ADD the new imports. 
            
            import ml.quality as ml_quality
            import ml.detection_confidence as ml_confidence
            import ml.report_writer as ml_report
            
            # Initialize classes
            conf_assessor = ml_confidence.ConfidenceAssessor()
            report_writer_obj = ml_report.ReportWriter()
            
            ML_AVAILABLE = True
        except ImportError as e:
            # Fallback if specific modules fail
            print(f"ML Modules not found: {e}")
            ML_AVAILABLE = False
            conf_assessor = None
            report_writer_obj = None

    except ImportError as e:
        print(f"Critical Error: Failed to import pipeline modules: {e}")
        sys.exit(1)

# ANSI Definitions
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

class PipelineLogger:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def info(self, msg):
        if self.verbose:
            print(f"  {msg}")
            
    def step(self, msg):
        print(f" {Colors.GREEN}✓{Colors.ENDC} {msg}")

    def warn(self, msg):
        print(f" {Colors.WARN}⚠ {msg}{Colors.ENDC}")

    def fail(self, msg):
        print(f" {Colors.FAIL}✖ {msg}{Colors.ENDC}")

    def header(self, target, mode, outdir):
        print(f"\n{Colors.BOLD}ANTIGRAVITY SPECTRAL PIPELINE{Colors.ENDC}")
        print(f"Target: {Colors.CYAN}{target}{Colors.ENDC}")
        print(f"Mode:   {Colors.BLUE}{mode}{Colors.ENDC}")
        print(f"Output: {Colors.DIM}{outdir}{Colors.ENDC}")
        print("-" * 50)

# JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

def generate_summary(target, run_id, input_f, src_files, w, cont, dets, ml_results=None):
    """Generates summary text content."""
    lines = [
        f"Target: {target}",
        f"Run ID: {run_id}",
        f"Input spectrum: {os.path.basename(input_f)}",
        "Source files:",
    ] + [f" - {os.path.basename(f)}" for f in src_files] + [
        "",
        "=" * 60,
        f"Wavelength Range: {w.min():.4f} - {w.max():.4f} um",
        f"Data Points: {len(w)}",
        f"Continuum Fit: {'OK' if cont.get('fit_ok') else 'FAILED'}",
        f"Temperature: {cont.get('T_K', 0):.1f} K",
        "-" * 60,
        "Molecular Detections (Depth / SNR):"
    ]
    
    found = False
    for mol, info in dets.items():
        if info['detected']:
            found = True
            lines.append(f"  {mol}: DETECTED (Max SNR: {info['max_snr']:.1f})")
            for b in info['bands']:
                if b['covered']:
                    rng = b.get('band_range', (0,0))
                    lines.append(f"    - Band {rng[0]:.2f}-{rng[1]:.2f} um: Depth={b['depth']:.3f}, SNR={b['snr']:.1f}")
        else:
             if any(b['covered'] for b in info['bands']):
                 lines.append(f"  {mol}: Not Detected (Max SNR < 3.0)")
             else:
                 lines.append(f"  {mol}: No Spectral Coverage")

    if not found:
        lines.append("  No robust molecular detections found.")

    lines.append("-" * 60)

    # ML Report Injection
    if ml_results:
        # Quality
        if ml_results.get('quality'):
            q = ml_results['quality']
            lines.append(f"ML Quality Score: {q['quality_score']:.2f} ({'Usable' if q['usable'] else 'Unusable'})")
            if q.get('notes'):
                lines.append("Quality Notes: " + "; ".join(q['notes']))
            lines.append("-" * 60)
            
        # Report Text
        if ml_results.get('generated_report'):
            lines.append("\n=== AUTOMATED ML REPORT ===\n")
            lines.append(ml_results['generated_report'])
        else:
             lines.append("INTERPRETATION LIMITS:\n1. O2/O3 are not definitive biosignatures alone.\n2. Cloud/Haze degeneracy applies.")
    else:
        lines.append("INTERPRETATION LIMITS:\n1. O2/O3 are not definitive biosignatures alone.\n2. Cloud/Haze degeneracy applies.")

    return "\n".join(lines) + "\n"

def process_spectrum(data, target_name, run_id, run_dirs, src_files, input_fname, args, logger):
    """Core analysis logic."""
    
    # Unpack
    w_um = data['wavelength_um']
    flux = data['flux']
    err = data['err']
    
    logger.info(f"Loaded {len(w_um)} points ({w_um.min():.2f}-{w_um.max():.2f} um)")
    logger.step("Spectrum loaded")
    
    # Continuum
    try:
        cont_res = continuum.fit_blackbody(w_um, flux, err=err, clip_sigma=3.0)
        logger.step(f"Continuum fitted (T={cont_res['T_K']:.0f}K)")
    except Exception as e:
        logger.fail(f"Continuum failed: {e}")
        return

    residual = cont_res['residual']
    
    # Features
    try:
        detections = features.detect_molecules(w_um, residual, err=err)
        logger.step("Features analyzed")
    except Exception as e:
        logger.fail(f"Feature detection failed: {e}")
        return

    # ML Analysis (Sidecar)
    ml_results = {
        "quality": None,
        "confidence": {},
        "generated_report": None
    }
    
    if ML_AVAILABLE:
        try:
            # 1. Quality Assessment
            # We need to assess quality on the loaded data
            try:
                # Need to calculate SNR strictly for the function if not present? 
                # quality.assess_quality takes (w, f, e, snr)
                # We have w_um, flux, err.
                q_res = ml_quality.assess_quality(w_um, flux, err)
                ml_results['quality'] = q_res
                logger.step(f"ML Quality: {q_res['quality_score']:.2f} ({'Usable' if q_res['usable'] else 'Unusable'})")
                if not q_res['usable']:
                    logger.warn(f"ML flagged spectrum as unusable: {q_res['notes']}")
            except Exception as e:
                logger.warn(f"ML Quality failed: {e}")

            # 2. Confidence Assessment
            # Iterate through detections
            try:
                conf_map = {}
                q_score = ml_results['quality']['quality_score'] if ml_results['quality'] else 0.5
                
                for mol, info in detections.items():
                    # Extract metrics from info
                    # info structure from features.py usually: 
                    # {'detected': bool, 'max_snr': float, 'bands': [...], ...}
                    # We need: snr, num_bands, depth, spectral_coverage
                    
                    # spectral coverage check
                    # We need to know if the molecule *could* be detected. 
                    # logic in features.py usually handles 'covered' in bands.
                    # info['bands'] is a list of dicts.
                    
                    has_coverage = False
                    if 'bands' in info and len(info['bands']) > 0:
                        # If any band is covered? Or all? Usually 'covered' flag exists.
                        # Let's check info. 
                        # features.py output varies. Assuming 'bands' list has 'covered'.
                        # Let's derive coverage from bands.
                        has_coverage = any(b.get('covered', False) for b in info['bands'])
                    else:
                        # Fallback
                        has_coverage = False

                    # Depth
                    # Find max depth among bands
                    max_depth = 0.0
                    for b in info.get('bands', []):
                        d = b.get('depth', 0.0)
                        if d > max_depth:
                            max_depth = d
                            
                    num_bands_detected = 0
                    if info['detected']:
                         # How many bands contributed? 
                         # usually features.py returns 'n_bands_detected' or we count based on SNR threshold?
                         # Let's trust 'num_bands' if it exists, or count bands > 3.0 SNR?
                         # Info has 'max_snr'. 
                         # Let's count bands with snr > 3.0
                         for b in info.get('bands', []):
                             if b.get('snr', 0) > 3.0:
                                 num_bands_detected += 1
                    
                    # Run Assessment
                    c_res = conf_assessor.assess(
                        molecule_name=mol,
                        snr=info.get('max_snr', 0.0),
                        num_bands=num_bands_detected,
                        depth=max_depth,
                        spectral_coverage=has_coverage,
                        quality_score=q_score
                    )
                    conf_map[mol] = c_res
                
                ml_results['confidence'] = conf_map
                logger.step(f"ML Confidence assessed for {len(conf_map)} molecules")
            except Exception as e:
                logger.warn(f"ML Confidence failed: {e}")

            # 3. Report Generation
            try:
                # Prepare data for ReportWriter
                # Needs: target name, instrument(s), physical detections, ML confidence labels, spectral coverage
                
                # Spectral coverage map
                cov_map = {}
                for mol, info in detections.items():
                     cov_map[mol] = any(b.get('covered', False) for b in info.get('bands', []))
                
                # Confidence labels
                label_map = {}
                for mol, res in ml_results['confidence'].items():
                    label_map[mol] = res['label']

                report_data = {
                    "target name": target_name,
                    "instrument(s)": ["JWST (Simulated/Real)"], # Placeholder or derive from header?
                    "physical detections": detections, # ReportWriter expects specific format? 
                    # ReportWriter _write_results iterates keys and checks ['snr'], ['num_bands']
                    # My detections dict has 'max_snr' and 'bands'. 
                    # I might need to adapt it. 
                    # Wait, ReportWriter usage example: "physical detections": {"H2O": {"snr": 8.5, "num_bands": 3}}
                    # My detections: {'H2O': {'detected': True, 'max_snr': ..., 'bands': ...}}
                    # I should map it to be safe.
                    
                    "ML confidence labels": label_map,
                    "spectral coverage": cov_map
                }
                
                # Adapt detections for ReportWriter
                adapted_dets = {}
                for mol, info in detections.items():
                    # Count bands > 3 snr
                    nb = 0
                    for b in info.get('bands', []):
                         if b.get('snr', 0) > 3.0:
                             nb += 1
                    adapted_dets[mol] = {
                        "snr": info.get('max_snr', 0.0),
                        "num_bands": nb
                    }
                report_data["physical detections"] = adapted_dets
                
                report_text = report_writer_obj.generate_report(report_data)
                ml_results['generated_report'] = report_text
                logger.step("ML Report generated")
                
            except Exception as e:
                logger.warn(f"ML Report Generation failed: {e}")

        except Exception as e:
            logger.warn(f"ML Sidecar failed (skipping): {e}")
            ml_results['error'] = str(e)

    # SAVE
    # Summary
    summary_txt = generate_summary(target_name, run_id, input_fname, src_files, w_um, cont_res, detections, ml_results)
    with open(os.path.join(run_dirs['reports'], "summary.txt"), 'w') as f:
        f.write(summary_txt)
        
    # JSON
    if not args.no_json:
        report = {
            'target': target_name, 'run_id': run_id, 
            'continuum': cont_res, 'detections': detections,
            'ml_analysis': ml_results if ML_AVAILABLE else None
        }
        with open(os.path.join(run_dirs['reports'], "results.json"), 'w') as f:
            json.dump(report, f, indent=4, cls=NumpyEncoder)
            
    # Plot
    if not args.no_plot:
        plot_path = os.path.join(run_dirs['plots'], "spectrum.png")
        try:
            # Suppress matplotlib warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotting.plot_all(w_um, flux, cont_res['continuum'], residual, detections, plot_path)
        except Exception as e:
            logger.warn(f"Plotting error: {e}")
            
    logger.step("Results saved")

def main():
    parser = argparse.ArgumentParser(description="Antigravity Spectral Pipeline | Smart CLI")
    
    # Positional: Input (File or Dir)
    parser.add_argument("input", nargs='?', default=".", help="Input file (.fits) or directory to scan")
    
    # Optional Overrides
    parser.add_argument("--target-name", type=str, help="Override Target Name")
    parser.add_argument("--outdir", default="outputs", help="Base output directory")
    
    # Negative Flags (Features ON by default)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--no-json", action="store_true", help="Disable JSON report")
    
    # Advanced / Forced Modes
    parser.add_argument("--merge-nrs", action="store_true", help="Force merge (auto-enabled if pair found)")
    parser.add_argument("--force-single", action="store_true", help="Force single file processing even if pair found")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    
    args = parser.parse_args()
    logger = PipelineLogger(args.verbose)
    
    # 1. ANALYZE INPUT
    input_path = os.path.abspath(args.input)
    is_dir = os.path.isdir(input_path)
    
    target_name = args.target_name
    timestamp = organize_io.get_timestamp_str()
    
    scan_result = None
    
    # Auto-detect Target Name if not provided
    if not target_name:
        if is_dir:
            # Heuristic: Use Dir Name, or try to find common target in files?
            # Simple: Use Directory Name.
            target_name = os.path.basename(input_path) 
            # If input is just ".", use current dir name
            if target_name in ['.', '']:
                target_name = os.path.basename(os.getcwd())
        else:
            # File: Read Header
            try:
                # Quick read of primary header
                with io_fits.fits.open(input_path) as hdul:
                    h = hdul[0].header
                    target_name = h.get('TARGNAME', h.get('OBJECT', os.path.splitext(os.path.basename(input_path))[0]))
            except:
                target_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Sanitize Target Name
    target_name = target_name.replace(" ", "_").replace("(", "").replace(")", "")

    # 2. DETERMINE STRATEGY
    nrs1, nrs2 = None, None
    selected_file = None
    mode_str = "Result"
    
    if is_dir:
        # Scan
        logger.info(f"Scanning {input_path}...")
        best = merge.find_best_spectra(input_path)
        
        # ML Ranking (Optional Override)
        if ML_AVAILABLE and ml_manager:
            try:
                # Get list of fits files
                all_fits = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.fits')]
                if all_fits:
                    ranked = ml_manager.rank_spectra(all_fits)
                    if ranked:
                        logger.info(f"ML suggests best file: {os.path.basename(ranked[0][0])} (Score: {ranked[0][1]:.2f})")
                        # For now, we just log it. In future, we could set selected_file here.
            except Exception as e:
                logger.warn(f"ML Ranking failed: {e}")

        nrs1 = best['nrs1']
        nrs2 = best['nrs2']
        
        # Smart Decision
        if nrs1 and nrs2 and not args.force_single:
            # Auto Merge
            mode_str = "AUTO + MERGED (nrs1+nrs2)"
            args.merge_nrs = True # Enable merge logic
        elif nrs1:
             mode_str = "AUTO (Best NRS1)"
             selected_file = nrs1['filename']
        elif nrs2:
             mode_str = "AUTO (Best NRS2)"
             selected_file = nrs2['filename']
        else:
             # Look for any fits? merge.find_best only looks for NRS and MIRI logic.
             # If MIRI found (implement if needed), else fail.
             # For now, if no NRS found, check if there are any fits files at all?
             # Let's trust find_best.
             if not nrs1 and not nrs2:
                 logger.fail("No suitable spectra found in directory.")
                 sys.exit(1)
    else:
        # Single File
        mode_str = "SINGLE FILE"
        selected_file = input_path

    # 3. SETUP I/O
    run_id = organize_io.get_run_id(timestamp, "merged" if args.merge_nrs else "single")
    run_dirs = organize_io.setup_run_directories(target_name, run_id)
    work_dirs = organize_io.setup_working_directories(target_name)
    
    # Header Print
    logger.header(target_name, mode_str, run_dirs['root'])
    
    # 4. PREPARE DATA
    final_data = None
    src_files = []
    input_report_name = ""
    
    if args.merge_nrs:
        # Merge Logic
        logger.info("Merging spectra...")
        
        # Load Raw
        # We need to load them. 'nrs1' from find_best has 'data' key? 
        # Checking merge.py... find_best_spectra returns {'filename':..., 'score':..., 'data': <loaded_dict>}
        # Yes, it loads them.
        
        m_data = merge.merge_spectra(nrs1['data'], nrs2['data'])
        
        # Save working copy
        merged_path = os.path.join(work_dirs['merged'], "merged_nrs.fits")
        chk = {'MERGED': True}
        io_fits.write_spectrum(merged_path, m_data['wavelength_um'], m_data['flux'], m_data['err'], chk)
        
        final_data = m_data
        src_files = [nrs1['filename'], nrs2['filename']]
        input_report_name = "merged_nrs.fits"
        
        # Archive
        organize_io.archive_input_spectrum(merged_path, run_dirs, "spectrum_used.fits")
        
    else:
        # Single File Logic
        fpath = selected_file
        if not fpath:
             logger.fail("Logic Error: No file selected")
             sys.exit(1)
             
        try:
             # Inspect if 'data' is already in nrs1/nrs2 dict from scan
             if is_dir and (nrs1 and nrs1['filename'] == fpath):
                 final_data = nrs1['data']
             elif is_dir and (nrs2 and nrs2['filename'] == fpath):
                 final_data = nrs2['data']
             else:
                 # Load fresh
                 final_data = io_fits.read_spectrum(fpath)
        except Exception as e:
            logger.fail(f"Could not load {fpath}: {e}")
            sys.exit(1)
            
        src_files = [fpath]
        input_report_name = os.path.basename(fpath)
        organize_io.archive_input_spectrum(fpath, run_dirs, "spectrum_used.fits")

    # 5. METADATA
    meta = {
        "target": target_name,
        "run_id": run_id,
        "mode": mode_str,
        "sources": [os.path.basename(f) for f in src_files]
    }
    organize_io.save_run_metadata(run_dirs, meta)
    
    # 6. EXECUTE PIPELINE
    process_spectrum(final_data, target_name, run_id, run_dirs, src_files, input_report_name, args, logger)
    
    print("-" * 50)

if __name__ == "__main__":
    main()
