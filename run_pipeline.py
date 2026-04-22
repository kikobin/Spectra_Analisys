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
from pathlib import Path

# Ensure local modules are importable
try:
    import io_fits
    import continuum
    import features
    import plotting
    import merge
    import organize_io
    import demo_data
    # ML Integration (Optional)
    # ML Integration (Optional)
    try:
        import ml.quality as ml_quality
        import ml.detection_confidence as ml_confidence
        import ml.report_writer as ml_report

        conf_assessor = ml_confidence.ConfidenceAssessor()
        report_writer_obj = ml_report.ReportWriter()

        ML_AVAILABLE = True
    except ImportError as e:
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
        import demo_data
        # ML Integration (Optional)
        try:
            import ml.quality as ml_quality
            import ml.detection_confidence as ml_confidence
            import ml.report_writer as ml_report

            conf_assessor = ml_confidence.ConfidenceAssessor()
            report_writer_obj = ml_report.ReportWriter()

            ML_AVAILABLE = True
        except ImportError:
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
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

def _classify_object(T_K):
    """Returns spectral type label and H2O/O2 expectation based on temperature."""
    if T_K <= 0:
        return "Unknown", False, False
    if T_K < 500:
        return "Y dwarf (very cold substellar)", True, False
    if T_K < 1400:
        return "T dwarf (cool substellar)", True, False
    if T_K < 2300:
        return "L dwarf (warm substellar)", True, False
    if T_K < 3900:
        return "M dwarf (red star)", True, False
    if T_K < 7500:
        return "FGK star", False, False
    return "Hot star (>7500 K)", False, False


def generate_summary(target, run_id, input_f, src_files, w, cont, dets, ml_results=None):
    """Generates summary text content."""
    T_K = cont.get('T_K', 0) or 0
    obj_class, h2o_expected, _ = _classify_object(T_K)

    lines = [
        "=" * 70,
        f"  SPECTRAL ANALYSIS REPORT",
        "=" * 70,
        f"  Target  : {target}",
        f"  Run ID  : {run_id}",
        f"  Input   : {os.path.basename(input_f)}",
        f"  Sources : {', '.join(os.path.basename(f) for f in src_files)}",
        "-" * 70,
        f"  Wavelength range : {w.min():.4f} – {w.max():.4f} μm  ({len(w)} pts)",
        f"  Continuum fit    : {'OK' if cont.get('fit_ok') else 'FAILED (polynomial fallback)'}",
        f"  Temperature      : {T_K:.0f} K",
        f"  Chi2 reduced     : {cont['chi2_reduced']:.2f}" if np.isfinite(cont.get('chi2_reduced', float('nan'))) else "  Chi2 reduced     : N/A (no errors)",
        f"  Object class     : {obj_class}",
        f"  H2O expected     : {'Yes' if h2o_expected else 'No / uncertain'}",
        "=" * 70,
        "",
        "  BAND-BY-BAND ANALYSIS",
        "",
        f"  {'Mol':<5} {'Band (μm)':<13} {'Depth':>7} {'±err':>6} {'EQW':>8} {'SNR':>6} {'FAP':>9}",
        "  " + "-" * 65,
    ]

    for mol, info in dets.items():
        for b in info['bands']:
            rng = b.get('band_range', (0, 0))
            band_str = f"{rng[0]:.2f}–{rng[1]:.2f}"
            if not b.get('covered', False):
                lines.append(f"  {mol:<5} {band_str:<13} {'—':>7} {'—':>6} {'—':>8} {'—':>6} {'—':>9}  NO COVERAGE")
            else:
                depth_str = f"{b['depth']:.3f}"
                derr_str  = f"±{b.get('depth_err', 0):.3f}"
                eqw_str   = f"{b.get('eqw', 0):.4f}"
                snr_str   = f"{b['snr']:.1f}"
                fap       = b.get('fap', 1.0)
                fap_str   = f"{fap:.2e}" if fap < 0.01 else f"{fap:.4f}"
                lines.append(f"  {mol:<5} {band_str:<13} {depth_str:>7} {derr_str:>6} "
                             f"{eqw_str:>8} {snr_str:>6} {fap_str:>9}")

    lines += [
        "  " + "-" * 65,
        "",
        "  DETECTION SUMMARY",
        "",
    ]

    found = False
    for mol, info in dets.items():
        status   = info.get('status', 'NOT DETECTED')
        conf     = info.get('confidence', 0.0)
        max_snr  = info.get('max_snr', 0.0)
        mol_fap  = info.get('fap', 1.0)
        expl     = info.get('explanation', '')
        if status not in ('NO SPECTRAL COVERAGE',):
            found = found or info.get('detected', False)
        fap_str = f"{mol_fap:.2e}" if mol_fap < 0.01 else f"{mol_fap:.4f}"
        lines.append(f"  {mol:<5} {status:<22} conf={conf:.3f}  SNR={max_snr:.1f}  FAP={fap_str}")
        if expl:
            lines.append(f"        → {expl}")

    if not found:
        lines.append("  → No robust molecular detections found above threshold.")

    lines += ["", "-" * 70]

    # ML section
    if ml_results:
        if ml_results.get('quality'):
            q = ml_results['quality']
            lines.append(f"  ML Quality Score : {q['quality_score']:.2f}  ({'Usable' if q['usable'] else 'UNUSABLE'})")
            if q.get('notes'):
                lines.append("  Quality notes    : " + "; ".join(q['notes']))
        if ml_results.get('generated_report'):
            lines += ["", "=== AUTOMATED REPORT ===", "", ml_results['generated_report']]

    lines += [
        "",
        "-" * 64,
        "  CAVEATS",
        "  1. O2/O3 alone are not definitive atmospheric markers.",
        "  2. Cloud/haze degeneracy may suppress or mimic features.",
        "  3. Continuum model uncertainty propagates into band depths.",
        "=" * 70,
    ]

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
    
    # Build object context from continuum temperature
    T_K = cont_res.get('T_K', 0) or 0
    object_params = {
        'temperature': T_K,
        'type': 'Y' if T_K < 500 else ('T' if T_K < 1400 else ('L' if T_K < 2300 else 'M'))
    }

    # Features
    try:
        detections = features.detect_molecules(w_um, residual, err=err,
                                               object_params=object_params)
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

            # 2. Confidence — already computed by features.detect_molecules() via
            # ConfidenceAssessor. Re-run here only to apply the data-quality ceiling,
            # which is not available inside features.py.
            try:
                q_score = ml_results['quality']['quality_score'] if ml_results['quality'] else None
                conf_map = {}
                for mol, info in detections.items():
                    if q_score is not None:
                        # Re-assess with quality ceiling using correct SNR threshold (> 2.0)
                        num_bands = sum(1 for b in info.get('bands', [])
                                        if b.get('snr', 0) > 2.0)
                        c_res = conf_assessor.assess(
                            molecule_name=mol,
                            snr=info.get('max_snr', 0.0),
                            num_bands=num_bands,
                            depth=info.get('max_depth', 0.0),
                            spectral_coverage=info.get('spectral_coverage', False),
                            object_params=object_params,
                            quality_score=q_score,
                            combined_snr=info.get('combined_snr', 0.0),
                        )
                        conf_map[mol] = c_res
                    else:
                        # Pull already-computed values directly from detections
                        conf_map[mol] = {
                            'status':      info.get('status', 'NOT DETECTED'),
                            'confidence':  info.get('confidence', 0.0),
                            'label':       info.get('label', 'NOT DETECTED'),
                            'explanation': info.get('explanation', ''),
                        }
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
                conf_detail_map = {}
                for mol, res in ml_results['confidence'].items():
                    label_map[mol] = res['label']
                    conf_detail_map[mol] = {
                        "label": res.get("label"),
                        "explanation": res.get("explanation", ""),
                    }

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
                    "ML confidence": conf_detail_map,
                    "spectral coverage": cov_map
                }
                
                # Adapt detections for ReportWriter
                adapted_dets = {}
                for mol, info in detections.items():
                    nb = sum(1 for b in info.get('bands', []) if b.get('snr', 0) > 2.0)
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
                plotting.plot_all(w_um, flux, cont_res['continuum'], residual, detections, plot_path,
                                              err=err, T_K=cont_res.get('T_K'),
                                              target_name=target_name)
        except Exception as e:
            logger.warn(f"Plotting error: {e}")
            
    logger.step("Results saved")

def main():
    parser = argparse.ArgumentParser(description="Antigravity Spectral Pipeline | Smart CLI")
    
    # Positional: Input (File or Dir)
    parser.add_argument("input", nargs='?', default=".", help="Input file (.fits) or directory to scan")

    # Demo mode (synthetic FITS)
    parser.add_argument("--demo", action="store_true", help="Run on a synthetic FITS spectrum (no downloads needed)")
    parser.add_argument("--demo-seed", type=int, default=42, help="Random seed for --demo spectrum generation")
    
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

    # 0. DEMO MODE (generate a small synthetic FITS)
    if args.demo:
        repo_root = Path(os.path.dirname(os.path.abspath(__file__)))
        demo_dir = repo_root / "data" / "inputs" / "demo"
        demo_file = demo_dir / "demo_dummy_x1d.fits"
        try:
            demo_data.make_dummy_fits(demo_file, seed=int(args.demo_seed))
        except Exception as e:
            logger.fail(f"Demo spectrum generation failed: {e}")
            sys.exit(2)
        args.input = str(demo_file)
        if not args.target_name:
            args.target_name = "DEMO"
    
    # 1. ANALYZE INPUT
    input_path = os.path.abspath(args.input)
    is_dir = os.path.isdir(input_path)
    
    target_name = args.target_name
    timestamp = organize_io.get_timestamp_str()
    
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
        mode_str = "DEMO (synthetic)" if args.demo else "SINGLE FILE"
        selected_file = input_path

    # 3. SETUP I/O
    run_tag = "demo" if args.demo else ("merged" if args.merge_nrs else "single")
    run_id = organize_io.get_run_id(timestamp, run_tag)
    run_dirs = organize_io.setup_run_directories(target_name, run_id, outputs_dir=args.outdir)
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
