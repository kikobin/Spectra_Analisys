"""
organize_io.py

Manages the directory structure for the spectral analysis pipeline.
Enforces a strict separation of inputs, working files, and outputs.
"""

import os
import datetime
import json
import shutil
from pathlib import Path

# Base Paths (Relative to project root, configurable)
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"

def get_timestamp_str():
    """Returns formatted timestamp for Run ID."""
    return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

def get_run_id(timestamp_str, tag):
    """Generates a Run ID (e.g., 2026-01-31T14-22-merged_nrs)."""
    clean_tag = tag.replace(" ", "_").replace("/", "-")
    return f"{timestamp_str}-{clean_tag}"

def setup_run_directories(target_name, run_id):
    """
    Creates the output directory structure for a specific run.
    
    Structure:
    outputs/
        <TARGET_NAME>/
            <RUN_ID>/
                input/
                plots/
                reports/
                logs/
    
    Returns:
        dict: Paths to the created directories.
    """
    base_run_dir = os.path.join(OUTPUTS_DIR, target_name, run_id)
    
    dirs = {
        "root": base_run_dir,
        "input": os.path.join(base_run_dir, "input"),
        "plots": os.path.join(base_run_dir, "plots"),
        "reports": os.path.join(base_run_dir, "reports"),
        "logs": os.path.join(base_run_dir, "logs")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    return dirs

def setup_working_directories(target_name):
    """
    Creates working directories for intermediate files.
    
    Structure:
    data/
        working/
            <TARGET_NAME>/
                selected/
                merged/
    """
    base_work = os.path.join(DATA_DIR, "working", target_name)
    dirs = {
        "selected": os.path.join(base_work, "selected"),
        "merged": os.path.join(base_work, "merged")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def save_run_metadata(run_dirs, metadata):
    """Saves metadata.json to the run's input directory."""
    path = os.path.join(run_dirs['input'], "metadata.json")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        return path
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return None

def archive_input_spectrum(source_path, run_dirs, new_name="spectrum_used.fits"):
    """Copies the actual FITS file used for analysis to the run folder for reproducibility."""
    dest = os.path.join(run_dirs['input'], new_name)
    try:
        shutil.copy2(source_path, dest)
        return dest
    except Exception as e:
        print(f"Warning: Could not archive input spectrum: {e}")
        return None
