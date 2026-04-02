"""
ml — Machine-learning quality and confidence layer for the Spectra pipeline.

Public API
----------
from ml import assess_quality, ConfidenceAssessor, ReportWriter

Functions
---------
assess_quality(w_um, flux, err) -> dict
    Multi-axis data quality score: SNR, coverage, gap penalty, smoothness.
    Returns {'quality_score', 'usable', 'snr_estimate', 'notes'}.

Classes
-------
ConfidenceAssessor
    Computes a probabilistic confidence score and status label (STRONG /
    LIKELY / MARGINAL / WEAK / NOT DETECTED) for a single molecular detection.

ReportWriter
    Generates a structured scientific text report (Abstract, Results,
    Limitations) from pipeline detections, with vocabulary constraints.
"""

from .quality import assess_quality
from .detection_confidence import ConfidenceAssessor
from .report_writer import ReportWriter

__all__ = ["assess_quality", "ConfidenceAssessor", "ReportWriter"]
