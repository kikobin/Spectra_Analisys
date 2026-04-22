from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.report_writer import ReportWriter


def test_forbidden_words_redaction() -> None:
    data = {
        "target name": "Test Planet",
        "instrument(s)": ["Test"],
        "physical detections": {"O2": {"snr": 3.0, "num_bands": 1}},
        "ML confidence labels": {},
        "ML confidence": {"O2": {"label": "LIKELY", "explanation": "This is a biosignature of life."}},
        "spectral coverage": {"O2": True},
    }

    report = ReportWriter().generate_report(data)
    assert "life" not in report.lower()
    assert "biosignature" not in report.lower()
    assert "[REDACTED]" in report


def test_no_detections_message() -> None:
    data = {
        "target name": "Empty Planet",
        "instrument(s)": ["Test"],
        "physical detections": {},
        "ML confidence labels": {},
        "spectral coverage": {"H2O": True},
    }

    report = ReportWriter().generate_report(data)
    assert "No definitive molecular absorption features" in report
