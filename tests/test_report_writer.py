
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.report_writer import ReportWriter, AnalysisResult

def test_forbidden_words():
    print("Testing forbidden words redaction...")
    # Inject forbidden words into the "explanation" field which is passed through
    data = AnalysisResult(
        target_name="Test Planet",
        instruments=["Test"],
        physical_detections={},
        ml_confidence={
            "O2": {"label": "LIKELY", "explanation": "This is a biosignature of life."}
        },
        spectral_coverage=[(1.0, 2.0)]
    )
    
    report = ReportWriter.generate_report(data)
    
    if "life" in report.lower() or "biosignature" in report.lower():
        print("FAILED: Forbidden words found in report.")
        print(report)
        sys.exit(1)
        
    if "[REDACTED]" not in report:
        print("FAILED: Redaction marker not found.")
        print(report)
        sys.exit(1)
        
    print("PASSED: Forbidden words redacted.")

def test_no_detections():
    print("\nTesting no detections...")
    data = AnalysisResult(
        target_name="Empty Planet",
        instruments=["Test"],
        physical_detections={},
        ml_confidence={},
        spectral_coverage=[(1.0, 2.0)]
    )
    
    report = ReportWriter.generate_report(data)
    
    if "No statistically significant molecular absorption features" not in report:
        print("FAILED: Did not report no detections correctly.")
        sys.exit(1)
        
    print("PASSED: No detections handled.")

if __name__ == "__main__":
    test_forbidden_words()
    test_no_detections()
    print("\nAll tests passed!")
