import sys
from .ranker import SpectrumRanker
from .classifier import DetectionClassifier
from .generator import ReportGenerator

class MLManager:
    def __init__(self):
        # Initialize sub-components
        # In a real system, we might lazy load heavy ML libs here
        self.ranker = SpectrumRanker()
        self.classifier = DetectionClassifier()
        self.generator = ReportGenerator()
        
    def rank_spectra(self, file_list):
        """
        Pass-through to Ranker.
        """
        return self.ranker.rank(file_list)

    def evaluate_detection(self, w_um, residual, detections):
        """
        Pass-through to Classifier.
        """
        return self.classifier.evaluate(w_um, residual, detections)

    def draft_report(self, full_context):
        """
        Pass-through to Generator.
        """
        return self.generator.generate_text(full_context)
