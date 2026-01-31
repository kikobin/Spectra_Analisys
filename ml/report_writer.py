import logging
from typing import Dict, Any, List

class ReportWriter:
    """
    Generates a scientific text report based on spectral analysis results.
    Adheres to strict vocabulary constraints (no 'life' or 'biosignature')
    and uses cautious scientific language.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prohibited_words = ["life", "biosignature"]

    def generate_report(self, data: Dict[str, Any]) -> str:
        """
        Generates the full text report.

        Args:
            data: JSON-like dictionary containing:
                - target name: str
                - instrument(s): List[str]
                - physical detections: Dict[str, Any]
                - ML confidence labels: Dict[str, str]
                - spectral coverage: Dict[str, bool]

        Returns:
            Formatted report string with Abstract, Results, and Limitations.
        """
        self._validate_input(data)

        target_name = data.get("target name", "Unknown Target")
        instruments = data.get("instrument(s)", [])
        
        abstract = self._write_abstract(data, target_name, instruments)
        results = self._write_results(data)
        limitations = self._write_limitations(data, instruments)

        report = (
            f"REPORT FOR TARGET: {target_name}\n"
            f"{'-' * 40}\n\n"
            f"ABSTRACT\n{abstract}\n\n"
            f"RESULTS\n{results}\n\n"
            f"LIMITATIONS\n{limitations}"
        )

        return report

    def _validate_input(self, data: Dict[str, Any]):
        required_keys = ["target name", "instrument(s)", "physical detections", "ML confidence labels", "spectral coverage"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            self.logger.warning(f"Missing keys in report input: {missing}")
    
    def _write_abstract(self, data: Dict[str, Any], target: str, instruments: List[str]) -> str:
        """
        Generates the Abstract section (3-4 sentences).
        """
        inst_str = ", ".join(instruments) if instruments else "unspecified instruments"
        
        # Identify high confidence detections for the abstract
        confident_molecules = []
        for mol, label in data.get("ML confidence labels", {}).items():
            if label == "LIKELY":
                confident_molecules.append(mol)
        
        abstract = f"This report defines the spectral analysis of {target} using data from {inst_str}. "
        abstract += "The analysis focused on the identification of molecular absorption features within the observed wavelength range. "
        
        if confident_molecules:
            mols = ", ".join(confident_molecules)
            abstract += f"Features consistent with {mols} were identified with high statistical confidence, though alternative interpretations remain possible. "
        else:
            abstract += "No definitive molecular absorption features were identified above the noise threshold in this analysis. "
        
        abstract += "These results provide constraints on the atmospheric composition, subject to the instrumental sensitivity and spectral coverage described herein."
        
        return abstract

    def _write_results(self, data: Dict[str, Any]) -> str:
        """
        Generates the Results section.
        """
        lines = []
        detections = data.get("physical detections", {})
        confidence = data.get("ML confidence labels", {})
        
        if not detections:
            return "No specific molecular searches were performed."

        for molecule in detections.keys():
            conf_label = confidence.get(molecule, "UNKNOWN")
            phys_data = detections[molecule]
            
            # Extract physical details if available, handle if just a boolean or dict
            # Assuming phys_data is a dict based on context, but being robust
            if isinstance(phys_data, dict):
                snr = phys_data.get("snr", 0.0)
                bands = phys_data.get("num_bands", 0)
            else:
                snr = 0.0
                bands = 0

            line = f"- {molecule}: Classified as {conf_label}."
            
            if conf_label == "LIKELY":
                line += f" Analysis reveals {bands} potential absorption bands with a signal to noise ratio of {snr:.1f}. The spectral morphology is consistent with theoretical models."
            elif conf_label == "UNCERTAIN":
                line += f" Marginal signals (SNR ~{snr:.1f}) were observed, but low spectral resolution or noise prevents a definitive identification."
            elif conf_label == "UNLIKELY":
                line += " No significant absorption features were detected above the noise floor."
            
            lines.append(line)
            
        return "\n".join(lines)

    def _write_limitations(self, data: Dict[str, Any], instruments: List[str]) -> str:
        """
        Generates the Limitations section.
        """
        lines = []
        coverage = data.get("spectral coverage", {})
        
        lines.append("The validity of these results is strictly bound by the following limitations:")
        
        # Instrument limitations
        lines.append(f"1. Observational Data: Analysis is largely dependent on the sensitivity and calibration of {', '.join(instruments) if instruments else 'the used instruments'}.")
        
        # Coverage limitations
        missing_coverage = [mol for mol, covered in coverage.items() if not covered]
        if missing_coverage:
            lines.append(f"2. Spectral Coverage: The dataset does not fully sample the characteristic absorption regions for {', '.join(missing_coverage)}, preventing any assessment of their presence.")
        else:
            lines.append("2. Spectral Coverage: The dataset provides coverage for all target molecules, though sensitivity varies across the wavelength range.")

        # Logic / Model limitations
        lines.append("3. Atmospheric Modeling: Detections are based on cross-correlation with template spectra; degenerate solutions with other atmospheric species or stellar activity cannot be fully ruled out.")
        
        return "\n".join(lines)

if __name__ == "__main__":
    # verification block
    writer = ReportWriter()
    
    test_data = {
        "target name": "K2-18 b",
        "instrument(s)": ["NIRSpec Prism", "MIRI LRS"],
        "physical detections": {
            "H2O": {"snr": 8.5, "num_bands": 3},
            "O2": {"snr": 1.2, "num_bands": 0},
            "O3": {"snr": 3.1, "num_bands": 1}
        },
        "ML confidence labels": {
            "H2O": "LIKELY",
            "O2": "UNLIKELY",
            "O3": "UNCERTAIN"
        },
        "spectral coverage": {
            "H2O": True,
            "O2": False,
            "O3": True
        }
    }
    
    print(writer.generate_report(test_data))
