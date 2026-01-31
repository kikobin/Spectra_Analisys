import logging
from typing import Dict, Any, Optional

class ConfidenceAssessor:
    """
    Assesses the confidence of a physical detection based on spectral metrics.
    Now supports multi-level confidence (STRONG, LIKELY, MARGINAL, WEAK) and physical context.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def assess(self, 
               molecule_name: str, 
               snr: float, 
               num_bands: int, 
               depth: float, 
               spectral_coverage: bool, 
               object_params: Optional[Dict[str, Any]] = None,
               quality_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculates confidence score and label for a given molecule.
        
        Args:
            molecule_name: "H2O", "O2", or "O3"
            snr: Signal-to-Noise Ratio (max or avg of bands)
            num_bands: Number of detected bands
            depth: Max absorption depth
            spectral_coverage: Boolean, true if the instrument covers the molecule's bio-signature region
            object_params: Dictionary with keys 'temperature', 'type' (e.g., {'temperature': 161.3, 'type': 'Y'})
            quality_score: Optional external quality metric (0-1)

        Returns:
            Dict containing:
                - status: "STRONG", "LIKELY", "MARGINAL", "WEAK", "NOT DETECTED", "NO SPECTRAL COVERAGE"
                - confidence: float (0.0 - 1.0)
                - explanation: str
        """
        if not spectral_coverage:
            return {
                "status": "NO SPECTRAL COVERAGE",
                "confidence": 0.0,
                "label": "NO COVERAGE", # Legacy field
                "explanation": "No spectral coverage for this molecule."
            }

        # Default params
        if object_params is None:
            object_params = {}

        score = 0.0
        explanations = []

        # --- 1. Base Score from SNR (Max 0.5) ---
        # Lower thresholds for exploratory science
        # SNR >= 5.0 -> 0.5 (Solid)
        # SNR >= 3.0 -> 0.4 (Good)
        # SNR >= 2.0 -> 0.3 (Acceptable weak)
        # SNR >= 1.5 -> 0.1 (Marginal)
        
        if snr >= 5.0:
            score += 0.5
            explanations.append(f"High SNR ({snr:.1f} >= 5.0)")
        elif snr >= 3.0:
            score += 0.4
            explanations.append(f"Moderate SNR ({snr:.1f} >= 3.0)")
        elif snr >= 2.0:
            score += 0.3
            explanations.append(f"Weak SNR ({snr:.1f} >= 2.0)")
        elif snr >= 1.5:
            score += 0.1
            explanations.append(f"Marginal SNR ({snr:.1f} >= 1.5)")
        else:
            explanations.append(f"Noise floor SNR ({snr:.1f} < 1.5)")

        # --- 2. Band Count Bonus (Max 0.2) ---
        if num_bands >= 3:
            score += 0.2
            explanations.append(f"Multi-band ({num_bands}+)")
        elif num_bands == 2:
            score += 0.1
            explanations.append("Two bands")
        elif num_bands == 1:
            # No bonus, but not penalty
            explanations.append("Single band")

        # --- 3. Depth Bonus (Max 0.1) ---
        if depth >= 0.10: # 10%
            score += 0.1
            explanations.append(f"Deep Feature ({depth:.2f})")
        elif depth >= 0.05: # 5%
            score += 0.05
        
        # --- 4. Physical Context Modifiers ---
        # "Priors"
        temp = object_params.get('temperature', 1000) # Default to hot if unknown
        obj_type = object_params.get('type', 'Unknown')
        
        # H2O Expectation: High for T/Y dwarfs (Temp < 1000K)
        is_cold = temp < 1000
        
        if molecule_name == "H2O":
            if is_cold:
                score += 0.1
                explanations.append("Expected molecule (Cold)")
                
        # O2/O3 Skepticism: Penalize if weak signal
        if molecule_name in ["O2", "O3"]:
            # If signal is already weak, apply penalty to prevent false positives
            if snr < 3.0:
                score -= 0.1
                explanations.append("Penalty: Unexpected weak bio-marker")

        # --- 5. Quality Score integration ---
        if quality_score is not None:
            # If data quality is low, cap confidence
            if quality_score < 0.5:
                score = min(score, 0.4)
                explanations.append(f"Low Data Quality ({quality_score:.1f})")

        # Cap score
        score = min(score, 1.0)
        score = max(score, 0.0)

        # --- Determine Status ---
        # STRONG: > 0.8
        # LIKELY: > 0.6
        # MARGINAL: > 0.4
        # WEAK/TENTATIVE: > 0.2 (Only if SNR is physically plausible)
        # NOT DETECTED: < 0.2
        
        if score >= 0.8:
            status = "STRONG"
        elif score >= 0.6:
            status = "LIKELY"
        elif score >= 0.4:
            status = "MARGINAL"
        elif score >= 0.2 and snr >= 1.5:
            status = "WEAK"
        else:
            status = "NOT DETECTED"
            # Force score to 0 if not detected
            score = 0.0

        return {
            "status": status,
            "confidence": round(score, 2),
            "label": status, # Legacy compatibility
            "explanation": "; ".join(explanations)
        }

if __name__ == "__main__":
    # verification block
    model = ConfidenceAssessor()
    
    test_cases = [
        # (Name, SNR, Bands, Depth, Cov, Params)
        ("H2O", 2.2, 2, 0.12, True, {'temperature': 160.0, 'type': 'Y'}),   # Weak but expect H2O
        ("O2", 2.2, 1, 0.05, True, {'temperature': 160.0, 'type': 'Y'}),    # Weak unexpected O2
        ("H2O", 0.9, 1, 0.02, True, {'temperature': 160.0, 'type': 'Y'}),   # Noise
        ("O3", 5.0, 1, 0.1, True, {'temperature': 160.0, 'type': 'Y'}),     # Strong O3 (should be likely/strong despite penalty)
        ("H2O", 10.0, 3, 0.5, False, {}),                                    # No Coverage
    ]

    print(f"{'MOL':<5} | {'SNR':<4} | {'STATUS':<12} | {'CONF':<4} | {'EXPLANATION'}")
    print("-" * 100)
    for name, snr, bands, depth, cov, params in test_cases:
        res = model.assess(name, snr, bands, depth, cov, object_params=params)
        print(f"{name:<5} | {snr:<4.1f} | {res['status']:<12} | {res['confidence']:<4} | {res['explanation']}")
