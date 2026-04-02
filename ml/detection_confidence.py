import logging
import numpy as np
from typing import Dict, Any, Optional


class ConfidenceAssessor:
    """
    Computes a confidence score and status label for a molecular detection.

    Scoring model (additive, capped at 1.0):
        Base SNR score    — up to 0.50   (single-band max SNR)
        Combined SNR      — up to 0.10   (multi-band quadrature bonus)
        Band-count bonus  — up to 0.20
        Depth bonus       — up to 0.10
        Physical prior    — ±0.15        (expected / unexpected molecule)
        Quality cap       — ceiling on total score when data quality is low

    Status thresholds:
        STRONG   ≥ 0.80
        LIKELY   ≥ 0.60
        MARGINAL ≥ 0.40
        WEAK     ≥ 0.20  (and SNR ≥ 1.5)
        NOT DETECTED otherwise
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def assess(
        self,
        molecule_name: str,
        snr: float,
        num_bands: int,
        depth: float,
        spectral_coverage: bool,
        object_params: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None,
        combined_snr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            molecule_name:    'H2O', 'O2', or 'O3'.
            snr:              Max per-band SNR.
            num_bands:        Number of bands with SNR > 2.
            depth:            Max absorption depth (fractional, 0–1).
            spectral_coverage: True if any band of this molecule is in range.
            object_params:    {'temperature': K, 'type': str}.
            quality_score:    Overall data quality [0–1] from ml.quality.
            combined_snr:     Quadrature sum of per-band SNRs (√Σsnr²).

        Returns:
            {status, confidence, label, explanation}
        """
        if not spectral_coverage:
            return {
                "status": "NO SPECTRAL COVERAGE",
                "confidence": 0.0,
                "label": "NO SPECTRAL COVERAGE",
                "explanation": "No spectral coverage for this molecule.",
            }

        if object_params is None:
            object_params = {}

        score = 0.0
        explanations = []

        # ── 1. Base SNR score (max 0.50) ───────────────────────────────────
        # Use the better of per-band max SNR and combined quadrature SNR.
        # Combined SNR reflects matched-filter significance across all bands.
        eff_snr = max(snr, combined_snr or 0.0)

        if eff_snr >= 7.0:
            score += 0.50
            explanations.append(f"Very high SNR ({eff_snr:.1f})")
        elif eff_snr >= 5.0:
            score += 0.45
            explanations.append(f"High SNR ({eff_snr:.1f})")
        elif eff_snr >= 3.0:
            score += 0.35
            explanations.append(f"Moderate SNR ({eff_snr:.1f})")
        elif eff_snr >= 2.0:
            score += 0.20
            explanations.append(f"Weak SNR ({eff_snr:.1f})")
        elif eff_snr >= 1.5:
            score += 0.08
            explanations.append(f"Marginal SNR ({eff_snr:.1f})")
        else:
            explanations.append(f"Below noise floor (SNR={eff_snr:.1f})")

        # ── 2. Combined-SNR multi-band bonus (max 0.10) ────────────────────
        if combined_snr and combined_snr > snr + 0.5:
            bonus = min(0.10, 0.05 * (combined_snr - snr))
            score += bonus
            explanations.append(f"Multi-band combined SNR={combined_snr:.1f} (+{bonus:.2f})")

        # ── 3. Band-count bonus (max 0.20) ─────────────────────────────────
        if num_bands >= 4:
            score += 0.20
            explanations.append(f"Strong multi-band ({num_bands} bands)")
        elif num_bands == 3:
            score += 0.15
            explanations.append(f"Multi-band ({num_bands} bands)")
        elif num_bands == 2:
            score += 0.10
            explanations.append("Two bands")
        elif num_bands == 1:
            explanations.append("Single band only")

        # ── 4. Depth bonus (max 0.10) ───────────────────────────────────────
        if depth >= 0.20:
            score += 0.10
            explanations.append(f"Very deep feature ({depth:.2f})")
        elif depth >= 0.10:
            score += 0.07
            explanations.append(f"Deep feature ({depth:.2f})")
        elif depth >= 0.05:
            score += 0.03

        # ── 5. Physical prior (±0.15) ───────────────────────────────────────
        temp = float(object_params.get("temperature", 1500))

        if molecule_name == "H2O":
            # H2O is ubiquitous in substellar atmospheres; strength increases at T < 2300 K
            if temp < 500:
                score += 0.15
                explanations.append("Strong prior: Y-dwarf (H2O dominant)")
            elif temp < 1400:
                score += 0.12
                explanations.append("Strong prior: T-dwarf (H2O expected)")
            elif temp < 2300:
                score += 0.08
                explanations.append("Prior: L-dwarf (H2O expected)")
            elif temp < 4000:
                score += 0.04
                explanations.append("Prior: M-dwarf (H2O present)")
            # Hot stars: no H2O prior bonus

        elif molecule_name == "CH4":
            # Dominant carbon carrier for T < 1400 K; very strong in T/Y dwarfs
            if temp < 500:
                score += 0.15
                explanations.append("Strong prior: CH4 dominant in Y-dwarf (T < 500 K)")
            elif temp < 1400:
                score += 0.12
                explanations.append("Strong prior: CH4 expected in T-dwarf (T < 1400 K)")
            elif temp < 1800:
                score += 0.04
                explanations.append("Weak prior: CH4 at L/T boundary — marginal abundance")
            else:
                score -= 0.08
                explanations.append("Penalty: CH4 unlikely at T > 1800 K (thermochemistry)")

        elif molecule_name == "CO":
            # CO dominant over CH4 for T > 1200 K; anti-correlated with CH4
            if temp > 1400:
                score += 0.08
                explanations.append("Prior: CO expected in warm atmosphere (T > 1400 K)")
            elif temp > 1200:
                score += 0.03
                explanations.append("Weak prior: CO/CH4 in chemical equilibrium near T~1200 K")
            else:
                score -= 0.08
                explanations.append("Penalty: CO unlikely at T < 1200 K (favors CH4)")

        elif molecule_name == "CO2":
            # Trace but detectable at 4.3 μm even at low mixing ratios
            score += 0.03
            explanations.append("Weak prior: CO2 trace abundance expected in all objects")

        elif molecule_name == "O2":
            # O2 is very rare; single-band detections are almost certainly noise
            if num_bands < 2:
                score -= 0.15
                explanations.append("Strong penalty: O2 requires ≥2 bands for credibility")
            elif eff_snr < 5.0:
                score -= 0.10
                explanations.append("Penalty: O2 claim needs SNR ≥ 5 across multiple bands")

        elif molecule_name == "O3":
            # O3 9.6 μm band is real but easily confused with silicate features
            if eff_snr < 4.0:
                score -= 0.10
                explanations.append("Penalty: O3 at 9.6 μm needs high SNR (silicate confusion)")
            if num_bands < 2:
                score -= 0.05
                explanations.append("Penalty: single-band O3 detection — low reliability")

        # ── 6. Data quality ceiling ─────────────────────────────────────────
        if quality_score is not None:
            if quality_score < 0.25:
                score = min(score, 0.25)
                explanations.append(f"Quality ceiling: poor data (q={quality_score:.2f})")
            elif quality_score < 0.50:
                score = min(score, 0.50)
                explanations.append(f"Quality ceiling: moderate data (q={quality_score:.2f})")

        score = float(np.clip(score, 0.0, 1.0))

        # ── Determine status ────────────────────────────────────────────────
        if score >= 0.80:
            status = "STRONG"
        elif score >= 0.60:
            status = "LIKELY"
        elif score >= 0.40:
            status = "MARGINAL"
        elif score >= 0.20 and eff_snr >= 1.5:
            status = "WEAK"
        else:
            status = "NOT DETECTED"
            score = 0.0

        return {
            "status": status,
            "confidence": round(score, 3),
            "label": status,
            "explanation": "; ".join(explanations),
        }


if __name__ == "__main__":
    model = ConfidenceAssessor()

    tests = [
        # (name, snr, bands, depth, cov, params, quality, combined_snr)
        ("H2O", 4.5, 3, 0.15, True, {"temperature": 650, "type": "T"}, 0.8, 6.2),
        ("H2O", 2.2, 2, 0.08, True, {"temperature": 160, "type": "Y"}, 0.6, 3.1),
        ("O2",  2.2, 1, 0.05, True, {"temperature": 300, "type": "Y"}, 0.7, 2.2),
        ("O3",  5.5, 1, 0.12, True, {"temperature": 300, "type": "Y"}, 0.8, 5.5),
        ("H2O", 0.9, 0, 0.01, True, {"temperature": 800, "type": "T"}, 0.4, 0.0),
        ("H2O", 8.0, 4, 0.30, False, {}, None, 0.0),  # no coverage
    ]

    print(f"{'MOL':<5} | {'SNR':>5} | {'cSNR':>5} | {'STATUS':<15} | {'CONF':>5} | EXPLANATION")
    print("-" * 100)
    for name, snr, bands, depth, cov, params, qual, csnr in tests:
        r = model.assess(name, snr, bands, depth, cov,
                         object_params=params, quality_score=qual, combined_snr=csnr)
        print(f"{name:<5} | {snr:>5.1f} | {csnr:>5.1f} | {r['status']:<15} | "
              f"{r['confidence']:>5.3f} | {r['explanation']}")
