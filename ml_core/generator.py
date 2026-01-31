class ReportGenerator:
    def __init__(self):
        pass

    def generate_text(self, context):
        """
        Generates a text summary based on the analysis context.
        """
        # Context contains physics results and ML opinions
        
        detections = context.get('detections', {})
        ml_review = context.get('ml_analysis', {}).get('detections_review', {})
        
        lines = []
        lines.append("\n**ML ASSISTANT REPORT**")
        lines.append(f"Analysis Version: {context.get('ml_analysis', {}).get('model_version', 'Unknown')}")
        lines.append("-----------------------------------------------------")
        
        detected_mols = [m for m, d in detections.items() if d['detected']]
        
        if detected_mols:
            lines.append(f"The system detected {len(detected_mols)} molecule(s): {', '.join(detected_mols)}.")
            for mol in detected_mols:
                review = ml_review.get(mol, {})
                conf = review.get('confidence', 'UNKNOWN')
                desc = review.get('description', '')
                lines.append(f"- **{mol}**: Confidence is **{conf}**. {desc}")
        else:
            lines.append("No significant molecular features were detected above the physical thresholds.")
            
        lines.append("-----------------------------------------------------")
        
        return "\n".join(lines)
