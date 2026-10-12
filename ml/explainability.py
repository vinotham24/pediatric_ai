class AcousticExplainer:
    def __init__(self):
        pass

    def explain(self, sed_type, confidence, screening_result, risk_result):
        """
        Generates structured acoustic feature contribution breakdown for model explainability.
        """
        evidence_points = []
        
        if sed_type == "CRY":
            top_pattern = screening_result.get("top_pattern", "")
            abnormality = screening_result.get("abnormality_score", 0)
            intensity = screening_result.get("intensity", "Moderate")
            
            evidence_points.append(f"DeepInfant STFT & Log-Mel Spectrogram pipeline identified infant cry with {int(confidence*100)}% acoustic confidence.")
            evidence_points.append(f"Vocal intensity evaluated as '{intensity}' with pitch stability index yielding an abnormality score of {abnormality}/100.")
            evidence_points.append(f"Acoustic feature vectors (F0 pitch, spectral flatness, Mel-energy distribution) indicate '{top_pattern}'.")

        elif sed_type == "COUGH":
            top_pattern = screening_result.get("top_pattern", "")
            events = screening_result.get("cough_events", 0)
            freq = screening_result.get("cough_frequency", "Low")
            
            evidence_points.append(f"DeepInfant STFT & Log-Mel Spectrogram classifier detected explosive pediatric cough burst with {int(confidence*100)}% confidence.")
            evidence_points.append(f"Detected {events} distinct cough bursts yielding a '{freq}' repetition cadence.")
            evidence_points.append(f"Spectral acoustic features (centroid, rolloff & energy phase ratio) align with '{top_pattern}'.")

        elif sed_type == "BREATHING":
            top_pattern = screening_result.get("top_pattern", "")
            evidence_points.append(f"DeepInfant acoustic classification detected '{top_pattern}' with {int(confidence*100)}% confidence.")

        else:
            evidence_points.append("Audio signal categorized as background acoustic activity or non-pediatric sound.")

        risk_level = risk_result.get("risk_level", "LOW")
        risk_score = risk_result.get("risk_score", 0)
        drivers = risk_result.get("risk_drivers", [])

        summary = (
            f"DeepInfant Acoustic Analysis: Evaluated overall acoustic risk as {risk_level} (Score: {risk_score}/100). "
            f"Key contributing signals: {', '.join(drivers)}."
        )

        return {
            "summary": summary,
            "evidence_points": evidence_points,
            "drivers": drivers
        }
