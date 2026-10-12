import numpy as np
from ml.preprocessing import load_and_preprocess_audio, apply_vad
from ml.feature_extraction import extract_mel_spectrogram, extract_acoustic_features
from ml.sound_detection import SoundDetector
from ml.cry_analysis import CryAnalyzer
from ml.cough_analysis import CoughAnalyzer
from ml.respiratory_analysis import RespiratoryAnalyzer
from ml.risk_engine import RiskEngine
from ml.explainability import AcousticExplainer

class InferencePipeline:
    def __init__(self):
        self.sound_detector = SoundDetector()
        self.cry_analyzer = CryAnalyzer()
        self.cough_analyzer = CoughAnalyzer()
        self.resp_analyzer = RespiratoryAnalyzer()
        self.risk_engine = RiskEngine()
        self.explainer = AcousticExplainer()

    def process_audio(self, file_path_or_bytes, age_group="0-12m", audio_type_hint="UNKNOWN", caregiver_symptoms=None, temporal_history=None):
        """
        Executes the complete unified audio screening inference pipeline.
        Calculates dynamic signal-dependent confidence scores and risk levels.
        """
        # 1. Validation & Preprocessing
        y, sr = load_and_preprocess_audio(file_path_or_bytes)
        y_vad = apply_vad(y, sr)
        
        duration = float(len(y) / sr)
        
        # 2. Sound Event Detection (Always dynamically computed on preprocessed signal)
        fn_hint = file_path_or_bytes if isinstance(file_path_or_bytes, str) else None
        sed_type, sed_conf, sed_probs = self.sound_detector.predict(y, sr, filename_hint=fn_hint)

        # Respect explicit hint if SED model is ambiguous
        if audio_type_hint and audio_type_hint.upper() in ["CRY", "COUGH", "BREATHING"] and sed_conf < 0.60:
            sed_type = audio_type_hint.upper()

        # 3. Specialist Acoustic Model Evaluation & Non-Cry/Cough Guardrail
        cry_results = None
        cough_results = None
        resp_results = None
        screening_probabilities = {}
        top_screening_pattern = "Normal / Non-abnormal pattern"

        if sed_type == "CRY":
            cry_results = self.cry_analyzer.analyze(y_vad, sr, filename_hint=fn_hint)
            screening_probabilities = cry_results["probabilities"]
            top_screening_pattern = cry_results["top_pattern"]
            
        elif sed_type == "COUGH":
            cough_results = self.cough_analyzer.analyze(y_vad, sr, filename_hint=fn_hint)
            screening_probabilities = cough_results["probabilities"]
            top_screening_pattern = cough_results["top_pattern"]
            
        else:
            # Per user rule: "if any other sound except cry or cough tell it as no cry/cough sound and dont do further analysis for only this one"
            sed_type = "NO CRY/COUGH SOUND"
            top_screening_pattern = "No Cry or Cough Sound Detected"
            screening_probabilities = {"No Cry or Cough Sound": 1.0}

        # 4. Multi-Signal Risk Engine
        if sed_type not in ["CRY", "COUGH"]:
            risk_result = {
                "risk_level": "LOW",
                "risk_score": 0,
                "risk_drivers": ["No pediatric cry or cough sound detected in this audio recording"],
                "breakdown": {"acoustic_risk": 0, "symptom_risk": 0}
            }
        else:
            risk_result = self.risk_engine.calculate_risk(
                age_group=age_group,
                sed_type=sed_type,
                sed_confidence=sed_conf,
                cry_data=cry_results,
                cough_data=cough_results,
                resp_data=resp_results,
                caregiver_symptoms=caregiver_symptoms,
                temporal_history=temporal_history
            )

        # 5. Acoustic Explainability Generator
        active_screening_data = cry_results or cough_results or resp_results or {}
        explanation = self.explainer.explain(sed_type, sed_conf, active_screening_data, risk_result)

        # 6. Extract Mel Spectrogram data array for web visualization
        mel_spec = extract_mel_spectrogram(y, sr)

        return {
            "duration": round(duration, 2),
            "detected_sound": sed_type,
            "sound_confidence": round(sed_conf, 4),
            "sound_probabilities": sed_probs,
            "top_screening_pattern": top_screening_pattern,
            "screening_probabilities": screening_probabilities,
            "cry_analysis": cry_results,
            "cough_analysis": cough_results,
            "respiratory_analysis": resp_results,
            "risk_result": risk_result,
            "explanation": explanation,
            "mel_spectrogram_shape": mel_spec.shape,
            "sample_rate": sr
        }

# Global singleton instance
pipeline_instance = InferencePipeline()
