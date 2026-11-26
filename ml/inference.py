from ml.preprocessing import load_and_preprocess_audio, apply_vad
from ml.feature_extraction import extract_mel_spectrogram, extract_acoustic_features
from ml.sound_detection import SoundDetector
from ml.cry_analysis import CryAnalyzer
from ml.cough_analysis import CoughAnalyzer
from ml.respiratory_analysis import RespiratoryAnalyzer
from services.prediction_status import screening_pattern_for

class InferencePipeline:
    def __init__(self):
        self.sound_detector = SoundDetector()
        self.cry_analyzer = CryAnalyzer()
        self.cough_analyzer = CoughAnalyzer()
        self.resp_analyzer = RespiratoryAnalyzer()

    def process_audio(self, file_path_or_bytes, age_group="0-12m", audio_type_hint="UNKNOWN", caregiver_symptoms=None, temporal_history=None):
        """
        Executes the complete unified audio screening inference pipeline.
        Calculates dynamic signal-dependent confidence scores and risk levels.
        """
        # 1. Validation & Preprocessing
        y, sr = load_and_preprocess_audio(file_path_or_bytes)
        y_vad = apply_vad(y, sr)
        
        duration = float(len(y) / sr)
        
        # 2. Only an accepted model result may establish a sound type.
        sed_candidate, sed_conf, sed_probs = self.sound_detector.predict(y, sr)
        detection_method = getattr(self.sound_detector, "last_method", "UNKNOWN")
        detection_status = getattr(self.sound_detector, "last_status", "UNKNOWN")
        detector_model_available = getattr(self.sound_detector, "model", None) is not None
        accepted_detection = (
            detector_model_available
            and detection_status == "CONFIDENT"
            and sed_candidate in {"CRY", "COUGH", "OTHER"}
        )
        sed_type = sed_candidate if accepted_detection else "UNKNOWN"
        experimental_sound_candidate = None

        # 3. Specialist Acoustic Model Evaluation & Non-Cry/Cough Guardrail
        cry_results = None
        cough_results = None
        screening_probabilities = {}
        possible_category = None
        model_score = None
        model_score_interpretation = "No cry-category model score is available for this sound."
        prediction_status = "UNKNOWN"
        top_screening_pattern = screening_pattern_for(prediction_status)

        if detection_status == "ERROR":
            prediction_status = "ERROR"
            top_screening_pattern = screening_pattern_for(prediction_status)
        elif not accepted_detection:
            # Without an accepted first-stage model result, sound type is unknown.
            prediction_status = "UNCERTAIN" if detection_status == "UNCERTAIN" else "UNKNOWN"
            top_screening_pattern = screening_pattern_for(prediction_status)
        elif sed_type == "CRY":
            filename_for_decode = file_path_or_bytes if isinstance(file_path_or_bytes, str) else None
            cry_results = self.cry_analyzer.analyze(y_vad, sr, filename_hint=filename_for_decode)
            prediction_status = cry_results.get("prediction_status", "UNKNOWN")
            possible_category = cry_results.get("possible_category")
            model_score = cry_results.get("model_score")
            model_score_interpretation = cry_results.get(
                "model_score_interpretation", model_score_interpretation
            )
            top_screening_pattern = cry_results["top_pattern"]
            
        elif sed_type == "COUGH":
            if getattr(self.cough_analyzer, "model", None) is not None:
                cough_results = self.cough_analyzer.analyze(y_vad, sr)
                prediction_status = cough_results.get("prediction_status", "UNKNOWN")
            else:
                cough_results = {
                    "classification_status": "UNAVAILABLE",
                    "prediction_status": "UNKNOWN",
                    "possible_category": None,
                    "model_score": None,
                    "message": "Unavailable because a verified cough-event classifier has not been trained.",
                }
                prediction_status = "UNKNOWN"
            top_screening_pattern = screening_pattern_for(prediction_status)

        else:
            # Accepted OTHER and all unknown sounds never invoke specialists.
            prediction_status = "UNKNOWN" if detection_status != "ERROR" else "ERROR"
            top_screening_pattern = screening_pattern_for(prediction_status)

        audio_analysis_status = (
            "ERROR" if detection_status == "ERROR" or prediction_status == "ERROR" else
            "UNCERTAIN" if not accepted_detection else
            "EXPERIMENTAL"
        )
        sound_classification = (
            f"{sed_type} (experimental detector result)" if accepted_detection
            else "Unavailable: sound not reliably determined."
        )
        cry_classification_status = (
            "Not performed because CRY was not reliably established."
            if sed_type != "CRY" else
            prediction_status
        )
        cough_classification_status = (
            "Unavailable because a verified cough-event classifier has not been trained."
        )

        # No clinically validated risk function is available. Do not derive a
        # clinical risk score from heuristic sound scores or cry model output.
        risk_result = {
            "risk_available": False,
            "risk_level": "NOT CALCULATED",
            "risk_score": None,
            "risk_drivers": [],
        }
        explanation = {
            "summary": (
                "Experimental audio analysis only. The sound detector and cry "
                "category model are not clinically validated."
            ),
            "evidence_points": [],
            "drivers": [],
        }

        # 6. Extract Mel Spectrogram data array for web visualization
        mel_spec = extract_mel_spectrogram(y, sr)

        return {
            "duration": round(duration, 2),
            "detected_sound": sed_type,
            "experimental_sound_candidate": experimental_sound_candidate,
            "audio_analysis_status": audio_analysis_status,
            "sound_classification": sound_classification,
            "cry_classification_status": cry_classification_status,
            "cough_classification_status": cough_classification_status,
            "medical_diagnosis": "Not provided.",
            "sound_confidence": round(float(sed_conf), 4) if sed_conf is not None else None,
            "sound_confidence_is_calibrated": False,
            "sound_confidence_interpretation": (
                "No sound-event detector is available. Any future detector score must be validated before interpretation."
            ),
            "sound_probabilities": sed_probs,
            "sound_detection_status": detection_status,
            "sound_detection_method": detection_method,
            "prediction_status": prediction_status,
            "possible_category": possible_category,
            "model_score": model_score,
            "model_score_interpretation": model_score_interpretation,
            "top_screening_pattern": top_screening_pattern,
            "screening_probabilities": screening_probabilities,
            "cry_analysis": cry_results,
            "cough_analysis": cough_results,
            "respiratory_analysis": None,
            "risk_result": risk_result,
            "explanation": explanation,
            "mel_spectrogram_shape": mel_spec.shape,
            "sample_rate": sr
        }

# Global singleton instance
pipeline_instance = InferencePipeline()
