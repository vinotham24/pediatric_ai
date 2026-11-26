"""Cry category inference using the same production artifact as the API."""

from pathlib import Path

import librosa
import numpy as np

from services.cry_model_service import CryModelService
from services.prediction_status import screening_pattern_for


# These are the contributor-supplied dataset categories. They are not clinical
# interpretations or diagnoses.
CRY_CLASS_DISPLAY = {
    "belly_pain": "belly_pain (dataset category)",
    "burping": "burping (dataset category)",
    "cold_hot": "cold_hot (dataset category)",
    "discomfort": "discomfort (dataset category)",
    "hungry": "hungry (dataset category)",
    "lonely": "lonely (dataset category)",
    "scared": "scared (dataset category)",
    "tired": "tired (dataset category)",
}


def map_cry_probabilities(probabilities):
    """Map scores by class name, never by an assumed numeric class index."""
    return {
        CRY_CLASS_DISPLAY.get(str(label), str(label)): float(score)
        for label, score in probabilities.items()
    }


def candidate_pattern(label, status="UNCERTAIN"):
    display = CRY_CLASS_DISPLAY.get(str(label))
    if not display:
        return screening_pattern_for("UNKNOWN")
    return screening_pattern_for(status, display)


class CryAnalyzer:
    def __init__(self, model_service=None):
        self.model_service = model_service
        self.model_service_error = False
        if self.model_service is None:
            try:
                self.model_service = CryModelService()
            except (FileNotFoundError, ValueError) as exc:
                print(f"Cry category model unavailable: {exc}")
                self.model_service_error = True

    def analyze(self, y, sr=16000, filename_hint=None):
        duration = float(len(y) / sr) if sr else 0.0
        rms = float(np.mean(librosa.feature.rms(y=y)[0])) if len(y) else 0.0
        if rms > 0.08:
            intensity = "High"
        elif rms > 0.03:
            intensity = "Moderate"
        else:
            intensity = "Low"

        try:
            pitch = librosa.yin(y, fmin=80, fmax=1000, sr=sr) if len(y) else np.array([])
            pitch = pitch[np.isfinite(pitch)]
            pitch_std = float(np.std(pitch)) if pitch.size else 0.0
            pitch_max = float(np.max(pitch)) if pitch.size else 0.0
        except Exception:
            pitch_std, pitch_max = 0.0, 0.0
        try:
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))
        except Exception:
            flatness = 0.0
        abnormality_raw = (
            (pitch_std / 12.0) * 30.0
            + (flatness * 1200) * 40.0
            + (rms * 250)
            + (pitch_max / 500.0) * 15.0
        )
        abnormality = min(98, max(12, int(abnormality_raw)))

        prediction = None
        prediction_error = False
        # filename_hint is used only as the source audio path. Its name is never
        # parsed for a label or included among model features.
        if self.model_service is not None and filename_hint:
            try:
                source = Path(filename_hint)
                if source.is_file():
                    prediction = self.model_service.predict(source)
            except Exception as exc:
                print(f"Cry category inference unavailable: {exc}")
                prediction_error = True

        if prediction:
            candidate_class = prediction.get("top_label")
            candidate = CRY_CLASS_DISPLAY.get(str(candidate_class)) if candidate_class else None
            status = prediction.get("prediction_status", "UNKNOWN")
            model_score = prediction.get("model_score")
        else:
            status = "ERROR" if prediction_error or self.model_service_error else "UNKNOWN"
            candidate_class = None
            candidate = None
            model_score = None

        top_pattern = candidate_pattern(candidate_class, status) if candidate_class else screening_pattern_for(status)

        return {
            "duration": round(duration, 2),
            "intensity": intensity,
            "abnormality_score": abnormality,
            "top_pattern": top_pattern,
            "candidate_class": candidate_class,
            "possible_category": candidate,
            "model_score": model_score,
            "prediction_status": status,
            "model_score_interpretation": (
                prediction.get("model_score_interpretation")
                if prediction else "Model score unavailable."
            ),
        }
