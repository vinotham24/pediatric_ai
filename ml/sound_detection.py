"""Experimental first-stage sound detector with explicit model availability."""

import os

import joblib

from config import MODELS_DIR
from ml.feature_extraction import extract_acoustic_features


SOUND_ALIASES = {
    "CRY": "CRY",
    "COUGH": "COUGH",
    "OTHER": "OTHER",
    "NO CRY/COUGH SOUND": "OTHER",
    "SPEECH": "OTHER",
    "NOISE": "OTHER",
    "BREATHING": "OTHER",
    "UNKNOWN": "UNKNOWN",
}


class SoundDetector:
    """Loads a sound-event model when available; never guesses from signal heuristics."""

    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, "sound_model", "sed_model.joblib")
        self.model = None
        self.model_error = None
        self.last_method = "UNAVAILABLE"
        self.last_status = "UNKNOWN"
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            return
        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            self.model_error = str(exc)
            self.model = None

    def predict(self, y, sr=16000, filename_hint=None):
        """Return (candidate, raw score or None, class scores).

        `filename_hint` is accepted for call compatibility but intentionally ignored.
        """
        _ = filename_hint
        if self.model is None:
            self.last_method = "UNAVAILABLE (SED model unavailable)"
            self.last_status = "UNKNOWN"
            return "UNKNOWN", None, {}

        try:
            features = extract_acoustic_features(y, sr).reshape(1, -1)
            if hasattr(self.model, "predict_proba"):
                raw_scores = self.model.predict_proba(features)[0]
                classes = [str(cls).upper() for cls in self.model.classes_]
                scores = {}
                for label, score in zip(classes, raw_scores):
                    canonical = SOUND_ALIASES.get(label, "UNKNOWN")
                    scores[canonical] = scores.get(canonical, 0.0) + float(score)
                prediction = str(self.model.predict(features)[0]).upper()
                candidate = SOUND_ALIASES.get(prediction, "UNKNOWN")
                score = scores.get(candidate)
            else:
                prediction = str(self.model.predict(features)[0]).upper()
                candidate = SOUND_ALIASES.get(prediction, "UNKNOWN")
                score = None
                scores = {}
            if candidate not in {"CRY", "COUGH", "OTHER", "UNKNOWN"}:
                candidate, score, scores = "UNKNOWN", None, {}
            self.last_method = "UNVALIDATED SED MODEL"
            self.last_status = "UNCERTAIN" if candidate != "UNKNOWN" else "UNKNOWN"
            return candidate, score, scores
        except Exception as exc:
            self.model_error = str(exc)
            self.last_method = "SED inference error"
            self.last_status = "ERROR"
            return "UNKNOWN", None, {}
