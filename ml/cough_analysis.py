"""Cough analysis availability gate.

This project currently has no verified cough-event annotations or trained cough
classifier. The West China pneumonia/bronchitis groups are not event labels.
"""

from pathlib import Path


class CoughAnalyzer:
    def __init__(self):
        self.model_path = Path(__file__).resolve().parents[1] / "models" / "cough_model" / "cough_model.joblib"
        self.model = None

    def analyze(self, y, sr=16000, filename_hint=None):
        _ = filename_hint
        duration = float(len(y) / sr) if sr else 0.0
        return {
            "classification_status": "UNAVAILABLE",
            "prediction_status": "UNKNOWN",
            "possible_category": None,
            "model_score": None,
            "message": (
                "No cough classifier trained on verified cough-event annotations is available. "
                "The pneumonia/bronchitis recording groups are not cough-event labels."
            ),
            "duration": round(duration, 2),
        }
