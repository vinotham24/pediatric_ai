import os
import numpy as np
import joblib
import librosa
from config import MODELS_DIR, CRY_CLASSES
from ml.feature_extraction import extract_acoustic_features

class CryAnalyzer:
    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, "cry_model", "cry_model.joblib")
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"Warning loading Cry model: {e}")

    def analyze(self, y, sr=16000, filename_hint=None):
        duration = float(len(y) / sr)
        rms = float(np.mean(librosa.feature.rms(y=y)[0]))
        fn_lower = str(filename_hint).lower() if filename_hint else ""
        
        # Determine intensity
        if rms > 0.08:
            intensity = "High"
        elif rms > 0.03:
            intensity = "Moderate"
        else:
            intensity = "Low"

        # Pitch & harmonicity features for abnormality score
        try:
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                pitch_std = float(np.std(f0_clean))
                pitch_max = float(np.max(f0_clean))
                pitch_mean = float(np.mean(f0_clean))
            else:
                pitch_std, pitch_max, pitch_mean = 0.0, 0.0, 0.0
        except Exception:
            pitch_std, pitch_max, pitch_mean = 0.0, 0.0, 0.0

        spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))
        
        # Calculate Abnormality Score (0-100) dynamically based on pitch instability & spectral entropy
        abnormality_raw = (pitch_std / 12.0) * 30.0 + (spec_flatness * 1200) * 40.0 + (rms * 250) + (pitch_max / 500.0) * 15.0
        abnormality_score = min(98, max(12, int(abnormality_raw)))

        probs = {}
        if self.model is not None:
            try:
                feat = extract_acoustic_features(y, sr).reshape(1, -1)
                p_arr = self.model.predict_proba(feat)[0]
                classes = list(self.model.classes_)
                probs = {cls: round(float(p), 4) for cls, p in zip(classes, p_arr)}
            except Exception as e:
                print(f"Error in ML cry inference: {e}")

        if not probs:
            probs = self._acoustic_heuristic_cry_probs(abnormality_score, pitch_max, pitch_std, rms)

        probs = self._normalize_cry_probs(probs)
        top_pattern = max(probs, key=probs.get)
        confidence = probs[top_pattern]

        return {
            "duration": round(duration, 2),
            "intensity": intensity,
            "abnormality_score": abnormality_score,
            "top_pattern": top_pattern,
            "confidence": round(confidence, 4),
            "probabilities": probs
        }

    def _normalize_cry_probs(self, probs):
        norm_map = {
            "Pain-associated": "Severe Belly Pain / Colic Cry",
            "Discomfort-associated": "Physical Irritation Cry",
            "Burping-associated": "Gas & Burping Discomfort Cry",
            "Hunger-associated": "Hunger Cry",
            "Fatigue-associated": "Fatigue & Sleepiness Cry",
            "Normal/Other": "Normal / Baseline Cry"
        }
        normalized = {}
        for k, v in probs.items():
            norm_k = norm_map.get(k, k)
            normalized[norm_k] = v
        return normalized

    def _acoustic_heuristic_cry_probs(self, abnormality_score, pitch_max, pitch_std, rms):
        """
        Dynamic Softmax distribution over cry patterns based on exact acoustic signals.
        """
        score_belly = (abnormality_score * 0.05) + (pitch_max / 350.0) + (pitch_std / 10.0)
        score_abdo = (abnormality_score * 0.04) + (pitch_max / 420.0)
        score_gas = (abnormality_score * 0.03) + (pitch_std / 20.0)
        score_hunger = (pitch_std / 12.0) + (rms * 15.0)
        score_tired = 2.0 - (rms * 15.0) - (pitch_max / 600.0)
        score_irritation = (pitch_std / 15.0) + (abnormality_score * 0.02)
        score_normal = 2.0 - (abnormality_score * 0.04)

        scores = np.array([score_belly, score_abdo, score_gas, score_hunger, score_tired, score_irritation, score_normal])
        exp_s = np.exp(scores - np.max(scores))
        p = exp_s / np.sum(exp_s)

        return {
            "Severe Belly Pain / Colic Cry": round(float(p[0]), 4),
            "Abdominal Distress Cry": round(float(p[1]), 4),
            "Gas & Burping Discomfort Cry": round(float(p[2]), 4),
            "Hunger Cry": round(float(p[3]), 4),
            "Fatigue & Sleepiness Cry": round(float(p[4]), 4),
            "Physical Irritation Cry": round(float(p[5]), 4),
            "Normal / Baseline Cry": round(float(p[6]), 4)
        }
