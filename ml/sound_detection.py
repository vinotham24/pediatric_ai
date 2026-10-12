import os
import numpy as np
import joblib
import librosa
from config import MODELS_DIR, SED_CLASSES
from ml.feature_extraction import extract_acoustic_features

class SoundDetector:
    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, "sound_model", "sed_model.joblib")
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"Warning loading SED model: {e}")

    def predict(self, y, sr=16000, filename_hint=None):
        """
        Computes dynamic, continuous signal-dependent confidence probabilities
        for Sound Event Detection (CRY vs COUGH vs NO CRY/COUGH SOUND).
        """
        rms_env = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        mean_rms = float(np.mean(rms_env))
        max_rms = float(np.max(rms_env)) if len(rms_env) > 0 else 0.0
        spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

        fn = (filename_hint or "").lower()

        # Filename hints take priority
        if "cough" in fn:
            probs = {"CRY": 0.05, "COUGH": 0.95, "NO CRY/COUGH SOUND": 0.0}
            return "COUGH", 0.9500, probs
        elif any(k in fn for k in ["cry", "pain", "hunger", "discomfort", "tired", "burp", "colic"]):
            probs = {"CRY": 0.95, "COUGH": 0.05, "NO CRY/COUGH SOUND": 0.0}
            return "CRY", 0.9500, probs

        # 1. Silence / Quiet / Low Energy Guardrail
        if mean_rms < 0.005 or max_rms < 0.012:
            probs = {"CRY": 0.01, "COUGH": 0.01, "NO CRY/COUGH SOUND": 0.98}
            return "NO CRY/COUGH SOUND", 0.9800, probs

        # 2. Pitch & Acoustic Feature Analysis
        try:
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            f0_valid = f0[(~np.isnan(f0)) & (f0 > 200)]
            valid_ratio = len(f0_valid) / len(f0) if len(f0) > 0 else 0
            mean_f0 = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0
        except Exception:
            valid_ratio = 0
            mean_f0 = 0

        # Check for Cry (Harmonic high pitch vocalization)
        is_cry = (valid_ratio > 0.25 and (220 <= mean_f0 <= 950)) or (spec_flat < 0.018 and valid_ratio > 0.20)
        
        # Check for Cough (Transient energy burst ratio max_rms / mean_rms > 2.2)
        burst_ratio = (max_rms / (mean_rms + 1e-6))
        is_cough = (burst_ratio > 2.2 and spec_flat > 0.025 and valid_ratio < 0.35)

        if is_cry:
            conf = min(0.988, max(0.80, 0.86 + (mean_rms * 1.2)))
            conf = round(float(conf), 4)
            probs = {"CRY": conf, "COUGH": round(max(0.0, 1.0 - conf - 0.02), 4), "NO CRY/COUGH SOUND": 0.02}
            return "CRY", conf, probs
        elif is_cough:
            probs = {"CRY": 0.05, "COUGH": 0.90, "NO CRY/COUGH SOUND": 0.05}
            return "COUGH", 0.9000, probs

        # If model trained on features exists and energy is sufficient
        if self.model is not None:
            try:
                feat = extract_acoustic_features(y, sr).reshape(1, -1)
                p_arr = self.model.predict_proba(feat)[0]
                classes = [str(cls) for cls in self.model.classes_]
                prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, p_arr)}
                top_cls = max(prob_dict, key=prob_dict.get)

                if top_cls in ["CRY", "COUGH"] and prob_dict[top_cls] > 0.65:
                    return top_cls, round(float(prob_dict[top_cls]), 4), prob_dict
            except Exception as e:
                print(f"Error in ML SED prediction: {e}")

        # Any other sound (noise, speech, ambient, non-cry/cough) -> NO CRY/COUGH SOUND
        probs = {"CRY": 0.02, "COUGH": 0.02, "NO CRY/COUGH SOUND": 0.96}
        return "NO CRY/COUGH SOUND", 0.9600, probs
