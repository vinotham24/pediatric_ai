import os
import numpy as np
import librosa
from config import RESPIRATORY_CLASSES

class RespiratoryAnalyzer:
    def __init__(self):
        pass

    def analyze(self, y, sr=16000):
        duration = float(len(y) / sr)
        rms = np.mean(librosa.feature.rms(y=y)[0])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])

        # High frequency tonal components signal wheezing / stridor
        if spec_cent > 3000 and zcr > 0.08:
            top_pattern = "Wheezing-like sound"
            probs = {
                "Wheezing-like sound": 0.76,
                "Abnormal breathing": 0.14,
                "Stridor-like sound": 0.06,
                "Normal breathing": 0.04
            }
        elif spec_cent > 2200 and rms > 0.03:
            top_pattern = "Stridor-like sound"
            probs = {
                "Stridor-like sound": 0.68,
                "Abnormal breathing": 0.18,
                "Wheezing-like sound": 0.10,
                "Normal breathing": 0.04
            }
        elif rms > 0.025:
            top_pattern = "Abnormal breathing"
            probs = {
                "Abnormal breathing": 0.65,
                "Wheezing-like sound": 0.18,
                "Stridor-like sound": 0.10,
                "Normal breathing": 0.07
            }
        else:
            top_pattern = "Normal breathing"
            probs = {
                "Normal breathing": 0.88,
                "Abnormal breathing": 0.07,
                "Wheezing-like sound": 0.03,
                "Stridor-like sound": 0.02
            }

        return {
            "duration": round(duration, 2),
            "top_pattern": top_pattern,
            "confidence": round(probs[top_pattern], 4),
            "probabilities": probs
        }
