
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

from services.prediction_status import determine_prediction_status


SAMPLE_RATE = 16000
DURATION = 5
TARGET_SAMPLES = SAMPLE_RATE * DURATION


def validate_class_order(model_classes, declared_classes):
    """Reject packages whose saved class order disagrees with predict_proba."""
    model_order = [str(label) for label in model_classes]
    declared_order = [str(label) for label in declared_classes]
    if model_order != declared_order:
        raise ValueError(
            "Cry model class order mismatch: package classes must match "
            "estimator.classes_ exactly."
        )
    return model_order


class CryModelService:

    def __init__(self, model_path=None):
        project_root = Path(__file__).resolve().parent.parent

        candidates = []
        if model_path is not None:
            candidates.append(Path(model_path))

        # Inference is pinned to the production artifact; experiments are never
        # promoted implicitly if it is absent.
        candidates.append(project_root / "models" / "cry_random_forest.joblib")

        model_file = None
        for candidate in candidates:
            if candidate and candidate.exists():
                model_file = candidate
                break

        if model_file is None:
            raise FileNotFoundError(
                "Cry model not found. Expected a trained model in the project models directory."
            )

        package = joblib.load(model_file)
        self.model = package["model"]
        self.feature_columns = list(package.get("feature_columns", []))
        package_classes = package.get("classes", list(self.model.classes_))
        self.classes = [str(label) for label in package_classes]
        self.model_name = package.get("model_name", model_file.name)
        self.screening_validation = package.get("screening_validation", {})
        if not self.feature_columns:
            raise ValueError("Model package does not contain feature columns.")

        if len(self.feature_columns) != 91:
            raise ValueError(
                f"Expected 91 features, found {len(self.feature_columns)}"
            )

        if hasattr(self.model, "classes_"):
            self.classes = validate_class_order(self.model.classes_, self.classes)

        print(f"Cry model loaded successfully from {model_file.name}.")
        print("Number of features:", len(self.feature_columns))
        print("Classes:", self.classes)

    def extract_features(self, audio_path):
        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio, sr = librosa.load(
                str(audio_path),
                sr=SAMPLE_RATE,
                mono=True,
                duration=DURATION,
            )
        except Exception as exc:
            raise ValueError(f"Unable to decode converted WAV audio: {exc}") from exc

        if audio.size == 0:
            raise ValueError("The uploaded audio file is empty.")

        if len(audio) < TARGET_SAMPLES:
            audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
        else:
            audio = audio[:TARGET_SAMPLES]

        features = {}

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=40,
            n_fft=2048,
            hop_length=512
        )

        for i in range(40):
            features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

        mel_power = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        mel_db = librosa.power_to_db(mel_power, ref=np.max)

        features["mel_mean"] = float(np.mean(mel_db))
        features["mel_std"] = float(np.std(mel_db))

        pitch = librosa.yin(audio, fmin=80, fmax=1000, sr=SAMPLE_RATE)
        features["pitch_mean"] = float(np.mean(pitch))
        features["pitch_std"] = float(np.std(pitch))

        zcr = librosa.feature.zero_crossing_rate(audio)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        rms = librosa.feature.rms(y=audio)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)

        features["spectral_centroid_mean"] = float(np.mean(centroid))
        features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))

        missing = [name for name in self.feature_columns if name not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        for key in self.feature_columns:
            value = features[key]
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                features[key] = 0.0
            else:
                features[key] = numeric_value

        return features

    def predict(self, audio_path):
        features = self.extract_features(audio_path)

        input_data = pd.DataFrame(
            [[float(features[name]) for name in self.feature_columns]],
            columns=self.feature_columns
        )

        for column in self.feature_columns:
            if not np.isfinite(input_data[column].to_numpy()).all():
                input_data[column] = 0.0

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(input_data)[0]
            class_labels = [str(label) for label in self.model.classes_]
            predicted_class = str(self.model.predict(input_data)[0])
        else:
            predictions = self.model.predict(input_data)[0]
            class_labels = [str(label) for label in self.classes]
            probabilities = np.zeros(len(class_labels), dtype=float)
            predicted_class = str(predictions)
            if predicted_class not in class_labels:
                raise ValueError(f"Model predicted an unsupported class: {predicted_class}")
            probabilities[class_labels.index(predicted_class)] = 1.0

        if len(probabilities) != len(class_labels) or not np.isfinite(probabilities).all():
            raise ValueError("Model returned invalid class probabilities.")

        probability_results = {
            label: round(float(score), 4)
            for label, score in zip(class_labels, probabilities)
        }

        if predicted_class not in probability_results:
            raise ValueError(
                f"Model predicted a class missing from its probability output: {predicted_class}"
            )
        # SVC's calibrated probability argmax can disagree with its classifier
        # decision. Use predict() for the label and report that class's score.
        top_label = predicted_class
        top_score = probability_results[top_label]

        status = determine_prediction_status(
            top_label, top_score, validation=getattr(self, "screening_validation", {})
        )
        predicted_label = top_label if status == "CONFIDENT" else status.lower()
        score_interpretation = (
            "Uncalibrated SVC score; not a probability of correctness. "
            "The saved evaluation does not justify a screening threshold."
        )

        return {
            "predicted_label": predicted_label,
            "top_label": top_label,
            "prediction_status": status,
            "possible_category": top_label if status in {"CONFIDENT", "UNCERTAIN"} else None,
            "model_score": round(float(top_score), 4),
            "model_score_interpretation": score_interpretation,
            "class_scores": probability_results,
            "disclaimer": (
                "Experimental AI output only. "
                "This is not a medical diagnosis."
            ),
            "warning": score_interpretation if status != "CONFIDENT" else "Experimental model prediction only; not a confirmed finding.",
        }
