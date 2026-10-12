import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, DATASETS_DIR
from ml.preprocessing import load_and_preprocess_audio
from ml.feature_extraction import extract_acoustic_features

def train_all_models():
    print("Training pediatric AI audio classification models with full acoustic feature vectors...")
    manifest_path = os.path.join(DATASETS_DIR, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        from training.download_datasets import setup_demo_datasets
        setup_demo_datasets()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 1. Train Sound Event Detection (SED) Model
    X_sed, y_sed = [], []
    for item in manifest:
        filepath = item["filepath"]
        audio_type = item["type"].upper()
        if os.path.exists(filepath):
            y_audio, sr = load_and_preprocess_audio(filepath)
            feat = extract_acoustic_features(y_audio, sr)
            X_sed.append(feat)
            y_sed.append(audio_type)

    if X_sed:
        sed_clf = RandomForestClassifier(n_estimators=50, random_state=42)
        sed_clf.fit(X_sed, y_sed)
        sed_dir = os.path.join(MODELS_DIR, "sound_model")
        os.makedirs(sed_dir, exist_ok=True)
        joblib.dump(sed_clf, os.path.join(sed_dir, "sed_model.joblib"))
        print("[OK] Saved SED model to models/sound_model/sed_model.joblib")

    # 2. Train Cry Model
    X_cry, y_cry = [], []
    for item in manifest:
        if item["type"] == "cry" and os.path.exists(item["filepath"]):
            y_audio, sr = load_and_preprocess_audio(item["filepath"])
            feat = extract_acoustic_features(y_audio, sr)
            X_cry.append(feat)
            y_cry.append(item["pattern"])

    if X_cry:
        cry_clf = RandomForestClassifier(n_estimators=50, random_state=42)
        cry_clf.fit(X_cry, y_cry)
        cry_dir = os.path.join(MODELS_DIR, "cry_model")
        os.makedirs(cry_dir, exist_ok=True)
        joblib.dump(cry_clf, os.path.join(cry_dir, "cry_model.joblib"))
        print("[OK] Saved Cry model to models/cry_model/cry_model.joblib")

    # 3. Train Cough Model
    X_cough, y_cough = [], []
    for item in manifest:
        if item["type"] == "cough" and os.path.exists(item["filepath"]):
            y_audio, sr = load_and_preprocess_audio(item["filepath"])
            feat = extract_acoustic_features(y_audio, sr)
            X_cough.append(feat)
            y_cough.append(item["pattern"])

    if X_cough:
        cough_clf = RandomForestClassifier(n_estimators=50, random_state=42)
        cough_clf.fit(X_cough, y_cough)
        cough_dir = os.path.join(MODELS_DIR, "cough_model")
        os.makedirs(cough_dir, exist_ok=True)
        joblib.dump(cough_clf, os.path.join(cough_dir, "cough_model.joblib"))
        print("[OK] Saved Cough model to models/cough_model/cough_model.joblib")

    print("All models trained and synchronized successfully!")

if __name__ == '__main__':
    train_all_models()
