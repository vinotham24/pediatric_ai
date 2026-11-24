"""Build a deduplicated archive supplement and evaluate a separate model candidate.

The production artifact is never written. Inputs overlapping the existing held-out
test split are excluded before feature extraction and training.
"""
import json
import sys
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from werkzeug.datastructures import FileStorage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.audio_service import AudioService
from services.cry_model_service import CryModelService
from training.audit_cry_archive import audit_archive


DEFAULT_ARCHIVE = Path.home() / "Downloads" / "archive.zip"
OUTPUT_DIR = ROOT / "training" / "cry_archive_audit_v1"
AUDIO_DIR = ROOT / "datasets" / "cry_archive_clean_v1"
EXPERIMENT_DIR = ROOT / "models" / "cry_experiments_v3"


def build_candidate(archive_path: Path = DEFAULT_ARCHIVE) -> dict:
    audit = audit_archive(archive_path, OUTPUT_DIR)
    manifest = pd.read_csv(OUTPUT_DIR / "archive_audio_audit.csv")
    eligible = manifest[manifest["audit_status"] == "include_unique_unseen_audio"]

    production = joblib.load(ROOT / "models" / "cry_random_forest.joblib")
    feature_columns = list(production["feature_columns"])
    cry_service = CryModelService()
    audio_service = AudioService()
    audio_rows = []
    errors = []

    with ZipFile(archive_path) as archive:
        for _, record in eligible.iterrows():
            archive_path_name = str(record["archive_path"])
            blob = archive.read(archive_path_name)
            digest = sha256(blob).hexdigest()
            label = str(record["resolved_label"])
            suffix = PurePosixPath(archive_path_name).suffix.lower()
            output_path = AUDIO_DIR / "audio" / label / f"{digest[:16]}{suffix}"
            source_path = wav_path = None
            try:
                source_path, wav_path = audio_service.decode_cry_upload(
                    FileStorage(
                        stream=BytesIO(blob),
                        filename=PurePosixPath(archive_path_name).name,
                    )
                )
                extracted = cry_service.extract_features(wav_path)
                if set(feature_columns) - set(extracted):
                    raise ValueError("Feature extraction did not match the production schema.")
                if not np.isfinite([extracted[name] for name in feature_columns]).all():
                    raise ValueError("Non-finite feature found after audio conversion.")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(blob)
                audio_rows.append({
                    "filename": f"archive_{digest[:16]}",
                    "label": label,
                    "source_archive_path": archive_path_name,
                    "sha256": digest,
                    **{name: float(extracted[name]) for name in feature_columns},
                })
            except Exception as exc:
                errors.append({
                    "archive_path": archive_path_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                for temp_path in (source_path, wav_path):
                    if temp_path:
                        audio_service.delete_recording(temp_path)

    if not audio_rows:
        raise RuntimeError("No eligible archive recordings produced valid model features.")

    archive_features = pd.DataFrame(audio_rows)
    train_path = ROOT / "training" / "cry_train.csv"
    test_path = ROOT / "training" / "cry_test.csv"
    original_train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    original_counts = original_train["label"].value_counts().to_dict()
    added_counts = archive_features["label"].value_counts().to_dict()

    combined_x = pd.concat(
        [original_train[feature_columns], archive_features[feature_columns]],
        ignore_index=True,
    )
    combined_y = pd.concat(
        [original_train["label"], archive_features["label"]],
        ignore_index=True,
    )
    if not np.isfinite(combined_x.to_numpy()).all():
        raise ValueError("Combined training features contain missing or infinite values.")

    baseline_model = production["model"]
    candidate = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    )
    candidate.fit(combined_x, combined_y)

    test_x = test[feature_columns]
    test_y = test["label"]
    baseline_pred = baseline_model.predict(test_x)
    candidate_pred = candidate.predict(test_x)
    labels = list(production["classes"])

    report = {
        "source_archive": str(archive_path),
        "audio_validation": {
            "eligible_unique_unseen_audio": int(len(eligible)),
            "feature_rows_created": int(len(archive_features)),
            "errors": errors,
        },
        "features": {
            "count": len(feature_columns),
            "ordered_names_match_production": feature_columns == list(production["feature_columns"]),
        },
        "training_counts_before_archive": {k: int(v) for k, v in original_counts.items()},
        "new_training_counts": {k: int(v) for k, v in added_counts.items()},
        "training_counts_after_archive": {
            str(k): int(v) for k, v in combined_y.value_counts().to_dict().items()
        },
        "class_balancing": (
            "SVC class_weight='balanced'; preserves all genuine examples without "
            "duplicating recordings. This changes class influence, not the raw counts."
        ),
        "untouched_existing_test_set": int(len(test_y)),
        "baseline_production_model": {
            "model_name": production.get("model_name"),
            "accuracy": float(accuracy_score(test_y, baseline_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, baseline_pred)),
            "macro_f1": float(f1_score(test_y, baseline_pred, average="macro", zero_division=0)),
            "predicted_counts": {str(k): int(v) for k, v in Counter(baseline_pred).items()},
            "classification_report": classification_report(
                test_y, baseline_pred, labels=labels, output_dict=True, zero_division=0
            ),
        },
        "archive_augmented_candidate": {
            "accuracy": float(accuracy_score(test_y, candidate_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, candidate_pred)),
            "macro_f1": float(f1_score(test_y, candidate_pred, average="macro", zero_division=0)),
            "predicted_counts": {str(k): int(v) for k, v in Counter(candidate_pred).items()},
            "classification_report": classification_report(
                test_y, candidate_pred, labels=labels, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(test_y, candidate_pred, labels=labels).tolist(),
        },
        "promotion": "not promoted; production inference artifact remains unchanged",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = OUTPUT_DIR / "cry_archive_features.csv"
    archive_features.to_csv(feature_path, index=False)
    combined_frame = pd.concat([
        original_train[["filename", "label", *feature_columns]].assign(source_type="existing_train"),
        archive_features[["filename", "label", *feature_columns]].assign(source_type="archive_new"),
    ], ignore_index=True)
    combined_frame.to_csv(OUTPUT_DIR / "cry_train_archive_augmented.csv", index=False)

    candidate_path = EXPERIMENT_DIR / "archive_augmented_svc.joblib"
    joblib.dump({
        "model": candidate,
        "feature_columns": feature_columns,
        "classes": labels,
        "model_name": "archive_augmented_svc_experimental",
        "training_counts": report["training_counts_after_archive"],
        "source_archive": str(archive_path),
    }, candidate_path)
    report["artifacts"] = {
        "clean_audio_dir": str(AUDIO_DIR / "audio"),
        "archive_features_csv": str(feature_path),
        "combined_training_csv": str(OUTPUT_DIR / "cry_train_archive_augmented.csv"),
        "candidate_model": str(candidate_path),
    }
    (OUTPUT_DIR / "candidate_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.archive), indent=2))
