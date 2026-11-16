from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "training" / "cry_train.csv"
TEST_CSV = PROJECT_ROOT / "training" / "cry_test.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "cry_random_forest.joblib"
REPORT_PATH = PROJECT_ROOT / "training" / "cry_model_report.txt"
CONFUSION_PATH = PROJECT_ROOT / "training" / "cry_confusion_matrix.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training and testing datasets...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

EXPECTED_LABELS = [
    "belly_pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired",
]
missing_train = set(EXPECTED_LABELS) - set(train_df["label"].unique())
missing_test = set(EXPECTED_LABELS) - set(test_df["label"].unique())
if missing_train or missing_test:
    raise ValueError(f"Missing labels in split: train={sorted(missing_train)} test={sorted(missing_test)}")

excluded_columns = ["filename", "label", "age_group", "gender", "mel_file", "app_instance_id"]
feature_columns = [
    column for column in train_df.columns if column not in excluded_columns and pd.api.types.is_numeric_dtype(train_df[column])
]
if not feature_columns:
    raise ValueError("No numerical audio features found.")

# Ensure the model sees the exact same feature ordering during inference.
missing_test_features = [column for column in feature_columns if column not in test_df.columns]
if missing_test_features:
    raise ValueError(f"Missing test features: {missing_test_features}")

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()
y_train = train_df["label"]
y_test = test_df["label"]

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Invalid values found in training features.")
if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Invalid values found in testing features.")

print("\nNumber of audio features:", len(feature_columns))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nCreating SVC model with class weighting and reproducible split.")
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
)

print("\nTraining started...")
model.fit(X_train, y_train)
print("Training completed.")

print("\nPredicting test recordings...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
report = classification_report(y_test, y_pred, zero_division=0)

print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")
print("\nClassification Report:")
print(report)

class_labels = model.classes_ if hasattr(model, "classes_") else sorted(y_test.unique())
matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
confusion_df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
confusion_df.to_csv(CONFUSION_PATH)

model_package = {
    "model": model,
    "feature_columns": feature_columns,
    "classes": list(class_labels),
    "model_name": "svc_balanced",
    "random_seed": 42,
}

if MODEL_PATH.exists():
    backup_path = MODEL_DIR / "cry_random_forest.joblib.bak"
    joblib.dump(joblib.load(MODEL_PATH), backup_path)
joblib.dump(model_package, MODEL_PATH)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("CRY MODEL EVALUATION\n")
    file.write("=" * 45 + "\n\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"Audio features: {len(feature_columns)}\n")
    file.write(f"Model: {model_package['model_name']}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
    file.write(f"Macro F1: {macro_f1:.4f}\n")
    file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    file.write(report)

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved evaluation report to: {REPORT_PATH}")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
    file.write(f"Macro F1-score: {macro_f1:.4f}\n")
    file.write(f"Weighted F1-score: {weighted_f1:.4f}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write(report)

    file.write("\n\nTOP 15 FEATURES\n")
    file.write(importance_df.head(15).to_string(index=False))


# --------------------------------------------------
# 13. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED")
print("=" * 50)

print("\nSaved model:", MODEL_PATH)
print("Saved report:", REPORT_PATH)
print("Saved confusion matrix:", CONFUSION_PATH)

print("\nAll operations completed successfully.")
