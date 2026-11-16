from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "training" / "cry_train.csv"
TEST_PATH = ROOT / "training" / "cry_test.csv"

OUTPUT_DIR = ROOT / "models" / "cry_experiments"
REPORT_DIR = ROOT / "training" / "cry_experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load original datasets
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

EXCLUDED = {
    "filename", "label", "age_group",
    "gender", "mel_file", "app_instance_id"
}

feature_columns = [
    col for col in train_df.select_dtypes(include=[np.number]).columns
    if col not in EXCLUDED
]

X_train = train_df[feature_columns].copy()
y_train = train_df["label"].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df["label"].copy()

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Feature columns do not match.")

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training data contains invalid values.")

if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Testing data contains invalid values.")

print("\n========== ORIGINAL TRAINING COUNTS ==========")
print(y_train.value_counts().to_string())

# Oversample training data only
oversampler = RandomOverSampler(random_state=42)

X_balanced, y_balanced = oversampler.fit_resample(
    X_train,
    y_train
)

print("\n========== BALANCED TRAINING COUNTS ==========")
print(pd.Series(y_balanced).value_counts().to_string())

print("\nOriginal training rows:", len(X_train))
print("Balanced training rows:", len(X_balanced))
print("Testing rows (unchanged):", len(X_test))

# Train a new Random Forest
model = RandomForestClassifier(
    n_estimators=500,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

print("\nTraining oversampled Random Forest...")
model.fit(X_balanced, y_balanced)

# Evaluate on original untouched test data
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)
weighted_f1 = f1_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)

print("\n========== OVERSAMPLED MODEL RESULTS ==========")
print("Accuracy:", round(accuracy, 4))
print("Balanced accuracy:", round(balanced_accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))

report = classification_report(
    y_test,
    predictions,
    labels=model.classes_,
    target_names=model.classes_,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

pd.DataFrame(
    matrix,
    index=model.classes_,
    columns=model.classes_
).to_csv(REPORT_DIR / "oversampled_confusion_matrix.csv")

(REPORT_DIR / "oversampled_report.txt").write_text(
    report,
    encoding="utf-8"
)

# Save separately; do not replace the production model
model_path = OUTPUT_DIR / "random_forest_oversampled.joblib"

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "classes": list(model.classes_)
    },
    model_path
)

print("\nExperimental model saved:", model_path)
print("Evaluation report saved:", REPORT_DIR / "oversampled_report.txt")
print("\nOVERSAMPLING EXPERIMENT COMPLETED")
