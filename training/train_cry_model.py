
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "training" / "cry_train.csv"
TEST_CSV = PROJECT_ROOT / "training" / "cry_test.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "cry_random_forest.joblib"
REPORT_PATH = PROJECT_ROOT / "training" / "cry_model_report.txt"
CONFUSION_PATH = PROJECT_ROOT / "training" / "cry_confusion_matrix.csv"


# --------------------------------------------------
# 2. CREATE MODEL FOLDER
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. LOAD TRAINING AND TESTING DATA
# --------------------------------------------------

print("Loading training and testing datasets...")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# --------------------------------------------------
# 4. SELECT ACOUSTIC FEATURES
# --------------------------------------------------

# Exclude metadata and identifiers.
# We only want numerical audio features.

excluded_columns = [
    "filename",
    "label",
    "age_group",
    "gender",
    "mel_file",
    "app_instance_id"
]

feature_columns = [
    column
    for column in train_df.columns
    if column not in excluded_columns
]

# Keep only numerical columns.
feature_columns = [
    column
    for column in feature_columns
    if pd.api.types.is_numeric_dtype(train_df[column])
]

if not feature_columns:
    raise ValueError("No numerical audio features found.")

# Ensure training and testing have the same features.
missing_test_features = [
    column
    for column in feature_columns
    if column not in test_df.columns
]

if missing_test_features:
    raise ValueError(
        f"Missing test features: {missing_test_features}"
    )

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

y_train = train_df["label"]
y_test = test_df["label"]

# Check for invalid feature values.
if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Invalid values found in training features.")

if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Invalid values found in testing features.")

print("\nNumber of audio features:", len(feature_columns))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. BUILD RANDOM FOREST MODEL
# --------------------------------------------------

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 6. TRAIN THE MODEL
# --------------------------------------------------

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 7. MAKE PREDICTIONS
# --------------------------------------------------

print("\nPredicting test recordings...")

y_pred = model.predict(X_test)


# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + "=" * 50)
print("RANDOM FOREST MODEL RESULTS")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nClassification Report:")
print(report)


# --------------------------------------------------
# 9. SAVE CONFUSION MATRIX
# --------------------------------------------------

class_labels = model.classes_

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

confusion_df = pd.DataFrame(
    matrix,
    index=class_labels,
    columns=class_labels
)

confusion_df.to_csv(CONFUSION_PATH)


# --------------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------------

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 Important Audio Features:")

print(
    importance_df.head(15).to_string(index=False)
)


# --------------------------------------------------
# 11. SAVE TRAINED MODEL
# --------------------------------------------------

model_package = {
    "model": model,
    "feature_columns": feature_columns,
    "classes": list(model.classes_)
}

joblib.dump(model_package, MODEL_PATH)


# --------------------------------------------------
# 12. SAVE EVALUATION REPORT
# --------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as file:

    file.write("RANDOM FOREST MODEL EVALUATION\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"Audio features: {len(feature_columns)}\n\n")

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
