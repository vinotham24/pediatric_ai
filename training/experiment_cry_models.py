from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "training" / "cry_train.csv"
TEST_PATH = ROOT / "training" / "cry_test.csv"
OUTPUT_DIR = ROOT / "models" / "cry_experiments"
REPORT_DIR = ROOT / "training" / "cry_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

missing_train = set(EXPECTED_LABELS) - set(train_df["label"].unique())
missing_test = set(EXPECTED_LABELS) - set(test_df["label"].unique())
if missing_train or missing_test:
    raise ValueError(f"Missing labels in split: train={sorted(missing_train)} test={sorted(missing_test)}")

EXCLUDED_COLUMNS = {"filename", "label", "age_group", "gender", "mel_file", "app_instance_id"}
feature_columns = [
    col for col in train_df.select_dtypes(include=[np.number]).columns
    if col not in EXCLUDED_COLUMNS
]

X_train = train_df[feature_columns].copy()
y_train = train_df["label"].copy()
X_test = test_df[feature_columns].copy()
y_test = test_df["label"].copy()

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing feature columns differ.")
if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training features contain NaN or infinity values.")
if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Testing features contain NaN or infinity values.")

experiments = {
    "logistic_regression_balanced": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    ),
    "svc_balanced": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    ),
    "random_forest_balanced": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "extra_trees_balanced": ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "balanced_random_forest": BalancedRandomForestClassifier(
        n_estimators=500,
        sampling_strategy="all",
        replacement=True,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
for name, model in experiments.items():
    print("\n" + "=" * 55)
    print("EXPERIMENT:", name)
    print("=" * 55)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    report = classification_report(y_test, predictions, zero_division=0)
    print("\nCLASSIFICATION REPORT")
    print(report)

    labels = model.classes_ if hasattr(model, "classes_") else sorted(y_test.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(REPORT_DIR / f"{name}_confusion_matrix.csv")
    (REPORT_DIR / f"{name}_report.txt").write_text(report, encoding="utf-8")

    results.append({
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "classes": list(labels),
            "model_name": name,
        },
        OUTPUT_DIR / f"{name}.joblib",
    )

results_df = pd.DataFrame(results).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["model"]
production_model_path = ROOT / "models" / "cry_random_forest.joblib"
backup_path = ROOT / "models" / "cry_random_forest.joblib.bak"
if production_model_path.exists():
    joblib.dump(joblib.load(production_model_path), backup_path)
joblib.dump(joblib.load(OUTPUT_DIR / f"{best_name}.joblib"), production_model_path)

print(f"\nPromoted best model: {best_name} -> {production_model_path}")
print(f"Backup saved to: {backup_path}")
