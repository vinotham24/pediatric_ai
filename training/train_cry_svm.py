from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

train_df = pd.read_csv(ROOT / "training" / "cry_train.csv")
test_df = pd.read_csv(ROOT / "training" / "cry_test.csv")

excluded = {
    "filename", "label", "age_group",
    "gender", "mel_file", "app_instance_id"
}

feature_columns = [
    c for c in train_df.select_dtypes(include=[np.number]).columns
    if c not in excluded
]

X_train = train_df[feature_columns].copy()
y_train = train_df["label"].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df["label"].copy()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])

print("Training SVM...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced = balanced_accuracy_score(y_test, predictions)
macro = f1_score(y_test, predictions, average="macro", zero_division=0)
weighted = f1_score(y_test, predictions, average="weighted", zero_division=0)

print("\n========== SVM RESULTS ==========")
print("Accuracy:", round(accuracy, 4))
print("Balanced accuracy:", round(balanced, 4))
print("Macro F1:", round(macro, 4))
print("Weighted F1:", round(weighted, 4))

report = classification_report(
    y_test,
    predictions,
    labels=model.classes_,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

output_dir = ROOT / "models" / "cry_experiments"
report_dir = ROOT / "training" / "cry_experiments"

output_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "classes": list(model.classes_)
    },
    output_dir / "svm_balanced.joblib"
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

pd.DataFrame(
    matrix,
    index=model.classes_,
    columns=model.classes_
).to_csv(report_dir / "svm_confusion_matrix.csv")

(report_dir / "svm_report.txt").write_text(
    report,
    encoding="utf-8"
)

print("\nSVM experiment completed.")
print("Model saved: models/cry_experiments/svm_balanced.joblib")
