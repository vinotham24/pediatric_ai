
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "training" / "cry_test.csv"

MODEL_PATH = (
    ROOT / "models" / "cry_experiments" /
    "random_forest.joblib"
)

OUTPUT_PATH = (
    ROOT / "training" / "cry_experiments" /
    "random_forest_predictions.csv"
)


# Load model and test data
package = joblib.load(MODEL_PATH)

model = package["model"]
feature_columns = package["feature_columns"]

test_df = pd.read_csv(TEST_PATH)

X_test = test_df[feature_columns]
y_test = test_df["label"]

# Get predicted classes and probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Maximum probability for each recording
confidence = probabilities.max(axis=1)

# Create detailed output
results = pd.DataFrame({
    "filename": test_df["filename"],
    "actual_label": y_test,
    "predicted_label": predictions,
    "confidence": confidence
})

# Add probability for each category
for index, class_name in enumerate(model.classes_):
    results[f"prob_{class_name}"] = probabilities[:, index]

results.to_csv(OUTPUT_PATH, index=False)

print("\nCONFIDENCE STATISTICS")
print(results["confidence"].describe().round(4))

print("\nAVERAGE CONFIDENCE BY ACTUAL CATEGORY")
print(
    results.groupby("actual_label")["confidence"]
    .mean()
    .round(4)
    .to_string()
)

print("\nPREDICTION DISTRIBUTION")
print(
    results["predicted_label"]
    .value_counts()
    .to_string()
)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nSaved predictions to:", OUTPUT_PATH)
