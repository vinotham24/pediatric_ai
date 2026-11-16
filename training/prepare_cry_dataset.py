
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "training" / "cry_features.csv"
TRAIN_CSV = PROJECT_ROOT / "training" / "cry_train.csv"
TEST_CSV = PROJECT_ROOT / "training" / "cry_test.csv"
UNKNOWN_CSV = PROJECT_ROOT / "training" / "cry_unknown.csv"

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Feature dataset not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print("Original dataset:", df.shape)

if "filename" not in df.columns or "label" not in df.columns:
    raise ValueError("The cry feature dataset must include filename and label columns.")

missing_files = df["filename"].isna() | (df["filename"].astype(str).str.strip() == "")
if missing_files.any():
    raise ValueError(f"Found {missing_files.sum()} recordings with a missing filename.")

if df["filename"].duplicated().any():
    duplicated = df.loc[df["filename"].duplicated(), "filename"].tolist()
    raise ValueError(f"Duplicate filenames detected: {duplicated[:10]}")

# The first five filename segments form a stable installation identifier when present.
df["app_instance_id"] = (
    df["filename"].astype(str).str.split("-").str[:5].str.join("-")
)

unknown_df = df[df["label"] == "unknown"].copy()
known_df = df[df["label"] != "unknown"].copy()

missing_labels = sorted(set(EXPECTED_LABELS) - set(known_df["label"]))
if missing_labels:
    raise ValueError(f"Missing required labels in the supervised dataset: {missing_labels}")

unexpected_labels = sorted(set(known_df["label"]) - set(EXPECTED_LABELS))
if unexpected_labels:
    raise ValueError(f"Unexpected labels present in the supervised dataset: {unexpected_labels}")

unknown_df.to_csv(UNKNOWN_CSV, index=False)

X = known_df.drop(columns=["label"])
y = known_df["label"]
groups = known_df["app_instance_id"]

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_indices, test_indices = next(splitter.split(X, y, groups))
train_df = known_df.iloc[train_indices].copy()
test_df = known_df.iloc[test_indices].copy()

train_groups = set(train_df["app_instance_id"])
test_groups = set(test_df["app_instance_id"])
overlapping_groups = train_groups.intersection(test_groups)
if overlapping_groups:
    raise ValueError(
        "Data leakage detected: recordings from the same app_instance_id appear in both splits."
    )

train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("\n" + "=" * 45)
print("DATASET PREPARATION COMPLETED")
print("=" * 45)
print("\nKnown recordings:", len(known_df))
print("Unknown recordings:", len(unknown_df))
print("Training recordings:", len(train_df))
print("Testing recordings:", len(test_df))
print("Training app installations:", train_df["app_instance_id"].nunique())
print("Testing app installations:", test_df["app_instance_id"].nunique())
print("Overlapping app installations:", len(overlapping_groups))
print("\nTraining label distribution:")
print(train_df["label"].value_counts())
print("\nTesting label distribution:")
print(test_df["label"].value_counts())
print("\nTraining file:", TRAIN_CSV)
print("Testing file:", TEST_CSV)
print("Unknown file:", UNKNOWN_CSV)
print("\nDataset preparation finished successfully.")
