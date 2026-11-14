
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# -----------------------------------------
# 1. PROJECT PATHS
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "training" / "cry_features.csv"

TRAIN_CSV = PROJECT_ROOT / "training" / "cry_train.csv"
TEST_CSV = PROJECT_ROOT / "training" / "cry_test.csv"
UNKNOWN_CSV = PROJECT_ROOT / "training" / "cry_unknown.csv"


# -----------------------------------------
# 2. LOAD FEATURE DATASET
# -----------------------------------------

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

print("Original dataset:", df.shape)


# -----------------------------------------
# 3. EXTRACT APP INSTALLATION ID
# -----------------------------------------

# The first five filename sections form the UUID.
# Recordings from the same installation must
# stay together in either training or testing.

df["app_instance_id"] = (
    df["filename"]
    .str.split("-")
    .str[:5]
    .str.join("-")
)


# -----------------------------------------
# 4. SEPARATE UNKNOWN RECORDINGS
# -----------------------------------------

unknown_df = df[df["label"] == "unknown"].copy()

known_df = df[df["label"] != "unknown"].copy()

print("\nKnown recordings:", len(known_df))
print("Unknown recordings:", len(unknown_df))

# Keep unknown recordings separately.
unknown_df.to_csv(UNKNOWN_CSV, index=False)


# -----------------------------------------
# 5. GROUP-AWARE STRATIFIED SPLIT
# -----------------------------------------

X = known_df.drop(columns=["label"])

y = known_df["label"]

groups = known_df["app_instance_id"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

train_df = known_df.iloc[train_indices].copy()

test_df = known_df.iloc[test_indices].copy()


# -----------------------------------------
# 6. CHECK FOR DATA LEAKAGE
# -----------------------------------------

train_groups = set(train_df["app_instance_id"])

test_groups = set(test_df["app_instance_id"])

overlapping_groups = train_groups.intersection(
    test_groups
)

if overlapping_groups:
    raise ValueError(
        "Data leakage detected! App installations "
        "are present in both datasets."
    )


# -----------------------------------------
# 7. SAVE TRAINING AND TESTING DATASETS
# -----------------------------------------

train_df.to_csv(TRAIN_CSV, index=False)

test_df.to_csv(TEST_CSV, index=False)


# -----------------------------------------
# 8. DISPLAY RESULTS
# -----------------------------------------

print("\n" + "=" * 45)
print("DATASET PREPARATION COMPLETED")
print("=" * 45)

print("\nTraining recordings:", len(train_df))
print("Testing recordings:", len(test_df))

print(
    "\nTraining app installations:",
    train_df["app_instance_id"].nunique()
)

print(
    "Testing app installations:",
    test_df["app_instance_id"].nunique()
)

print(
    "Overlapping app installations:",
    len(overlapping_groups)
)

print("\nTraining label distribution:")
print(train_df["label"].value_counts())

print("\nTesting label distribution:")
print(test_df["label"].value_counts())

print("\nTraining file:", TRAIN_CSV)
print("Testing file:", TEST_CSV)
print("Unknown file:", UNKNOWN_CSV)

print("\nDataset preparation finished successfully.")
