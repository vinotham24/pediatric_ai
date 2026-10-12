import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS_DIR

def create_subject_level_split(manifest_path=None, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    PREVENT DATA LEAKAGE: Splits dataset by unique subject ID (not simple file split).
    """
    if manifest_path is None:
        manifest_path = os.path.join(DATASETS_DIR, "dataset_manifest.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Group by subject
    subjects = list(set(item["subject"] for item in manifest))
    random.seed(42)
    random.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])
    test_subjects = set(subjects[n_train + n_val :])

    train_split, val_split, test_split = [], [], []

    for item in manifest:
        sbj = item["subject"]
        if sbj in train_subjects:
            train_split.append(item)
        elif sbj in val_subjects:
            val_split.append(item)
        else:
            test_split.append(item)

    splits = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }

    splits_path = os.path.join(DATASETS_DIR, "dataset_splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Subject-aware split created: Train={len(train_split)}, Val={len(val_split)}, Test={len(test_split)} samples.")
    return splits

if __name__ == '__main__':
    create_subject_level_split()
