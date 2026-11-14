
import pandas as pd
from pathlib import Path

DATASET_PATH = Path("donateacry-corpus")
OUTPUT_PATH = Path("training/cry_metadata.csv")

reason_labels = {
    "hu": "hungry",
    "bu": "burping",
    "bp": "belly_pain",
    "dc": "discomfort",
    "ti": "tired",
    "lo": "lonely",
    "ch": "cold_hot",
    "sc": "scared",
    "dk": "unknown"
}

age_labels = {
    "04": "0-4 weeks",
    "48": "4-8 weeks",
    "26": "2-6 months",
    "72": "7 months-2 years",
    "22": "above 2 years"
}

extensions = {".wav", ".3gp", ".caf"}

records = []
invalid_files = []

for file in DATASET_PATH.rglob("*"):

    if not file.is_file():
        continue

    if file.suffix.lower() not in extensions:
        continue

    parts = file.stem.rsplit("-", 5)

    if len(parts) != 6:
        invalid_files.append(file.name)
        continue

    uuid, timestamp, version, gender, age, reason = parts

    if reason not in reason_labels or age not in age_labels:
        invalid_files.append(file.name)
        continue

    records.append({
        "filename": file.name,
        "file_path": str(file),
        "format": file.suffix.lower(),
        "timestamp": timestamp,
        "app_version": version,
        "gender": gender,
        "age_code": age,
        "age_group": age_labels[age],
        "reason_code": reason,
        "label": reason_labels[reason]
    })

df = pd.DataFrame(records)

df.to_csv(OUTPUT_PATH, index=False)

print("\nDataset preparation completed!")
print("Valid recordings:", len(df))
print("Invalid filenames:", len(invalid_files))

print("\nCry category distribution:")
print(df["label"].value_counts())

print("\nAudio format distribution:")
print(df["format"].value_counts())

print("\nSaved metadata to:", OUTPUT_PATH)

if invalid_files:
    print("\nSample invalid filenames:")
    print(invalid_files[:10])
