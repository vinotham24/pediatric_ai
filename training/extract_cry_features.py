
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_CSV = PROJECT_ROOT / "training" / "processed_cry_metadata.csv"
OUTPUT_CSV = PROJECT_ROOT / "training" / "cry_features.csv"

PROCESSED_AUDIO_DIR = PROJECT_ROOT / "recordings" / "cry_processed"
MEL_OUTPUT_DIR = PROJECT_ROOT / "recordings" / "cry_mel"

SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 40
N_MELS = 128

TARGET_SAMPLES = SAMPLE_RATE * DURATION


# --------------------------------------------------
# 2. CREATE OUTPUT FOLDER
# --------------------------------------------------

MEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove old Mel spectrogram files so that the
# regenerated dataset contains only current files.

for old_file in MEL_OUTPUT_DIR.glob("*.npy"):
    old_file.unlink()

print("Starting cry feature extraction...")
print("Old Mel spectrogram files cleared.")


# --------------------------------------------------
# 3. LOAD PROCESSED METADATA
# --------------------------------------------------

if not METADATA_CSV.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_CSV}"
    )

metadata = pd.read_csv(METADATA_CSV)

required_columns = [
    "filename",
    "file_path",
    "label",
    "age_group",
    "gender"
]

missing_columns = [
    column for column in required_columns
    if column not in metadata.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in metadata CSV: {missing_columns}"
    )

print("Total recordings:", len(metadata))


# --------------------------------------------------
# 4. HELPER FUNCTION TO LOAD AUDIO
# --------------------------------------------------

def load_audio(audio_path):

    audio, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    # Make every recording exactly 5 seconds.
    if len(audio) < TARGET_SAMPLES:

        audio = np.pad(
            audio,
            (0, TARGET_SAMPLES - len(audio))
        )

    else:

        audio = audio[:TARGET_SAMPLES]

    return audio


# --------------------------------------------------
# 5. EXTRACT AUDIO FEATURES
# --------------------------------------------------

def extract_features(audio, filename):

    features = {}

    # ----------------------------------------------
    # MFCC FEATURES
    # ----------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    for i in range(N_MFCC):

        features[f"mfcc_{i + 1}_mean"] = float(
            np.mean(mfcc[i])
        )

        features[f"mfcc_{i + 1}_std"] = float(
            np.std(mfcc[i])
        )

    # ----------------------------------------------
    # MEL SPECTROGRAM
    # ----------------------------------------------

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(
        mel_power,
        ref=np.max
    )

    mel_mean = float(np.mean(mel_db))
    mel_std = float(np.std(mel_db))

    features["mel_mean"] = mel_mean
    features["mel_std"] = mel_std

    # Save the complete Mel spectrogram separately.
    mel_filename = Path(filename).stem + ".npy"

    mel_path = MEL_OUTPUT_DIR / mel_filename

    np.save(mel_path, mel_db)

    # ----------------------------------------------
    # PITCH FEATURES
    # ----------------------------------------------

    pitch = librosa.yin(
        audio,
        fmin=80,
        fmax=1000,
        sr=SAMPLE_RATE
    )

    features["pitch_mean"] = float(np.mean(pitch))
    features["pitch_std"] = float(np.std(pitch))

    # ----------------------------------------------
    # ZERO CROSSING RATE
    # ----------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(audio)

    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # ----------------------------------------------
    # ROOT MEAN SQUARE ENERGY
    # ----------------------------------------------

    rms = librosa.feature.rms(y=audio)

    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # ----------------------------------------------
    # SPECTRAL FEATURES
    # ----------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE
    )

    features["spectral_centroid_mean"] = float(
        np.mean(centroid)
    )

    features["spectral_bandwidth_mean"] = float(
        np.mean(bandwidth)
    )

    features["spectral_rolloff_mean"] = float(
        np.mean(rolloff)
    )

    # ----------------------------------------------
    # METADATA
    # ----------------------------------------------

    features["filename"] = filename

    # Save a relative path so the CSV is portable
    # within the project folder.
    features["mel_file"] = (
        mel_path.relative_to(PROJECT_ROOT).as_posix()
    )

    return features


# --------------------------------------------------
# 6. PROCESS ALL RECORDINGS
# --------------------------------------------------

all_features = []
failed_files = []

for index, row in metadata.iterrows():

    filename = str(row["filename"])

    # Use the path recorded in the metadata first.
    audio_path = Path(str(row["file_path"]))

    if not audio_path.is_absolute():
        audio_path = PROJECT_ROOT / audio_path

    # If that path is unavailable, try the processed
    # audio folder using the unique output filename.
    if not audio_path.exists():
        audio_path = PROCESSED_AUDIO_DIR / filename

    try:

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        audio = load_audio(audio_path)

        features = extract_features(
            audio,
            filename
        )

        # Add the recording's metadata.
        features["label"] = row["label"]
        features["age_group"] = row["age_group"]
        features["gender"] = row["gender"]

        all_features.append(features)

    except Exception as error:

        failed_files.append(
            (filename, str(error))
        )

    # Display progress every 100 recordings.
    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/{len(metadata)}"
        )


# --------------------------------------------------
# 7. SAVE FEATURE DATASET
# --------------------------------------------------

if not all_features:
    raise RuntimeError(
        "No features were extracted. Check the audio paths."
    )

features_df = pd.DataFrame(all_features)

# Keep metadata columns at the beginning.
metadata_columns = [
    "filename",
    "label",
    "age_group",
    "gender",
    "mel_file"
]

other_columns = [
    column for column in features_df.columns
    if column not in metadata_columns
]

features_df = features_df[
    metadata_columns + other_columns
]

# Replace infinite values and missing numeric values.
features_df = features_df.replace(
    [np.inf, -np.inf],
    np.nan
)

numeric_columns = features_df.select_dtypes(
    include=[np.number]
).columns

features_df[numeric_columns] = features_df[
    numeric_columns
].fillna(0)

features_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# --------------------------------------------------
# 8. FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 45)
print("CRY FEATURE EXTRACTION COMPLETED")
print("=" * 45)

print("Total metadata recordings:", len(metadata))
print("Successfully processed:", len(all_features))
print("Failed recordings:", len(failed_files))

print("Feature CSV:", OUTPUT_CSV)
print("Mel spectrogram folder:", MEL_OUTPUT_DIR)

print("\nDataset shape:", features_df.shape)

print(
    "Unique filenames:",
    features_df["filename"].nunique()
)

print(
    "Unique Mel files:",
    features_df["mel_file"].nunique()
)

print("\nLabel distribution:")

print(features_df["label"].value_counts())

if failed_files:

    print("\nFailed files:")

    for filename, error in failed_files[:10]:
        print(filename, "->", error)

print("\nFeature extraction finished.")
