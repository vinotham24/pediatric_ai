
import pandas as pd
import subprocess
import imageio_ffmpeg
from pathlib import Path

METADATA_PATH = Path("training/cry_metadata.csv")
OUTPUT_DIR = Path("recordings/cry_processed")
OUTPUT_METADATA = Path("training/processed_cry_metadata.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

df = pd.read_csv(METADATA_PATH)

# Remove only previously generated WAV files.
# Original dataset files are untouched.
for old_file in OUTPUT_DIR.glob("*.wav"):
    old_file.unlink()

successful = []
failed = []

print("Starting corrected preprocessing...")
print("Total recordings:", len(df))

for index, row in df.iterrows():

    source = Path(row["file_path"])

    # Include original extension to prevent filename collisions.
    unique_name = (
        source.stem
        + "_"
        + source.suffix.lower().replace(".", "")
        + ".wav"
    )

    output_file = OUTPUT_DIR / unique_name

    try:
        command = [
            FFMPEG,
            "-y",
            "-i", str(source),
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            str(output_file)
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )

        successful.append({
            "filename": output_file.name,
            "file_path": str(output_file),
            "label": row["label"],
            "age_group": row["age_group"],
            "gender": row["gender"]
        })

    except Exception as e:
        failed.append({
            "filename": source.name,
            "error": str(e)
        })

    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(df)}")

processed_df = pd.DataFrame(successful)

processed_df.to_csv(
    OUTPUT_METADATA,
    index=False
)

if failed:
    pd.DataFrame(failed).to_csv(
        "training/failed_cry_files.csv",
        index=False
    )

print("\nCorrected preprocessing completed!")
print("Successful:", len(successful))
print("Failed:", len(failed))
print("Unique output files:", processed_df["filename"].nunique())
