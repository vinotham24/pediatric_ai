# Infant audio dataset intake

This intake format is for building separate Baby Cry and Baby Cough models. It contains no sample data. Do not train from a dataset until its recording-level age, event labels, source identity, and reuse terms have been verified.

## Manifest

Start from `training/infant_audio_manifest_template.csv`. Use one row per annotated event. Keep the original audio files in a local dataset directory (the `datasets/` directory is git-ignored) and set `audio_path` relative to that directory. Do not put names or other personal information in IDs.

Required fields:

- `event_id`: unique anonymous event identifier.
- `recording_id`: anonymous recording identifier; repeated when multiple events come from one recording.
- `source_id`: anonymous infant/subject identifier used to keep all of one infant's audio in one validation split.
- `audio_path`: path to the preserved source audio, relative to the dataset root.
- `age_days`: verified age at recording; the intake validator accepts only ages 0–365 days.
- `event_label`: exactly `cry` or `cough`. Hiccups and all other sounds must not be labeled as cough.
- `classification_label`: genuine dataset label for the event-specific classifier (for example, a documented cry category or cough diagnosis). Do not infer this from a filename.
- `annotation_start_ms` and `annotation_end_ms`: verified event boundaries within the source recording.
- `annotation_method`: how the event and label were verified (for example, expert annotation or published dataset annotation).
- `dataset_source` and `license`: source citation/URL and the license or written access terms for the audio.

## Intake validation

From the project root, run:

```powershell
.\venv\Scripts\python.exe training\validate_infant_audio_manifest.py `
  --manifest datasets\infant_audio\manifest.csv `
  --dataset-root datasets\infant_audio
```

The validator checks metadata completeness, infant age, permitted event labels, source/recording identity consistency, event intervals, path containment, and non-empty audio files. It does not decode, listen to, relabel, split, or train on audio. It never changes a model.

Before group-aware evaluation, each target class needs enough independent `source_id` values to put different infants in training and test groups. Confirm cohort consent and dataset-specific reuse terms; a paper's license does not automatically grant rights to audio collected from third parties.

The eventual application must keep event detection (exactly `cry` and `cough`) separate from the existing cry classifier and a separately trained cough classifier. Do not create either model artifact until the data is verified and evaluation is complete. The production artifact `models/cry_random_forest.joblib` must remain unchanged.
