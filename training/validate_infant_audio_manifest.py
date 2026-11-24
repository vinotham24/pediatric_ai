"""Validate verified infant cry/cough annotations before any model training.

This script only checks a CSV manifest and file references. It does not decode,
label, split, train, or modify model artifacts.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "event_id",
    "recording_id",
    "source_id",
    "audio_path",
    "age_days",
    "event_label",
    "classification_label",
    "annotation_start_ms",
    "annotation_end_ms",
    "annotation_method",
    "dataset_source",
    "license",
}
EVENT_LABELS = {"cry", "cough"}


def validate_manifest(manifest_path, dataset_root):
    """Return (errors, summary) for the given manifest and audio root."""
    errors = []
    summary = {
        "rows": 0,
        "recordings": set(),
        "sources": set(),
        "newborn_sources": set(),
        "event_counts": Counter(),
        "class_sources": defaultdict(set),
    }

    manifest_path = Path(manifest_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    if not manifest_path.is_file():
        return [f"Manifest not found: {manifest_path}"], summary
    if not dataset_root.is_dir():
        return [f"Dataset root not found: {dataset_root}"], summary

    event_ids = set()
    recording_metadata = {}
    source_ages = {}

    try:
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                return [f"Missing columns: {', '.join(missing)}"], summary

            for row_number, row in enumerate(reader, start=2):
                summary["rows"] += 1
                prefix = f"row {row_number}"
                values = {key: (row.get(key) or "").strip() for key in REQUIRED_COLUMNS}

                for field in REQUIRED_COLUMNS:
                    if not values[field]:
                        errors.append(f"{prefix}: {field} is required")

                event_id = values["event_id"]
                if event_id:
                    if event_id in event_ids:
                        errors.append(f"{prefix}: duplicate event_id {event_id!r}")
                    event_ids.add(event_id)

                event_label = values["event_label"].lower()
                if event_label and event_label not in EVENT_LABELS:
                    errors.append(
                        f"{prefix}: event_label must be exactly cry or cough; "
                        f"got {values['event_label']!r}"
                    )

                age_days = None
                try:
                    age_days = int(values["age_days"])
                    if age_days < 0 or age_days > 365:
                        errors.append(f"{prefix}: age_days must be between 0 and 365")
                except (TypeError, ValueError):
                    if values["age_days"]:
                        errors.append(f"{prefix}: age_days must be a whole number")

                source_id = values["source_id"]
                recording_id = values["recording_id"]
                if source_id:
                    summary["sources"].add(source_id)
                    if age_days is not None:
                        previous_age = source_ages.setdefault(source_id, age_days)
                        if previous_age != age_days:
                            errors.append(
                                f"{prefix}: source_id {source_id!r} has conflicting ages"
                            )
                        if age_days <= 28:
                            summary["newborn_sources"].add(source_id)

                if recording_id:
                    summary["recordings"].add(recording_id)
                    metadata = (source_id, age_days)
                    previous = recording_metadata.setdefault(recording_id, metadata)
                    if previous != metadata:
                        errors.append(
                            f"{prefix}: recording_id {recording_id!r} has conflicting "
                            "source_id or age_days"
                        )

                if event_label in EVENT_LABELS:
                    summary["event_counts"][event_label] += 1
                    class_label = values["classification_label"]
                    if class_label and source_id:
                        summary["class_sources"][(event_label, class_label)].add(source_id)

                try:
                    start_ms = float(values["annotation_start_ms"])
                    end_ms = float(values["annotation_end_ms"])
                    if start_ms < 0 or end_ms <= start_ms:
                        errors.append(
                            f"{prefix}: event interval must satisfy 0 <= start < end"
                        )
                except (TypeError, ValueError):
                    if values["annotation_start_ms"] or values["annotation_end_ms"]:
                        errors.append(f"{prefix}: event boundaries must be numeric milliseconds")

                audio_path = values["audio_path"]
                if audio_path:
                    relative_path = Path(audio_path)
                    if relative_path.is_absolute():
                        errors.append(f"{prefix}: audio_path must be relative to dataset-root")
                    else:
                        candidate = (dataset_root / relative_path).resolve()
                        try:
                            candidate.relative_to(dataset_root)
                        except ValueError:
                            errors.append(f"{prefix}: audio_path escapes dataset-root")
                        else:
                            if not candidate.is_file():
                                errors.append(f"{prefix}: audio file not found: {audio_path}")
                            elif candidate.stat().st_size == 0:
                                errors.append(f"{prefix}: audio file is empty: {audio_path}")
    except (OSError, UnicodeError, csv.Error) as exc:
        return [f"Unable to read manifest: {exc}"], summary

    return errors, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "datasets" / "infant_audio" / "manifest.csv"),
        help="CSV manifest path",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "datasets" / "infant_audio"),
        help="Directory against which relative audio_path values are resolved",
    )
    args = parser.parse_args()

    errors, summary = validate_manifest(args.manifest, args.dataset_root)
    print(f"Annotated events: {summary['rows']}")
    print(f"Unique recordings: {len(summary['recordings'])}")
    print(f"Unique infant sources: {len(summary['sources'])}")
    print(f"Newborn sources (<=28 days): {len(summary['newborn_sources'])}")
    print(f"Event labels: {dict(summary['event_counts'])}")
    for (event_label, class_label), sources in sorted(summary["class_sources"].items()):
        print(f"{event_label} / {class_label}: {len(sources)} infant sources")

    if errors:
        print(f"Manifest invalid ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        return 1
    if summary["rows"] == 0:
        print("Manifest is empty; no training data is available.")
        return 1
    print("Manifest and referenced files pass intake checks. Training is not performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
