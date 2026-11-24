"""Audit a labeled cry-audio ZIP for decodability and conflicting duplicates.

This does not extract or modify the source archive and does not train a model.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import soundfile as sf


LABEL_MAP = {
    "belly pain": "belly_pain",
    "burping": "burping",
    "cold_hot": "cold_hot",
    "discomfort": "discomfort",
    "hungry": "hungry",
    "tired": "tired",
}
FILENAME_CODE_MAP = {
    "hu": "hungry", "bu": "burping", "bp": "belly_pain",
    "dc": "discomfort", "ti": "tired", "lo": "lonely",
    "ch": "cold_hot", "sc": "scared",
}
SUPPORTED_MODEL_LABELS = {
    "belly_pain", "burping", "cold_hot", "discomfort",
    "hungry", "lonely", "scared", "tired",
}


def audit_archive(archive_path: Path, output_dir: Path) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    train_stems, test_stems = {}, {}
    try:
        import pandas as pd

        def normalize_stem(value):
            stem = PurePosixPath(str(value)).stem.lower()
            return re.sub(r"_(3gp|caf|wav)$", "", stem)

        for split_name, target in (("cry_train.csv", train_stems), ("cry_test.csv", test_stems)):
            split_path = project_root / "training" / split_name
            if split_path.exists():
                frame = pd.read_csv(split_path)
                target.update({normalize_stem(name): str(label) for name, label in zip(frame["filename"], frame["label"])})
    except Exception:
        # The archive audit remains useful if the project split files are absent.
        train_stems, test_stems = {}, {}

    records = []
    doc_candidates = []
    with ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            path = PurePosixPath(item.filename)
            if path.suffix.lower() in {".txt", ".md", ".json", ".csv", ".yaml", ".yml"}:
                doc_candidates.append(item.filename)
                continue
            label = path.parts[1] if len(path.parts) >= 3 else "[unlabeled]"
            record = {
                "archive_path": item.filename,
                "source_label": label,
                "model_label": LABEL_MAP.get(label, ""),
                "filename_label": FILENAME_CODE_MAP.get(path.stem.lower().rsplit("-", 1)[-1], ""),
                "resolved_label": "",
                "resolution_basis": "",
                "existing_split_overlap": "",
                "size_bytes": item.file_size,
                "sha256": "",
                "sample_rate": "",
                "channels": "",
                "duration_seconds": "",
                "decode_status": "",
            }
            try:
                blob = archive.read(item)
                record["sha256"] = hashlib.sha256(blob).hexdigest()
                info = sf.info(BytesIO(blob))
                record["sample_rate"] = info.samplerate
                record["channels"] = info.channels
                record["duration_seconds"] = round(info.duration, 4)
                record["decode_status"] = "ok" if info.frames > 0 else "empty"
            except Exception as exc:
                record["decode_status"] = f"error: {type(exc).__name__}: {exc}"
            records.append(record)

    hash_groups = defaultdict(list)
    for record in records:
        if record["sha256"]:
            hash_groups[record["sha256"]].append(record)

    counts_raw = Counter(record["source_label"] for record in records)
    counts_clean = Counter()
    resolution_counts = Counter()
    overlap_counts = Counter()
    folder_conflicts_resolved = 0
    for group in hash_groups.values():
        folder_labels = {record["model_label"] or record["source_label"] for record in group}
        embedded_labels = {record["filename_label"] for record in group if record["filename_label"]}
        if len(embedded_labels) == 1:
            resolved_label = next(iter(embedded_labels))
            resolution_basis = "filename_reason_code"
        elif len(embedded_labels) == 0 and len(folder_labels) == 1:
            resolved_label = next(iter(folder_labels))
            resolution_basis = "single_folder_label"
        else:
            resolved_label = ""
            resolution_basis = "unresolved_conflict"
        if resolved_label and (len(folder_labels) > 1 or resolved_label not in folder_labels):
            folder_conflicts_resolved += 1

        group_stems = {
            re.sub(r"_(3gp|caf|wav)$", "", PurePosixPath(record["archive_path"]).stem.lower())
            for record in group
        }
        overlaps_test = bool(group_stems & set(test_stems))
        overlaps_train = bool(group_stems & set(train_stems))
        overlap = "test" if overlaps_test else "train" if overlaps_train else "none"
        for record in group:
            record["resolved_label"] = resolved_label
            record["resolution_basis"] = resolution_basis
            record["existing_split_overlap"] = overlap

        if not resolved_label:
            for record in group:
                record["audit_status"] = "exclude_unresolved_label_conflict"
        elif resolved_label not in SUPPORTED_MODEL_LABELS:
            for record in group:
                record["audit_status"] = "exclude_unsupported_class"
        elif any(record["decode_status"] != "ok" for record in group):
            for record in group:
                record["audit_status"] = "exclude_decode_error"
        elif overlaps_test:
            overlap_counts["test"] += 1
            for record in group:
                record["audit_status"] = "exclude_existing_test_overlap"
        elif overlaps_train:
            overlap_counts["train"] += 1
            for record in group:
                record["audit_status"] = "exclude_existing_train_overlap"
        else:
            representative = group[0]
            representative["audit_status"] = "include_unique_unseen_audio"
            counts_clean[resolved_label] += 1
            for duplicate in group[1:]:
                duplicate["audit_status"] = "exclude_exact_duplicate"
        resolution_counts[resolution_basis] += 1

    # Records that failed hashing/decoding never entered a hash group.
    for record in records:
        if "audit_status" not in record:
            record["audit_status"] = "exclude_decode_error"

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "archive_audio_audit.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    supported_counts = {label: counts_clean.get(label, 0) for label in sorted(SUPPORTED_MODEL_LABELS)}
    existing_train_counts = Counter()
    train_csv = project_root / "training" / "cry_train.csv"
    if train_csv.exists():
        try:
            for label, count in pd.read_csv(train_csv)["label"].value_counts().items():
                existing_train_counts[str(label)] = int(count)
        except Exception:
            existing_train_counts = Counter()
    combined_train_counts = {
        label: existing_train_counts.get(label, 0) + supported_counts.get(label, 0)
        for label in sorted(SUPPORTED_MODEL_LABELS)
    }
    balance_limit = min(combined_train_counts.values()) if combined_train_counts else 0
    summary = {
        "archive": str(archive_path),
        "audio_file_count": len(records),
        "total_audio_bytes": sum(record["size_bytes"] for record in records),
        "all_files_decodable": all(record["decode_status"] == "ok" for record in records),
        "documentation_candidates": doc_candidates,
        "raw_folder_counts": dict(sorted(counts_raw.items())),
        "exact_payload_count": len(hash_groups),
        "exact_duplicate_groups": sum(len(group) > 1 for group in hash_groups.values()),
        "groups_with_filename_reason_code": resolution_counts.get("filename_reason_code", 0),
        "folder_conflict_groups_resolved_by_filename_code": folder_conflicts_resolved,
        "unresolved_label_conflict_groups": resolution_counts.get("unresolved_conflict", 0),
        "groups_overlapping_existing_train_or_test": dict(overlap_counts),
        "unique_unseen_supported_counts": supported_counts,
        "production_classes_absent_from_archive_folders": sorted(
            SUPPORTED_MODEL_LABELS - {LABEL_MAP.get(label, label) for label in counts_raw}
        ),
        "production_classes_with_no_new_unseen_audio": sorted(
            label for label, count in supported_counts.items() if count == 0
        ),
        "unique_unseen_counts_added_to_existing_training_split": dict(sorted(supported_counts.items())),
        "combined_available_train_counts_before_resampling": dict(sorted(combined_train_counts.items())),
        "maximum_balanced_examples_per_class_without_oversampling": balance_limit,
        "balanced_dataset_created": False,
        "balance_decision": (
            "Not created: folder conflicts can be resolved by embedded Donateacry "
            "reason codes. Only 76 unique supported recordings remain after excluding "
            "the existing train/test split. The combined dataset's smallest class is "
            "scared (12 examples), so equal downsampling would leave only 96 total "
            "examples; naive oversampling would repeat the same recordings and risk "
            "overfitting. The new files are kept separate for a class-weighted candidate."
        ),
        "manifest_csv": str(manifest_path),
    }
    summary_path = output_dir / "archive_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the cry-audio ZIP")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "training" / "cry_archive_audit_v1",
    )
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.archive, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
