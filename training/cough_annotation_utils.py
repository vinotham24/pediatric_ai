"""Shared local paths and CSV/progress helpers for the cough annotation tools."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "datasets" / "cough" / "metadata"
REPORTS_DIR = PROJECT_ROOT / "datasets" / "cough" / "reports"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "cough" / "processed"
SOURCE_CSV = METADATA_DIR / "source_recordings.csv"
ANNOTATIONS_CSV = METADATA_DIR / "cough_annotations.csv"
SEGMENTS_CSV = METADATA_DIR / "segment_manifest.csv"
UNKNOWN_CSV = METADATA_DIR / "unknown_annotations.csv"
PROGRESS_MD = REPORTS_DIR / "annotation_progress.md"

ANNOTATION_FIELDS = [
    "annotation_id",
    "recording_id",
    "source_filename",
    "patient_id",
    "start_time_seconds",
    "end_time_seconds",
    "label",
    "annotator",
    "verification_status",
    "notes",
]

SOURCE_FIELDS = [
    "recording_id",
    "original_filename",
    "source_group",
    "duration_seconds",
    "sample_rate",
    "number_of_channels",
    "file_validity",
    "audio_path",
    "patient_id",
    "age_months",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        Path(temp_name).replace(path)
    except Exception:
        try:
            Path(temp_name).unlink(missing_ok=True)
        finally:
            raise


def update_progress_report() -> dict[str, object]:
    sources = read_csv(SOURCE_CSV)
    annotations = read_csv(ANNOTATIONS_CSV)
    completed_statuses = {"VERIFIED", "REJECTED"}
    completed_recordings = {
        row.get("recording_id", "")
        for row in annotations
        if row.get("verification_status", "").upper() in completed_statuses
    }
    recordings_with_entries = {row.get("recording_id", "") for row in annotations}
    pending_recordings = {
        row.get("recording_id", "")
        for row in annotations
        if row.get("verification_status", "").upper() == "PENDING"
    }
    verified_counts = Counter(
        row.get("label", "").upper()
        for row in annotations
        if row.get("verification_status", "").upper() == "VERIFIED"
    )
    unknown_count = sum(row.get("label", "").upper() == "UNKNOWN" for row in annotations)
    extracted = sum(
        row.get("status", "").upper() == "EXTRACTED"
        for row in read_csv(SEGMENTS_CSV)
    )
    awaiting = [
        row.get("recording_id", "")
        for row in sources
        if row.get("recording_id", "") not in completed_recordings
    ]

    total = len(sources)
    reviewed_count = len(completed_recordings)
    percent = (reviewed_count / total * 100.0) if total else 0.0
    report = [
        "# Manual cough annotation progress",
        "",
        f"- Total source recordings: {total}",
        f"- Recordings with at least one VERIFIED or REJECTED annotation: {reviewed_count}",
        f"- Recordings with pending annotations: {len(pending_recordings)}",
        f"- Recordings with any annotation row: {len(recordings_with_entries)}",
        f"- Review progress (by recording-level annotation status): {percent:.1f}%",
        f"- Verified cough events: {verified_counts['COUGH']}",
        f"- Verified non-cough events: {verified_counts['NON_COUGH']}",
        f"- Verified unknown events: {verified_counts['UNKNOWN']}",
        f"- Unknown-labeled annotation rows (all statuses): {unknown_count}",
        f"- Extracted verified clips: {extracted}",
        "",
        "## Verified event class distribution",
        "",
        "| Label | Verified annotations |",
        "|---|---:|",
        f"| COUGH | {verified_counts['COUGH']} |",
        f"| NON_COUGH | {verified_counts['NON_COUGH']} |",
        f"| UNKNOWN | {verified_counts['UNKNOWN']} |",
        "",
        "The recording-level progress count means a source has at least one annotation marked VERIFIED or REJECTED. It does not prove that every part of that recording was reviewed. Clips remain grouped by source recording for any later participant-wise split.",
        "",
        "## Recording IDs awaiting a verified/rejected annotation",
        "",
    ]
    report.extend(f"- `{recording_id}`" for recording_id in awaiting)
    if not awaiting:
        report.append("- None")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_MD.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "total_recordings": total,
        "reviewed_recordings": reviewed_count,
        "pending_recordings": len(pending_recordings),
        "recordings_with_entries": len(recordings_with_entries),
        "verified_counts": dict(verified_counts),
        "unknown_rows": unknown_count,
        "extracted_clips": extracted,
        "awaiting": awaiting,
    }
