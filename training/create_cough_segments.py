"""Extract only manually verified annotation intervals into 16 kHz mono WAVs.

Run from the project root after saving annotations:
    .\\venv\\Scripts\\python.exe training\\create_cough_segments.py

COUGH and NON_COUGH annotations marked VERIFIED become clips. UNKNOWN intervals
are copied to metadata/unknown_annotations.csv and never become training clips.
The script does not trim or pad an event to fit configured duration limits.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf

from cough_annotation_utils import (
    ANNOTATIONS_CSV,
    ANNOTATION_FIELDS,
    METADATA_DIR,
    PROJECT_ROOT,
    PROCESSED_DIR,
    SEGMENTS_CSV,
    SOURCE_CSV,
    UNKNOWN_CSV,
    read_csv,
    update_progress_report,
    write_csv_atomic,
)


SEGMENT_FIELDS = [
    "annotation_id", "recording_id", "patient_id", "source_filename", "label",
    "start_time_seconds", "end_time_seconds", "duration_seconds", "sample_rate",
    "clip_path", "status", "error",
]


def validate_interval(start: float, end: float, source_duration: float,
                      minimum_duration: float, maximum_duration: float) -> str | None:
    """Return an explanatory reason when the interval should not be extracted."""
    values = (start, end, source_duration, minimum_duration, maximum_duration)
    if not all(math.isfinite(value) for value in values):
        return "Non-finite timestamp or duration."
    if start < 0 or end <= start:
        return "Invalid or zero-duration interval."
    if end > source_duration + 0.001:
        return "Interval exceeds source recording duration."
    duration = end - start
    if duration < minimum_duration:
        return f"Segment is shorter than {minimum_duration:g}s; manual review required."
    if duration > maximum_duration:
        return f"Segment exceeds {maximum_duration:g}s; manual review required."
    return None


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return result[:90] or "item"


def _decode_mono(source: Path, sample_rate: int, ffmpeg: str) -> np.ndarray:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-i", str(source),
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Audio decode failed: {detail or result.returncode}")
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not np.isfinite(audio).all():
        raise ValueError("Decoded audio contains non-finite samples.")
    return audio


def _safe_generated_path(relative_path: str, class_label: str) -> Path:
    class_dir = (PROCESSED_DIR / ("cough" if class_label == "COUGH" else "non_cough")).resolve()
    target = (PROJECT_ROOT / relative_path).resolve()
    if not target.is_relative_to(class_dir) or target == class_dir or target.suffix.lower() != ".wav":
        raise ValueError(f"Refusing to access generated path outside {class_dir}: {target}")
    return target


def _write_segment(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        import os
        os.close(fd)
        sf.write(temp_name, audio, sample_rate, format="WAV", subtype="PCM_16")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--minimum-duration", type=float, default=0.05)
    parser.add_argument("--maximum-duration", type=float, default=10.0,
                        help="Long intervals are logged for manual review; they are never truncated.")
    args = parser.parse_args()
    if args.sample_rate <= 0 or args.minimum_duration < 0 or args.maximum_duration <= args.minimum_duration:
        parser.error("Invalid sample rate or duration limits.")
    if not SOURCE_CSV.exists() or not ANNOTATIONS_CSV.exists():
        parser.error("Required source_recordings.csv or cough_annotations.csv is missing.")

    source_rows = read_csv(SOURCE_CSV)
    sources = {row["recording_id"]: row for row in source_rows}
    annotations = read_csv(ANNOTATIONS_CSV)
    previous = read_csv(SEGMENTS_CSV)
    previous_generated = {
        row.get("clip_path", "")
        for row in previous
        if row.get("status", "").upper() == "EXTRACTED" and row.get("clip_path")
    }
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cache_id = None
    cache_audio: np.ndarray | None = None
    result_rows: list[dict[str, str]] = []
    unknown_rows: list[dict[str, str]] = []
    new_generated: set[str] = set()

    for annotation in sorted(annotations, key=lambda r: (r.get("recording_id", ""), r.get("start_time_seconds", ""))):
        annotation_id = annotation.get("annotation_id", "")
        recording_id = annotation.get("recording_id", "")
        label = annotation.get("label", "").upper()
        status = annotation.get("verification_status", "").upper()
        base = {
            "annotation_id": annotation_id,
            "recording_id": recording_id,
            "patient_id": annotation.get("patient_id", ""),
            "source_filename": annotation.get("source_filename", ""),
            "label": label,
            "start_time_seconds": annotation.get("start_time_seconds", ""),
            "end_time_seconds": annotation.get("end_time_seconds", ""),
            "duration_seconds": "",
            "sample_rate": str(args.sample_rate),
            "clip_path": "",
            "status": "",
            "error": "",
        }
        if label == "UNKNOWN":
            unknown_rows.append(annotation)
            base["status"] = "UNKNOWN_STORED_SEPARATELY"
            result_rows.append(base)
            continue
        if status != "VERIFIED" or label not in {"COUGH", "NON_COUGH"}:
            base["status"] = "SKIPPED_NOT_VERIFIED"
            result_rows.append(base)
            continue
        source_record = sources.get(recording_id)
        if source_record is None:
            base.update(status="FAILED", error="Recording ID not found in source inventory.")
            result_rows.append(base)
            continue
        try:
            start = float(annotation.get("start_time_seconds", ""))
            end = float(annotation.get("end_time_seconds", ""))
            source_duration = float(source_record.get("duration_seconds", ""))
            reason = validate_interval(start, end, source_duration, args.minimum_duration, args.maximum_duration)
            if reason:
                base.update(status="NEEDS_MANUAL_REVIEW", error=reason)
                result_rows.append(base)
                continue
            source_path = Path(source_record["audio_path"])
            if not source_path.is_absolute():
                source_path = PROJECT_ROOT / source_path
            source_path = source_path.resolve()
            approved_root = (PROJECT_ROOT / "datasets" / "cough" / "raw" / "west_china").resolve()
            if not source_path.is_relative_to(approved_root) or not source_path.is_file():
                raise FileNotFoundError("Source audio is missing or outside the West China source directory.")
            if cache_id != recording_id:
                cache_audio = _decode_mono(source_path, args.sample_rate, ffmpeg)
                cache_id = recording_id
            assert cache_audio is not None
            actual_duration = len(cache_audio) / args.sample_rate
            if end > actual_duration + 0.001:
                raise ValueError("Annotation end exceeds decoded audio duration.")
            begin_sample = int(round(start * args.sample_rate))
            end_sample = int(round(end * args.sample_rate))
            segment = cache_audio[begin_sample:end_sample]
            actual_segment_duration = len(segment) / args.sample_rate
            reason = validate_interval(start, start + actual_segment_duration, actual_duration,
                                       args.minimum_duration, args.maximum_duration)
            if reason or segment.size == 0:
                base.update(status="NEEDS_MANUAL_REVIEW", error=reason or "Empty segment after timestamp conversion.")
                result_rows.append(base)
                continue

            target_dir = PROCESSED_DIR / ("cough" if label == "COUGH" else "non_cough")
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{_safe_name(recording_id)}__{_safe_name(annotation_id)}__{label}.wav"
            target = target_dir / filename
            relative_target = target.relative_to(PROJECT_ROOT).as_posix()
            if target.exists() and relative_target not in previous_generated:
                raise FileExistsError(f"Refusing to overwrite untracked output: {target}")
            _write_segment(target, segment, args.sample_rate)
            new_generated.add(relative_target)
            base.update(
                duration_seconds=f"{actual_segment_duration:.6f}",
                clip_path=relative_target,
                status="EXTRACTED",
            )
        except Exception as exc:
            base.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        result_rows.append(base)

    # Remove only clips previously listed as generated by this script that are
    # no longer represented by a valid VERIFIED annotation.
    for stale in previous_generated - new_generated:
        prior = next((row for row in previous if row.get("clip_path") == stale), None)
        if not prior or prior.get("label", "").upper() not in {"COUGH", "NON_COUGH"}:
            continue
        try:
            stale_path = _safe_generated_path(stale, prior["label"].upper())
            if stale_path.is_file():
                stale_path.unlink()
        except Exception as exc:
            print(f"Warning: could not remove stale generated clip {stale}: {exc}", file=sys.stderr)

    write_csv_atomic(SEGMENTS_CSV, SEGMENT_FIELDS, result_rows)
    write_csv_atomic(UNKNOWN_CSV, ANNOTATION_FIELDS, unknown_rows)
    progress = update_progress_report()
    extracted = sum(row["status"] == "EXTRACTED" for row in result_rows)
    review = sum(row["status"] == "NEEDS_MANUAL_REVIEW" for row in result_rows)
    failed = sum(row["status"] == "FAILED" for row in result_rows)
    print(f"Extracted verified clips: {extracted}")
    print(f"UNKNOWN annotations stored separately: {len(unknown_rows)}")
    print(f"Intervals requiring manual review: {review}; failures: {failed}")
    print(f"Progress report: {PROJECT_ROOT / 'datasets' / 'cough' / 'reports' / 'annotation_progress.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
