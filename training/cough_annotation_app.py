"""Run the stand-alone local browser tool for manual cough annotation.

Usage from the project root:
    .\\venv\\Scripts\\python.exe training\\cough_annotation_app.py

Then open http://127.0.0.1:5051/. This is a separate app and does not import
or modify the project's main Flask application.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import uuid
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from flask import Flask, jsonify, request, send_file

from cough_annotation_utils import (
    ANNOTATION_FIELDS,
    ANNOTATIONS_CSV,
    PROJECT_ROOT,
    SOURCE_CSV,
    read_csv,
    update_progress_report,
    write_csv_atomic,
)


HERE = Path(__file__).resolve().parent
HTML_FILE = HERE / "cough_annotator.html"
VALID_LABELS = {"COUGH", "NON_COUGH", "UNKNOWN"}
VALID_STATUSES = {"PENDING", "VERIFIED", "REJECTED"}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
app = Flask(__name__)


def source_map() -> dict[str, dict[str, str]]:
    return {row["recording_id"]: row for row in read_csv(SOURCE_CSV)}


def annotation_rows() -> list[dict[str, str]]:
    return read_csv(ANNOTATIONS_CSV)


def _source_path(record: dict[str, str]) -> Path:
    path = Path(record.get("audio_path", ""))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    allowed_root = (PROJECT_ROOT / "datasets" / "cough" / "raw" / "west_china").resolve()
    if not resolved.is_relative_to(allowed_root) or not resolved.is_file():
        raise FileNotFoundError("Source audio is missing or outside the West China source directory.")
    return resolved


@lru_cache(maxsize=24)
def _waveform_payload(recording_id: str) -> dict[str, object]:
    record = source_map().get(recording_id)
    if not record:
        raise KeyError(recording_id)
    source = _source_path(record)
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-i", str(source),
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "8000", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if result.returncode != 0 or not result.stdout:
        message = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"Could not decode waveform: {message}")
    mono = np.frombuffer(result.stdout, dtype="<f4")
    if not np.isfinite(mono).all():
        raise ValueError("Waveform contains non-finite samples.")
    duration = len(mono) / 8000.0
    bucket_count = min(1800, max(300, int(duration * 100)))
    width = int(math.ceil(len(mono) / bucket_count))
    peaks = []
    for start in range(0, len(mono), width):
        part = mono[start:start + width]
        peaks.append(round(float(np.max(np.abs(part))) if part.size else 0.0, 5))
    return {"duration_seconds": duration, "peaks": peaks}


@app.get("/")
def index():
    return send_file(HTML_FILE, mimetype="text/html")


@app.get("/api/recordings")
def api_recordings():
    return jsonify(read_csv(SOURCE_CSV))


@app.get("/api/progress")
def api_progress():
    progress = update_progress_report()
    counts = progress["verified_counts"]
    total = progress["total_recordings"]
    reviewed = progress["reviewed_recordings"]
    annotations = annotation_rows()
    pending = sum(row.get("verification_status", "").upper() == "PENDING" for row in annotations)
    return jsonify({
        "total_recordings": total,
        "reviewed_recordings": reviewed,
        "verified_cough_events": counts.get("COUGH", 0),
        "verified_non_cough_events": counts.get("NON_COUGH", 0),
        "verified_unknown_events": counts.get("UNKNOWN", 0),
        "unknown_rows": progress["unknown_rows"],
        "pending_annotation_rows": pending,
        "awaiting_recordings": len(progress["awaiting"]),
        "progress_percent": (reviewed / total * 100.0) if total else 0.0,
    })


@app.get("/api/annotations")
def api_get_annotations():
    recording_id = request.args.get("recording_id", "")
    if recording_id not in source_map():
        return jsonify({"error": "Unknown recording ID."}), 404
    rows = [row for row in annotation_rows() if row.get("recording_id") == recording_id]
    rows.sort(key=lambda row: (float(row.get("start_time_seconds") or 0), row.get("annotation_id", "")))
    return jsonify(rows)


@app.get("/api/waveform/<recording_id>")
def api_waveform(recording_id: str):
    if recording_id not in source_map():
        return jsonify({"error": "Unknown recording ID."}), 404
    try:
        return jsonify(_waveform_payload(recording_id))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422


@app.get("/audio/<recording_id>")
def serve_audio(recording_id: str):
    record = source_map().get(recording_id)
    if not record:
        return jsonify({"error": "Unknown recording ID."}), 404
    try:
        return send_file(_source_path(record), mimetype="audio/mpeg", conditional=True)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@app.post("/api/annotations")
def api_save_annotation():
    payload = request.get_json(silent=True) or {}
    sources = source_map()
    recording_id = str(payload.get("recording_id", "")).strip()
    source = sources.get(recording_id)
    if not source:
        return jsonify({"error": "Unknown recording ID."}), 404

    label = str(payload.get("label", "")).strip().upper()
    status = str(payload.get("verification_status", "PENDING")).strip().upper()
    annotator = str(payload.get("annotator", "")).strip()
    notes = str(payload.get("notes", "")).strip()
    if label not in VALID_LABELS:
        return jsonify({"error": "Choose COUGH, NON_COUGH, or UNKNOWN after listening."}), 400
    if status not in VALID_STATUSES:
        return jsonify({"error": "Invalid verification status."}), 400
    if not annotator:
        return jsonify({"error": "Enter the annotator name or initials."}), 400
    try:
        start = float(payload.get("start_time_seconds"))
        end = float(payload.get("end_time_seconds"))
        duration = float(source["duration_seconds"])
    except (TypeError, ValueError, KeyError):
        return jsonify({"error": "Enter valid start and end timestamps."}), 400
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return jsonify({"error": "End time must be greater than start time; both must be non-negative."}), 400
    if end > duration + 0.001:
        return jsonify({"error": f"End time exceeds the source duration ({duration:.3f} seconds)."}), 400
    if len(notes) > 2000:
        return jsonify({"error": "Notes must be 2,000 characters or fewer."}), 400

    rows = annotation_rows()
    annotation_id = str(payload.get("annotation_id", "")).strip()
    existing_index = next((i for i, row in enumerate(rows) if row.get("annotation_id") == annotation_id), None) if annotation_id else None
    if annotation_id and existing_index is None:
        return jsonify({"error": "Annotation no longer exists; reload the recording."}), 404
    if not annotation_id:
        annotation_id = str(uuid.uuid4())

    new_row = {
        "annotation_id": annotation_id,
        "recording_id": recording_id,
        "source_filename": source["original_filename"],
        "patient_id": source.get("patient_id", ""),
        "start_time_seconds": f"{start:.3f}",
        "end_time_seconds": f"{end:.3f}",
        "label": label,
        "annotator": annotator,
        "verification_status": status,
        "notes": notes,
    }
    if existing_index is None:
        rows.append(new_row)
    else:
        rows[existing_index] = new_row
    write_csv_atomic(ANNOTATIONS_CSV, ANNOTATION_FIELDS, rows)
    update_progress_report()
    return jsonify({"ok": True, "annotation": new_row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5051)
    args = parser.parse_args()
    if not SOURCE_CSV.exists() or not ANNOTATIONS_CSV.exists():
        parser.error("Create the source inventory and annotation CSV first.")
    print(f"Local annotator: http://127.0.0.1:{args.port}/")
    print("Use Ctrl+C in this terminal to stop the annotation server.")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
