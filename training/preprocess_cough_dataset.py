"""Preprocess only manually verified cough/non-cough event intervals.

This script never labels recordings, invents event boundaries, truncates audio
to a fixed window, or modifies source audio. It ignores rows unless their label,
annotation status, source file, and explicit event boundaries are present.

Run from the project root:
    python training/preprocess_cough_dataset.py

Required packages are already listed in requirements.txt: numpy, soundfile,
imageio-ffmpeg. FFmpeg decodes the input and resamples/monophonizes it in memory.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDINGS = PROJECT_ROOT / "datasets" / "cough" / "metadata" / "recordings.csv"
DEFAULT_EVENTS = PROJECT_ROOT / "datasets" / "cough" / "metadata" / "cough_events.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "cough" / "processed"
DEFAULT_REPORTS = PROJECT_ROOT / "datasets" / "cough" / "reports"
VALID_LABELS = {"COUGH", "NON_COUGH"}
VERIFIED_STATUSES = {"verified", "verified_manual", "verified_expert"}


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return stem[:100] or "recording"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _decode_mono_resampled(path: Path, sample_rate: int, ffmpeg: str) -> np.ndarray:
    command = [
        ffmpeg,
        "-nostdin",
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FFmpeg decode failed: {detail or result.returncode}")
    if not result.stdout:
        raise ValueError("Decoded audio is empty.")
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not np.isfinite(audio).all():
        raise ValueError("Decoded audio contains non-finite samples.")
    return audio


def _interval(row: dict[str, str], prefix: str = "event_") -> tuple[float, float]:
    start_key = f"{prefix}start_seconds"
    end_key = f"{prefix}end_seconds"
    try:
        start = float(row.get(start_key, ""))
        end = float(row.get(end_key, ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("Explicit event start/end seconds are required.") from exc
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Event interval must be finite, non-negative, and end after start.")
    return start, end


def _record_failure(results: list[dict[str, str]], *, recording_id: str, event_id: str,
                    label: str, status: str, error: str, audio_path: str = "") -> None:
    results.append({
        "recording_id": recording_id,
        "event_id": event_id,
        "label": label,
        "status": status,
        "audio_path": audio_path,
        "processed_path": "",
        "duration_seconds": "",
        "sample_rate": "",
        "error": error,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--cough-events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--minimum-duration", type=float, default=0.25,
                        help="Reject annotated clips shorter than this many seconds.")
    parser.add_argument("--silence-rms-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    if args.sample_rate <= 0 or args.minimum_duration <= 0:
        parser.error("sample-rate and minimum-duration must be positive")

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        print(f"Unable to locate FFmpeg through imageio-ffmpeg: {exc}", file=sys.stderr)
        return 2

    recordings = _read_csv(args.recordings)
    event_rows = _read_csv(args.cough_events)
    recording_by_id = {
        row.get("recording_id", "").strip(): row
        for row in recordings
        if row.get("recording_id", "").strip()
    }

    work: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for event in event_rows:
        if event.get("label", "").strip().upper() != "COUGH":
            continue
        if event.get("annotation_status", "").strip().lower() not in VERIFIED_STATUSES:
            continue
        rec_id = event.get("recording_id", "").strip()
        if rec_id in recording_by_id:
            work.append((event.get("event_id", "").strip(), recording_by_id[rec_id], event))

    # Non-cough events must have verified recording-level source labels and
    # explicit boundaries in recordings.csv. No entire recording is assumed
    # to be an event unless those boundaries cover it explicitly.
    for recording in recordings:
        if recording.get("label", "").strip().upper() != "NON_COUGH":
            continue
        if recording.get("annotation_status", "").strip().lower() not in VERIFIED_STATUSES:
            continue
        work.append((recording.get("recording_id", "").strip(), recording, recording))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    seen_output_names: set[str] = set()

    for item_number, (event_id, recording, annotation) in enumerate(work, start=1):
        recording_id = recording.get("recording_id", "").strip()
        label = annotation.get("label", recording.get("label", "")).strip().upper()
        if label not in VALID_LABELS:
            _record_failure(results, recording_id=recording_id, event_id=event_id,
                            label=label, status="rejected", error="Unsupported binary label.")
            continue

        rel_path = recording.get("audio_path", "").strip()
        source = Path(rel_path)
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        if not source.is_file():
            _record_failure(results, recording_id=recording_id, event_id=event_id,
                            label=label, status="failed", audio_path=rel_path,
                            error="Source audio file does not exist.")
            continue

        try:
            start, end = _interval(annotation)
            audio = _decode_mono_resampled(source, args.sample_rate, ffmpeg)
            begin = int(round(start * args.sample_rate))
            finish = int(round(end * args.sample_rate))
            if finish > len(audio):
                raise ValueError(
                    f"Event end ({end:.3f}s) exceeds decoded duration "
                    f"({len(audio) / args.sample_rate:.3f}s)."
                )
            segment = audio[begin:finish]
            duration = len(segment) / args.sample_rate
            if duration < args.minimum_duration:
                raise ValueError(f"Annotated event is too short ({duration:.3f}s).")
            rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
            peak = float(np.max(np.abs(segment))) if segment.size else 0.0
            if not segment.size or peak == 0.0 or rms < args.silence_rms_threshold:
                raise ValueError("Annotated event is silent or below the RMS threshold.")

            output_dir = args.output_dir / ("cough" if label == "COUGH" else "non_cough")
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{_safe_stem(recording_id)}_{_safe_stem(event_id)}"
            output_name = f"{stem}.wav"
            if output_name.casefold() in seen_output_names:
                output_name = f"{stem}_{item_number}.wav"
            seen_output_names.add(output_name.casefold())
            output_path = output_dir / output_name
            sf.write(output_path, segment, args.sample_rate, subtype="PCM_16")
            results.append({
                "recording_id": recording_id,
                "event_id": event_id,
                "label": label,
                "status": "processed",
                "audio_path": rel_path,
                "processed_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "duration_seconds": f"{duration:.6f}",
                "sample_rate": str(args.sample_rate),
                "error": "",
            })
        except Exception as exc:
            _record_failure(results, recording_id=recording_id, event_id=event_id,
                            label=label, status="failed", audio_path=rel_path,
                            error=f"{type(exc).__name__}: {exc}")

    results_path = args.reports_dir / "preprocessing_results.csv"
    result_fields = [
        "recording_id", "event_id", "label", "status", "audio_path",
        "processed_path", "duration_seconds", "sample_rate", "error",
    ]
    with results_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(results)

    processed_count = sum(row["status"] == "processed" for row in results)
    failed_count = sum(row["status"] == "failed" for row in results)
    rejected_count = sum(row["status"] == "rejected" for row in results)
    summary_path = args.reports_dir / "preprocessing_summary.md"
    summary_path.write_text(
        "# Cough dataset preprocessing summary\n\n"
        f"- Run time (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"- Recordings manifest: `{args.recordings}`\n"
        f"- Cough events manifest: `{args.cough_events}`\n"
        f"- Target sample rate: {args.sample_rate} Hz, mono\n"
        f"- Explicit verified event intervals considered: {len(work)}\n"
        f"- Successfully processed: {processed_count}\n"
        f"- Failed: {failed_count}\n"
        f"- Rejected: {rejected_count}\n"
        f"- Details: `{results_path}`\n\n"
        "Only rows labeled COUGH or NON_COUGH with a verified annotation status "
        "and explicit start/end times are eligible. Original recordings are read-only. "
        "Segments are not padded or fixed-length truncated. A zero-row run means no "
        "verified event intervals were available; it does not synthesize samples.\n",
        encoding="utf-8",
    )
    print(f"Verified intervals considered: {len(work)}")
    print(f"Processed: {processed_count}; failed: {failed_count}; rejected: {rejected_count}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
