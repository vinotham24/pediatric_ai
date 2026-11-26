"""Local manual review UI for contributor cry tags; source metadata/audio are read-only."""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = None

try:
    import soundfile as sf
except Exception:
    sf = None

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "training" / "processed_cry_metadata_clean.csv"
REVIEWS = ROOT / "training" / "reports" / "cry_label_review_progress_phase8.csv"
STATUSES = {"PENDING", "VERIFIED", "QUESTIONABLE", "REJECTED"}
MINORITY = {"belly_pain", "burping", "cold_hot", "discomfort", "lonely", "scared", "tired"}
DECODABLE_EXTENSIONS = {".wav", ".caf", ".3gp"}
SILENCE_RMS_THRESHOLD = 1e-4
DECODE_TIMEOUT_SECONDS = 30
STATE_LOCK = threading.RLock()


def _decode_for_review(path: Path) -> tuple[str, bytes | None, str]:
    """Validate and decode audio. Non-WAV formats become in-memory PCM WAV for browsers."""
    if not path.is_file():
        return "MISSING", None, "Source audio file is missing."
    if path.suffix.lower() not in DECODABLE_EXTENSIONS:
        return "UNSUPPORTED FORMAT", None, f"File extension {path.suffix or '(none)'} is not supported."
    if FFMPEG_EXE is None:
        if path.suffix.lower() != ".wav" or sf is None:
            return "UNSUPPORTED FORMAT", None, "No installed decoder is available for this format."
        try:
            import soundfile as soundfile_module
            samples, sample_rate = soundfile_module.read(str(path), dtype="float32", always_2d=True)
            import io
            buffer = io.BytesIO()
            soundfile_module.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = buffer.getvalue()
        except Exception as exc:
            return "CORRUPT", None, f"WAV decode failed: {exc}"
    else:
        command = [FFMPEG_EXE, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=DECODE_TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired:
            return "CORRUPT", None, "Audio decode timed out."
        except OSError as exc:
            return "UNSUPPORTED FORMAT", None, f"Installed decoder could not be started: {exc}"
        if result.returncode != 0 or not result.stdout.startswith(b"RIFF") or b"WAVE" not in result.stdout[:16]:
            return "CORRUPT", None, result.stderr.decode("utf-8", errors="replace")[:500] or "Audio decoder rejected the file."
        wav_bytes = result.stdout

    if not wav_bytes:
        return "CORRUPT", None, "Decoder returned no audio."
    if sf is None:
        # No signal measurement library: a valid decode alone is not enough to mark it reviewable.
        return "UNSUPPORTED FORMAT", None, "Decoded audio could not be checked for silence (soundfile unavailable)."
    try:
        import io
        samples, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    except Exception as exc:
        return "CORRUPT", None, f"Decoded audio validation failed: {exc}"
    if samples.size == 0:
        return "SILENT", None, "Decoded audio contains no samples."
    if not __import__("numpy").isfinite(samples).all():
        return "CORRUPT", None, "Decoded audio contains non-finite samples."
    rms = float((samples.astype("float64") ** 2).mean() ** 0.5)
    if rms <= SILENCE_RMS_THRESHOLD:
        return "SILENT", None, f"Audio RMS {rms:.2g} is at or below the silence threshold."
    # Original WAV is served byte-for-byte. CAF/3GP playback is an in-memory PCM WAV derivative.
    return "READY", (path.read_bytes() if path.suffix.lower() == ".wav" else wav_bytes), ""


def _safe_original_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    try:
        path = (ROOT / raw_path.replace("\\", "/")).resolve()
        if ROOT.resolve() not in path.parents:
            return None
        return path
    except (OSError, RuntimeError):
        return None


def load_records():
    """Load all metadata rows in scope, retaining missing/invalid files in the queue."""
    records = {}
    with METADATA.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("label") not in MINORITY:
                continue
            key = f"{row.get('app_instance_id', '')}-{row.get('timestamp', '')}"
            path = _safe_original_path(row.get("original_file_path", ""))
            is_clean = "donateacry_corpus_cleaned_and_updated_data" in row.get("original_file_path", "")
            candidate = {
                "recording_key": key,
                "original_filename": row.get("original_filename") or Path(row.get("original_file_path", "")).name or "(filename missing)",
                "original_path": str(path) if path else "",
                "format": row.get("format") or (path.suffix if path else ""),
                "contributor_label": row["label"],
                "reason_code": row.get("reason_code", ""),
                "folder_label": Path(row.get("original_file_path", "").replace("\\", "/")).parent.name or "(folder missing)",
                "age_group": row.get("age_group", ""),
                "gender": row.get("gender", ""),
                "app_instance_id": row.get("app_instance_id", ""),
                "is_clean": is_clean,
            }
            previous = records.get(key)
            # Keep one event per source key. Prefer an existing representation, then cleaned copy.
            if previous is None:
                records[key] = candidate
            else:
                previous_exists = bool(previous["original_path"] and Path(previous["original_path"]).is_file())
                candidate_exists = bool(candidate["original_path"] and Path(candidate["original_path"]).is_file())
                if (candidate_exists and not previous_exists) or (candidate_exists == previous_exists and is_clean and not previous["is_clean"]):
                    records[key] = candidate

    for record in records.values():
        source = Path(record["original_path"]) if record["original_path"] else None
        status, playback, detail = _decode_for_review(source) if source else ("MISSING", None, "Source path is missing or outside the project directory.")
        record["availability_status"] = status
        record["availability_detail"] = detail
        record["playback_bytes"] = playback
        record["is_clean"] = bool(record.pop("is_clean"))
    return records


RECORDS = {}


def load_reviews():
    data = {}
    if REVIEWS.exists():
        with REVIEWS.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                data[row["recording_key"]] = row
    return data


REVIEWS_STATE = {}


def persist_reviews():
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    fields = ["recording_key", "original_filename", "contributor_label", "folder_label", "review_status", "reviewer", "notes", "updated_at_utc"]
    temp = REVIEWS.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(REVIEWS_STATE):
            w.writerow({k: REVIEWS_STATE[key].get(k, "") for k in fields})
    os.replace(temp, REVIEWS)


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>Cry label review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18212b}header,.card{border:1px solid #ccd4dc;border-radius:10px;padding:1rem;margin-bottom:1rem}label{display:block;margin:.7rem 0 .25rem}select,input,textarea,button{font:inherit;padding:.55rem}textarea{width:98%;min-height:5rem}audio{width:100%;margin:1rem 0}button{cursor:pointer}small{color:#4a5663}.row{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}.status{font-weight:700;padding:.5rem;background:#fff3cd;border-radius:5px}</style></head><body>
<h1>Manual cry-tag review</h1><header><b>Review only.</b> Contributor tags are not medically verified ground truth. The reviewer decision is separate from source metadata. Listen to the available audio before making a decision. CAF/3GP audio is decoded to a temporary in-memory WAV for browser playback; it does not replace the original.</header>
<div class="card"><div class="row"><label>Filter class <select id="filter"><option value="">All minority classes</option></select></label><label>Recording <select id="record"></select></label></div><p id="progress"></p><div id="availability" class="status" role="status"></div><div id="info"></div><audio id="audio" controls preload="none"></audio><p><a id="download">Open/download playback audio</a></p>
<label>Review status <select id="status"><option>PENDING</option><option>VERIFIED</option><option>QUESTIONABLE</option><option>REJECTED</option></select></label>
<label>Reviewer (optional) <input id="reviewer" maxlength="120"></label><label>Notes <textarea id="notes"></textarea></label><button id="save">Save review</button> <span id="saved" role="status"></span></div><small>If a record is marked MISSING, CORRUPT, SILENT, or UNSUPPORTED FORMAT, its review controls are disabled. Do not review a file unless its audio plays successfully.</small>
<script>
const filterEl=document.getElementById('filter'),recordEl=document.getElementById('record'),progressEl=document.getElementById('progress'),infoEl=document.getElementById('info'),availabilityEl=document.getElementById('availability'),audioEl=document.getElementById('audio'),downloadEl=document.getElementById('download'),statusEl=document.getElementById('status'),reviewerEl=document.getElementById('reviewer'),notesEl=document.getElementById('notes'),savedEl=document.getElementById('saved'),saveEl=document.getElementById('save');
let data=[];let selected=null;let playbackConfirmed=false;
async function init(){data=await(await fetch('/api/records')).json();let classes=[...new Set(data.map(x=>x.contributor_label))].sort();for(let c of classes){let o=document.createElement('option');o.value=c;o.textContent=c;filterEl.append(o)}filterEl.onchange=render;recordEl.onchange=change;render()}
function render(){let cls=filterEl.value;let arr=data.filter(x=>!cls||x.contributor_label===cls);arr.sort((a,b)=>((a.contributor_label==='cold_hot'&&a.folder_label==='discomfort')?-1:0)-((b.contributor_label==='cold_hot'&&b.folder_label==='discomfort')?-1:0)||a.contributor_label.localeCompare(b.contributor_label)||a.original_filename.localeCompare(b.original_filename));recordEl.innerHTML='';for(let r of arr){let o=document.createElement('option');o.value=r.recording_key;o.textContent=`${r.availability_status} | ${r.contributor_label} | ${r.original_filename}`;recordEl.append(o)}if(!arr.length)return;recordEl.value=selected&&arr.some(x=>x.recording_key===selected)?selected:arr[0].recording_key;change()}
function setReviewEnabled(enabled){statusEl.disabled=!enabled;reviewerEl.disabled=!enabled;notesEl.disabled=!enabled;saveEl.disabled=!enabled}
audioEl.addEventListener('playing',()=>{playbackConfirmed=true;availabilityEl.textContent='Audio status: READY — playback started; review controls enabled.';setReviewEnabled(true)});
audioEl.addEventListener('error',()=>{playbackConfirmed=false;let r=data.find(x=>x.recording_key===selected);if(r)r.availability_status='UNSUPPORTED FORMAT';availabilityEl.textContent='Audio status: UNSUPPORTED FORMAT — browser playback failed; review controls disabled.';setReviewEnabled(false)});
function change(){selected=recordEl.value;let r=data.find(x=>x.recording_key===selected);if(!r)return;playbackConfirmed=false;availabilityEl.textContent=`Audio status: ${r.availability_status}${r.availability_detail?` — ${r.availability_detail}`:''}`;infoEl.innerHTML=`<b>Contributor-submitted label:</b> ${esc(r.contributor_label)} (code ${esc(r.reason_code)})<br><b>Source folder:</b> ${esc(r.folder_label)}<br><b>Age group in source metadata:</b> ${esc(r.age_group||'not supplied')} (not independently verified)<br><b>Gender field:</b> ${esc(r.gender||'not supplied')}<br><b>Recording key:</b> ${esc(r.recording_key)}<br><b>File:</b> ${esc(r.original_filename)} (${esc(r.format)})`;let ready=r.availability_status==='READY';audioEl.pause();audioEl.removeAttribute('src');audioEl.load();audioEl.hidden=!ready;if(ready)audioEl.src='/audio/'+encodeURIComponent(r.recording_key);downloadEl.href=ready?audioEl.src:'#';downloadEl.hidden=!ready;let state=r.review||{};statusEl.value=state.review_status||'PENDING';setReviewEnabled(false);if(ready)availabilityEl.textContent+=' — play audio to enable review.';savedEl.textContent='';let n=data.filter(x=>x.review&&x.review.review_status&&x.review.review_status!=='PENDING').length;progressEl.textContent=`${n} of ${data.length} unique minority-class recording keys reviewed. Source copies with the same app-installation/timestamp key are grouped.`}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
saveEl.onclick=async()=>{let out=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({recording_key:selected,review_status:statusEl.value,reviewer:reviewerEl.value,notes:notesEl.value,playback_confirmed:playbackConfirmed})});let j=await out.json();savedEl.textContent=j.ok?'Saved separately to review-progress CSV.':j.error;if(j.ok){let r=data.find(x=>x.recording_key===selected);r.review=j.review;change()}};
init();</script></body></html>'''


def _range_bounds(range_header: str, size: int) -> tuple[int, int] | None:
    """Return an inclusive single byte range, or None for malformed/unsatisfiable ranges."""
    if size <= 0 or "," in range_header:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    try:
        if not first:
            suffix = int(last)
            if suffix <= 0:
                return None
            return max(0, size - suffix), size - 1
        start = int(first)
        end = int(last) if last else size - 1
    except ValueError:
        return None
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


class Handler(BaseHTTPRequestHandler):
    def send_body(self, body, content_type, status=200, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.send_body(PAGE.encode(), "text/html; charset=utf-8")
        if parsed.path == "/api/records":
            rows = []
            with STATE_LOCK:
                for rec in RECORDS.values():
                    x = {k: v for k, v in rec.items() if k not in {"original_path", "playback_bytes", "is_clean"}}
                    x["review"] = REVIEWS_STATE.get(rec["recording_key"], {})
                    rows.append(x)
            return self.send_body(json.dumps(rows).encode(), "application/json")
        if parsed.path.startswith("/audio/"):
            key = unquote(parsed.path.removeprefix("/audio/"))
            rec = RECORDS.get(key)
            if not rec:
                return self.send_body(b"Not found", "text/plain", 404)
            if rec["availability_status"] != "READY" or rec.get("playback_bytes") is None:
                return self.send_body(rec["availability_status"].encode(), "text/plain", 409)
            data = rec["playback_bytes"]
            size = len(data)
            common = {"Accept-Ranges": "bytes", "X-Content-Type-Options": "nosniff"}
            range_header = self.headers.get("Range")
            if range_header:
                bounds = _range_bounds(range_header, size)
                if bounds is None:
                    return self.send_body(b"", "audio/wav", 416, {**common, "Content-Range": f"bytes */{size}"})
                start, end = bounds
                chunk = data[start:end + 1]
                return self.send_body(chunk, "audio/wav", 206, {**common, "Content-Range": f"bytes {start}-{end}/{size}"})
            return self.send_body(data, "audio/wav", 200, common)
        self.send_body(b"Not found", "text/plain", 404)

    def do_HEAD(self):
        # Full-file metadata request; ranges are handled by GET for browser media playback.
        parsed = urlparse(self.path)
        if parsed.path.startswith("/audio/"):
            key = unquote(parsed.path.removeprefix("/audio/"))
            rec = RECORDS.get(key)
            if not rec:
                return self.send_body(b"", "text/plain", 404)
            if rec["availability_status"] != "READY" or rec.get("playback_bytes") is None:
                return self.send_body(b"", "text/plain", 409)
            data = rec["playback_bytes"]
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return
        self.send_body(b"", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/review":
            return self.send_body(b'{"error":"not found"}', "application/json", 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            key = payload.get("recording_key", "")
            status = payload.get("review_status", "")
            if key not in RECORDS or status not in STATUSES:
                raise ValueError("Invalid recording or review status")
            with STATE_LOCK:
                if RECORDS[key]["availability_status"] != "READY":
                    return self.send_body(json.dumps({"ok": False, "error": f"Cannot review audio marked {RECORDS[key]['availability_status']}"}).encode(), "application/json", 409)
                if payload.get("playback_confirmed") is not True:
                    return self.send_body(json.dumps({"ok": False, "error": "Play the recording successfully before saving a review."}).encode(), "application/json", 409)
                row = {
                    "recording_key": key,
                    "original_filename": RECORDS[key]["original_filename"],
                    "contributor_label": RECORDS[key]["contributor_label"],
                    "folder_label": RECORDS[key]["folder_label"],
                    "review_status": status,
                    "reviewer": str(payload.get("reviewer", ""))[:120],
                    "notes": str(payload.get("notes", ""))[:4000],
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                REVIEWS_STATE[key] = row
                persist_reviews()
            return self.send_body(json.dumps({"ok": True, "review": row}).encode(), "application/json")
        except Exception as exc:
            return self.send_body(json.dumps({"ok": False, "error": str(exc)}).encode(), "application/json", 400)

    def log_message(self, fmt, *args):
        print("[cry-review] " + fmt % args)


def main():
    global RECORDS, REVIEWS_STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("For privacy, bind only to localhost")
    RECORDS = load_records()
    REVIEWS_STATE = load_reviews()
    print(f"Review {len(RECORDS)} unique minority-class recording keys at http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
