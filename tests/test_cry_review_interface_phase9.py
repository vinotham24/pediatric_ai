"""Isolated tests for the manual cry review app; all files live in temp dirs."""
import csv
import importlib.util
import json
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cry_review_phase9_target", ROOT / "training" / "cry_label_review_app.py")
review_app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_app)


class CryReviewInterfacePhase9(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temp.name)
        (self.temp_root / "training").mkdir()
        (self.temp_root / "audio").mkdir()
        self.metadata_path = self.temp_root / "training" / "metadata.csv"
        self.reviews_path = self.temp_root / "training" / "reviews.csv"
        self.original_metadata_bytes = self._write_fixture_metadata()
        self.old = (review_app.ROOT, review_app.METADATA, review_app.REVIEWS,
                    review_app.RECORDS, review_app.REVIEWS_STATE)
        review_app.ROOT = self.temp_root
        review_app.METADATA = self.metadata_path
        review_app.REVIEWS = self.reviews_path
        review_app.RECORDS = review_app.load_records()
        review_app.REVIEWS_STATE = review_app.load_reviews()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), review_app.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        (review_app.ROOT, review_app.METADATA, review_app.REVIEWS,
         review_app.RECORDS, review_app.REVIEWS_STATE) = self.old
        self.temp.cleanup()

    def _wav(self, name, silence=False, folder="misc"):
        path = self.temp_root / "audio" / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(8000)
            sample = b"\x00\x00" * 800 if silence else b"\x10\x00" * 800
            out.writeframes(sample)
        return path

    def _write_fixture_metadata(self):
        fields = ["processed_filename", "processed_file_path", "original_filename", "original_file_path", "format", "timestamp", "app_version", "gender", "age_code", "age_group", "reason_code", "label", "app_instance_id"]
        rows = []
        # Seven cold_hot-coded WAV files physically under a discomfort folder.
        for i in range(7):
            fn = f"cold{i}.wav"
            self._wav(fn, folder="discomfort")
            rows.append(self._row(f"cold-{i}", str(100+i), fn, "cold_hot", "ch", "discomfort"))
        # Two representations of one recording key: only one review queue item.
        self._wav("dup_android.wav", folder="android")
        self._wav("dup_clean.wav", folder="donateacry_corpus_cleaned_and_updated_data/tired")
        rows.append(self._row("dup", "200", "dup_android.wav", "tired", "ti", "android"))
        rows.append(self._row("dup", "200", "dup_clean.wav", "tired", "ti", "donateacry_corpus_cleaned_and_updated_data/tired"))
        # A usable supported audio file.
        self._wav("valid.wav", folder="burping")
        rows.append(self._row("valid", "300", "valid.wav", "burping", "bu", "burping"))
        # Existing path, but corrupt bytes; and silent bytes for behavior audit.
        (self.temp_root / "audio" / "scared").mkdir(parents=True, exist_ok=True)
        (self.temp_root / "audio" / "scared" / "corrupt.wav").write_bytes(b"not-a-wave")
        rows.append(self._row("corrupt", "400", "corrupt.wav", "scared", "sc", "scared"))
        self._wav("silent.wav", silence=True, folder="lonely")
        rows.append(self._row("silent", "500", "silent.wav", "lonely", "lo", "lonely"))
        # Unsupported extension is visible but cannot be reviewed.
        (self.temp_root / "audio" / "unsupported").mkdir(parents=True, exist_ok=True)
        (self.temp_root / "audio" / "unsupported" / "unknown.xyz").write_bytes(b"unknown container")
        rows.append(self._row("unsupported", "550", "unknown.xyz", "discomfort", "dc", "unsupported"))
        # Valid CAF and 3GP fixtures exercise the installed FFmpeg decoder and WAV playback conversion.
        self._make_encoded_fixture("caf", "pcm_s16le", "caf")
        self._make_encoded_fixture("3gp", "aac", "3gp")
        rows.append(self._row("caf", "560", "encoded.caf", "discomfort", "dc", "caf_source"))
        rows.append(self._row("3gp", "570", "encoded.3gp", "discomfort", "dc", "3gp_source"))
        rows.append(self._row("missing", "600", "missing.wav", "belly_pain", "bp", "belly_pain"))
        with self.metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return self.metadata_path.read_bytes()

    def _make_encoded_fixture(self, extension, codec, output_format):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        source = self._wav(f"encode_{extension}.wav", folder="fixtures")
        output = self.temp_root / "audio" / f"{extension}_source" / f"encoded.{extension}"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source), "-c:a", codec, "-f", output_format, str(output)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if result.returncode != 0:
            self.skipTest(f"Installed FFmpeg cannot generate the {extension} temporary fixture: {result.stderr.decode(errors='replace')}")

    def _row(self, uuid, timestamp, filename, label, code, folder):
        rel = f"audio/{filename}"
        rel = f"audio/{folder}/{filename}"
        return {
            "processed_filename": filename, "processed_file_path": rel,
            "original_filename": filename,
            "original_file_path": rel,
            "format": Path(filename).suffix, "timestamp": timestamp, "app_version": "1.0",
            "gender": "f", "age_code": "04", "age_group": "0-4 weeks",
            "reason_code": code, "label": label, "app_instance_id": uuid,
        }

    def _post(self, key, status, reviewer="R", notes="note", extra=None, playback_confirmed=True):
        payload = {"recording_key": key, "review_status": status, "reviewer": reviewer, "notes": notes, "playback_confirmed": playback_confirmed}
        if extra:
            payload.update(extra)
        request = urllib.request.Request(self.base + "/api/review", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    def _records(self):
        with urllib.request.urlopen(self.base + "/api/records") as response:
            return json.loads(response.read())

    def test_supported_wav_original_is_served_unchanged(self):
        key = "valid-300"
        expected = (self.temp_root / "audio" / "burping" / "valid.wav").read_bytes()
        with urllib.request.urlopen(self.base + "/audio/" + key) as response:
            self.assertEqual(response.headers.get_content_type(), "audio/wav")
            self.assertEqual(response.read(), expected)

    def test_caf_and_3gp_decode_to_browser_wav_without_changing_source(self):
        for key, extension in (("caf-560", ".caf"), ("3gp-570", ".3gp")):
            record = review_app.RECORDS[key]
            self.assertEqual(record["availability_status"], "READY")
            original = Path(record["original_path"]).read_bytes()
            with urllib.request.urlopen(self.base + "/audio/" + key) as response:
                payload = response.read()
                self.assertEqual(response.headers.get_content_type(), "audio/wav")
                self.assertTrue(payload.startswith(b"RIFF") and b"WAVE" in payload[:16])
            self.assertEqual(Path(record["original_path"]).suffix.lower(), extension)
            self.assertEqual(Path(record["original_path"]).read_bytes(), original)

    def test_status_notes_read_only_label_independence_and_restart(self):
        key = "valid-300"
        for status in ("VERIFIED", "QUESTIONABLE", "REJECTED"):
            result = self._post(key, status, notes=f"notes {status}", extra={"contributor_label": "changed", "label": "changed"})
            self.assertTrue(result["ok"])
            self.assertEqual(result["review"]["contributor_label"], "burping")
            self.assertEqual(result["review"]["review_status"], status)
            self.assertEqual(result["review"]["notes"], f"notes {status}")
        self.assertEqual(self.metadata_path.read_bytes(), self.original_metadata_bytes)
        other = "cold-0-100"
        self._post(other, "QUESTIONABLE", notes="separate decision")
        self.assertEqual(review_app.REVIEWS_STATE[key]["review_status"], "REJECTED")
        # Simulate process restart by reloading persisted CSV into fresh state.
        review_app.REVIEWS_STATE = review_app.load_reviews()
        self.assertEqual(review_app.REVIEWS_STATE[key]["notes"], "notes REJECTED")
        self.assertEqual(review_app.REVIEWS_STATE[other]["notes"], "separate decision")
        self.assertEqual(self.metadata_path.read_bytes(), self.original_metadata_bytes)

    def test_duplicate_representations_are_grouped_and_clean_copy_preferred(self):
        grouped = [r for r in review_app.RECORDS.values() if r["recording_key"] == "dup-200"]
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["original_filename"], "dup_clean.wav")

    def test_seven_discrepancies_are_prioritized_by_frontend_sort(self):
        page = review_app.PAGE
        marker = "arr.sort("
        start = page.index(marker) + len(marker)
        end = page.index(");recordEl.innerHTML", start) + 1
        comparator = page[start:end]
        items = [{"contributor_label": "burping", "folder_label": "burping", "original_filename": "aaa.wav"}]
        items += [{"contributor_label": "cold_hot", "folder_label": "discomfort", "original_filename": f"cold{i}.wav"} for i in range(7)]
        script = f"const a={json.dumps(items)};a.sort(({comparator});process.stdout.write(JSON.stringify(a.map(x=>x.original_filename)))"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        ordered = json.loads(result.stdout)
        self.assertEqual(ordered[:7], [f"cold{i}.wav" for i in range(7)])

    def test_missing_corrupt_silent_and_unknown_format_are_visible_with_status(self):
        rows = {r["recording_key"]: r for r in self._records()}
        expected = {"missing-600": "MISSING", "corrupt-400": "CORRUPT", "silent-500": "SILENT", "unsupported-550": "UNSUPPORTED FORMAT"}
        for key, status in expected.items():
            self.assertIn(key, rows)
            self.assertEqual(rows[key]["availability_status"], status)
        self.assertIn("Audio status:", review_app.PAGE)
        self.assertIn("MISSING", review_app.PAGE)

    def test_full_and_partial_audio_responses(self):
        path = (self.temp_root / "audio" / "burping" / "valid.wav")
        full = path.read_bytes()
        with urllib.request.urlopen(self.base + "/audio/valid-300") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get("Accept-Ranges"), "bytes")
            self.assertEqual(int(response.headers["Content-Length"]), len(full))
            self.assertEqual(response.read(), full)
        request = urllib.request.Request(self.base + "/audio/valid-300", headers={"Range": "bytes=10-25"})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], f"bytes 10-25/{len(full)}")
            self.assertEqual(int(response.headers["Content-Length"]), 16)
            self.assertEqual(response.read(), full[10:26])
        for header, expected in (("bytes=10-", full[10:]), (f"bytes=-12", full[-12:])):
            request = urllib.request.Request(self.base + "/audio/valid-300", headers={"Range": header})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), expected)

    def test_invalid_byte_range_returns_416(self):
        size = (self.temp_root / "audio" / "burping" / "valid.wav").stat().st_size
        for header in (f"bytes={size}-", "bytes=20-10", "bytes=-0", "bytes=0-1,4-6", "items=0-2"):
            request = urllib.request.Request(self.base + "/audio/valid-300", headers={"Range": header})
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request)
            response = raised.exception
            self.assertEqual(response.code, 416)
            self.assertEqual(response.headers["Content-Range"], f"bytes */{size}")

    def test_invalid_audio_cannot_be_annotated(self):
        keys = ("missing-600", "corrupt-400", "silent-500", "unsupported-550")
        for key in keys:
            record = review_app.RECORDS[key]
            self.assertNotEqual(record["availability_status"], "READY")
            request = urllib.request.Request(
                self.base + "/api/review",
                data=json.dumps({"recording_key": key, "review_status": "VERIFIED", "reviewer": "test", "notes": "must not save"}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request)
            self.assertEqual(raised.exception.code, 409)
            self.assertNotIn(key, review_app.REVIEWS_STATE)

    def test_ready_audio_requires_playback_confirmation_before_review(self):
        key = "valid-300"
        request = urllib.request.Request(
            self.base + "/api/review",
            data=json.dumps({"recording_key": key, "review_status": "VERIFIED", "playback_confirmed": False}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code, 409)
        self.assertNotIn(key, review_app.REVIEWS_STATE)
        result = self._post(key, "VERIFIED", notes="heard recording", playback_confirmed=True)
        self.assertTrue(result["ok"])

    def test_unrecognized_format_is_visible_and_not_streamed(self):
        self.assertIn("unsupported-550", review_app.RECORDS)
        self.assertEqual(review_app.RECORDS["unsupported-550"]["availability_status"], "UNSUPPORTED FORMAT")
        request = urllib.request.Request(self.base + "/audio/unsupported-550")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code, 409)


if __name__ == "__main__":
    unittest.main()
