"""Integration checks for honest sound routing and shared upload/live analysis."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import app as application
from ml.cough_analysis import CoughAnalyzer
from ml.inference import InferencePipeline
from ml.sound_detection import SoundDetector
import services.audio_service as audio_module


class FakeCursor:
    def __init__(self):
        self.lastrowid = 0
        self.calls = []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        self.lastrowid += 1


class FakeConnection:
    def __init__(self):
        self._cursor = FakeCursor()
        self.committed = False
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        self.closed = True


def sample_result():
    return {
        "detected_sound": "UNKNOWN",
        "sound_confidence": None,
        "sound_confidence_is_calibrated": False,
        "sound_confidence_interpretation": "No validated detector score is available.",
        "sound_probabilities": {},
        "sound_detection_status": "UNKNOWN",
        "sound_detection_method": "UNAVAILABLE (SED model unavailable)",
        "experimental_sound_candidate": None,
        "audio_analysis_status": "UNCERTAIN",
        "sound_classification": "Unavailable: sound not reliably determined.",
        "cry_classification_status": "Not performed because CRY was not reliably established.",
        "cough_classification_status": "Unavailable because a verified cough-event classifier has not been trained.",
        "medical_diagnosis": "Not provided.",
        "prediction_status": "UNKNOWN",
        "possible_category": None,
        "model_score": None,
        "model_score_interpretation": "Unavailable.",
        "top_screening_pattern": "Not available - sound could not be classified reliably.",
        "risk_result": {"risk_available": False, "risk_level": "NOT CALCULATED", "risk_score": None, "risk_drivers": []},
        "duration": 1.0,
    }


class SubmissionWorkflowTests(unittest.TestCase):
    def setUp(self):
        application.app.config.update(TESTING=True)
        self.client = application.app.test_client()

    def test_missing_first_stage_model_does_not_infer_from_filename(self):
        detector = SoundDetector()
        detector.model = None
        zeros = np.zeros(16000, dtype=np.float32)
        pure_tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
        cough_like = np.zeros(16000, dtype=np.float32)
        for start in (2000, 6000, 10000):
            cough_like[start:start + 800] = np.random.default_rng(start).normal(0, 0.2, 800)

        for signal in (zeros, pure_tone, cough_like):
            with self.subTest(signal_peak=float(np.max(np.abs(signal)))):
                cry_name = detector.predict(signal, 16000, filename_hint="cry_sample.wav")
                cough_name = detector.predict(signal, 16000, filename_hint="cough_sample.wav")
                neutral_name = detector.predict(signal, 16000, filename_hint="ordinary.wav")
                self.assertEqual(cry_name[0], "UNKNOWN")
                self.assertEqual(cough_name[0], "UNKNOWN")
                self.assertEqual(neutral_name[0], "UNKNOWN")
                self.assertEqual(cry_name, cough_name)
                self.assertEqual(cough_name, neutral_name)
                self.assertIsNone(cry_name[1])
                self.assertEqual(cry_name[2], {})
        self.assertEqual(detector.last_status, "UNKNOWN")
        self.assertEqual(detector.last_method, "UNAVAILABLE (SED model unavailable)")

    def test_cough_classifier_unavailable_returns_no_disease_probabilities(self):
        result = CoughAnalyzer().analyze(np.zeros(16000, dtype=np.float32))
        self.assertEqual(result["classification_status"], "UNAVAILABLE")
        self.assertEqual(result["prediction_status"], "UNKNOWN")
        self.assertIsNone(result["possible_category"])
        self.assertIsNone(result["model_score"])
        self.assertNotIn("probabilities", result)
        self.assertNotIn("top_pattern", result)

    def test_tonal_and_cough_like_audio_remain_unknown_without_detector(self):
        rate = 16000
        samples = np.arange(rate, dtype=np.float32) / rate
        tonal = (0.2 * np.sin(2 * np.pi * 440 * samples)).astype(np.float32)
        cough_like = np.zeros(rate, dtype=np.float32)
        for start in (2000, 6000, 10000):
            cough_like[start:start + 800] = np.random.default_rng(start).normal(0, 0.2, 800)

        for signal in (tonal, cough_like):
            with self.subTest(peak=float(np.max(np.abs(signal)))):
                pipeline = InferencePipeline()
                pipeline.cry_analyzer = mock.Mock()
                pipeline.cough_analyzer = mock.Mock()
                pipeline.cough_analyzer.model = None
                with mock.patch(
                    "ml.inference.load_and_preprocess_audio",
                    return_value=(signal, rate),
                ), mock.patch(
                    "ml.inference.apply_vad", return_value=signal
                ), mock.patch(
                    "ml.inference.extract_mel_spectrogram",
                    return_value=np.zeros((128, 20)),
                ):
                    result = pipeline.process_audio("cough-like-audio.wav")

                self.assertEqual(result["detected_sound"], "UNKNOWN")
                self.assertEqual(result["prediction_status"], "UNKNOWN")
                self.assertEqual(
                    result["sound_classification"],
                    "Unavailable: sound not reliably determined.",
                )
                self.assertIsNone(result["possible_category"])
                self.assertIsNone(result["cough_analysis"])
                pipeline.cry_analyzer.analyze.assert_not_called()
                pipeline.cough_analyzer.analyze.assert_not_called()

    def test_routing_requires_accepted_model_backed_sound_detection(self):
        y = np.ones(16000, dtype=np.float32) * 0.1
        cases = [
            ("CRY", "UNCERTAIN", False, None, False),
            ("COUGH", "UNCERTAIN", False, None, False),
            ("CRY", "CONFIDENT", True, object(), False),
            ("COUGH", "CONFIDENT", True, None, False),
            ("COUGH", "CONFIDENT", True, object(), True),
            ("OTHER", "CONFIDENT", True, object(), False),
            ("UNKNOWN", "UNKNOWN", False, None, False),
        ]
        for sound, detector_status, detector_available, cough_model, expect_cough_call in cases:
            with self.subTest(sound=sound):
                pipeline = InferencePipeline.__new__(InferencePipeline)
                pipeline.sound_detector = mock.Mock()
                pipeline.sound_detector.predict.return_value = (sound, None, {})
                pipeline.sound_detector.model = object() if detector_available else None
                pipeline.sound_detector.last_method = "FIXTURE"
                pipeline.sound_detector.last_status = detector_status
                pipeline.cry_analyzer = mock.Mock()
                pipeline.cry_analyzer.analyze.return_value = {
                    "prediction_status": "UNCERTAIN", "possible_category": "tired (dataset category)",
                    "model_score": 0.4, "model_score_interpretation": "uncalibrated",
                    "top_pattern": "Not available - prediction is uncertain.", "candidate_class": "tired",
                }
                pipeline.cough_analyzer = mock.Mock()
                pipeline.cough_analyzer.model = cough_model
                pipeline.cough_analyzer.analyze.return_value = {
                    "classification_status": "UNAVAILABLE", "prediction_status": "UNKNOWN",
                    "message": "No verified cough model.",
                }
                with mock.patch("ml.inference.load_and_preprocess_audio", return_value=(y, 16000)), \
                     mock.patch("ml.inference.apply_vad", return_value=y), \
                     mock.patch("ml.inference.extract_mel_spectrogram", return_value=np.zeros((128, 20))):
                    result = pipeline.process_audio("fixture.wav")
                if detector_status == "CONFIDENT" and detector_available and sound == "CRY":
                    pipeline.cry_analyzer.analyze.assert_called_once()
                    pipeline.cough_analyzer.analyze.assert_not_called()
                    self.assertEqual(result["detected_sound"], "CRY")
                elif detector_status == "CONFIDENT" and detector_available and sound == "COUGH" and cough_model is not None:
                    pipeline.cough_analyzer.analyze.assert_called_once()
                    pipeline.cry_analyzer.analyze.assert_not_called()
                    self.assertEqual(result["detected_sound"], "COUGH")
                elif detector_status == "CONFIDENT" and detector_available and sound == "COUGH":
                    pipeline.cough_analyzer.analyze.assert_not_called()
                    self.assertEqual(result["cough_analysis"]["classification_status"], "UNAVAILABLE")
                    self.assertIsNone(result["cough_analysis"]["possible_category"])
                elif detector_status == "UNCERTAIN" and sound in {"CRY", "COUGH"}:
                    pipeline.cry_analyzer.analyze.assert_not_called()
                    pipeline.cough_analyzer.analyze.assert_not_called()
                    self.assertEqual(result["detected_sound"], "UNKNOWN")
                    self.assertIsNone(result["experimental_sound_candidate"])
                    self.assertIsNone(result["possible_category"])
                    self.assertIsNone(result["model_score"])
                    self.assertEqual(result["prediction_status"], "UNCERTAIN")
                else:
                    pipeline.cough_analyzer.analyze.assert_not_called()
                    pipeline.cry_analyzer.analyze.assert_not_called()
                if result["detected_sound"] != "CRY":
                    self.assertEqual(
                        result["cry_classification_status"],
                        "Not performed because CRY was not reliably established.",
                    )

    def test_upload_and_live_share_analysis_and_persist_history(self):
        with tempfile.TemporaryDirectory() as temp:
            original_dir = audio_module.UPLOAD_RECORDINGS_DIR
            audio_module.UPLOAD_RECORDINGS_DIR = temp
            try:
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, np.ones(8000, dtype=np.float32) * 0.1, 8000, format="WAV")
                wav_bytes = wav_buffer.getvalue()
                connections = []

                def connection_factory():
                    conn = FakeConnection()
                    connections.append(conn)
                    return conn

                mock_pipeline = mock.Mock()
                mock_pipeline.process_audio.return_value = sample_result()
                with mock.patch.object(application, "InferencePipeline", return_value=mock_pipeline), \
                     mock.patch.object(application, "get_db_connection", side_effect=connection_factory):
                    responses = []
                    for endpoint, source in (("/api/audio/upload", "UPLOAD"), ("/api/audio/live_analyze", "LIVE")):
                        response = self.client.post(endpoint, data={
                            "audio_file": (io.BytesIO(wav_bytes), "mic.wav"),
                            "age_group": "0-12m",
                        }, content_type="multipart/form-data")
                        self.assertEqual(response.status_code, 200, response.get_json())
                        data = response.get_json()
                        self.assertEqual(data["detected_sound"], "UNKNOWN")
                        self.assertEqual(data["prediction_status"], "UNKNOWN")
                        self.assertEqual(
                            data["sound_classification"],
                            "Unavailable: sound not reliably determined.",
                        )
                        self.assertEqual(
                            data["cough_classification_status"],
                            "Unavailable because a verified cough-event classifier has not been trained.",
                        )
                        self.assertEqual(data["medical_diagnosis"], "Not provided.")
                        self.assertIsNone(data["sound_confidence"])
                        self.assertIn("/recordings/uploads/", data["audio_url"])
                        self.assertTrue(data.get("prediction_id"))
                        responses.append(data)
                        self.assertEqual(connections[-1]._cursor.calls[0][1][0], source)
                        self.assertTrue(connections[-1].committed)
                        self.assertTrue(connections[-1].closed)
                self.assertEqual(mock_pipeline.process_audio.call_count, 2)
                self.assertEqual(responses[0]["prediction_status"], responses[1]["prediction_status"])
                self.assertEqual(len(list(Path(temp).glob("*.wav"))), 2)
            finally:
                audio_module.UPLOAD_RECORDINGS_DIR = original_dir

    def test_upload_rejects_empty_corrupt_and_unsupported_files(self):
        with tempfile.TemporaryDirectory() as temp:
            original_dir = audio_module.UPLOAD_RECORDINGS_DIR
            audio_module.UPLOAD_RECORDINGS_DIR = temp
            try:
                cases = [("empty.wav", b""), ("bad.wav", b"not audio"), ("bad.txt", b"audio")]
                for endpoint in ("/api/audio/upload", "/api/audio/live_analyze"):
                    for filename, content in cases:
                        with self.subTest(endpoint=endpoint, filename=filename):
                            response = self.client.post(endpoint, data={
                                "audio_file": (io.BytesIO(content), filename),
                            }, content_type="multipart/form-data")
                            self.assertEqual(response.status_code, 400, response.get_json())
                            self.assertEqual(response.get_json()["prediction_status"], "ERROR")
                self.assertEqual(list(Path(temp).iterdir()), [])
            finally:
                audio_module.UPLOAD_RECORDINGS_DIR = original_dir

    def test_main_pages_render(self):
        for route in ("/", "/dashboard", "/upload", "/live", "/history", "/cry-model-test"):
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
