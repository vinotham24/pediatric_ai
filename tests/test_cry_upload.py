"""End-to-end cry upload checks using generated tones, never patient audio."""
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import app as application
import imageio_ffmpeg
from services.cry_model_service import CryModelService
import services.audio_service as audio_module


class CryUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp_dir.name)
        cls.wav_path = cls.directory / "tone.wav"
        rate = 8000
        t = np.arange(rate, dtype=np.float32) / rate
        tone = (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(cls.wav_path, tone, rate, subtype="PCM_16")
        cls.formats = {"wav": cls.wav_path}
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        for ext, codec in (
            ("mp3", "libmp3lame"),
            ("m4a", "aac"),
            ("ogg", "libvorbis"),
            ("flac", "flac"),
        ):
            target = cls.directory / f"tone.{ext}"
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-i", str(cls.wav_path),
                 "-c:a", codec, str(target)], capture_output=True, text=True
            )
            if result.returncode == 0 and target.exists():
                cls.formats[ext] = target

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        application.app.config.update(TESTING=True)
        self.cry_model = CryModelService()
        self.client = application.app.test_client()

    def upload(self, path, filename=None):
        name = filename or path.name
        with open(path, "rb") as audio_stream:
            content = audio_stream.read()
        return self.client.post(
            "/api/cry-model/test",
            data={"audio_file": (io.BytesIO(content), name)},
            content_type="multipart/form-data",
        )

    def test_cry_test_route_does_not_bypass_sound_detection_gate(self):
        response = self.upload(self.formats["wav"])
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(body["detected_sound"], "UNKNOWN")
        self.assertEqual(body["sound_detection_status"], "UNKNOWN")
        self.assertIsNone(body["possible_category"])
        self.assertIsNone(body["top_label"])
        self.assertIsNone(body["experimental_sound_candidate"])
        self.assertEqual(body["class_scores"], {})
        self.assertEqual(body["prediction_status"], "UNKNOWN")
        self.assertEqual(
            body["cry_classification_status"],
            "Not performed because CRY was not reliably established.",
        )

    def test_supported_audio_formats_use_the_shared_gate_for_upload_and_live_api(self):
        original_dir = audio_module.UPLOAD_RECORDINGS_DIR
        with tempfile.TemporaryDirectory() as upload_dir:
            audio_module.UPLOAD_RECORDINGS_DIR = upload_dir
            try:
                for endpoint in ("/api/audio/upload", "/api/audio/live_analyze"):
                    for extension in ("wav", "mp3", "m4a", "ogg", "flac"):
                        with self.subTest(endpoint=endpoint, extension=extension):
                            if extension not in self.formats:
                                self.skipTest(f"FFmpeg cannot encode {extension} in this environment")
                            with self.formats[extension].open("rb") as stream:
                                content = stream.read()
                            with mock.patch.object(application, "_persist_analysis_result", return_value=1), \
                                 mock.patch.object(application.assistant_service, "explain_results", return_value={}):
                                response = self.client.post(
                                    endpoint,
                                    data={
                                        "audio_file": (
                                            io.BytesIO(content),
                                            f"baby-cough-181204_1298-02.{extension}",
                                        ),
                                    },
                                    content_type="multipart/form-data",
                                )
                            self.assertEqual(response.status_code, 200, response.get_json())
                            body = response.get_json()
                            self.assertEqual(body["audio_analysis_status"], "UNCERTAIN")
                            self.assertEqual(body["detected_sound"], "UNKNOWN")
                            self.assertEqual(
                                body["sound_classification"],
                                "Unavailable: sound not reliably determined.",
                            )
                            self.assertIsNone(body["possible_category"])
                            self.assertIsNone(body["model_score"])
                            self.assertIsNone(body["cry_analysis"])
                            self.assertIsNone(body["experimental_sound_candidate"])
                            self.assertEqual(
                                body["cry_classification_status"],
                                "Not performed because CRY was not reliably established.",
                            )
                            self.assertEqual(
                                body["cough_classification_status"],
                                "Unavailable because a verified cough-event classifier has not been trained.",
                            )
                            self.assertEqual(body["medical_diagnosis"], "Not provided.")
            finally:
                audio_module.UPLOAD_RECORDINGS_DIR = original_dir

    def test_generated_compressed_formats_decode_in_cry_test_route(self):
        for extension in ("mp3", "m4a", "ogg", "flac"):
            with self.subTest(extension=extension):
                if extension not in self.formats:
                    self.skipTest(f"FFmpeg cannot encode {extension} in this environment")
                response = self.upload(self.formats[extension])
                self.assertEqual(response.status_code, 200, response.get_json())
                body = response.get_json()
                self.assertIsNone(body["possible_category"])
                self.assertEqual(body["prediction_status"], "UNKNOWN")

    def test_model_feature_extraction_resamples_8khz_without_optional_resampy(self):
        features = self.cry_model.extract_features(self.formats["wav"])
        self.assertEqual(len(features), 91)
        self.assertTrue(np.isfinite([features[name] for name in self.cry_model.feature_columns]).all())

    def test_svc_decision_label_wins_when_probability_argmax_disagrees(self):
        class FakeSvc:
            classes_ = np.array(["hungry", "burping", "tired"])

            def predict_proba(self, _features):
                return np.array([[0.44, 0.43, 0.13]])

            def predict(self, _features):
                return np.array(["burping"])

        service = CryModelService.__new__(CryModelService)
        service.model = FakeSvc()
        service.feature_columns = ["feature"]
        service.classes = ["hungry", "burping", "tired"]
        service.extract_features = lambda _path: {"feature": 1.0}

        result = service.predict("unused.wav")
        self.assertEqual(result["top_label"], "burping")
        self.assertEqual(result["predicted_label"], "uncertain")
        self.assertEqual(result["prediction_status"], "UNCERTAIN")
        self.assertEqual(result["model_score"], 0.43)

    def test_empty_corrupt_and_unsupported_uploads_are_rejected(self):
        for filename, content in (
            ("empty.wav", b""),
            ("broken.wav", b"this is not audio"),
            ("recording.txt", b"audio"),
        ):
            with self.subTest(filename=filename):
                response = self.client.post(
                    "/api/cry-model/test",
                    data={"audio_file": (io.BytesIO(content), filename)},
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.get_json()["success"])

    def test_temporary_files_are_removed_after_success_and_inference_failure(self):
        service = application.audio_service
        original_decode = service.decode_cry_upload
        temporary_paths = []

        def capture_paths(upload):
            result = original_decode(upload)
            temporary_paths.extend(result)
            return result

        with mock.patch.object(service, "decode_cry_upload", side_effect=capture_paths):
            response = self.upload(self.formats["wav"])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(temporary_paths)
        self.assertTrue(all(not os.path.exists(path) for path in temporary_paths))

        temporary_paths.clear()
        with mock.patch.object(service, "decode_cry_upload", side_effect=capture_paths), \
             mock.patch.object(application, "InferencePipeline") as pipeline_class:
            pipeline_class.return_value.process_audio.side_effect = RuntimeError("test failure")
            response = self.upload(self.formats["wav"])
        self.assertEqual(response.status_code, 500)
        self.assertTrue(temporary_paths)
        self.assertTrue(all(not os.path.exists(path) for path in temporary_paths))

    def test_missing_cry_model_does_not_create_prediction_when_sound_is_unverified(self):
        with mock.patch("ml.cry_analysis.CryModelService", side_effect=FileNotFoundError("model missing")):
            response = self.upload(self.formats["wav"])
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertTrue(result["success"])
        self.assertIsNone(result["possible_category"])
        self.assertEqual(result["prediction_status"], "UNKNOWN")
        self.assertEqual(result["detected_sound"], "UNKNOWN")
        self.assertEqual(
            result["cry_classification_status"],
            "Not performed because CRY was not reliably established.",
        )


if __name__ == "__main__":
    unittest.main()
