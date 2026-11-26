"""Regression checks for cry label mapping, uncertainty, and upload inference."""

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as application
from ml.cry_analysis import candidate_pattern, map_cry_probabilities
from ml.inference import InferencePipeline
from services.cry_model_service import CryModelService, validate_class_order


SAMPLE_NAME = "b4c460d8-bd6b-4017-9bb8-2ae4ff0c5af1-1430418266487-1.7-f-26-ti.wav"
SAMPLE_PATH = ROOT / "recordings" / "uploads" / f"upload_20796051_{SAMPLE_NAME}"
DATASET_METADATA = ROOT / "training" / "processed_cry_metadata_clean.csv"


class CryPredictionConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SAMPLE_PATH.is_file() or not DATASET_METADATA.is_file():
            raise unittest.SkipTest("Verified tired sample or cry metadata is unavailable")
        cls.service = CryModelService()

    def test_saved_package_class_order_matches_estimator(self):
        self.assertEqual(
            validate_class_order(self.service.model.classes_, self.service.classes),
            self.service.classes,
        )

    def test_class_order_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "class order mismatch"):
            validate_class_order(["hungry", "tired"], ["tired", "hungry"])

    def test_scores_and_display_mapping_follow_class_names(self):
        mapped = map_cry_probabilities({"tired": 0.8, "hungry": 0.2})
        self.assertEqual(mapped["tired (dataset category)"], 0.8)
        self.assertEqual(mapped["hungry (dataset category)"], 0.2)
        self.assertEqual(candidate_pattern("tired"), "Not available - prediction is uncertain.")

    def test_supplied_sample_label_is_verified_from_metadata(self):
        metadata = pd.read_csv(DATASET_METADATA)
        rows = metadata[metadata["original_file_path"].astype(str).str.endswith(SAMPLE_NAME)]
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(set(rows["label"].dropna().astype(str)), {"tired"})
        self.assertEqual(set(rows["reason_code"].dropna().astype(str)), {"ti"})

    def test_supplied_sample_is_tired_candidate_and_marked_uncertain(self):
        result = self.service.predict(SAMPLE_PATH)
        self.assertEqual(result["top_label"], "tired")
        self.assertEqual(result["predicted_label"], "uncertain")
        self.assertEqual(result["prediction_status"], "UNCERTAIN")
        self.assertIn("not a probability", result["model_score_interpretation"])
        self.assertNotIn("confidence", result)

    def test_main_pipeline_does_not_route_unavailable_detector_result_to_cry_model(self):
        result = InferencePipeline().process_audio(str(SAMPLE_PATH))
        self.assertEqual(result["detected_sound"], "UNKNOWN")
        self.assertIsNone(result["experimental_sound_candidate"])
        self.assertEqual(
            result["sound_classification"],
            "Unavailable: sound not reliably determined.",
        )
        self.assertIsNone(result["possible_category"])
        self.assertIsNone(result["model_score"])
        self.assertIsNone(result["cry_analysis"])
        self.assertEqual(result["prediction_status"], "UNKNOWN")
        self.assertEqual(result["audio_analysis_status"], "UNCERTAIN")
        self.assertEqual(
            result["cry_classification_status"],
            "Not performed because CRY was not reliably established.",
        )
        self.assertEqual(
            result["cough_classification_status"],
            "Unavailable because a verified cough-event classifier has not been trained.",
        )
        self.assertFalse(result["sound_confidence_is_calibrated"])

    def test_main_flask_upload_does_not_show_cry_category_without_detector(self):
        class Cursor:
            def __init__(self):
                self.lastrowid = 0

            def execute(self, *_args, **_kwargs):
                self.lastrowid += 1

        class Connection:
            def __init__(self):
                self._cursor = Cursor()

            def cursor(self):
                return self._cursor

            def commit(self):
                pass

            def close(self):
                pass

        application.app.config.update(TESTING=True)
        with mock.patch.object(application.audio_service, "save_uploaded_file",
                               return_value=(SAMPLE_NAME, str(SAMPLE_PATH))), \
             mock.patch.object(application, "get_db_connection", return_value=Connection()):
            client = application.app.test_client()
            with SAMPLE_PATH.open("rb") as stream:
                response = client.post(
                    "/api/audio/upload",
                    data={
                        "audio_file": (io.BytesIO(stream.read()), SAMPLE_NAME),
                        "age_group": "0-12m",
                        "audio_type_hint": "UNKNOWN",
                    },
                    content_type="multipart/form-data",
                )
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertIsNone(result["experimental_sound_candidate"])
        self.assertIsNone(result["cry_analysis"])
        self.assertIsNone(result["possible_category"])
        self.assertIsNone(result["model_score"])
        self.assertEqual(result["prediction_status"], "UNKNOWN")
        self.assertEqual(
            result["cry_classification_status"],
            "Not performed because CRY was not reliably established.",
        )
        self.assertEqual(
            result["top_screening_pattern"],
            "Not available - sound could not be classified reliably.",
        )

    def test_cry_test_route_does_not_bypass_first_stage_sound_gate(self):
        application.app.config.update(TESTING=True)
        client = application.app.test_client()
        with SAMPLE_PATH.open("rb") as stream:
            response = client.post(
                "/api/cry-model/test",
                data={"audio_file": (io.BytesIO(stream.read()), "neutral_audio.wav")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body["detected_sound"], "UNKNOWN")
        self.assertEqual(body["prediction_status"], "UNKNOWN")
        self.assertIsNone(body["possible_category"])
        self.assertIsNone(body["top_label"])
        self.assertEqual(body["prediction_status"], "UNKNOWN")
        self.assertEqual(
            body["cry_classification_status"],
            "Not performed because CRY was not reliably established.",
        )

    def test_main_inference_does_not_use_filename_as_a_label_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            misleading = root / "cough_tired.wav"
            neutral = root / "ordinary.wav"
            shutil.copyfile(SAMPLE_PATH, misleading)
            shutil.copyfile(SAMPLE_PATH, neutral)
            pipeline = InferencePipeline()
            first = pipeline.process_audio(str(misleading))
            second = pipeline.process_audio(str(neutral))
        self.assertEqual(first["detected_sound"], second["detected_sound"])
        self.assertEqual(first["sound_probabilities"], second["sound_probabilities"])
        self.assertEqual(first["experimental_sound_candidate"], second["experimental_sound_candidate"])
        self.assertEqual(first["possible_category"], second["possible_category"])
        self.assertEqual(first["top_screening_pattern"], second["top_screening_pattern"])
        self.assertEqual(first["screening_probabilities"], second["screening_probabilities"])


if __name__ == "__main__":
    unittest.main()
