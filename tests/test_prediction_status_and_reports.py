"""Phase 5 checks for status gating, upload/live parity, and safe reports."""
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as application
import services.report_service as report_module
from services.prediction_status import (
    UNCERTAIN_PATTERN,
    determine_prediction_status,
    normalize_prediction_record,
    screening_pattern_for,
)
from services.report_service import ReportService


class PredictionStatusTests(unittest.TestCase):
    def test_uncalibrated_candidate_is_uncertain_even_with_high_score(self):
        self.assertEqual(determine_prediction_status("tired", 0.999), "UNCERTAIN")

    def test_confident_requires_explicit_validation_and_calibration_metadata(self):
        validation = {
            "validated_for_screening": True,
            "scores_calibrated": True,
            "validation_reference": "independent prospective validation",
            "acceptance_threshold": 0.8,
        }
        self.assertEqual(determine_prediction_status("tired", 0.8, validation), "CONFIDENT")
        self.assertEqual(determine_prediction_status("tired", 0.79, validation), "UNCERTAIN")

    def test_confident_candidate_is_labeled_experimental_not_clinical(self):
        text = screening_pattern_for("CONFIDENT", "tired (dataset category)")
        self.assertIn("Experimental category prediction", text)
        self.assertIn("not a confirmed finding", text)
        self.assertNotIn("risk", text.lower())

    def test_unknown_error_and_missing_score(self):
        self.assertEqual(determine_prediction_status(None, 0.9), "UNKNOWN")
        self.assertEqual(determine_prediction_status("tired", 0.9, error="decode"), "ERROR")
        self.assertEqual(determine_prediction_status("tired", None), "UNCERTAIN")

    def test_uncertain_record_has_no_definitive_pattern_or_risk_score(self):
        record = normalize_prediction_record({
            "sound_type": "CRY", "confidence": 0.988, "risk_level": "HIGH",
            "risk_score": 98, "screening_prediction": "Severe Belly Pain / Colic Cry",
            "probabilities_json": '{"tired": 0.99}',
        })
        self.assertEqual(record["prediction_status"], "UNCERTAIN")
        self.assertIsNone(record["possible_category"])
        self.assertEqual(record["screening_prediction"], UNCERTAIN_PATTERN)
        self.assertIsNone(record["risk_score"])
        self.assertEqual(record["risk_level"], "NOT CALCULATED")
        self.assertNotIn("confidence", record)
        self.assertNotIn("probabilities_json", record)

    def test_uncertain_candidate_is_not_promoted_to_a_definitive_pattern(self):
        record = normalize_prediction_record({
            "sound_type": "CRY",
            "probabilities_json": '{"prediction_status":"UNCERTAIN","possible_category":"tired","model_score":0.91,"sound_detection_status":"UNCERTAIN","sound_detection_method":"HEURISTIC"}',
        })
        self.assertIsNone(record["possible_category"])
        self.assertEqual(record["screening_prediction"], UNCERTAIN_PATTERN)
        self.assertIn("HISTORICAL RESULT", record["audio_analysis_status"])
        self.assertIn("not revalidated", record["legacy_result_note"])

    def test_model_score_never_generates_a_risk_score(self):
        record = normalize_prediction_record({
            "sound_type": "CRY",
            "probabilities_json": '{"prediction_status":"CONFIDENT","possible_category":"tired","model_score":0.99}',
            "sound_detection_status": "CONFIDENT",
            "risk_score": 99,
            "risk_level": "HIGH",
        })
        self.assertEqual(record["model_score"], 0.99)
        self.assertEqual(record["possible_category"], "tired")
        self.assertIsNone(record["risk_score"])
        self.assertEqual(record["risk_level"], "NOT CALCULATED")


class EndpointConsistencyTests(unittest.TestCase):
    def test_upload_and_live_use_identical_status_result(self):
        expected = {
            "detected_sound": "CRY", "prediction_status": "UNCERTAIN",
            "possible_category": "tired (dataset category)", "model_score": 0.91,
            "model_score_interpretation": "Uncalibrated; not probability of correctness.",
            "top_screening_pattern": UNCERTAIN_PATTERN,
            "risk_result": {"risk_available": False, "risk_level": "NOT CALCULATED", "risk_score": None},
            "sound_confidence": 0.9, "sound_confidence_is_calibrated": False,
            "duration": 1.0,
        }

        class Cursor:
            lastrowid = 1
            def execute(self, *_args, **_kwargs): pass

        class Connection:
            def cursor(self): return Cursor()
            def commit(self): pass
            def close(self): pass

        class Pipeline:
            def process_audio(self, *_args, **_kwargs): return dict(expected)

        client = application.app.test_client()
        application.app.config.update(TESTING=True)
        with mock.patch.object(application.audio_service, "save_uploaded_file", return_value=("x.wav", "x.wav")), \
             mock.patch.object(application, "InferencePipeline", return_value=Pipeline()), \
             mock.patch.object(application, "get_db_connection", return_value=Connection()):
            upload = client.post("/api/audio/upload", data={"audio_file": (io.BytesIO(b"audio"), "x.wav")})
            live = client.post("/api/audio/live_analyze", data={"audio_file": (io.BytesIO(b"audio"), "x.wav")})
        self.assertEqual(upload.status_code, 200, upload.get_json())
        self.assertEqual(live.status_code, 200, live.get_json())
        a, b = upload.get_json(), live.get_json()
        for key in ("detected_sound", "prediction_status", "possible_category", "model_score", "top_screening_pattern", "risk_result"):
            self.assertEqual(a[key], b[key], key)


class ReportConsistencyTests(unittest.TestCase):
    def test_current_uncertain_status_messages_survive_report_normalization(self):
        record = {
            "sound_type": "UNKNOWN",
            "audio_analysis_status": "UNCERTAIN",
            "sound_classification": "Unavailable: sound not reliably determined.",
            "sound_detection_status": "UNCERTAIN",
            "sound_detection_method": "HEURISTIC (SED model unavailable)",
            "prediction_status": "UNCERTAIN",
            "possible_category": None,
            "experimental_sound_candidate": "CRY",
            "cry_classification_status": "Not performed because CRY was not reliably established.",
            "cough_classification_status": "Unavailable because a verified cough-event classifier has not been trained.",
            "medical_diagnosis": "Not provided.",
            "top_screening_pattern": UNCERTAIN_PATTERN,
        }
        normalized = normalize_prediction_record(record)
        self.assertEqual(normalized["audio_analysis_status"], "UNCERTAIN")
        self.assertEqual(
            normalized["sound_classification"],
            "Unavailable: sound not reliably determined.",
        )
        self.assertEqual(
            normalized["cry_classification_status"],
            "Not performed because CRY was not reliably established.",
        )
        self.assertEqual(
            normalized["cough_classification_status"],
            "Unavailable because a verified cough-event classifier has not been trained.",
        )
        self.assertEqual(normalized["medical_diagnosis"], "Not provided.")
        self.assertIsNone(normalized["experimental_sound_candidate"])

        row = next(csv.DictReader(io.StringIO(ReportService().generate_csv_report([record]))))
        self.assertEqual(row["Audio_Analysis_Status"], "UNCERTAIN")
        self.assertEqual(row["Cry_Reason_Classification"], record["cry_classification_status"])
        self.assertEqual(row["Cough_Classification"], record["cough_classification_status"])
        self.assertEqual(row["Medical_Diagnosis"], "Not provided.")
        self.assertNotIn("Unverified_Heuristic_Candidate", row)
        self.assertNotIn("Unverified heuristic candidate", row)

        class FakeDoc:
            elements = None

            def __init__(self, *_args, **_kwargs):
                pass

            def build(self, elements):
                FakeDoc.elements = elements

        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(report_module, "REPORTS_DIR", temp), \
             mock.patch.object(report_module, "SimpleDocTemplate", FakeDoc):
            ReportService().generate_pdf_report(record)
        pdf_text = " ".join(
            getattr(element, "text", "") + " " + str(getattr(element, "_cellvalues", ""))
            for element in FakeDoc.elements
        )
        self.assertIn("Audio analysis status", pdf_text)
        self.assertIn("Not performed because CRY was not reliably established.", pdf_text)
        self.assertIn(
            "Unavailable because a verified cough-event classifier has not been trained.",
            pdf_text,
        )
        self.assertNotIn("Unverified heuristic candidate", pdf_text)

    def test_csv_report_uses_uncertain_status_no_candidate_pattern_or_risk(self):
        service = ReportService()
        record = {
            "sound_type": "CRY", "confidence": 0.988, "risk_level": "HIGH",
            "risk_score": 98, "screening_prediction": "Severe Belly Pain / Colic Cry",
            "probabilities_json": '{"prediction_status":"UNCERTAIN","possible_category":"tired","model_score":0.91,"sound_detection_status":"UNCERTAIN","sound_detection_method":"HEURISTIC"}',
        }
        rows = list(csv.DictReader(io.StringIO(service.generate_csv_report([record]))))
        row = rows[0]
        self.assertEqual(row["Prediction_Status"], "UNCERTAIN")
        self.assertEqual(row["Possible_Category"], "")
        self.assertEqual(
            row["Sound_Classification"],
            "Unavailable: sound not reliably determined.",
        )
        self.assertNotIn("Unverified_Heuristic_Candidate", row)
        self.assertIn("not revalidated", row["Historical_Result_Note"])
        self.assertEqual(row["Screening_Pattern"], UNCERTAIN_PATTERN)
        self.assertEqual(row["Risk_Status"], "No validated risk calculation is available.")
        self.assertIn("not clinically validated", row["Research_Disclaimer"])
        self.assertNotIn("Risk_Score", row)
        self.assertNotIn("tired", row["Possible_Category"])
        self.assertEqual(row["Medical_Diagnosis"], "Not provided.")
        self.assertIn("unavailable", row["Cough_Classification"].lower())

    def test_pdf_report_uses_same_uncertain_fields_and_disclaimer(self):
        class FakeDoc:
            elements = None
            def __init__(self, *_args, **_kwargs): pass
            def build(self, elements): FakeDoc.elements = elements

        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(report_module, "REPORTS_DIR", temp), \
             mock.patch.object(report_module, "SimpleDocTemplate", FakeDoc):
            service = ReportService()
            _, path = service.generate_pdf_report({
                "sound_type": "CRY", "confidence": 0.988, "risk_level": "HIGH",
                "risk_score": 98, "screening_prediction": "Severe Belly Pain / Colic Cry",
                "probabilities_json": '{"prediction_status":"UNCERTAIN","possible_category":"tired","model_score":0.91}',
            })
            self.assertTrue(Path(path).parent == Path(temp))
        content = " ".join(
            getattr(element, "text", "") + " " + str(getattr(element, "_cellvalues", ""))
            for element in FakeDoc.elements
        )
        self.assertIn("UNCERTAIN", content)
        self.assertIn("Not available", content)
        self.assertIn("Not calculated", content)
        self.assertNotIn("Severe Belly Pain", content)
        self.assertIn("not clinically validated", content)
        self.assertNotIn("tired", content)
        self.assertIn("Cough classification", content)
        self.assertIn("Medical diagnosis", content)


if __name__ == "__main__":
    unittest.main()
