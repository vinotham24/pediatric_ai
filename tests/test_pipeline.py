import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR
from ml.preprocessing import load_and_preprocess_audio
from ml.feature_extraction import extract_acoustic_features, extract_mel_spectrogram
from ml.inference import pipeline_instance
from ml.risk_engine import RiskEngine
from services.report_service import ReportService
from database.models import init_db, get_db_connection

class TestPediatricAI(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_feature_extraction(self):
        y = np.random.normal(0, 0.1, 16000).astype(np.float32)
        features = extract_acoustic_features(y, 16000)
        self.assertTrue(len(features) > 50)
        
        mel = extract_mel_spectrogram(y, 16000)
        self.assertEqual(mel.shape[0], 128)

    def test_inference_pipeline(self):
        dataset_sample = os.path.join(BASE_DIR, "datasets", "cry", "cry_pain_01.wav")
        if os.path.exists(dataset_sample):
            res = pipeline_instance.process_audio(dataset_sample, age_group="0-12m")
            self.assertIn("detected_sound", res)
            self.assertIn("risk_result", res)
            self.assertIn("top_screening_pattern", res)

    def test_risk_engine(self):
        engine = RiskEngine()
        risk = engine.calculate_risk(
            age_group="0-12m",
            sed_type="COUGH",
            sed_confidence=0.92,
            cough_data={"abnormality_score": 80, "cough_events": 6, "top_pattern": "Pneumonia-associated pattern", "probabilities": {"Pneumonia-associated pattern": 0.75}},
            caregiver_symptoms={"temperature": 38.8, "spo2": 94}
        )
        self.assertIn(risk["risk_level"], ["HIGH", "MODERATE", "LOW"])
        self.assertTrue(risk["risk_score"] > 50)

    def test_pdf_report(self):
        rs = ReportService()
        sample_record = {
            "id": 999,
            "age_group": "0-12m",
            "source": "TEST",
            "filename": "test.wav",
            "duration": 3.5,
            "sound_type": "COUGH",
            "confidence": 0.94,
            "screening_prediction": "Pneumonia-associated pattern",
            "risk_level": "MODERATE",
            "risk_score": 55,
            "probabilities": {"Pneumonia-associated pattern": 0.72, "Bronchitis-associated pattern": 0.18},
            "explanation_summary": "Test explanation summary."
        }
        filename, filepath = rs.generate_pdf_report(sample_record)
        self.assertTrue(os.path.exists(filepath))

if __name__ == '__main__':
    unittest.main()
