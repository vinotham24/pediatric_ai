import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sampling & Processing Config
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
WINDOW_DURATION_SEC = 3.0
HOP_DURATION_SEC = 1.0
FRAME_SIZE = 2048
HOP_LENGTH = 512
N_MEL_BANDS = 128
N_MFCC = 20

# Risk Thresholds
CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_LABEL = "UNKNOWN"

# Age Groups
AGE_GROUPS = {
    "0-12m": {"label": "0–12 months (Infants)", "cry_weight": 0.5, "cough_weight": 0.3, "resp_weight": 0.2},
    "1-3y":  {"label": "1–3 years (Toddlers)", "cry_weight": 0.3, "cough_weight": 0.4, "resp_weight": 0.3},
    "3-5y":  {"label": "3–5 years (Preschool)", "cry_weight": 0.1, "cough_weight": 0.5, "resp_weight": 0.4},
    "5-10y": {"label": "5–10 years (School Age)", "cry_weight": 0.05, "cough_weight": 0.55, "resp_weight": 0.4}
}

# Directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATABASE_PATH = os.path.join(BASE_DIR, "database", "pediatric_ai.db")

LIVE_RECORDINGS_DIR = os.path.join(RECORDINGS_DIR, "live")
UPLOAD_RECORDINGS_DIR = os.path.join(RECORDINGS_DIR, "uploads")
PROCESSED_RECORDINGS_DIR = os.path.join(RECORDINGS_DIR, "processed")

# Sound Event Detection Classes
SED_CLASSES = ["CRY", "COUGH", "BREATHING", "SPEECH", "NOISE", "UNKNOWN"]

# Cry Pattern Classes (Clear, Specific, Faculty-Ready)
CRY_CLASSES = [
    "Severe Belly Pain / Colic Cry",
    "Abdominal Distress Cry",
    "Gas & Burping Discomfort Cry",
    "Hunger Cry",
    "Fatigue & Sleepiness Cry",
    "Physical Irritation Cry",
    "Normal / Baseline Cry"
]

# Cough Disease-Associated Screening Classes (Clear, Specific, Faculty-Ready)
COUGH_CLASSES = [
    "Croup Barking Cough Pattern",
    "Whooping Cough (Pertussis) Pattern",
    "Infant Bronchiolitis Wet Cough Pattern",
    "Pneumonia Chesty Cough Pattern",
    "Deep Bronchitis Cough Pattern",
    "Viral Wheezing / Asthmatic Cough Pattern",
    "Upper Respiratory Infection / Cold Cough Pattern",
    "Normal / Non-Abnormal Cough"
]

# Respiratory Sound Classes
RESPIRATORY_CLASSES = [
    "Normal breathing",
    "Abnormal breathing",
    "Wheezing-like sound",
    "Stridor-like sound"
]

# Disclaimer text
MEDICAL_DISCLAIMER = (
    "This application is an AI-assisted research screening system. "
    "Audio analysis cannot independently confirm a medical diagnosis. "
    "Results should not replace evaluation by a qualified healthcare professional. "
    "If a child has severe breathing difficulty, blue/gray lips or face, severe lethargy, "
    "inability to feed/drink, or another emergency symptom, seek urgent medical care."
)
