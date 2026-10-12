# Real-Time Pediatric Health Risk and Disease-Associated Screening Using Deep Learning-Based Cry and Cough Sound Analysis

A complete, research-oriented, real-time web application for pediatric audio analysis (infant crying, pediatric coughing, and respiratory sounds) for children aged **0–10 years**.

> ⚠️ **MEDICAL DISCLAIMER:** This application is an AI-assisted research screening system. Audio analysis cannot independently confirm a medical diagnosis. Results should not replace evaluation by a qualified healthcare professional. If a child has severe breathing difficulty, cyanosis (blue/gray lips or face), severe lethargy, or inability to feed/drink, seek urgent medical care.

---

## 🌟 Key Features

1. **Three Input Modes**:
   - 🎙️ **Real-Time Microphone**: Continuous 3-second overlapping sliding window analysis, live canvas waveform visualizer, WebSocket event streaming, auto-saving of detected audio snippets (`recordings/live/`).
   - 📁 **Audio File Upload**: Drag-and-drop upload (WAV, MP3, M4A, OGG, FLAC) with auto-detection of sound event type and optional caregiver symptom checklist.
   - 🧪 **Research Dataset / Demo Mode**: Pre-packaged research dataset explorer (Donate-a-Cry, Baby Chillanto, Coswara Pediatric Cough) with ground-truth vs AI prediction evaluation.

2. **Multi-Stage Machine Learning Pipeline**:
   - **Sound Event Detection (SED)**: Multi-class classifier (`CRY`, `COUGH`, `BREATHING`, `SPEECH`, `NOISE`, `UNKNOWN`).
   - **Infant Cry Model**: Evaluates cry duration, intensity, cry abnormality score (0–100), and pattern probabilities (`Pain-associated`, `Hunger-associated`, `Discomfort`, `Tiredness`, `Burping`, `Normal Cry`).
   - **Pediatric Cough Screening Model**: Evaluates cough burst count, cough frequency, and disease-associated screening probabilities (`Pneumonia-associated pattern`, `Bronchitis-associated pattern`, `Bronchiolitis-associated pattern`, `Viral wheezing-associated pattern`, `Upper respiratory infection-associated pattern`, `Normal/Other`).
   - **Respiratory Model**: Evaluates normal breathing vs `Wheezing-like sound`, `Stridor-like sound`, and `Abnormal breathing`.

3. **Multi-Signal Risk Fusion Engine**:
   - Fuses child age weights (0-12m, 1-3y, 3-5y, 5-10y), acoustic abnormality scores, screening probabilities, event repetition rates, and caregiver reported symptoms (Temp °C, SpO2 %, Resp Rate, Fever, Retractions, Lethargy).
   - Incorporates a **Temporal Persistence Filter** to avoid false alarms from isolated low-confidence noise bursts.
   - Outputs: `LOW` (Green), `MODERATE` (Yellow), `HIGH` (Red) risk levels.

4. **AI Assistant & Guideline Explanations**:
   - Grounded in WHO, CDC, NIH, and NHS pediatric clinical reference guidelines.
   - Explains findings, provides caregiver observation checklists, lists red-flag emergency symptoms, and outlines notes for pediatrician consultations.

5. **Clinical PDF & CSV Reporting**:
   - Downloadable research PDF reports with embedded risk summary, screening probabilities table, and disclaimers.
   - Exportable CSV history logs for statistical analysis.

---

## 📁 Directory Structure

```text
pediatric-ai/
├── app.py                      # Flask main entrypoint & Socket.IO server
├── config.py                   # Configuration constants & model thresholds
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/                     # Trained ML model weights (.joblib)
│   ├── sound_model/
│   ├── cry_model/
│   └── cough_model/
│
├── ml/                         # Machine Learning Pipeline
│   ├── preprocessing.py        # Resampling, mono, VAD, audio windowing
│   ├── feature_extraction.py   # STFT, Mel Spectrogram, MFCC, Pitch, Spectral metrics
│   ├── sound_detection.py      # Sound Event Detector inference
│   ├── cry_analysis.py         # Infant Cry Classifier & Abnormality Scorer
│   ├── cough_analysis.py       # Pediatric Cough Classifier & Event Counter
│   ├── respiratory_analysis.py # Respiratory sound analyzer
│   ├── risk_engine.py          # Multi-Signal Risk Fusion Engine
│   ├── explainability.py       # Acoustic Explainer
│   └── inference.py            # Unified Pipeline Coordinator
│
├── training/                   # Dataset & Model Training Utilities
│   ├── download_datasets.py    # Setup dataset samples & manifests
│   ├── prepare_dataset.py      # Subject-level dataset splitter (no data leakage)
│   └── train_models.py         # Model training & serializer script
│
├── services/                   # Application Services
│   ├── audio_service.py        # File saving & management
│   ├── ai_assistant.py         # WHO/CDC grounded AI Assistant
│   └── report_service.py       # PDF & CSV generator
│
├── database/                   # SQLite Storage
│   └── models.py               # ORM Schema & connection manager
│
├── datasets/                   # Research Dataset Audio Files
│   ├── cry/
│   ├── cough/
│   └── respiratory/
│
├── recordings/                 # Saved Audio Snippets
│   ├── live/
│   └── uploads/
│
├── templates/                  # Modular HTML Templates
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── live.html
│   ├── upload.html
│   ├── dataset.html
│   ├── history.html
│   └── analysis.html
│
└── static/                     # Web Assets
    ├── css/
    └── js/
```

---

## 🚀 Quick Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Datasets & Train Models
```bash
python training/download_datasets.py
python training/train_models.py
```

### 3. Run Unit Tests
```bash
python -m unittest tests/test_pipeline.py
```

### 4. Launch Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## 🧪 Demonstration Scenarios

1. **Live Microphone Mode**: Navigate to `/dashboard` or `/live`, select child age, click **START MONITORING**, produce audio, and observe real-time waveform, sound event detection, cumulative counters, and auto-saved snippets.
2. **Upload Mode**: Navigate to `/upload`, upload an audio file (WAV/MP3/M4A), select metadata, and click **Analyze Audio File** to view spectrograms, probability bar charts, and AI assistant explanations.
3. **Dataset Demo Mode**: Navigate to `/dataset` to view research samples (Donate-a-Cry, Baby Chillanto, Coswara) and run live AI benchmark comparisons.
4. **PDF & CSV Export**: Navigate to `/history` to download generated clinical PDF reports and full CSV logs.
