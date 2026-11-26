# Demo and Viva Guide

## Start the application

From PowerShell:

```powershell
cd C:\Users\sruja\pediatric_ai
.\venv\Scripts\Activate.ps1
python app.py
```

Open **http://127.0.0.1:5051**. If the environment is not already provisioned, see the checklist for installing only the existing `requirements.txt` dependencies. Use localhost for microphone permission; browser/device policy may still block capture.

## Suggested demonstration

1. Show the home page and research/non-clinical disclaimer.
2. Open Upload Audio. Submit a short permitted WAV, MP3, M4A, OGG, or FLAC test clip. Explain that sound classification is unavailable without a trained first-stage detector, the actual sound type is UNKNOWN, no cry-reason category is run, and cough classification has not been trained.
3. Open Live Record & Analyze. Grant microphone access, record a short clip, stop, and submit. Point out that the backend uses the same analysis route as upload and stores a history entry.
4. Open History and select a result. Demonstrate PDF and CSV exports if available. Explain that confidence is not a clinical probability and risk is not calculated.
5. Show `training/cry_label_review_app.py` separately only if requested; do not imply that any labels have already been manually reviewed.
6. If showing the cry-test page, note that it shares the first-stage sound gate and will not produce a cry-reason category while sound type is UNKNOWN.
7. Explain cough classification is unavailable until verified cough-event annotations exist.

## Screenshots to capture manually

- Home/research disclaimer.
- Upload page with supported formats visible.
- Upload result showing prediction status and model-score limitation.
- Live page before and after a local test recording.
- History list and a detail page.
- PDF and CSV report opened locally, with no identifying data exposed.
- Cry review queue showing a normal item and a visible invalid/silent status if demonstrating fixtures.
- Submission checklist and project documentation tree.

Do not fabricate successful outputs for absent models. Use fixture-only test output only when clearly labeled as a test. Redact UUIDs, names, recordings, and other personal data.

## Viva talking points

- The production cry artifact was preserved; no training occurred in this final integration.
- Cry evaluation is imbalanced and weak; predictions remain experimental and uncertain.
- A trained sound detector and cough classifier are absent; the system does not fabricate them.
- Pneumonia/bronchitis cohorts are not cough-event labels.
- Automated tests verify routes and reports but do not demonstrate clinical performance or a real microphone session.
