# Final Submission Checklist

## Run locally

From PowerShell:

```powershell
cd C:\Users\sruja\pediatric_ai
.\venv\Scripts\Activate.ps1
```

Use the existing project environment; do not install dependencies without approval. Start Flask:

```powershell
python app.py
```

Open **http://127.0.0.1:5051**.

## Demonstration steps

1. Open Upload Audio, choose a supported audio file, submit, and inspect status and uncertainty wording.
2. Open Live Record & Analyze on localhost, grant microphone permission, record, stop, and submit.
3. Open History, inspect an upload/live record, and try PDF/CSV report export.
4. Do not present any score as a diagnosis or probability of correctness. The interface is not clinically validated.

## Feature status

| Feature | Status | Evidence/limitation |
|---|---|---|
| Flask pages/navigation | WORKING | Flask page-render tests pass; inspect local browser before submission. |
| Supported audio upload validation | WORKING | Generated WAV/MP3/M4A/OGG/FLAC files pass through both upload and live-analysis routes; empty, corrupt and unsupported inputs are rejected on both. |
| Shared upload/live backend analysis and persistence | WORKING | Fixture test confirms same pipeline and history insert. |
| Physical browser microphone flow | PARTIALLY WORKING | Frontend and API path exist; physical device/browser session not tested. |
| First-stage trained CRY/COUGH/OTHER detector | NOT IMPLEMENTED | No SED model artifact is installed; the unreliable heuristic is disabled, so the actual sound type remains UNKNOWN and classification is reported unavailable. |
| Cry category candidate | PARTIALLY WORKING | Existing model loads, but poor metrics/unverified labels mean output stays uncertain and experimental. |
| Cough-event classification | NOT IMPLEMENTED | No verified event rows/clips or trained classifier. |
| OTHER/UNKNOWN specialist routing | WORKING | Tests confirm no second-stage classifier invoked. |
| History persistence | WORKING | Shared integration test validates upload/live inserts; inspect app DB/browser manually. |
| PDF/CSV report status consistency | WORKING | Automated tests pass; exports are research-only. |
| Clinical risk score | NOT IMPLEMENTED | Intentionally unavailable; no validated risk function. |
| Cry label review safeguards | WORKING | 11 fixture tests pass; no real annotations completed. |
| Patient-independent evaluation | BLOCKED | No reliable patient IDs. |
| Cough event annotations | BLOCKED | Manual annotation is required; current event files have zero verified rows. |

## Test commands

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_submission_workflow.py -v
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_cry_upload.py -v
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_cry_prediction_consistency.py -v
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_prediction_status_and_reports.py -v
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_cry_review_interface_phase9.py -v
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Observed result on 26 November 2026: focused and full-suite results are recorded in [TESTING_AND_RESULTS.md](TESTING_AND_RESULTS.md). The physical microphone/browser flow and screenshot steps remain manual.

## Screenshots to capture before submission

- Home page/disclaimer.
- Upload page and an actual test-file result.
- Live capture page and an actual microphone result.
- History list/detail page.
- PDF export and CSV export.
- Cry review app status/queue, if included in demonstration.
- Documentation directory.

Capture these manually. Do not substitute mock output for the real browser/microphone demonstration.

## Remaining manual steps

- Confirm the browser grants microphone access and that a real microphone recording can be submitted.
- Verify one real local audio upload and inspect its history/detail/PDF/CSV result.
- Capture/redact screenshots.
- Retain dataset attribution and license information with any redistributed dataset copy.
- Explain the missing trained detector, unavailable cough classifier, weak cry evaluation, lack of patient IDs, and non-clinical status during submission.

## Protection confirmation

No model was trained, no external dataset was downloaded, no new dependencies were installed, no original audio/labels were edited, and no Git commit or push was made during this final completion. Pre-existing working-tree changes were preserved.
