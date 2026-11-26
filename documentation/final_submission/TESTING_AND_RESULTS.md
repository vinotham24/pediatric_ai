# Testing and Results

## Executed test suites

Commands were run under the existing project virtual environment on 26 November 2026:

| Suite | Command pattern | Passed | Failed |
|---|---|---:|---:|
| Submission workflow integration | `.\venv\Scripts\python.exe -m unittest discover -s tests -p test_submission_workflow.py -v` | 7 | 0 |
| Cry upload regression | `.\venv\Scripts\python.exe -m unittest discover -s tests -p test_cry_upload.py -v` | 8 | 0 |
| Cry prediction consistency | `.\venv\Scripts\python.exe -m unittest discover -s tests -p test_cry_prediction_consistency.py -v` | 9 | 0 |
| Prediction status and reports | `.\venv\Scripts\python.exe -m unittest discover -s tests -p test_prediction_status_and_reports.py -v` | 11 | 0 |
| Cry review interface Phase 9/10 | `.\venv\Scripts\python.exe -m unittest discover -s tests -p test_cry_review_interface_phase9.py -v` | 11 | 0 |
| Existing core pipeline | `.\venv\Scripts\python.exe -m unittest discover -s tests -p test_pipeline.py -v` | 4 | 0 |
| **All discovered tests** | `.\venv\Scripts\python.exe -m unittest discover -s tests -v` | **50** | **0** |

## Coverage and meaning

- Workflow suite: page rendering; missing-model tone/cough-like inputs remain UNKNOWN; no filename-driven classification; specialist routing; cough unavailability; shared upload/live analysis and persistence; empty/corrupt/unsupported rejection through both APIs.
- Cry upload suite: generated WAV/MP3/M4A/OGG/FLAC files through both shared upload and live-analysis routes; the experimental cry-test endpoint cannot bypass first-stage routing; empty/corrupt/unsupported validation; missing-model/error behavior; feature resampling; temporary cleanup.
- Cry consistency suite: class order and label mapping, supplied tired label against metadata, uncertain status, and no filename as a model feature.
- Status/report suite: candidate/unknown/error/missing-score outcomes, no clinical risk score, required uncertain-result wording, PDF/CSV consistency, upload/live consistency.
- Review interface suite: temporary fixture queue/state persistence, source-label separation, invalid audio blocking, duplicate grouping, CAF/3GP decode-to-WAV, full/range HTTP responses and 416 handling.

Audio-format fixtures are generated signals, not patient recordings; review fixture files and progress are temporary. The final full run passed all 50 tests with 0 failures. The focused routing/status/report run passed all 35 tests with 0 failures. No model training, dependency installation, or model download occurred.

The legacy core suite's risk-engine unit test verifies a standalone legacy utility. It does not mean that the live application uses or reports a clinical risk score; the integrated inference and report path explicitly marks risk as not calculated.

## Not verified by these tests

Automated tests do not verify a human microphone/browser session, browser permissions, real device audio quality, cross-browser UI rendering, clinical utility, patient-independent generalization, or calibrated probabilities. The five audio formats are validated through generated fixtures on both APIs, not actual browser microphone capture. Cough classifier tests confirm explicit unavailability; they are not cough model performance tests. A manual local demonstration remains necessary before screenshots/submission.
