# Cough Sound Detection Fix and Investigation

## Summary

The reported `free-sound-1674744248.mp3` was not present in the workspace, so its contents were not listened to or independently classified. Its filename was not used as evidence. The active pitch/flatness fallback has been removed. When the trained first-stage sound-event detector is absent, the application now returns `UNKNOWN`, reports sound classification as unavailable, does not show a heuristic CRY candidate, and does not invoke either specialist. Cough classification remains unavailable because no cough-event classifier has been trained.

No pretrained sound-event model or compatible inference runtime was available locally. YAMNet was investigated as a possible future candidate, but it was not installed or integrated. No package installation, model download, or network/environment change was made.

## Root cause

The first-stage SED model artifact was missing, so the application used a hand-built signal heuristic based on RMS, pitch estimates, and spectral flatness. Its broad voiced/high-pitch and low-flatness rules could label tonal or voiced audio as CRY without learned sound-event evidence. In a local synthetic reproduction, a generated 440 Hz tone triggered CRY; a separate generated cough-like pulse fixture remained UNKNOWN. These fixtures demonstrate a plausible false-positive mechanism only; neither establishes the contents of the absent user MP3.

The heuristic candidate was not a confirmed detector result, but it remained visible in the primary UI as an unverified CRY candidate. Routing was already gated against invoking the cry-reason model for uncertain heuristic results; this fix removes the heuristic output entirely when no supported SED model is present and keeps the actual sound type UNKNOWN. The cry-reason model remains unchanged and is only invoked for an accepted CRY result from a supported detector.

## Local model and dependency investigation

- The project environment includes the existing audio/preprocessing and scikit-learn dependencies, but no TensorFlow, TensorFlow Hub, `tf-keras`, Torch, torchaudio, Transformers, TFLite runtime, ONNX Runtime, PANNs, OpenL3, or Hugging Face Hub inference stack was installed.
- The project model directory contained cry-model artifacts but no YAMNet, SED, or cough-event model weights. The checked common model caches contained no YAMNet or sound-event weights.
- YAMNet is a plausible general audio-event model to evaluate in a future, approved integration. Its official 521-class map includes `Baby cry, infant cry` (index 20) and `Cough` (index 42). It returns scores across AudioSet classes, not a medical diagnosis. `OTHER` would require an explicit policy over other classes, and `UNKNOWN` is not a native output class; conservative rejection thresholds and validation would be needed.
- YAMNet expects normalized mono audio at 16 kHz. The official README describes overlapping 0.96-second windows and requires at least about 975 ms to produce the first score frame. The project preprocessing already produces mono 16 kHz input.
- The official README lists TensorFlow, `tf-keras`, `resampy`, NumPy, and SoundFile and requires separate YAMNet weights. The README warns of a Keras 2 compatibility requirement and incompatibility with Keras 3 in the default TensorFlow 2.16 setup. The model is small (3.7M weights; 69.2M multiplies per 960 ms frame), but CPU execution was not measured in this environment.
- The TensorFlow Models repository's `research` code is covered by its Apache 2.0 license. Distribution terms and attribution requirements for the separate weights/model and any AudioSet-derived materials must be confirmed before use; they were not treated as cleared by this investigation.
- References: [YAMNet official README](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet), [official class map](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv), [TensorFlow Models license](https://github.com/tensorflow/models/blob/master/LICENSE), and [TensorFlow YAMNet tutorial](https://www.tensorflow.org/hub/tutorials/yamnet).

## Integration decision

No model was integrated. There were no local weights/runtime to run, and the task did not authorize downloading weights or installing dependencies. Class mapping, licensing/attribution for the separate weights, CPU behavior, and calibrated rejection thresholds also remain to be verified before an integration could be responsibly accepted. The existing experimental cry model was preserved; no model was trained, and no Phase 6 or Phase 7 experiment artifacts were intentionally modified by this fix.

## Behavior after the fix

- Missing SED: actual sound type and prediction status are `UNKNOWN`; the UI, history, and reports say `Unavailable: sound not reliably determined.`
- No heuristic candidate or filename/metadata-based prediction is exposed.
- Cry-reason classification is not performed for UNKNOWN.
- Cough classification says that a verified cough-event classifier has not been trained; no cough category or diagnosis is fabricated.
- Upload and live recording continue to share the same inference and persistence path.
- The existing medical/research disclaimer remains visible. No medical diagnosis or clinical validation is claimed.

## Files changed for this fix

- Detection and routing: `ml/sound_detection.py`, `ml/inference.py`, `app.py`.
- History, explanation, and export behavior: `services/prediction_status.py`, `services/ai_assistant.py`, `services/report_service.py`.
- User-facing rendering: `static/js/prediction_result.js`, `static/js/history.js`, `templates/upload.html`, `templates/live.html`, `templates/analysis.html`, `templates/index.html`.
- Regression coverage: `tests/test_submission_workflow.py`, `tests/test_cry_upload.py`, `tests/test_cry_prediction_consistency.py`, `tests/test_prediction_status_and_reports.py`.
- Related project documentation: `documentation/final_submission/FINAL_PROJECT_REPORT.md`, `KNOWN_LIMITATIONS_AND_FUTURE_WORK.md`, `MODEL_TRAINING_AND_EVALUATION.md`, `SYSTEM_ARCHITECTURE.md`, `DEMO_AND_VIVA_GUIDE.md`, `FINAL_SUBMISSION_CHECKLIST.md`, and `TESTING_AND_RESULTS.md`.

## Tests

Final validation used the existing Python 3.13.5 project virtual environment:

- Focused routing/status/report modules: **35 tests passed, 0 failed**.
- Full suite: `.\venv\Scripts\python.exe -m unittest discover -s tests -v` — **50 tests passed, 0 failed**.
- `git diff --check` completed without whitespace errors.

The new regressions cover missing-detector behavior for a generated tonal signal and a generated cough-like signal, filename invariance, no cry classifier invocation for UNKNOWN, no fabricated cough output, upload/live consistency, and uncertainty in history/PDF/CSV. Synthetic signals are not patient recordings and do not establish clinical or real-world model performance.

## Remaining limitations and exact next steps

1. Keep sound classification UNKNOWN/unavailable until a suitable detector is available; do not infer labels from filenames or user-selected metadata.
2. Before any model acquisition, obtain approval to install compatible inference dependencies and download a pinned YAMNet (or equivalent) model artifact. Confirm the model-weight license, attribution, model provenance, and class-map version.
3. Implement a separate first-stage detector that maps only supported model classes to CRY/COUGH/OTHER, with explicit confidence/rejection handling for UNKNOWN. Select thresholds on suitable held-out data; do not treat YAMNet class scores as calibrated probabilities.
4. Verify CPU runtime, input duration handling, ambiguous/background/unsupported sounds, and conservative behavior with representative validation clips. Do not claim clinical validity.
5. Continue to report cough classification as unavailable until suitable cough-event annotations and an appropriate trained classifier exist. Pneumonia/bronchitis cohort labels are not cough-event labels.
6. Manually verify upload and browser microphone behavior in a local browser after an approved detector integration. This was not tested with a physical microphone in this fix.
