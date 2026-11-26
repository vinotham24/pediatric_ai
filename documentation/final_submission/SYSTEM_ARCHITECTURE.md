# System Architecture

## Runtime flow

```text
Browser upload ─┐
                ├─> shared Flask audio request ─> validate/save/decode ─> InferencePipeline
MediaRecorder ──┘                                                   │
                                                                   v
                                         accepted first-stage sound detector result
                                                ┌───────────┼───────────┐
                                               CRY         COUGH     OTHER/UNKNOWN
                                                │             │             │
                                      existing cry model  unavailable gate  no specialist
                                                └─────────────┴─────────────┘
                                                                   v
                                   status-gated response + shared SQLite history
                                                                   v
                                                page / history / PDF / CSV
```

## Components

- `app.py`: Flask pages, shared upload/live APIs, gated cry-test endpoint, detail/history/report endpoints, and Socket.IO backend hooks.
- `services/audio_service.py`: format checks, safe opaque filenames, audio decoding, and temporary-file cleanup.
- `ml/inference.py`: single routing and status path shared by upload and live analysis.
- `ml/sound_detection.py`: loads a first-stage model only if an artifact exists; none is present in this checkout. When unavailable, it returns UNKNOWN with no score; the unreliable pitch/flatness heuristic has been removed.
- `ml/cry_analysis.py`, `services/cry_model_service.py`: existing cry model integration. The artifact is preserved and its result remains experimental/uncertain.
- `ml/cough_analysis.py`: explicit unavailable result; no trained cough classifier exists.
- `services/prediction_status.py`: shared status, history sanitization, uncertainty text, and removal of unsupported legacy risk/confidence fields.
- `database/`: SQLite persistence for audio records, sessions, and predictions.
- `services/report_service.py`: PDF and CSV exports from the same normalized prediction metadata.
- `static/js/live_monitor.js`, `static/js/upload.js`, `static/js/prediction_result.js`: browser capture/submission and rendering.

## Routing and status contract

Only a model-backed first-stage result explicitly marked `CONFIDENT` can be accepted. If the detector is absent or uncertain, the detected sound is `UNKNOWN`, the interface says sound classification is unavailable, no candidate label is displayed, and neither specialist is invoked. Accepted CRY invokes the cry analyzer; accepted COUGH reaches the cough availability gate; OTHER invokes neither specialist. The same gate also applies to the experimental cry-test page. ERROR is used when analysis fails. A candidate without explicit validation/calibration evidence cannot become `CONFIDENT`. No medical risk score is derived from model confidence. An absent detector model never triggers filename-based classification.

## Storage

Upload and live APIs call the same persistence helper and record source type in the existing SQLite history structure. When the legacy database requires non-null detector confidence but no detector score exists, a storage sentinel is used internally and stripped from history/API/report output. Prediction metadata records the score interpretation and research disclaimer.

## Operational boundaries

The Socket.IO server handlers exist, but the current browser flow submits a completed MediaRecorder blob through the live HTTP API. Live microphone capture, browser permissions, and device playback need a manual check at localhost. The live API integration is automated-test verified. Dashboard legacy live-summary widgets are not connected to live capture; the Live page is the operative workflow.
