# Pediatric Cry and Cough Audio Research Prototype — Final Project Report

**Project:** Pediatric Cry and Cough AI  
**Submission preparation:** 26 November 2026  
**Status:** Academic research prototype; not clinically validated

## Abstract

This project combines audio upload and microphone capture with a staged sound-analysis workflow and a Flask interface for history and research reports. The existing cry-category estimator remains in place, but evaluation shows weak minority-class performance and its score is not calibrated as a probability of correctness. The first-stage sound-event model is absent; the former pitch/flatness heuristic was disabled after a generated 440 Hz tone reproduced an unsupported CRY label. The application now reports UNKNOWN and states that sound classification is unavailable when no supported detector is present. No verified cough-event annotations or cough classifier are available, so cough classification is unavailable. The completed integration gives upload and live API requests the same analysis and persistence path, removes unsupported risk scoring, and reports uncertainty. It does not provide a medical diagnosis.

## Introduction and problem statement

Infant vocal audio is difficult to classify reliably from a small, imbalanced, contributor-labeled dataset. A useful research prototype needs to preserve source provenance, group repeated recordings, report uncertainty, and distinguish model outputs from medical findings. This project demonstrates an audio analysis workflow while documenting where the evidence does not support classification.

## Objectives

- Accept supported audio uploads and microphone recordings.
- Run the existing cry-category model only after a supported first-stage detector accepts CRY; keep cough analysis unavailable until verified event labels and a model exist.
- Avoid specialist classification for OTHER and UNKNOWN.
- Keep a shared status, history, and report path for uploads and live submissions.
- Preserve existing cry and Phase 7 artifacts; do not train a new model or manufacture labels.

## Existing and proposed system

The original application contained separate upload and live handling and legacy clinical-style risk/result fields. The current research workflow shares the same `InferencePipeline` for both submission types, persists through the existing SQLite history schema, strips obsolete score/risk interpretations from displayed and exported history, and adds explicit research disclaimers. Upload/live integration is exercised by automated Flask tests. Physical microphone and browser playback remain manual demonstration checks.

## System architecture

See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md). In short: browser upload or MediaRecorder submission → audio validation/decode → first-stage detector → optional CRY or COUGH specialist → status-gated response → SQLite history → HTML/PDF/CSV.

## Technologies

Python, Flask, Flask-SocketIO, NumPy, SciPy, librosa, SoundFile/FFmpeg utilities, scikit-learn/joblib, SQLite, JavaScript, HTML/CSS, ReportLab. Versions and declared dependencies are in the existing `requirements.txt`; no dependency was installed as part of this completion.

## Datasets and sources

### Cry corpus

The local Donate-a-Cry-derived corpus has 1,583 feature/metadata rows, including 1,389 rows in the eight model categories and 194 `unknown` rows. Two modeled rows are silent, leaving 1,387 usable labeled rows; these represent 930 source-recording keys after collapsing alternate copies. Hungry accounts for 1,124 of 1,389 modeled rows (about 80.9%). The tags are contributor-submitted suspected reasons, not verified medical ground truth. App-installation UUID is available as a grouping proxy, not a patient ID. Seven `cold_hot` code files in the cleaned `discomfort` folder remain for manual review. See the Phase 6/8 reports.

### Pediatric cough recordings

The verified West China collection contains 82 pneumonia-group and 91 bronchitis-group MP3 recordings (173 total). These are source disease-group cohorts, not cough-event labels; the paper reports a cohort age range of 0 to 11 years, but the released P1 through P82 and B1 through B91 inventory has no per-file age mapping, and cough-event timestamps are unavailable. The event annotation CSVs currently contain zero verified event rows and processed cough/non-cough directories contain only placeholders. No cough classifier is trained or claimed. The source's official record is [Figshare DOI 10.6084/m9.figshare.21176197.v1](https://doi.org/10.6084/m9.figshare.21176197.v1); the prior download verification/attribution record should accompany reuse. Cite Liao, Siqi; Song, Chao; Wang, Xiaoqin; and Wang, Yanyun (2022), link CC BY 4.0, identify changes, and do not imply author endorsement.

## Preprocessing and features

The existing cry preparation converts supported source audio to 16 kHz mono. The feature table contains 91 numeric features; the audit found the earlier five-second feature window truncates some recordings whose median duration is about 6.96 seconds. No new cry features were generated for this completion. Cough annotations/segmentation tools are separate and preserve source recordings; no verified segments exist. Live/upload preprocessing validates and decodes audio before inference.

## Models and evaluation

The production-path artifact `models/cry_random_forest.joblib` is a legacy filename: the saved estimator is a StandardScaler plus class-weighted RBF SVC. It was preserved. Phase 6 reproduced its historical test metrics: accuracy 0.6727, balanced accuracy 0.1609, macro F1 0.1470, weighted F1 0.6817 on 275 rows. Hungry recall was 0.8044; recall was zero for burping, cold_hot, lonely, and scared, and 0.0588 for tired. This is not reliable multiclass performance.

Phase 7 compared five experimental model families on a grouped split; no experiment artifact was deployed. Balanced Logistic Regression led validation macro F1 at 0.2048 and balanced accuracy at 0.3304, but test macro F1 was 0.1301. The test partition had one tired and one cold_hot example. These results do not support promotion. Probabilities are not independently calibrated. Full metrics and split details are in `training/reports/cry_dataset_audit.md` and `training/reports/cry_model_comparison_phase7.md`.

No first-stage SED model file is present. The former pitch/flatness heuristic could label a pure 440 Hz tone as CRY and was removed from active detection. A separate generated cough-like pulse fixture remained UNKNOWN; neither fixture establishes the content of any user recording. No pretrained sound-event inference runtime or YAMNet weights were found locally. YAMNet is a possible future candidate, but it requires approval before downloading weights/installing compatible dependencies and requires further license/attribution, threshold, and CPU validation. The cough specialist explicitly returns unavailable/unknown and does not infer disease from pneumonia/bronchitis cohort membership.

## Application implementation and workflow

Upload and browser MediaRecorder submissions use the same backend processing helper and inference pipeline. Supported files are validated and decoded; saved filenames are opaque and sanitized. If a supported detector is absent or uncertain, the output sound type is UNKNOWN, the interface reports classification unavailable, and neither specialist runs. CRY analysis runs only after a supported detector accepts CRY; the cry-test page uses the same gate. COUGH classification is unavailable because there is no verified event-labeled cough model. OTHER/UNKNOWN do not invoke either specialist. Analysis history is written through the shared SQLite schema for upload and live paths. Results and PDF/CSV reports identify uncertain or unknown output, explain score limitations, and do not derive clinical risk from model confidence. Historical database rows are not rewritten; their presentation identifies results whose routing was not revalidated.

The manual cry label-review interface is separate from Flask. Phase 10 fixture tests verify visible invalid-file states, playback gating, source-label separation, persistence, duplicate grouping, and byte-range seeking. No actual manual decisions have been entered. The cough annotation interface is likewise not evidence of completed annotation.

## Testing and results

See [TESTING_AND_RESULTS.md](TESTING_AND_RESULTS.md). All six discovered unittest suites ran successfully: 50 tests passed, 0 failed. They cover the core pipeline, workflow integration, synthetic tone/cough-like UNKNOWN behavior, five upload formats, invalid inputs, cry routing, class mapping, statuses/reports, and review-interface safety. They do not prove clinical validity, physical microphone behavior, or patient-independent performance.

## Results and discussion

The application now offers a consistent and testable research demonstration with explicit missing-model and uncertainty behavior. Cry classification runs only after an accepted CRY result from a supported detector; the existing cry model's output must not be interpreted as a reliable reason or clinical finding. The detector is not a trained first-stage model in the current checkout. Cough event classification remains unimplemented because verified event labels are absent. No new test-set metric or model improvement is claimed.

## Screenshots checklist

Capture screenshots manually using the list in [DEMO_AND_VIVA_GUIDE.md](DEMO_AND_VIVA_GUIDE.md): home/dashboard disclaimer, upload form, uncertain result, live result, history row, PDF/CSV export, and cry review queue status. Redact identifying audio/file information before submission.

## Limitations

- Cry labels are contributor supplied, not independently verified; class imbalance is severe.
- Patient IDs are unavailable; app-instance grouping cannot guarantee subject-independent evaluation.
- Test partitions are small for minority classes and historical test sets have been reused.
- Cry probabilities are uncalibrated and predictions are explicitly uncertain.
- The first-stage trained sound detector and cough classifier are absent.
- The cough dataset has diagnosis-group labels, no verified cough-event labels, no event boundaries, and no per-recording ages.
- A real browser/microphone end-to-end session was not executed in this environment.

## Future work

Complete manual cry label review without overwriting source tags; acquire independently recorded minority-class examples under documented licenses and participant grouping; establish a sealed patient-independent evaluation set and calibration protocol; annotate cough/non-cough/unknown events with timestamp review; only then conduct pre-registered model comparisons. Clinical use would require separate prospective validation, governance, and regulatory review.

## Conclusion

The current deliverable is an academic research prototype, not a diagnostic or screening system. Upload/live integration and reporting are covered by passing automated tests. Existing cry artifacts are preserved, cough classification remains unavailable, and limitations are presented directly.

## References

- Donate-a-Cry project/corpus materials included locally; dataset-specific attribution and provenance should be retained with source redistribution.
- Liao, Siqi; Song, Chao; Wang, Xiaoqin; Wang, Yanyun (2022). *Data from: A classification framework for identifying bronchitis and pneumonia in children based on a small-scale cough sounds dataset*. Figshare. https://doi.org/10.6084/m9.figshare.21176197.v1
- CC BY 4.0 license: https://creativecommons.org/licenses/by/4.0/
- West China dataset paper: https://www.nature.com/articles/s41597-023-02266-0
- Coswara Data repository (not the source used for the West China recordings): https://github.com/iiscleap/Coswara-Data
- Project audit and experiment reports: `training/reports/cry_dataset_audit.md`, `training/reports/cry_model_comparison_phase7.md`, and Phase 8–10 reports.
