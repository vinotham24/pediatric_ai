# Model Training and Evaluation

## Existing cry artifact

The file `models/cry_random_forest.joblib` retains its historical name, but audit inspection identifies its estimator as a StandardScaler plus class-weighted RBF SVC (`svc_balanced`). It was not replaced, modified, or retrained during final completion. It expects the existing feature representation and outputs one of eight contributor-tagged categories.

The reproduced historical test evaluation on 275 rows was: overall accuracy 0.6727, balanced accuracy 0.1609, macro F1 0.1470, weighted F1 0.6817. Tired recall was 0.0588; burping, cold_hot, lonely, and scared recall were 0. These scores demonstrate that overall accuracy is dominated by hungry examples and do not support reliable screening.

## Phase 7 experiments

Five experimental families were compared without deployment. Balanced Logistic Regression had the strongest validation macro F1 (0.2048) and balanced accuracy (0.3304). Its test macro F1 was 0.1301. The test partition contained only one tired and one cold_hot example. No tested model is ready for deployment. Phase 7 artifacts and results were preserved.

## Confidence and interpretation

The cry model's output score is uncalibrated and must not be described as probability of correctness. `CONFIDENT` requires explicit calibrated, screening-validated evidence and threshold metadata; such evidence is not available, so cry candidates remain UNCERTAIN. The sound heuristic has been removed; when the first-stage model is unavailable the sound is UNKNOWN with no score. No model score is converted to clinical risk.

## Sound detection and cough

There is no `models/sound_model/sed_model.joblib` artifact in the inspected checkout. The former pitch/flatness fallback was disabled because it could label a generated pure tone as CRY. There is no `models/cough_model/cough_model.joblib`, no verified event annotation rows, and no cough classifier training/evaluation result. Diagnosis cohort membership is not used as event-level ground truth.

## Future experiment requirements

Do not retrain until labels and source groups are reviewed. Use group-aware cross-validation for selection, reserve a new sealed independent evaluation cohort, report per-class precision/recall/F1, macro and weighted F1, balanced accuracy, confusion matrix, and calibration. Keep all formats/segments from a source event and all recordings from a participant in one split. Any threshold must be selected on validation data and justified by a defined use case. None of these steps establishes clinical validity by itself.
