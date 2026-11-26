# Known Limitations and Future Work

## Current limitations

1. **Cry labels:** Source tags are contributor-submitted suspected reasons and are not medically verified. Seven cold_hot-coded files conflict with cleaned-folder taxonomy and require human review.
2. **Class imbalance:** Hungry is about 80.9% of modeled rows. Minority class usable recording-key counts range from 11 to 56. Historical model balanced accuracy and macro F1 are poor.
3. **Participant independence:** Source IDs identify app installations, not babies. Grouping prevents some copy leakage but cannot prove subject-level separation.
4. **Evaluation:** Minority test supports are tiny; Phase 7 had one tired and one cold_hot example. Prior test reuse limits claims. No model is promoted.
5. **Sound detector:** No trained first-stage SED artifact is installed. The heuristic fallback was removed because voiced/tonal signals could be labeled CRY based on pitch/flatness thresholds. Without a detector, sound classification is unavailable and results remain UNKNOWN; neither specialist is invoked. The cry-test page also uses the same first-stage gate.
6. **Cough:** 173 source MP3s are grouped by pneumonia/bronchitis cohort. No verified event annotations, timestamps, or age mapping exist; cough classification is unavailable.
7. **Live capture:** Generated WAV/MP3/M4A/OGG/FLAC files and invalid inputs are tested through the shared upload/live-analysis APIs, but physical microphone access, browser permission and real-world listening were not tested end-to-end.
8. **Risk:** No validated risk function exists. Risk scores are not calculated.
9. **Review UI:** Interface safeguards pass fixtures; an actual reviewer must still listen attentively. `playing` event is not proof of human listening. Patient grouping is unknown.

## Future work

- Complete manual review of cry labels without changing source tags; adjudicate ambiguous recordings.
- Obtain independently recorded and licensed recordings for minority classes, with participant IDs and label methodology.
- Create a new participant-held-out evaluation set; do not reuse the historical test split for tuning.
- Evaluate calibrated probabilities on group-held-out validation data, and define any acceptance criteria before testing.
- Obtain approval before installing an inference runtime or downloading pretrained weights; verify the model/license, supported classes, conservative rejection policy, CPU behavior, and performance on suitable validation data.
- Annotate cough/non-cough/unknown events with timestamp boundaries and reviewer provenance; keep source cohort labels separate.
- Train and compare a first-stage detector and cough classifier only after valid data exists; report uncertainty and per-class metrics.
- Perform external prospective validation and relevant governance/regulatory review before any clinical use.
