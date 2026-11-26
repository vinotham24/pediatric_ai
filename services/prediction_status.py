"""Shared, evidence-gated prediction status and legacy-record sanitization."""

import json
import math


STATUSES = {"CONFIDENT", "UNCERTAIN", "UNKNOWN", "ERROR"}
UNCERTAIN_PATTERN = "Not available - prediction is uncertain."
UNKNOWN_PATTERN = "Not available - sound could not be classified reliably."
ERROR_PATTERN = "Not available - analysis could not be completed."
EXPERIMENTAL_NOTICE = (
    "Experimental research output only. This model is not clinically validated "
    "and does not confirm a medical finding or diagnosis."
)


def determine_prediction_status(candidate, score, validation=None, error=None):
    """Only validation metadata may authorize CONFIDENT; no default cutoff exists."""
    if error:
        return "ERROR"
    if not candidate:
        return "UNKNOWN"
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return "UNCERTAIN"
    if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
        return "UNCERTAIN"

    validation = validation or {}
    if not (
        validation.get("validated_for_screening") is True
        and validation.get("scores_calibrated") is True
        and validation.get("validation_reference")
    ):
        return "UNCERTAIN"
    try:
        threshold = float(validation["acceptance_threshold"])
    except (KeyError, TypeError, ValueError):
        return "UNCERTAIN"
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        return "UNCERTAIN"
    return "CONFIDENT" if numeric_score >= threshold else "UNCERTAIN"


def screening_pattern_for(status, candidate=None):
    status = str(status or "UNKNOWN").upper()
    if status == "CONFIDENT" and candidate:
        return f"Experimental category prediction: {candidate} (not a confirmed finding)."
    if status == "ERROR":
        return ERROR_PATTERN
    if status == "UNKNOWN":
        return UNKNOWN_PATTERN
    return UNCERTAIN_PATTERN


def prediction_metadata(result):
    """JSON-safe data persisted in the legacy probabilities_json column."""
    return {
        "prediction_status": result.get("prediction_status", "UNKNOWN"),
        "possible_category": result.get("possible_category"),
        "model_score": result.get("model_score"),
        "model_score_interpretation": result.get("model_score_interpretation", ""),
        "screening_pattern": result.get("top_screening_pattern", UNKNOWN_PATTERN),
        "sound_detection_status": result.get("sound_detection_status", "UNKNOWN"),
        "sound_detection_method": result.get("sound_detection_method", "UNKNOWN"),
        "experimental_sound_candidate": None,
        "audio_analysis_status": result.get("audio_analysis_status", result.get("prediction_status", "UNKNOWN")),
        "sound_classification": result.get(
            "sound_classification", "Unavailable: sound not reliably determined."
        ),
        "cry_classification_status": result.get("cry_classification_status", "Not performed because CRY was not reliably established."),
        "cough_classification_status": result.get("cough_classification_status", "Unavailable because a verified cough-event classifier has not been trained."),
        "medical_diagnosis": "Not provided.",
        "sound_confidence_interpretation": result.get("sound_confidence_interpretation", "No validated detector score is available."),
        "experimental_notice": EXPERIMENTAL_NOTICE,
    }


def normalize_prediction_record(record):
    """Sanitize current and pre-Phase-5 rows for website and export surfaces."""
    row = dict(record)
    metadata = {}
    raw = row.get("probabilities_json")
    if raw:
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(value, dict) and "prediction_status" in value:
                metadata = value
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    sound_type = str(row.get("sound_type") or row.get("detected_sound") or "UNKNOWN").upper()
    status = str(row.get("prediction_status") or metadata.get("prediction_status") or "").upper()
    if status not in STATUSES:
        status = "UNCERTAIN" if sound_type in {"CRY", "COUGH"} else "UNKNOWN"

    candidate = row.get("possible_category", metadata.get("possible_category"))
    detection_status = str(
        row.get("sound_detection_status") or metadata.get("sound_detection_status") or "UNKNOWN"
    ).upper()
    # A historical cry category cannot be shown as a result unless the
    # first-stage sound type was explicitly accepted. Stored source rows are
    # left untouched; only the presentation is marked as legacy/unverified.
    if status in {"UNKNOWN", "ERROR"} or detection_status != "CONFIDENT" or sound_type != "CRY":
        candidate = None
    pattern = screening_pattern_for(status, candidate)
    if status == "UNCERTAIN":
        pattern = UNCERTAIN_PATTERN
    if status == "ERROR":
        pattern = ERROR_PATTERN

    sound_classification = str(
        row.get("sound_classification")
        or metadata.get("sound_classification")
        or "Unavailable: sound not reliably determined."
    )
    if detection_status != "CONFIDENT" or status in {"UNKNOWN", "ERROR"}:
        sound_classification = "Unavailable: sound not reliably determined."

    row.update({
        "prediction_status": status,
        "possible_category": candidate,
        "model_score": row.get("model_score", metadata.get("model_score")),
        "model_score_interpretation": row.get(
            "model_score_interpretation",
            metadata.get("model_score_interpretation", "Uncalibrated model score; not a probability of correctness."),
        ),
        "sound_detection_status": str(row.get("sound_detection_status") or metadata.get("sound_detection_status") or "UNKNOWN"),
        "sound_detection_method": str(row.get("sound_detection_method") or metadata.get("sound_detection_method") or "Not recorded"),
        "sound_confidence_interpretation": str(row.get("sound_confidence_interpretation") or metadata.get("sound_confidence_interpretation") or "Detector score unavailable or unvalidated."),
        "experimental_sound_candidate": None,
        "audio_analysis_status": str(row.get("audio_analysis_status") or metadata.get("audio_analysis_status") or "HISTORICAL RESULT - ROUTING NOT REVALIDATED"),
        "sound_classification": sound_classification,
        "legacy_result_note": row.get("legacy_result_note") or (
            "Historical stored result; database values were not rewritten. Sound routing was not revalidated under the current policy."
            if not row.get("audio_analysis_status") and not metadata.get("audio_analysis_status") else ""
        ),
        "cry_classification_status": str(row.get("cry_classification_status") or metadata.get("cry_classification_status") or "Not recorded for this earlier result."),
        "cough_classification_status": str(row.get("cough_classification_status") or metadata.get("cough_classification_status") or "Unavailable because a verified cough-event classifier has not been trained."),
        "medical_diagnosis": "Not provided.",
        "screening_prediction": pattern,
        "top_pattern": pattern,
        "risk_available": False,
        "risk_level": "NOT CALCULATED",
        "risk_score": None,
        "risk_status": "No validated risk calculation is available.",
        "experimental_notice": EXPERIMENTAL_NOTICE,
        "probabilities": {},
    })
    # Legacy rows store detector confidence and heuristic probabilities here.
    # Neither is cry-model confidence, so keep them out of API and export data.
    row.pop("confidence", None)
    row.pop("probabilities_json", None)
    row.pop("class_scores", None)
    return row
