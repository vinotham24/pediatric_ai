from config import MEDICAL_DISCLAIMER


class AIAssistantService:
    """Provide non-diagnostic wording for the experimental model response."""

    def __init__(self):
        pass

    def explain_results(self, inference_result, age_group="0-12m"):
        status = str(inference_result.get("prediction_status", "UNKNOWN")).upper()
        sound_classification = inference_result.get(
            "sound_classification", "Unavailable: sound not reliably determined."
        )
        category = inference_result.get("possible_category")
        pattern = inference_result.get("top_screening_pattern", "Not available.")
        if status not in {"CONFIDENT", "UNCERTAIN", "UNKNOWN", "ERROR"}:
            status = "UNKNOWN"

        category_text = f" Possible category: {category}." if category else ""
        summary = (
            f"Experimental audio output: {sound_classification}; prediction status {status}."
            f"{category_text} "
            "Cry-reason classification is not performed unless CRY is reliably established. "
            "Cough classification is unavailable because no verified cough-event classifier has been trained. "
            "Medical diagnosis: not provided."
        )
        return {
            "prediction_status": status,
            "sound_classification": sound_classification,
            "experimental_sound_candidate": None,
            "cry_classification_status": inference_result.get(
                "cry_classification_status",
                "Not performed because CRY was not reliably established.",
            ),
            "cough_classification_status": inference_result.get(
                "cough_classification_status",
                "Unavailable because a verified cough-event classifier has not been trained.",
            ),
            "medical_diagnosis": "Not provided.",
            "detection_summary": summary,
            "screening_pattern": pattern,
            "model_score": inference_result.get("model_score"),
            "model_score_interpretation": inference_result.get(
                "model_score_interpretation", "No validated model score is available."
            ),
            "risk_status": "Not calculated; no validated risk calculation is available.",
            "recommendations": [],
            "disclaimer": MEDICAL_DISCLAIMER,
        }
