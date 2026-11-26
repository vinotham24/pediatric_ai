window.renderPredictionResult = function (data, errorMessage) {
    const status = errorMessage ? "ERROR" : String(data.prediction_status || "UNKNOWN").toUpperCase();
    const set = (id, value) => {
        const node = document.getElementById(id);
        if (node) node.textContent = value;
    };
    set("resultAnalysisStatus", data.audio_analysis_status || status);
    set("resultSound", data.sound_classification || "Unavailable: sound not reliably determined.");
    set("resultDetectionNote", `Detection method: ${data.sound_detection_method || "Not reported"}; status: ${data.sound_detection_status || "UNKNOWN"}. Any detector score is uncalibrated.`);
    set("resultStatus", status);
    const cryWasReliablyEstablished = String(data.detected_sound || "UNKNOWN").toUpperCase() === "CRY";
    set("resultCategory", cryWasReliablyEstablished && (status === "CONFIDENT" || status === "UNCERTAIN")
        ? (data.possible_category || "Not available") : "Not available");
    set("resultCryClassification", data.cry_classification_status ||
        "Not performed because CRY was not reliably established.");
    set("resultCoughClassification", data.cough_classification_status ||
        "Unavailable because a verified cough-event classifier has not been trained.");
    set("resultMedicalDiagnosis", data.medical_diagnosis || "Not provided.");

    const score = data.model_score;
    const hasScore = !errorMessage && cryWasReliablyEstablished && score !== null && score !== undefined && Number.isFinite(Number(score));
    set("resultModelScore", hasScore ? `${(Number(score) * 100).toFixed(2)}% raw model score (uncalibrated)` : "Unavailable");
    set("resultScoreNote", errorMessage || data.model_score_interpretation ||
        "This score is not a probability that the prediction is correct.");

    const pattern = errorMessage
        ? "Not available - analysis could not be completed."
        : (data.top_screening_pattern || "Not available - sound could not be classified reliably.");
    set("resultPattern", pattern);
    set("resultRisk", "Not calculated; no validated risk calculation is available.");
    set("resultNotice", data.ai_explanation?.disclaimer ||
        "Experimental research output only. This model is not clinically validated and does not confirm a diagnosis.");
};
