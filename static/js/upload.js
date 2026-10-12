document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const resultContainer = document.getElementById('uploadResultContainer');
    const btnSubmit = document.getElementById('btnUploadSubmit');

    const resSoundType = document.getElementById('resSoundType');
    const resConfidence = document.getElementById('resConfidence');
    const resPattern = document.getElementById('resPattern');
    const resRiskBadge = document.getElementById('resRiskBadge');
    const audioPlayer = document.getElementById('uploadAudioPlayer');
    const btnViewFullAnalysis = document.getElementById('btnViewFullAnalysis');
    const btnDownloadPDF = document.getElementById('btnDownloadPDF');

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            btnSubmit.disabled = true;
            btnSubmit.textContent = 'Analyzing...';

            const formData = new FormData(uploadForm);

            try {
                const res = await fetch('/api/audio/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) throw new Error('Failed to analyze uploaded file');
                const data = await res.json();

                // Populate Core UI Metrics
                resSoundType.textContent = data.detected_sound;
                resConfidence.textContent = `${(data.sound_confidence * 100).toFixed(1)}%`;
                resPattern.textContent = data.top_screening_pattern;

                const rLevel = data.risk_result ? data.risk_result.risk_level : 'LOW';
                const rScore = data.risk_result ? data.risk_result.risk_score : 20;
                resRiskBadge.className = `badge badge-${rLevel === 'HIGH' ? 'high' : rLevel === 'MODERATE' ? 'mod' : 'low'}`;
                resRiskBadge.textContent = `${rLevel} RISK (${rScore}/100)`;

                if (audioPlayer && data.audio_url) {
                    audioPlayer.src = data.audio_url;
                }

                if (btnViewFullAnalysis) btnViewFullAnalysis.href = `/analysis/${data.prediction_id}`;
                if (btnDownloadPDF) btnDownloadPDF.href = `/api/report/${data.prediction_id}/pdf`;

                // Populate Live Parent Caregiver Report Sections
                const aiExp = data.ai_explanation || {};
                
                const reportDetectionSummary = document.getElementById('reportDetectionSummary');
                const reportPrecisionMetric = document.getElementById('reportPrecisionMetric');
                const reportPathophysiology = document.getElementById('reportPathophysiology');
                const reportAcousticFeatures = document.getElementById('reportAcousticFeatures');
                const reportTriageGuidance = document.getElementById('reportTriageGuidance');
                const reportCaregiverChecklist = document.getElementById('reportCaregiverChecklist');
                const reportRedFlags = document.getElementById('reportRedFlags');

                if (reportDetectionSummary) {
                    reportDetectionSummary.innerHTML = aiExp.detection_summary || `Auto-detected <strong>${data.detected_sound}</strong> sound event with ${(data.sound_confidence * 100).toFixed(1)}% accuracy.`;
                }

                if (reportPrecisionMetric) {
                    reportPrecisionMetric.textContent = aiExp.accuracy_precision_text || `${(data.sound_confidence * 100).toFixed(1)}% Accuracy Score`;
                }

                if (reportPathophysiology) {
                    reportPathophysiology.textContent = aiExp.pathophysiology_explanation || 'Airway irritation causing cough or cry sounds.';
                }

                if (reportAcousticFeatures && aiExp.acoustic_feature_analysis) {
                    reportAcousticFeatures.innerHTML = aiExp.acoustic_feature_analysis.map(item => `<li>${item}</li>`).join('');
                }

                if (reportTriageGuidance) {
                    reportTriageGuidance.textContent = aiExp.triage_guidance || 'Manage at home and observe.';
                }

                if (reportCaregiverChecklist && aiExp.caregiver_checklist) {
                    reportCaregiverChecklist.innerHTML = aiExp.caregiver_checklist.map(item => `<li>${item}</li>`).join('');
                }

                const reportEmergencyAction = document.getElementById('reportEmergencyAction');
                const reportDoctorNotice = document.getElementById('reportDoctorNotice');

                if (reportRedFlags && aiExp.red_flags) {
                    reportRedFlags.innerHTML = aiExp.red_flags.map(item => `<li>${item}</li>`).join('');
                }

                if (reportEmergencyAction && aiExp.emergency_action) {
                    reportEmergencyAction.textContent = aiExp.emergency_action;
                }

                if (reportDoctorNotice && aiExp.doctor_contact_notice) {
                    reportDoctorNotice.textContent = aiExp.doctor_contact_notice;
                }

                resultContainer.style.display = 'block';
                resultContainer.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                alert('Error analyzing audio file. Please try a valid audio format.');
                console.error(err);
            } finally {
                btnSubmit.disabled = false;
                btnSubmit.textContent = '🔍 Auto-Detect & Analyze Audio File';
            }
        });
    }
});
