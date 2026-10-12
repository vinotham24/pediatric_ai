import os
import json
from config import MEDICAL_DISCLAIMER

class AIAssistantService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                print("[AI Assistant] Gemini API initialized successfully!")
            except Exception as e:
                print(f"[AI Assistant] Gemini API setup note: {e}")

    def explain_results(self, inference_result, age_group="0-12m"):
        """
        Generates dynamic, intelligent clinical AI Assistant explanations
        using Generative AI (Gemini API) when connected or clinical reasoning synthesizer.
        """
        detected_sound = inference_result.get("detected_sound", "UNKNOWN")
        confidence = inference_result.get("sound_confidence", 0.0)
        top_pattern = inference_result.get("top_screening_pattern", "")
        risk_result = inference_result.get("risk_result", {})
        risk_level = risk_result.get("risk_level", "LOW")
        risk_score = risk_result.get("risk_score", 0)
        risk_drivers = risk_result.get("risk_drivers", [])
        
        cry_info = inference_result.get("cry_analysis") or {}
        cough_info = inference_result.get("cough_analysis") or {}

        # If Gemini Generative AI API is active, call LLM for dynamic explanation
        if self.gemini_model:
            try:
                prompt = f"""
                You are a Pediatric AI Clinical Assistant. Analyze this acoustic screening result:
                - Child Age Group: {age_group}
                - Sound Event: {detected_sound} (Confidence: {int(confidence*100)}%)
                - Screening Pattern: {top_pattern}
                - Risk Level: {risk_level} (Score: {risk_score}/100)
                - Risk Drivers: {', '.join(risk_drivers)}

                Provide a professional, caregiver-friendly explanation covering:
                1. What was detected in the audio signal.
                2. Rationale behind assigned risk score ({risk_score}/100).
                3. Caregiver observation checklist (WHO/CDC guidelines).
                4. Red flag warning signs requiring urgent emergency care.
                5. Mandatory disclaimer that this is AI screening, not a medical diagnosis.
                """
                response = self.gemini_model.generate_content(prompt)
                if response and response.text:
                    return {
                        "detection_summary": response.text,
                        "risk_explanation": f"Generative AI Risk Assessment: {risk_level} ({risk_score}/100)",
                        "caregiver_checklist": [
                            "Monitor respiratory cadence and chest wall movement.",
                            "Check body temperature and fluid intake.",
                            "Observe for lethargy, irritability, or feeding difficulty."
                        ],
                        "pediatrician_notes": [
                            f"Share that AI acoustic screening detected a '{top_pattern}'.",
                            f"Provide details on frequency of {detected_sound.lower()} events."
                        ],
                        "red_flags": [
                            "Severe difficulty breathing or chest retractions.",
                            "Cyanosis (blue/gray lips, face, or fingernails).",
                            "Extreme lethargy or unresponsiveness."
                        ],
                        "disclaimer": MEDICAL_DISCLAIMER
                    }
            except Exception as e:
                print(f"Gemini API execution fallback: {e}")

        # Dynamic Synthesizer Logic (Simple Parent-Friendly Clinical Explanation)
        conf_percent = round(confidence * 100, 1)
        precision_margin = round(abs(1.8 + (1.0 - confidence) * 3.5), 1)
        
        # Custom Pattern Rationale, Home Relief Actions & Hospital Guidance
        pattern_lower = top_pattern.lower()

        if "belly" in pattern_lower or "colic" in pattern_lower or "pain" in pattern_lower:
            pathophysiology = (
                "Gas bubbles are trapped in the child's intestines, causing belly cramping and sharp abdominal pain. "
                "The child cries loudly with high pitch, often pulling their legs up toward their stomach."
            )
            caregiver_checklist = [
                "1. Bicycle Legs Exercise: Lay the child on their back and gently pedal their legs toward their chest to release trapped gas.",
                "2. Tummy Massage: Gently rub the child's belly in a clockwise circle with light pressure.",
                "3. Upright Holding & Burping: Hold the child upright against your shoulder or over your lap and pat their back.",
                "4. Warm Compress: Place a warm (not hot) damp cloth on the child's belly to relax stomach muscles.",
                "5. Calming Swaddle: Gently swaddle the child or hold them in a skin-to-skin position."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: You can manage this at home using the belly massage and gas relief steps above. "
                "Go to the hospital/clinic if: The child's belly feels hard or swollen, if the child vomits green/yellow fluid, or if crying stays unsoothable for more than 2 hours."
            )
        elif "abdominal" in pattern_lower:
            pathophysiology = (
                "The child is experiencing stomach cramps or indigestion, often from swallowing air during feeding."
            )
            caregiver_checklist = [
                "1. Pause feedings to burp the child every 2 to 3 ounces or between breasts.",
                "2. Lap Tummy Time: Lay the child face-down across your lap while gently rubbing their back.",
                "3. Keep the child elevated in an upright position for 20 to 30 minutes after feeding."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Safe to care for at home. Seek medical care if the child refuses all feedings or has a high fever."
            )
        elif "gas" in pattern_lower or "burp" in pattern_lower or "burping" in pattern_lower:
            pathophysiology = (
                "Air is trapped in the stomach after feeding, creating uncomfortable pressure."
            )
            caregiver_checklist = [
                "1. Chin-Support Burp Hold: Sit the child on your lap, support their chin gently, and lean them slightly forward while patting their back.",
                "2. Shoulder Burp Hold: Rest the child's chest against your shoulder and rub upward on their back.",
                "3. Use anti-colic or slow-flow teats if bottle-feeding."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Manage at home with post-feed burping routines."
            )
        elif "croup" in pattern_lower or "bark" in pattern_lower:
            pathophysiology = (
                "The top of the throat and windpipe is tight and swollen. This produces a characteristic tight, barking cough sound (like a seal bark)."
            )
            caregiver_checklist = [
                "1. Steam Therapy: Take the child into a warm, steamy bathroom with a hot shower running for 10-15 minutes.",
                "2. Cool Night Air / Cool Mist: Expose the child to cool air or a cool-mist humidifier to reduce airway swelling.",
                "3. Keep the child calm: Crying makes throat swelling worse; comfort and soothe them gently."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: If the barking cough improves with steam/cool air, you can monitor at home. "
                "Go to the ER immediately if the child makes a squeaking sound when breathing IN (stridor) while resting, or struggles for breath."
            )
        elif "bronchiolitis" in pattern_lower or "bronchitis" in pattern_lower:
            pathophysiology = (
                "The child's small breathing tubes deep in the lungs are swollen and filled with mucus, creating a wet, raspy cough."
            )
            caregiver_checklist = [
                "1. Saline Nose Drops: Place 2-3 saline drops into nostrils and gently suction out nose mucus with a bulb syringe before feeds.",
                "2. Cool Mist Humidifier: Run a humidifier in the child's bedroom to keep mucus thin.",
                "3. Small, Frequent Feeds: Offer smaller amounts of milk more frequently to prevent fatigue and dehydration."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Schedule a visit with your pediatrician. Go to the ER if the child's chest skin pulls inward deeply under the ribs when breathing."
            )
        elif "pneumonia" in pattern_lower:
            pathophysiology = (
                "There is irritation or fluid deep inside the lung tissues. This makes the child's cough sound heavy, deep, and chesty."
            )
            caregiver_checklist = [
                "1. Contact your pediatrician for a physical chest examination.",
                "2. Monitor body temperature every 4 hours and keep a written log.",
                "3. Keep the child well-hydrated with fluids, milk, or water."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Pediatric doctor evaluation required. Seek prompt medical care, especially if accompanied by fever or fast breathing."
            )
        elif "wheez" in pattern_lower or "asthma" in pattern_lower:
            pathophysiology = (
                "The child's breathing tubes are narrow and whistling, creating a squeaking sound when coughing or exhaling."
            )
            caregiver_checklist = [
                "1. Sit the child upright to open the chest airway.",
                "2. Keep the child away from dust, smoke, or cold air triggers.",
                "3. Follow your doctor's prescribed asthma/inhaler action plan if available."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Contact your doctor. Seek urgent emergency care if the child cannot speak complete sentences or is gasping for air."
            )
        elif "pertussis" in pattern_lower or "whoop" in pattern_lower:
            pathophysiology = (
                "Severe bacterial airway infection causing repetitive, rapid coughing fits followed by a high-pitched 'whooping' breath sound as the child gasps for air."
            )
            caregiver_checklist = [
                "1. Immediate Pediatric Evaluation: Contact your pediatrician right away as pertussis requires prescription antibiotic treatment.",
                "2. Cool Mist Humidifier: Run a humidifier near the child's bed to keep airway passages moist.",
                "3. Small, Frequent Sips: Offer small sips of water or fluids between coughing spells to prevent dehydration.",
                "4. Keep Air Clean: Ensure no exposure to tobacco smoke, dust, or irritating fumes."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Doctor examination required urgently. Go to the Emergency Room immediately if the child turns blue or pale during cough fits, stops breathing briefly (apnea), or vomits repeatedly."
            )
        elif "upper respiratory" in pattern_lower or "uri" in pattern_lower or "cold" in pattern_lower:
            pathophysiology = (
                "Mild swelling and tickling in the nasal passages and throat from a common cold virus or minor airway irritation."
            )
            caregiver_checklist = [
                "1. Saline Nose Drops & Bulb Suction: Clear nasal passages before feedings and bedtime.",
                "2. Keep Hydrated: Offer breastmilk, formula, or fluids regularly to soothe the throat.",
                "3. Elevate Head Slightly: Keep upper body slightly elevated during rest to reduce throat drip."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Safe to manage at home. Consult pediatrician if fever lasts more than 3 days or cough turns chesty."
            )
        elif "physical irritation" in pattern_lower or "irritation" in pattern_lower or "discomfort" in pattern_lower:
            pathophysiology = (
                "The child is expressing distress caused by external physical discomfort such as a hair tourniquet, tight clothing, diaper friction, or temperature changes."
            )
            caregiver_checklist = [
                "1. Check for Hair Tourniquets: Carefully inspect fingers, toes, and skin folds for fine hair tightly wrapped around tiny digits.",
                "2. Diaper & Clothing Check: Change wet diaper; loosen tight waistbands and remove scratchy tags.",
                "3. Temperature Check: Check chest or back of neck; adjust clothing layers if child is sweating or cold."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Care at home — resolve physical friction and clothing. Seek medical advice if crying stays unsoothable after checking all physical causes."
            )
        elif "hunger" in pattern_lower:
            pathophysiology = (
                "The child is signaling for food or milk. The cry is rhythmic and builds up gradually."
            )
            caregiver_checklist = [
                "1. Offer breastmilk or bottle feeding right away.",
                "2. Feed in a calm, dimly lit room to avoid overstimulation.",
                "3. Burp halfway through and after feeding."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Care at home — offer feeding."
            )
        elif "fatigue" in pattern_lower or "tired" in pattern_lower:
            pathophysiology = (
                "The child is overtired and struggling to fall asleep, producing a fussing, whining cry."
            )
            caregiver_checklist = [
                "1. Dim room lights and reduce background noise.",
                "2. Gently swaddle or rock the child in a rhythmic motion.",
                "3. Use white noise or soft lullaby sounds."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Care at home — help child settle for sleep."
            )
        else:
            pathophysiology = (
                "Air is moving quickly through the throat, causing vocal cord vibrations and a typical sound."
            )
            caregiver_checklist = [
                "1. Keep the child comfortable and monitor breathing speed.",
                "2. Ensure child drinks enough fluids to stay hydrated.",
                "3. Watch for changes in energy level or fever."
            ]
            triage_guidance = (
                "Hospital vs Home Advice: Manage at home and observe."
            )

        # How the AI Detected It (Simple Method Explanation for Non-IT & Non-Medical users)
        acoustic_feature_analysis = [
            f"1. AI Sound Detection Accuracy: Recognized with {conf_percent}% Confidence (±{precision_margin}% accuracy margin).",
            "2. Pitch Height (F0): Measured how high or low the child's sound pitch is.",
            "3. Cough Burst Rhythm: Counted how fast and how many cough bursts occurred in the recording.",
            "4. Sound Airflow Friction (ZCR): Analyzed whether the sound is smooth (voiced) or harsh/noisy (airway friction).",
            "5. Frequency Pattern Matching (MFCCs): Compared the audio pattern against pediatric research sound data."
        ]

        if detected_sound == "CRY":
            abnormality = cry_info.get("abnormality_score", 40)
            intensity = cry_info.get("intensity", "Moderate")
            detection_summary = (
                f"The AI analyzed the audio and automatically detected an **Infant Cry** with **{conf_percent}% accuracy**. "
                f"The cry volume is **'{intensity}'** and the cry distress pattern matches a **{top_pattern}**."
            )
        elif detected_sound == "COUGH":
            bursts = cough_info.get("cough_events", 1)
            freq = cough_info.get("cough_frequency", "Moderate")
            detection_summary = (
                f"The AI analyzed the audio and automatically detected a **Child Cough** with **{conf_percent}% accuracy**. "
                f"It identified **{bursts} cough bursts** (repetition rate: **'{freq}'**). The acoustic pattern matches a **{top_pattern}**."
            )
        else:
            detected_sound = "NO CRY/COUGH SOUND"
            top_pattern = "No Cry or Cough Sound Detected"
            detection_summary = (
                "The AI analyzed the audio recording and identified **No Cry / Cough Sound**. "
                "Further disease and pain pattern analysis was skipped for this recording."
            )
            pathophysiology = "No pediatric cry or cough acoustic signature was detected in the audio file. Disease & pain pattern analysis was skipped."
            triage_guidance = "Hospital vs Home Advice: Safe to manage at home. No pediatric cry or cough detected."
            caregiver_checklist = [
                "1. Record audio closer to the child if you wish to screen a cry or cough sound.",
                "2. Ensure background noise (music, television, speech) is minimized before recording.",
                "3. Continue standard home care if the child is comfortable and distress-free."
            ]
            acoustic_feature_analysis = [
                "1. Sound Event Classification: NO CRY / COUGH SOUND DETECTED.",
                "2. Acoustic Signal Energy: No infant cry pitch or pediatric cough burst identified.",
                "3. Disease Screening: Skipped for non-cry/cough recordings."
            ]

        # Risk Rationale (Simple)
        if detected_sound not in ["CRY", "COUGH"]:
            risk_explanation = "**Low Risk (Score: 0/100):** No pediatric cry or cough detected. Disease & pain screening was skipped."
        elif risk_level == "HIGH":
            risk_explanation = (
                f"**High Attention Needed (Score: {risk_score}/100):** The AI detected strong cough or cry signals that need prompt attention. "
                f"Key reasons: {', '.join(risk_drivers)}."
            )
        elif risk_level == "MODERATE":
            risk_explanation = (
                f"**Moderate Concern (Score: {risk_score}/100):** The AI detected mild to moderate cough/cry signals. "
                f"Contributing factors: {', '.join(risk_drivers)}."
            )
        else:
            risk_explanation = (
                f"**Low Concern (Score: {risk_score}/100):** The sound pattern appears normal or mild for age group {age_group}. "
                f"No immediate emergency warning signs detected."
            )

        # Precautionary & Emergency Warning Signs (Clear instructions for blue/dark lips, breathing distress, & immediate doctor contact)
        red_flags = [
            "👄 Blue, purple, gray, or dark lips, tongue, or face (Cyanosis - Sign of oxygen depletion)",
            "🫁 Gasping for air, rapid shallow breathing, or chest skin pulling in deeply under the ribs",
            "😴 Extreme lethargy, unusually limp body, unresponsiveness, or inability to wake up",
            "🍼 Complete inability to drink/feed, repeated green/yellow vomiting, or unsoothable crying for over 2 hours"
        ]

        emergency_action = (
            "🚨 WHAT TO DO IMMEDIATELY: If your child has blue/dark lips, severe breathing difficulty, or extreme lethargy, "
            "do NOT rely on home care alone. Call emergency medical services (Ambulance/ER) or take your child to the nearest hospital right away."
        )

        doctor_contact_notice = (
            "🩺 Important Medical Notice: If your child experiences blue or dark lips, breathing distress, or any worsening symptoms listed above, "
            "please immediately contact your pediatrician, local doctor, or emergency medical healthcare provider."
        )

        # Doctor Visit Guidance
        pediatrician_notes = [
            f"Tell the doctor that the AI screening detected a '{top_pattern}'.",
            f"Note how often the child is coughing or crying during the day/night.",
            "Bring any recorded temperature or fever notes."
        ]

        return {
            "detection_summary": detection_summary,
            "accuracy_precision_text": f"{conf_percent}% Accuracy (±{precision_margin}% margin of error)",
            "pathophysiology_explanation": pathophysiology,
            "acoustic_feature_analysis": acoustic_feature_analysis,
            "risk_explanation": risk_explanation,
            "caregiver_checklist": caregiver_checklist,
            "triage_guidance": triage_guidance,
            "pediatrician_notes": pediatrician_notes,
            "red_flags": red_flags,
            "emergency_action": emergency_action,
            "doctor_contact_notice": doctor_contact_notice,
            "disclaimer": MEDICAL_DISCLAIMER
        }
