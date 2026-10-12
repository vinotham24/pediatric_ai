from config import AGE_GROUPS

class RiskEngine:
    def __init__(self):
        pass

    def calculate_risk(self, age_group="0-12m", sed_type="UNKNOWN", sed_confidence=0.0,
                       cry_data=None, cough_data=None, resp_data=None,
                       caregiver_symptoms=None, temporal_history=None):
        """
        Fuses acoustic predictions, age factors, caregiver inputs, and temporal history.
        """
        if age_group not in AGE_GROUPS:
            age_group = "0-12m"
            
        weights = AGE_GROUPS[age_group]
        risk_score = 0.0
        risk_drivers = []
        
        # 1. Acoustic Model Risk Contribution
        if sed_type == "CRY" and cry_data:
            abnormality = cry_data.get("abnormality_score", 0)
            pattern = cry_data.get("top_pattern", "")
            
            score_contrib = (abnormality / 100.0) * 40.0 * weights["cry_weight"]
            risk_score += score_contrib
            
            if abnormality >= 50:
                risk_drivers.append(f"Elevated cry abnormality score ({abnormality}/100)")
            if pattern == "Pain-associated":
                risk_score += 25.0 * weights["cry_weight"]
                risk_drivers.append("Acoustic pattern consistent with Pain-associated Cry")
            elif pattern == "Discomfort-associated":
                risk_score += 15.0 * weights["cry_weight"]
                risk_drivers.append("Acoustic pattern consistent with Discomfort-associated Cry")
                
        elif sed_type == "COUGH" and cough_data:
            abnormality = cough_data.get("abnormality_score", 0)
            events = cough_data.get("cough_events", 0)
            top_pattern = cough_data.get("top_pattern", "")
            p_pneumonia = cough_data.get("probabilities", {}).get("Pneumonia-associated pattern", 0.0)
            
            score_contrib = (abnormality / 100.0) * 45.0 * weights["cough_weight"]
            risk_score += score_contrib
            
            if events >= 3:
                risk_score += 15.0 * weights["cough_weight"]
                risk_drivers.append(f"Repeated cough burst pattern ({events} events)")

            if "pneumonia" in top_pattern.lower() or p_pneumonia > 0.35:
                risk_score += 35.0 * weights["cough_weight"]
                risk_drivers.append(f"Pneumonia-associated acoustic pattern detected ({int(max(p_pneumonia, 0.45)*100)}%)")
            elif "bronchitis" in top_pattern.lower():
                risk_score += 28.0 * weights["cough_weight"]
                risk_drivers.append("Bronchitis-associated acoustic pattern detected")
            elif "wheezing" in top_pattern.lower():
                risk_score += 25.0 * weights["cough_weight"]
                risk_drivers.append("Viral wheezing-associated acoustic pattern detected")
            elif top_pattern != "Normal/Other":
                risk_score += 18.0 * weights["cough_weight"]
                risk_drivers.append(f"Detected {top_pattern}")
                
        elif sed_type == "BREATHING" and resp_data:
            top_pattern = resp_data.get("top_pattern", "")
            if top_pattern in ["Wheezing-like sound", "Stridor-like sound"]:
                risk_score += 35.0 * weights["resp_weight"]
                risk_drivers.append(f"Detected {top_pattern}")
            elif top_pattern == "Abnormal breathing":
                risk_score += 20.0 * weights["resp_weight"]
                risk_drivers.append("Abnormal respiratory cadence detected")

        # 2. Caregiver Symptoms Contribution (Optional Inputs)
        if caregiver_symptoms:
            # Temperature
            temp = caregiver_symptoms.get("temperature")
            if temp and temp >= 38.5:
                risk_score += 25.0
                risk_drivers.append(f"High fever reported ({temp}°C)")
            elif temp and temp >= 37.5:
                risk_score += 12.0
                risk_drivers.append(f"Low-grade fever reported ({temp}°C)")
                
            # SpO2
            spo2 = caregiver_symptoms.get("spo2")
            if spo2 and spo2 < 92:
                risk_score += 35.0
                risk_drivers.append(f"Critical low oxygen saturation ({spo2}%)")
            elif spo2 and spo2 < 95:
                risk_score += 18.0
                risk_drivers.append(f"Reduced oxygen saturation ({spo2}%)")
                
            # Emergency Symptoms
            if caregiver_symptoms.get("blue_lips"):
                risk_score += 40.0
                risk_drivers.append("EMERGENCY RED FLAG: Cyanosis (blue/gray lips or face)")
            if caregiver_symptoms.get("difficulty_breathing"):
                risk_score += 30.0
                risk_drivers.append("Reported severe breathing difficulty / chest retractions")
            if caregiver_symptoms.get("lethargy"):
                risk_score += 20.0
                risk_drivers.append("Reported unusual lethargy / unresponsiveness")
            if caregiver_symptoms.get("reduced_feeding"):
                risk_score += 15.0
                risk_drivers.append("Reported inability to feed or drink")

        # 3. Temporal Persistence Filter
        if temporal_history and len(temporal_history) >= 3:
            recent_highs = sum(1 for item in temporal_history[-3:] if item.get("risk_score", 0) >= 50)
            if recent_highs >= 2:
                risk_score += 10.0
                risk_drivers.append("Persistent acoustic risk detected across continuous time windows")

        # Final Risk Level Thresholding
        risk_score_int = min(100, max(0, int(risk_score)))
        
        if risk_score_int >= 65 or caregiver_symptoms.get("blue_lips") if caregiver_symptoms else False:
            risk_level = "HIGH"
        elif risk_score_int >= 35:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        if not risk_drivers:
            risk_drivers.append("No significant acoustic abnormalities detected")

        return {
            "risk_level": risk_level,
            "risk_score": risk_score_int,
            "risk_drivers": risk_drivers,
            "age_group": age_group
        }
