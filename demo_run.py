import requests
import json
import os

BASE_URL = "http://127.0.0.1:5000"

def run_demo():
    print("=" * 70)
    print(" PEDIATRIC AI HEALTH SCREENING - LIVE SYSTEM DEMONSTRATION")
    print("=" * 70)

    # 1. Fetch available research datasets
    print("\n[1] Fetching pre-packaged research dataset samples...")
    res = requests.get(f"{BASE_URL}/api/datasets")
    datasets = res.json()
    print(f"-> Retrieved {len(datasets)} dataset samples.")
    for d in datasets[:3]:
        print(f"   * Dataset: {d['dataset']} | File: {d['file']} | Known Label: {d['pattern']}")

    # 2. Analyze Cry Sample (Donate-a-Cry Pain Cry)
    pain_sample = [d for d in datasets if "pain" in d["file"]][0]
    print(f"\n[2] Executing Deep Learning Screening on Infant Cry Sample: {pain_sample['file']}")
    res_cry = requests.post(
        f"{BASE_URL}/api/datasets/analyze",
        json={"filepath": pain_sample["filepath"], "age_group": "0-12m"}
    ).json()

    print(f"   * Sound Event Detected : {res_cry['detected_sound']} (Confidence: {res_cry['sound_confidence']*100:.1f}%)")
    print(f"   * Screening Pattern    : {res_cry['top_screening_pattern']}")
    if res_cry.get('cry_analysis'):
        print(f"   * Cry Abnormality Score: {res_cry['cry_analysis']['abnormality_score']}/100")
    print(f"   * Overall Risk Level   : {res_cry['risk_result']['risk_level']} (Score: {res_cry['risk_result']['risk_score']}/100)")
    print(f"   * AI Explanation       : {res_cry['explanation']['summary']}")

    # 3. Analyze Cough Sample (Pediatric Pneumonia Cough)
    pneumonia_sample = [d for d in datasets if "pneumonia" in d["file"]][0]
    print(f"\n[3] Executing Deep Learning Screening on Pediatric Cough Sample: {pneumonia_sample['file']}")
    res_cough = requests.post(
        f"{BASE_URL}/api/audio/upload",
        files={"audio_file": (os.path.basename(pneumonia_sample["filepath"]), open(pneumonia_sample["filepath"], "rb"), "audio/wav")},
        data={"age_group": "1-3y", "audio_type_hint": "COUGH"}
    ).json()

    print(f"   * Sound Event Detected : {res_cough['detected_sound']} (Confidence: {res_cough['sound_confidence']*100:.1f}%)")
    if res_cough.get('cough_analysis'):
        print(f"   * Cough Events Count   : {res_cough['cough_analysis']['cough_events']} bursts ({res_cough['cough_analysis']['cough_frequency']} frequency)")
    print(f"   * Screening Pattern    : {res_cough['top_screening_pattern']}")
    print(f"   * Overall Risk Level   : {res_cough['risk_result']['risk_level']} (Score: {res_cough['risk_result']['risk_score']}/100)")
    print(f"   * Primary Risk Drivers : {', '.join(res_cough['risk_result']['risk_drivers'])}")

    # 4. Upload Audio File with Caregiver Symptoms
    print(f"\n[4] Simulating Audio Upload + Caregiver Symptom Checklist...")
    sample_file_path = pneumonia_sample["filepath"]
    with open(sample_file_path, "rb") as f:
        files = {"audio_file": (os.path.basename(sample_file_path), f, "audio/wav")}
        data = {
            "age_group": "0-12m",
            "audio_type_hint": "COUGH",
            "temperature": "38.6",
            "spo2": "94",
            "fever": "yes",
            "difficulty_breathing": "yes"
        }
        res_upload = requests.post(f"{BASE_URL}/api/audio/upload", files=files, data=data).json()

    print(f"   * Upload Saved As     : {os.path.basename(res_upload.get('audio_url', ''))}")
    print(f"   * Auto-Detected Sound : {res_upload['detected_sound']} ({res_upload['sound_confidence']*100:.1f}%)")
    print(f"   * Multi-Signal Risk    : {res_upload['risk_result']['risk_level']} (Score: {res_upload['risk_result']['risk_score']}/100)")
    print(f"   * Fused Risk Drivers  : {', '.join(res_upload['risk_result']['risk_drivers'])}")

    # 5. Fetch History Database Records
    print(f"\n[5] Fetching Analysis History Log from Database...")
    res_hist = requests.get(f"{BASE_URL}/api/history").json()
    print(f"-> Retrieved {len(res_hist)} analysis session records in SQLite database:")
    for h in res_hist[:4]:
        print(f"   * ID #{h['id']} | Mode: {h['source']:<7} | Sound: {h['sound_type']:<6} | Pattern: {h['screening_prediction']:<32} | Risk: {h['risk_level']} ({h['risk_score']}/100)")

    print("\n" + "=" * 70)
    print(" LIVE DEMONSTRATION COMPLETE - ALL PIPELINE STAGES OPERATIONAL")
    print("=" * 70)

if __name__ == '__main__':
    run_demo()
