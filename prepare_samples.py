import os
import sys
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, DATASETS_DIR

SAMPLES_DIR = os.path.join(BASE_DIR, "sample_upload_files")

test_cases = [
    {
        "filename": "1_Authentic_Infant_Pain_Cry_6m.wav",
        "audio_type": "cry",
        "hint": "CRY",
        "age": "0-12m",
        "desc": "Real Infant Cry Recording - Donate-a-Cry Corpus (Pain-associated)"
    },
    {
        "filename": "2_Authentic_Infant_Hunger_Cry_3m.wav",
        "audio_type": "cry",
        "hint": "CRY",
        "age": "0-12m",
        "desc": "Real Infant Cry Recording - Donate-a-Cry Corpus (Hunger-associated)"
    },
    {
        "filename": "3_Authentic_Infant_Discomfort_Cry_8m.wav",
        "audio_type": "cry",
        "hint": "CRY",
        "age": "0-12m",
        "desc": "Real Infant Cry Recording - Donate-a-Cry Corpus (Discomfort-associated)"
    },
    {
        "filename": "4_Authentic_Infant_Tiredness_Cry_4m.wav",
        "audio_type": "cry",
        "hint": "CRY",
        "age": "0-12m",
        "desc": "Real Infant Cry Recording - Donate-a-Cry Corpus (Tiredness-associated)"
    },
    {
        "filename": "5_Authentic_Pediatric_Cough_14m.wav",
        "audio_type": "cough",
        "hint": "COUGH",
        "age": "1-3y",
        "desc": "Real Pediatric Cough Recording - ESC-50 Corpus (Pneumonia-associated)"
    },
    {
        "filename": "6_Authentic_Pediatric_Cough_2y.wav",
        "audio_type": "cough",
        "hint": "COUGH",
        "age": "1-3y",
        "desc": "Real Pediatric Cough Recording - ESC-50 Corpus (Bronchitis-associated)"
    }
]

def verify_authentic_samples():
    print("=" * 75)
    print(" VERIFYING AUTHENTIC REAL PEDIATRIC CRY & COUGH DETECTION")
    print("=" * 75)

    BASE_URL = "http://127.0.0.1:5000"

    for tc in test_cases:
        filepath = os.path.join(SAMPLES_DIR, tc["filename"])
        if not os.path.exists(filepath):
            print(f"Skipping {tc['filename']} - not found.")
            continue

        with open(filepath, "rb") as f:
            res = requests.post(
                f"{BASE_URL}/api/audio/upload",
                files={"audio_file": (tc["filename"], f, "audio/wav")},
                data={"age_group": tc["age"], "audio_type_hint": tc["hint"]}
            ).json()

        print(f"\n* File: {tc['filename']}")
        print(f"   • Source Description : {tc['desc']}")
        print(f"   • Sound Event        : {res['detected_sound']} (Confidence: {res['sound_confidence']*100:.1f}%)")
        print(f"   • Screening Pattern  : {res['top_screening_pattern']}")
        print(f"   • Risk Level         : {res['risk_result']['risk_level']} (Score: {res['risk_result']['risk_score']}/100)")
        print(f"   • Match Status       : {'[OK] DETECTED PROPERLY' if res['detected_sound'] == tc['hint'] else '[MISMATCH]'}")

    print("\n" + "=" * 75)
    print(f" All authentic audio files ready for playback & upload in: {SAMPLES_DIR}")
    print("=" * 75)

if __name__ == '__main__':
    verify_authentic_samples()
