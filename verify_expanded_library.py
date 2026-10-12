import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR
from ml.inference import pipeline_instance

SAMPLES_DIR = os.path.join(BASE_DIR, "sample_upload_files")

def verify_all_files():
    print("=" * 80)
    print(" EXPANDED AUTHENTIC PEDIATRIC AUDIO LIBRARY SCREENING VERIFICATION")
    print("=" * 80)

    files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")])
    print(f"Found {len(files)} authentic child cry & cough audio files in sample_upload_files/\n")

    for f_name in files:
        filepath = os.path.join(SAMPLES_DIR, f_name)
        audio_hint = "CRY" if ("Cry" in f_name or "cry" in f_name or "Pain" in f_name or "Hunger" in f_name or "Discomfort" in f_name or "Tired" in f_name or "Burp" in f_name) else "COUGH"
        
        res = pipeline_instance.process_audio(
            filepath,
            age_group="0-12m",
            audio_type_hint=audio_hint
        )

        print(f"* File: {f_name}")
        print(f"   • Sound Event       : {res['detected_sound']} ({res['sound_confidence']*100:.1f}%)")
        print(f"   • Screening Pattern : {res['top_screening_pattern']}")
        print(f"   • Risk Level        : {res['risk_result']['risk_level']} (Score: {res['risk_result']['risk_score']}/100)")
        print("-" * 80)

if __name__ == '__main__':
    verify_all_files()
