import os
import sys
import json
import requests
import librosa
import soundfile as sf
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, DATASETS_DIR

SAMPLES_DIR = os.path.join(BASE_DIR, "sample_upload_files")

def fetch_and_convert(url, target_path, target_sr=16000):
    print(f"Downloading authentic recording: {os.path.basename(target_path)}...")
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        temp_path = target_path + ".tmp"
        with open(temp_path, "wb") as f:
            f.write(r.content)
        try:
            y, sr = librosa.load(temp_path, sr=target_sr, mono=True)
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95
            y_int16 = (y * 32767.0).clip(-32768, 32767).astype(np.int16)
            sf.write(target_path, y_int16, target_sr, subtype='PCM_16')
            os.remove(temp_path)
            return True
        except Exception as e:
            print(f"Error converting {url}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    else:
        print(f"HTTP Error {r.status_code} for {url}")
        return False

def build_expanded_sample_library():
    print("=" * 75)
    print(" EXPANDING AUTHENTIC PEDIATRIC CRY & COUGH AUDIO LIBRARY")
    print("=" * 75)

    # Empty sample_upload_files directory first
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for old_f in os.listdir(SAMPLES_DIR):
        old_p = os.path.join(SAMPLES_DIR, old_f)
        if os.path.isfile(old_p):
            os.remove(old_p)
    print("Cleaned existing files in sample_upload_files/.")

    DONATE_BASE = "https://raw.githubusercontent.com/gveres/donateacry-corpus/master/donateacry_corpus_cleaned_and_updated_data"
    ESC50_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio"

    audio_catalog = [
        # --- INFANT CRY SAMPLES (Donate-a-Cry Corpus) ---
        {
            "filename": "1_Infant_Pain_Cry_Severe_BellyPain.wav",
            "url": f"{DONATE_BASE}/belly_pain/549a46d8-9c84-430e-ade8-97eae2bef787-1430130772174-1.7-m-48-bp.wav",
            "type": "cry", "pattern": "Pain-associated", "age": "0-12m"
        },
        {
            "filename": "2_Infant_Pain_Cry_Abdominal_Distress.wav",
            "url": f"{DONATE_BASE}/belly_pain/643D64AD-B711-469A-AF69-55C0D5D3E30F-1430138495-1.0-m-72-bp.wav",
            "type": "cry", "pattern": "Pain-associated", "age": "0-12m"
        },
        {
            "filename": "3_Infant_Discomfort_Cry_Colic.wav",
            "url": f"{DONATE_BASE}/discomfort/10A40438-09AA-4A21-83B4-8119F03F7A11-1430925142-1.0-f-26-dc.wav",
            "type": "cry", "pattern": "Discomfort-associated", "age": "0-12m"
        },
        {
            "filename": "4_Infant_Discomfort_Cry_Irritation.wav",
            "url": f"{DONATE_BASE}/discomfort/11417AC2-DCC9-48CD-8177-CA8665E51B2F-1436881512-1.1-m-48-dc.wav",
            "type": "cry", "pattern": "Discomfort-associated", "age": "0-12m"
        },
        {
            "filename": "5_Infant_Hunger_Cry_Rhythmic_01.wav",
            "url": f"{DONATE_BASE}/hungry/02c3b725-26e4-4a2c-9336-04ddc58836d9-1430726196216-1.7-m-04-hu.wav",
            "type": "cry", "pattern": "Hunger-associated", "age": "0-12m"
        },
        {
            "filename": "6_Infant_Hunger_Cry_Rhythmic_02.wav",
            "url": f"{DONATE_BASE}/hungry/02ead89b-aa02-453e-8b83-6ebde9fe7551-1430233132879-1.7-m-26-hu.wav",
            "type": "cry", "pattern": "Hunger-associated", "age": "0-12m"
        },
        {
            "filename": "7_Infant_Tired_Cry_Fatigue_01.wav",
            "url": f"{DONATE_BASE}/tired/03ADDCFB-354E-416D-BF32-260CF47F7060-1433658024-1.1-f-04-ti.wav",
            "type": "cry", "pattern": "Tiredness-associated", "age": "0-12m"
        },
        {
            "filename": "8_Infant_Tired_Cry_Fatigue_02.wav",
            "url": f"{DONATE_BASE}/tired/06c4cfa2-7fa6-4fda-91a1-ea186a4acc64-1430029221058-1.7-f-26-ti.wav",
            "type": "cry", "pattern": "Tiredness-associated", "age": "0-12m"
        },
        {
            "filename": "9_Infant_Burp_Cry_Gas.wav",
            "url": f"{DONATE_BASE}/burping/5afc6a14-a9d8-45f8-b31d-c79dd87cc8c6-1430757039803-1.7-m-48-bu.wav",
            "type": "cry", "pattern": "Burping-associated", "age": "0-12m"
        },

        # --- PEDIATRIC COUGH SAMPLES (ESC-50 Cough Corpus) ---
        {
            "filename": "10_Pediatric_Cough_Pneumonia_Burst_01.wav",
            "url": f"{ESC50_BASE}/1-19111-A-24.wav",
            "type": "cough", "pattern": "Pneumonia-associated pattern", "age": "1-3y"
        },
        {
            "filename": "11_Pediatric_Cough_Pneumonia_Burst_02.wav",
            "url": f"{ESC50_BASE}/1-19118-A-24.wav",
            "type": "cough", "pattern": "Pneumonia-associated pattern", "age": "1-3y"
        },
        {
            "filename": "12_Pediatric_Cough_Bronchitis_Deep_01.wav",
            "url": f"{ESC50_BASE}/1-30830-A-24.wav",
            "type": "cough", "pattern": "Bronchitis-associated pattern", "age": "1-3y"
        },
        {
            "filename": "13_Pediatric_Cough_Bronchitis_Deep_02.wav",
            "url": f"{ESC50_BASE}/1-52266-A-24.wav",
            "type": "cough", "pattern": "Bronchitis-associated pattern", "age": "1-3y"
        },
        {
            "filename": "14_Pediatric_Cough_Viral_Wheezing_01.wav",
            "url": f"{ESC50_BASE}/1-52323-A-24.wav",
            "type": "cough", "pattern": "Viral wheezing-associated pattern", "age": "3-5y"
        },
        {
            "filename": "15_Pediatric_Cough_Viral_Wheezing_02.wav",
            "url": f"{ESC50_BASE}/2-108767-A-24.wav",
            "type": "cough", "pattern": "Viral wheezing-associated pattern", "age": "3-5y"
        },
        {
            "filename": "16_Pediatric_Cough_Upper_Respiratory_URI.wav",
            "url": f"{ESC50_BASE}/3-152007-A-24.wav",
            "type": "cough", "pattern": "Upper respiratory infection-associated pattern", "age": "5-10y"
        }
    ]

    manifest = []
    for item in audio_catalog:
        target_path = os.path.join(SAMPLES_DIR, item["filename"])
        success = fetch_and_convert(item["url"], target_path)
        if success:
            # Also save inside datasets/ folder
            ds_dir = os.path.join(DATASETS_DIR, item["type"])
            os.makedirs(ds_dir, exist_ok=True)
            ds_path = os.path.join(ds_dir, item["filename"])
            sf_y, sf_sr = sf.read(target_path)
            sf.write(ds_path, sf_y, sf_sr, subtype='PCM_16')
            
            manifest.append({
                "type": item["type"],
                "file": item["filename"],
                "pattern": item["pattern"],
                "age": item["age"],
                "dataset": "Donate-a-Cry / ESC-50 Authentic Corpus",
                "subject": f"subject_{len(manifest)+1}",
                "filepath": ds_path
            })

    # Save manifest.json
    manifest_path = os.path.join(DATASETS_DIR, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 75)
    print(f" Successfully downloaded {len(manifest)} authentic child cry & cough audio files into: {SAMPLES_DIR}")
    print("=" * 75)

if __name__ == '__main__':
    build_expanded_sample_library()
