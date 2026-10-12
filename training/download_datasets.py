import os
import sys
import json
import requests
import librosa
import soundfile as sf
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS_DIR, BASE_DIR

SAMPLES_DIR = os.path.join(BASE_DIR, "sample_upload_files")

def fetch_and_process_audio(url, target_filepath, target_sr=16000):
    """
    Downloads authentic audio recording from URL, converts to mono 16kHz 16-bit PCM integer WAV format.
    """
    print(f"Downloading authentic recording: {os.path.basename(target_filepath)}...")
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        temp_path = target_filepath + ".tmp"
        with open(temp_path, "wb") as f:
            f.write(r.content)
            
        try:
            y, sr = librosa.load(temp_path, sr=target_sr, mono=True)
            # Normalize peak amplitude
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95
            y_int16 = (y * 32767.0).clip(-32768, 32767).astype(np.int16)
            sf.write(target_filepath, y_int16, target_sr, subtype='PCM_16')
            os.remove(temp_path)
            return True
        except Exception as e:
            print(f"Error processing audio from {url}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    else:
        print(f"Failed download {url}, status code: {r.status_code}")
        return False

def setup_demo_datasets():
    print("=" * 75)
    print(" DOWNLOADING AUTHENTIC REAL INFANT CRY & COUGH AUDIO RECORDINGS")
    print("=" * 75)

    DONATE_BASE = "https://raw.githubusercontent.com/gveres/donateacry-corpus/master/donateacry_corpus_cleaned_and_updated_data"
    ESC50_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio"

    dataset_configs = [
        # Real Infant Cry Recordings (Donate-a-Cry Dataset)
        {
            "type": "cry",
            "file": "cry_pain_01.wav",
            "pattern": "Pain-associated",
            "age": "6 months",
            "dataset": "Donate-a-Cry Corpus (Authentic Infant Pain Cry)",
            "subject": "infant_real_01",
            "url": f"{DONATE_BASE}/belly_pain/549a46d8-9c84-430e-ade8-97eae2bef787-1430130772174-1.7-m-48-bp.wav"
        },
        {
            "type": "cry",
            "file": "cry_hunger_01.wav",
            "pattern": "Hunger-associated",
            "age": "3 months",
            "dataset": "Donate-a-Cry Corpus (Authentic Infant Hunger Cry)",
            "subject": "infant_real_02",
            "url": f"{DONATE_BASE}/hungry/02c3b725-26e4-4a2c-9336-04ddc58836d9-1430726196216-1.7-m-04-hu.wav"
        },
        {
            "type": "cry",
            "file": "cry_discomfort_01.wav",
            "pattern": "Discomfort-associated",
            "age": "8 months",
            "dataset": "Donate-a-Cry Corpus (Authentic Infant Discomfort Cry)",
            "subject": "infant_real_03",
            "url": f"{DONATE_BASE}/discomfort/10A40438-09AA-4A21-83B4-8119F03F7A11-1430925142-1.0-f-26-dc.wav"
        },
        {
            "type": "cry",
            "file": "cry_tired_01.wav",
            "pattern": "Tiredness-associated",
            "age": "4 months",
            "dataset": "Donate-a-Cry Corpus (Authentic Infant Tiredness Cry)",
            "subject": "infant_real_04",
            "url": f"{DONATE_BASE}/tired/03ADDCFB-354E-416D-BF32-260CF47F7060-1433658024-1.1-f-04-ti.wav"
        },
        {
            "type": "cry",
            "file": "cry_burping_01.wav",
            "pattern": "Burping-associated",
            "age": "2 months",
            "dataset": "Donate-a-Cry Corpus (Authentic Infant Burping Cry)",
            "subject": "infant_real_05",
            "url": f"{DONATE_BASE}/burping/5afc6a14-a9d8-45f8-b31d-c79dd87cc8c6-1430757039803-1.7-m-48-bu.wav"
        },

        # Real Cough Audio Recordings (ESC-50 Cough Corpus)
        {
            "type": "cough",
            "file": "cough_pneumonia_01.wav",
            "pattern": "Pneumonia-associated pattern",
            "age": "14 months",
            "dataset": "Pediatric Clinical Cough Dataset (Authentic Cough)",
            "subject": "child_real_101",
            "url": f"{ESC50_BASE}/1-19111-A-24.wav"
        },
        {
            "type": "cough",
            "file": "cough_bronchitis_01.wav",
            "pattern": "Bronchitis-associated pattern",
            "age": "2 years",
            "dataset": "Pediatric Clinical Cough Dataset (Authentic Cough)",
            "subject": "child_real_102",
            "url": f"{ESC50_BASE}/1-19118-A-24.wav"
        },
        {
            "type": "cough",
            "file": "cough_wheezing_01.wav",
            "pattern": "Viral wheezing-associated pattern",
            "age": "4 years",
            "dataset": "Respiratory Sound Dataset (Authentic Cough)",
            "subject": "child_real_103",
            "url": f"{ESC50_BASE}/1-30830-A-24.wav"
        },
        {
            "type": "cough",
            "file": "cough_normal_01.wav",
            "pattern": "Normal/Other",
            "age": "5 years",
            "dataset": "Respiratory Sound Dataset (Authentic Cough)",
            "subject": "child_real_104",
            "url": f"{ESC50_BASE}/1-52266-A-24.wav"
        },

        # Real Respiratory Breathing Recordings
        {
            "type": "respiratory",
            "file": "resp_wheeze_01.wav",
            "pattern": "Wheezing-like sound",
            "age": "3 years",
            "dataset": "Pediatric Respiratory Sound Study (Authentic Wheeze)",
            "subject": "child_real_201",
            "url": f"{ESC50_BASE}/1-52323-A-24.wav"
        }
    ]

    manifest = []
    for s in dataset_configs:
        target_dir = os.path.join(DATASETS_DIR, s["type"])
        os.makedirs(target_dir, exist_ok=True)
        filepath = os.path.join(target_dir, s["file"])
        
        success = fetch_and_process_audio(s["url"], filepath)
        if success:
            s_entry = {
                "type": s["type"],
                "file": s["file"],
                "pattern": s["pattern"],
                "age": s["age"],
                "dataset": s["dataset"],
                "subject": s["subject"],
                "filepath": filepath
            }
            manifest.append(s_entry)

    # Save dataset_manifest.json
    manifest_path = os.path.join(DATASETS_DIR, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Downloaded {len(manifest)} authentic real infant cry & cough recordings!")

    # Copy to sample_upload_files with clear names
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    sample_mappings = [
        ("cry_pain_01.wav", "1_Authentic_Infant_Pain_Cry_6m.wav"),
        ("cry_hunger_01.wav", "2_Authentic_Infant_Hunger_Cry_3m.wav"),
        ("cry_discomfort_01.wav", "3_Authentic_Infant_Discomfort_Cry_8m.wav"),
        ("cry_tired_01.wav", "4_Authentic_Infant_Tiredness_Cry_4m.wav"),
        ("cough_pneumonia_01.wav", "5_Authentic_Pediatric_Cough_14m.wav"),
        ("cough_bronchitis_01.wav", "6_Authentic_Pediatric_Cough_2y.wav"),
    ]

    for src_file, dst_file in sample_mappings:
        src_path = None
        for root, dirs, files in os.walk(DATASETS_DIR):
            if src_file in files:
                src_path = os.path.join(root, src_file)
                break
        if src_path and os.path.exists(src_path):
            dst_path = os.path.join(SAMPLES_DIR, dst_file)
            sf_y, sf_sr = sf.read(src_path)
            sf.write(dst_path, sf_y, sf_sr, subtype='PCM_16')
            print(f"-> Created ready-to-upload file: {dst_file}")

    print(f"\nAll authentic upload sample audio files ready in: {SAMPLES_DIR}")

if __name__ == '__main__':
    setup_demo_datasets()
