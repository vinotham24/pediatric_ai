import os
import uuid
import time
import soundfile as sf
import numpy as np
from config import LIVE_RECORDINGS_DIR, UPLOAD_RECORDINGS_DIR, PROCESSED_RECORDINGS_DIR

class AudioService:
    def __init__(self):
        for path in [LIVE_RECORDINGS_DIR, UPLOAD_RECORDINGS_DIR, PROCESSED_RECORDINGS_DIR]:
            os.makedirs(path, exist_ok=True)

    def save_live_recording(self, audio_data, sr=16000, prefix="audio"):
        """
        Saves float/int numpy audio array as browser-compatible 16-bit PCM integer WAV file.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{uuid.uuid4().hex[:6]}.wav"
        filepath = os.path.join(LIVE_RECORDINGS_DIR, filename)
        
        # Ensure 1D mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Normalize and scale to 16-bit signed PCM integer for 100% browser audio player compatibility
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_norm = audio_data / max_val * 0.9
        else:
            audio_norm = audio_data
            
        audio_int16 = (audio_norm * 32767.0).clip(-32768, 32767).astype(np.int16)
        sf.write(filepath, audio_int16, sr, subtype='PCM_16')
        return filename, filepath

    def save_uploaded_file(self, file_storage):
        """
        Saves uploaded FileStorage object to recordings/uploads/
        """
        filename = f"upload_{uuid.uuid4().hex[:8]}_{file_storage.filename}"
        filepath = os.path.join(UPLOAD_RECORDINGS_DIR, filename)
        file_storage.save(filepath)
        return filename, filepath

    def delete_recording(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
