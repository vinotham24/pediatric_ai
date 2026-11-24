import os
import uuid
import time
import tempfile
import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np
from config import LIVE_RECORDINGS_DIR, UPLOAD_RECORDINGS_DIR, PROCESSED_RECORDINGS_DIR

try:
    import imageio_ffmpeg
except ImportError:  # Report a useful error if the optional decoder is missing.
    imageio_ffmpeg = None

class AudioService:
    CRY_UPLOAD_EXTENSIONS = {
        '.wav', '.mp3', '.m4a', '.aac', '.ogg', '.oga', '.opus',
        '.flac', '.3gp', '.3g2', '.caf', '.aif', '.aiff', '.mp4',
        '.m4v', '.wma', '.webm', '.amr', '.au', '.snd', '.mka',
    }

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

    def decode_cry_upload(self, file_storage):
        """Save an upload using its claimed suffix and decode it to verified PCM WAV.

        Returns (source_path, wav_path); caller must remove both paths.
        FFmpeg validates the container and codec, so extensions and MIME types
        are never treated as proof that a file contains audio.
        """
        suffix = Path(file_storage.filename or '').suffix.lower()
        if suffix not in self.CRY_UPLOAD_EXTENSIONS:
            raise ValueError('Unsupported audio format. Choose a supported audio file.')
        if imageio_ffmpeg is None:
            raise RuntimeError('Audio decoder unavailable: install imageio-ffmpeg.')

        source = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        source_path, wav_path = source.name, wav.name
        source.close()
        wav.close()
        try:
            file_storage.save(source_path)
            if os.path.getsize(source_path) == 0:
                raise ValueError('The uploaded audio file is empty.')
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            command = [ffmpeg, '-nostdin', '-v', 'error', '-y', '-i', source_path,
                       '-map', '0:a:0', '-vn', '-sn', '-dn', '-ac', '1', '-ar', '16000',
                       '-c:a', 'pcm_s16le', '-f', 'wav', wav_path]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode:
                print(f"FFmpeg audio decode failed: {result.stderr.strip()}", flush=True)
                raise ValueError('Unable to decode this audio file. Check that it is a valid, supported recording.')
            info = sf.info(wav_path)
            if info.frames <= 0 or info.samplerate != 16000 or info.channels != 1 or info.subtype != 'PCM_16':
                raise ValueError('Audio conversion did not produce valid mono 16 kHz PCM audio.')
            # Independently decode the converted WAV before inference.
            samples, sr = sf.read(wav_path, dtype='float32')
            if sr != 16000 or samples.size == 0 or not np.isfinite(samples).all():
                raise ValueError('Converted audio could not be validated.')
            return source_path, wav_path
        except Exception:
            self.delete_recording(source_path)
            self.delete_recording(wav_path)
            raise

    def delete_recording(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
