import numpy as np
import librosa
import soundfile as sf
import os
import subprocess
from config import SAMPLE_RATE, AUDIO_CHANNELS, WINDOW_DURATION_SEC, HOP_DURATION_SEC

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = None

def decode_audio_with_ffmpeg(path_or_bytes, target_sr=SAMPLE_RATE):
    if not FFMPEG_PATH:
        raise RuntimeError("FFmpeg decoder executable not found.")
    
    cmd = [
        FFMPEG_PATH,
        '-i', 'pipe:0' if not isinstance(path_or_bytes, str) else path_or_bytes,
        '-f', 's16le',
        '-ac', '1',
        '-ar', str(target_sr),
        '-'
    ]
    input_data = path_or_bytes if not isinstance(path_or_bytes, str) else None
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if input_data else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate(input=input_data)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg decoding error: {err.decode(errors='ignore')}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, target_sr

def load_and_preprocess_audio(file_path_or_bytes, target_sr=SAMPLE_RATE):
    """
    Loads audio from filepath or bytes buffer, converts to mono, resamples to 16kHz, and normalizes.
    Supports WAV, WebM, OGG, MP3, AAC, FLAC, M4A browser recordings.
    """
    y = None
    sr = target_sr

    try:
        y, sr = librosa.load(file_path_or_bytes, sr=target_sr, mono=True)
    except Exception as e1:
        try:
            if isinstance(file_path_or_bytes, str):
                y_raw, sr_orig = sf.read(file_path_or_bytes)
                if len(y_raw.shape) > 1:
                    y_raw = np.mean(y_raw, axis=1)
                if sr_orig != target_sr:
                    y = librosa.resample(y_raw, orig_sr=sr_orig, target_sr=target_sr)
                else:
                    y = y_raw
            else:
                raise e1
        except Exception as e2:
            try:
                y, sr = decode_audio_with_ffmpeg(file_path_or_bytes, target_sr=target_sr)
            except Exception as e3:
                raise RuntimeError(f"Could not load audio. Librosa: {e1} | Soundfile: {e2} | FFmpeg: {e3}")

    if y is None or len(y) == 0:
        raise ValueError("Loaded audio signal is empty.")

    # Amplitude normalization
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    return y, target_sr

def apply_vad(y, sr=SAMPLE_RATE, top_db=25):
    """
    Voice/Audio Activity Detection to remove long silences.
    """
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    if len(non_silent_intervals) == 0:
        return y
    
    y_trimmed = np.concatenate([y[start:end] for start, end in non_silent_intervals])
    return y_trimmed

def segment_audio(y, sr=SAMPLE_RATE, window_sec=WINDOW_DURATION_SEC, hop_sec=HOP_DURATION_SEC):
    """
    Slices audio array into overlapping window segments.
    """
    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    
    if len(y) <= win_len:
        # Pad short audio
        padded = np.pad(y, (0, win_len - len(y)), mode='constant')
        return [padded]
        
    segments = []
    for start in range(0, len(y) - win_len + 1, hop_len):
        segments.append(y[start : start + win_len])
        
    return segments
