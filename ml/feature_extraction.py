import numpy as np
import librosa
from config import SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH, N_MEL_BANDS, N_MFCC

def extract_mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MEL_BANDS):
    """
    Computes log Mel Spectrogram array (n_mels, time_steps).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def extract_acoustic_features(y, sr=SAMPLE_RATE):
    """
    Extracts a comprehensive 1D acoustic feature vector (~130 dimensions)
    combining MFCCs, Spectral metrics, pitch tracking, and RMS energy.
    """
    features = []
    
    # 1. MFCCs + Deltas + Delta-Deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    for feat_mat in [mfcc, delta_mfcc, delta2_mfcc]:
        features.extend([
            np.mean(feat_mat, axis=1),
            np.std(feat_mat, axis=1),
            np.max(feat_mat, axis=1),
            np.min(feat_mat, axis=1)
        ])
        
    # 2. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    features.extend([np.mean(spec_cent), np.std(spec_cent), np.max(spec_cent), np.min(spec_cent)])
    
    # 3. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    features.extend([np.mean(spec_bw), np.std(spec_bw), np.max(spec_bw), np.min(spec_bw)])
    
    # 4. Spectral Rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    features.extend([np.mean(spec_roll), np.std(spec_roll), np.max(spec_roll), np.min(spec_roll)])
    
    # 5. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])
    
    # 6. RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])

    
    # 7. Pitch / F0 estimation (YIN algorithm)
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            pitch_mean, pitch_std, pitch_max = np.mean(f0_clean), np.std(f0_clean), np.max(f0_clean)
        else:
            pitch_mean, pitch_std, pitch_max = 0.0, 0.0, 0.0
    except Exception:
        pitch_mean, pitch_std, pitch_max = 0.0, 0.0, 0.0
        
    features.extend([pitch_mean, pitch_std, pitch_max])
    
    # Flatten all feature statistics into a single 1D vector
    feature_vector = np.concatenate([np.atleast_1d(f) for f in features])
    return feature_vector
