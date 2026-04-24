"""
Extracts a rich, human-readable dictionary of audio features for display
and raga parameter matching in the web UI.

Unlike the training feature extractor (which returns a fixed-size ML vector),
this module returns a labeled dict covering pitch, oscillation, spectral,
chroma, and energy dimensions.
"""

import warnings
import numpy as np
import librosa

from pipeline.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, PITCH_FMIN, PITCH_FMAX

warnings.filterwarnings("ignore", category=UserWarning)

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def extract_rich_features(path: str) -> dict | None:
    """
    Extract detailed audio features from a file.
    Returns a labeled dict, or None if the file cannot be loaded.
    """
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        print(f"[WARN] Cannot load {path}: {exc}")
        return None

    features: dict = {}
    features["duration_sec"] = round(float(len(y) / sr), 2)

    # ── Pitch ──────────────────────────────────────────────────────────────
    pitch_features = _pitch_features(y, sr)
    features.update(pitch_features)

    # ── Spectral ───────────────────────────────────────────────────────────
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_centroid"]     = round(float(sc.mean()), 2)
    features["spectral_centroid_std"] = round(float(sc.std()), 2)

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_bandwidth"]     = round(float(sb.mean()), 2)
    features["spectral_bandwidth_std"] = round(float(sb.std()), 2)

    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_rolloff"] = round(float(sr_feat.mean()), 2)

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features["zero_crossing_rate"] = round(float(zcr.mean()), 6)

    # ── Energy ─────────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features["rms_energy"]     = round(float(rms.mean()), 6)
    features["rms_energy_std"] = round(float(rms.std()), 6)

    # ── Chroma ─────────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma_mean = chroma.mean(axis=1)
    features["dominant_pitch_class"] = PITCH_CLASSES[int(np.argmax(chroma_mean))]
    features["chroma_energy"] = {
        pc: round(float(v), 4) for pc, v in zip(PITCH_CLASSES, chroma_mean)
    }

    # ── MFCCs (first 13) ───────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["mfcc"] = {
        f"mfcc_{i + 1}": round(float(mfcc[i].mean()), 4) for i in range(13)
    }

    return features


# ── helpers ────────────────────────────────────────────────────────────────────

def _pitch_features(y: np.ndarray, sr: int) -> dict:
    """Extract pitch-related features including oscillation and drift."""
    out: dict = {}
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr, hop_length=HOP_LENGTH
        )
        voiced_f0 = f0[voiced_flag == 1] if voiced_flag is not None else f0
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) > 1:
            out["min_pitch"]  = round(float(np.min(voiced_f0)), 4)
            out["max_pitch"]  = round(float(np.max(voiced_f0)), 4)
            out["mean_pitch"] = round(float(np.mean(voiced_f0)), 4)
            out["std_pitch"]  = round(float(np.std(voiced_f0)), 4)

            # Oscillation depth: peak-to-peak span of pitch variations
            out["oscillation_depth"] = round(float(np.max(voiced_f0) - np.min(voiced_f0)), 4)

            # Oscillation rate: Hz of pitch oscillation estimated via mean crossings
            centered     = voiced_f0 - np.mean(voiced_f0)
            sign_changes = int(np.sum(np.diff(np.sign(centered)) != 0))
            hop_sec      = HOP_LENGTH / sr
            voiced_dur   = len(voiced_f0) * hop_sec
            out["oscillation_rate"] = round(float(sign_changes / (2 * max(voiced_dur, 1e-3))), 4)

            # Pitch drift: absolute linear trend of pitch over time (Hz per second)
            t     = np.arange(len(voiced_f0)) * hop_sec
            slope = float(np.polyfit(t, voiced_f0, 1)[0]) if len(t) > 1 else 0.0
            out["pitch_drift"] = round(abs(slope), 6)

            # Pitch continuity: fraction of total frames that are voiced
            total_frames = len(f0) if f0 is not None else 1
            voiced_count = int(np.sum(voiced_flag == 1)) if voiced_flag is not None else len(voiced_f0)
            out["pitch_continuity"] = round(float(voiced_count / max(total_frames, 1)), 4)
        else:
            _zero_pitch(out)
    except Exception:
        _zero_pitch(out)

    return out


def _zero_pitch(out: dict) -> None:
    for key in [
        "min_pitch", "max_pitch", "mean_pitch", "std_pitch",
        "oscillation_depth", "oscillation_rate", "pitch_drift", "pitch_continuity",
    ]:
        out[key] = 0.0
