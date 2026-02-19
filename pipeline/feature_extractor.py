"""
Audio feature extraction for raga recognition.

For each audio segment we extract:
  - MFCC (mean + std) ............... 2 × N_MFCC  = 40 dims
  - Chroma (mean + std) ............. 2 × N_CHROMA = 24 dims
  - Spectral centroid (mean + std) .. 2 dims
  - Spectral bandwidth (mean + std) . 2 dims
  - Spectral rolloff (mean + std) ... 2 dims
  - Zero-crossing rate (mean + std) . 2 dims
  - RMS energy (mean + std) ......... 2 dims
  - Dominant pitch (mean + std) ..... 2 dims
  - Pitch std (a proxy for pitch continuity) .. 1 dim
  ─────────────────────────────────────────────
  Total per segment ................. 79 dims

File-level features are obtained by averaging segment vectors so that
variable-length audio files map to a fixed-size representation.
"""

import warnings
import numpy as np
import librosa

from pipeline.config import (
    SAMPLE_RATE, SEGMENT_SEC, SEGMENT_HOP_SEC,
    N_MFCC, N_CHROMA, N_FFT, HOP_LENGTH,
    PITCH_FMIN, PITCH_FMAX,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_load(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file, converting to mono at the project sample rate."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def _segment(y: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    Slice the waveform into overlapping windows.
    Returns at least one segment even for very short files.
    """
    seg_len = int(SEGMENT_SEC * sr)
    hop_len = int(SEGMENT_HOP_SEC * sr)

    if len(y) <= seg_len:
        return [y]

    segments = []
    start = 0
    while start + seg_len <= len(y):
        segments.append(y[start : start + seg_len])
        start += hop_len

    # include the tail if it is at least 5 s long
    tail = y[start:]
    if len(tail) >= 5 * sr:
        segments.append(tail)

    return segments if segments else [y]


# ── per-segment feature vector ─────────────────────────────────────────────────

def _extract_segment_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Return a 1-D feature vector for a single audio segment."""
    parts = []

    # MFCCs – capture timbral / vowel colour
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    parts += [mfcc.mean(axis=1), mfcc.std(axis=1)]

    # Chroma – pitch-class energy (core of raga identity)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH,
                                         n_chroma=N_CHROMA)
    parts += [chroma.mean(axis=1), chroma.std(axis=1)]

    # Spectral centroid – brightness
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT,
                                           hop_length=HOP_LENGTH)
    parts += [sc.mean(axis=1), sc.std(axis=1)]

    # Spectral bandwidth
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT,
                                            hop_length=HOP_LENGTH)
    parts += [sb.mean(axis=1), sb.std(axis=1)]

    # Spectral rolloff
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    parts += [ro.mean(axis=1), ro.std(axis=1)]

    # Zero-crossing rate – roughness / noisiness
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    parts += [zcr.mean(axis=1), zcr.std(axis=1)]

    # RMS energy – amplitude dynamics
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    parts += [rms.mean(axis=1), rms.std(axis=1)]

    # Pitch (fundamental frequency via pyin) – melodic contour
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=PITCH_FMIN, fmax=PITCH_FMAX,
            sr=sr, hop_length=HOP_LENGTH
        )
        voiced_f0 = f0[voiced_flag == 1] if voiced_flag is not None else f0
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) > 0:
            pitch_mean = np.array([voiced_f0.mean()])
            pitch_std  = np.array([voiced_f0.std()])
        else:
            pitch_mean = np.array([0.0])
            pitch_std  = np.array([0.0])
    except Exception:
        pitch_mean = np.array([0.0])
        pitch_std  = np.array([0.0])

    parts += [pitch_mean, pitch_std]

    return np.concatenate(parts)


# ── public API ─────────────────────────────────────────────────────────────────

def extract_features(path: str) -> np.ndarray | None:
    """
    Extract a fixed-size feature vector from an audio file.

    Segments the file, computes per-segment features, then returns the
    mean across all segments so variable-length files are handled gracefully.

    Returns None and prints a warning if loading fails.
    """
    try:
        y, sr = _safe_load(path)
    except Exception as exc:
        print(f"  [WARN] Cannot load {path}: {exc}")
        return None

    segments = _segment(y, sr)
    seg_feats = []
    for seg in segments:
        try:
            fv = _extract_segment_features(seg, sr)
            seg_feats.append(fv)
        except Exception as exc:
            print(f"  [WARN] Segment error in {path}: {exc}")

    if not seg_feats:
        return None

    return np.vstack(seg_feats).mean(axis=0)
