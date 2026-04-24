"""
Pipeline configuration: audio loading and feature-extraction parameters.
"""

# ── Audio loading ────────────────────────────────────────────────────────────
SAMPLE_RATE     = 22050          # Hz – standard for music analysis
SEGMENT_SEC     = 30             # seconds per analysis window
SEGMENT_HOP_SEC = 15             # seconds between windows (50 % overlap)

# ── Feature extraction ───────────────────────────────────────────────────────
N_MFCC      = 20    # MFCC coefficients
N_CHROMA    = 12    # pitch-class bins (one per semitone)
N_FFT       = 2048  # FFT window size
HOP_LENGTH  = 512   # frames between FFT windows

# ── Pitch extraction ─────────────────────────────────────────────────────────
PITCH_FMIN  = 60.0    # Hz – lower bound (roughly B1)
PITCH_FMAX  = 1000.0  # Hz – upper bound (roughly B5)
