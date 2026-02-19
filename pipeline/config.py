"""
Pipeline configuration: paths, feature-extraction params, model settings.
"""
import os

# ── Root paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
AUDIO_DIR   = os.path.join(DATASET_DIR, "audio_files")
PARAM_CSV   = os.path.join(DATASET_DIR, "param.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Saved artefact paths ─────────────────────────────────────────────────────
CLASSIFIER_PATH  = os.path.join(MODELS_DIR, "raga_classifier.joblib")
SCALER_PATH      = os.path.join(MODELS_DIR, "feature_scaler.joblib")
PROTOTYPES_PATH  = os.path.join(MODELS_DIR, "raga_prototypes.joblib")

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
PITCH_FMIN  = 60.0   # Hz – lower bound (roughly B1)
PITCH_FMAX  = 1000.0 # Hz – upper bound (roughly B5)

# ── Training ─────────────────────────────────────────────────────────────────
MIN_USABILITY    = 2.0   # skip files below this usability score
TEST_SIZE        = 0.20  # 20 % held out for evaluation
RANDOM_STATE     = 42
CV_FOLDS         = 5

# ── Scoring weights (must sum to 1.0) ────────────────────────────────────────
# SVM confidence vs. DTW proximity to raga prototype
SCORE_WEIGHT_SVM = 0.55
SCORE_WEIGHT_DTW = 0.45
