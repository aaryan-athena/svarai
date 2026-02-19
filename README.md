# Svarai – Raga Accuracy Pipeline

A machine learning pipeline that evaluates how accurately a user sang a given Indian classical raga. The model is trained on reference recordings and scores new performances based on pitch, amplitude, and melodic structure — without requiring a perfect match, since every singer has their own style.

## Ragas Supported

| Raga | Files |
|---|---|
| Asavari | 10 WAV |
| Sarang | 10 WAV |
| Yaman | 10 WAV |

## Project Structure

```
svarai/
├── dataset/
│   ├── param.csv             # metadata: raga labels, quality, usability scores
│   └── audio_files/          # reference WAV recordings
├── pipeline/
│   ├── config.py             # all tuneable settings (paths, feature params)
│   ├── feature_extractor.py  # audio → 79-dim feature vector
│   ├── data_loader.py        # reads param.csv, drives extraction
│   ├── model.py              # SVM training & persistence
│   └── evaluator.py          # scores a user recording
├── models/                   # saved artefacts (created after training)
├── train.py                  # entry point: train the model
├── evaluate.py               # entry point: score a recording
└── requirements.txt
```

## Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

## Usage

### 1. Train

```bash
python train.py
```

Reads all files from `dataset/param.csv`, extracts features, runs 5-fold cross-validation, and saves the trained model to `models/`.

Optional flag:

```bash
python train.py --quiet   # suppress per-file extraction messages
```

### 2. Evaluate a recording

```bash
python evaluate.py <path_to_audio> <raga_name>
```

Examples:

```bash
python evaluate.py my_recording.wav Yaman
python evaluate.py recordings/test.wav Asavari
python evaluate.py demo.wav sarang       # case-insensitive
```

### Example output

```
============================================================
  Result          : ✓ CORRECT
  Target raga     : Yaman
  Predicted raga  : Yaman
  Overall score   : 78.4 / 100
------------------------------------------------------------
  SVM confidence  : 84.2 %
  DTW similarity  : 70.5 %
------------------------------------------------------------
  Per-raga probabilities:
    Yaman        84.2%  █████████████████████████
    Asavari       9.1%  ██
    Sarang        6.7%  ██
============================================================

Feedback:
  Good performance. The core melodic phrases are recognisable,
  with minor deviations.
  Score breakdown — SVM confidence: 84.2%  |  DTW similarity: 70.5%
```

## How It Works

### Feature Extraction

Each audio file is sliced into overlapping 30-second segments. A 79-dimensional feature vector is computed per segment and then averaged across all segments, so files of any length map to the same fixed representation.

| Feature | Dims | Purpose |
|---|---|---|
| MFCC (mean + std) | 40 | Timbral / vowel colour |
| Chroma (mean + std) | 24 | Pitch-class energy — core of raga identity |
| Spectral centroid / bandwidth / rolloff | 6 | Brightness, spectral shape |
| Zero-crossing rate (mean + std) | 2 | Roughness |
| RMS energy (mean + std) | 2 | Amplitude dynamics |
| Fundamental pitch F0 (mean + std) | 2 | Melodic contour |
| Pitch std | 1 | Pitch continuity proxy |

### Model

- **SVM with RBF kernel** + isotonic calibration → per-class probability estimates
- `class_weight="balanced"` handles minor class imbalance automatically
- Training samples are weighted by the `Usability` score from `param.csv`
- 5-fold stratified cross-validation is run and reported at training time

### Scoring

```
Score = 0.55 × SVM_P(target_raga) + 0.45 × DTW_similarity_to_prototype
```

- **SVM component** — how confidently the model identifies the correct raga
- **DTW component** — how close the feature vector is to the raga's average profile (centroid across all training samples)

This combination tolerates individual variability: the user is never compared to a single reference recording, only to the statistical average of the raga.

## Configuration

All tuneable parameters live in `pipeline/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MIN_USABILITY` | `2.0` | Skip files below this usability score |
| `SEGMENT_SEC` | `30` | Analysis window length in seconds |
| `SEGMENT_HOP_SEC` | `15` | Step between windows (50% overlap) |
| `N_MFCC` | `20` | Number of MFCC coefficients |
| `N_CHROMA` | `12` | Chroma bins |
| `SCORE_WEIGHT_SVM` | `0.55` | Weight of SVM confidence in final score |
| `SCORE_WEIGHT_DTW` | `0.45` | Weight of DTW similarity in final score |

## Requirements

- Python 3.10+
- librosa, numpy, scipy, scikit-learn, pandas, joblib, soundfile

Install all with:

```bash
pip install -r requirements.txt
```
