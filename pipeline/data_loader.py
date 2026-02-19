"""
Dataset loader.

Reads param.csv, filters files by usability, extracts features,
and returns (X, y, weights) ready for model training.
"""

import os
import numpy as np
import pandas as pd

from pipeline.config import (
    AUDIO_DIR, PARAM_CSV, MIN_USABILITY,
)
from pipeline.feature_extractor import extract_features


def load_dataset(verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load audio files listed in param.csv and extract features.

    Returns
    -------
    X       : (n_samples, n_features)  feature matrix
    y       : (n_samples,)             raga label strings
    weights : (n_samples,)             per-sample usability weights
    labels  : sorted list of unique raga labels
    """
    df = _read_param_csv()
    df = _filter(df, verbose)

    X_list, y_list, w_list = [], [], []
    total = len(df)

    for idx, row in df.iterrows():
        filename   = row["File Name"].strip()
        raga_label = row["Raga Label"].strip()
        usability  = float(row["Usability"])
        filepath   = os.path.join(AUDIO_DIR, filename)

        if verbose:
            print(f"  [{idx+1}/{total}] {filename} ({raga_label}) ...", end=" ", flush=True)

        if not os.path.isfile(filepath):
            if verbose:
                print("MISSING - skipped")
            continue

        fv = extract_features(filepath)
        if fv is None:
            if verbose:
                print("FAILED  - skipped")
            continue

        X_list.append(fv)
        y_list.append(raga_label)
        w_list.append(usability)

        if verbose:
            print("OK")

    if not X_list:
        raise RuntimeError("No features could be extracted. Check AUDIO_DIR and param.csv.")

    X = np.vstack(X_list)
    y = np.array(y_list)
    w = np.array(w_list, dtype=float)

    # normalise weights to [0.5, 1.0] so low-quality files still contribute
    w = 0.5 + 0.5 * (w - w.min()) / (w.max() - w.min() + 1e-9)

    labels = sorted(set(y_list))

    if verbose:
        print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(labels)} ragas: {labels}")

    return X, y, w, labels


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_param_csv() -> pd.DataFrame:
    df = pd.read_csv(PARAM_CSV, usecols=range(10))
    df.columns = [
        "File Name", "Raga Label", "File Size (MB)", "Duration (sec)",
        "Quality (1-5)", "BG Noise?", "Vocal/Inst.", "Gamakas?",
        "Pitch Cont.", "Usability",
    ]
    # drop empty rows
    df = df.dropna(subset=["File Name", "Raga Label"])
    df = df[df["File Name"].str.strip() != ""]
    return df


def _filter(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    df["Usability"] = pd.to_numeric(df["Usability"], errors="coerce")
    before = len(df)
    df = df[df["Usability"] >= MIN_USABILITY].copy()
    after = len(df)
    if verbose and before != after:
        print(f"  Filtered {before - after} low-usability files "
              f"(usability < {MIN_USABILITY})")
    return df.reset_index(drop=True)
