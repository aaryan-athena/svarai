"""
Model training, evaluation, and persistence.

Architecture
───────────────────────────────────────────────────────────────────────────
1. StandardScaler  – zero-mean, unit-variance normalisation
2. SVM (RBF) with probability calibration (CalibratedClassifierCV)
   – produces per-class probability estimates used for scoring

We also store per-raga prototype vectors (centroid of all training
feature vectors belonging to each raga) for DTW-based similarity
scoring in the evaluator.
"""

import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

from pipeline.config import (
    CLASSIFIER_PATH, SCALER_PATH, PROTOTYPES_PATH,
    CV_FOLDS, RANDOM_STATE, TEST_SIZE,
)


# ── training ───────────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
    verbose: bool = True,
) -> dict:
    """
    Train the raga classifier and save all artefacts to disk.

    Returns a dict with CV accuracy stats.
    """
    # ── scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── cross-validation report ─────────────────────────────────────────────
    base_svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=False,   # base estimator for calibration must be non-prob
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if verbose:
        print(f"\nRunning {CV_FOLDS}-fold cross-validation …")

    cv_results = cross_validate(
        base_svm, X_scaled, y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
        verbose=0,
    )
    cv_acc  = cv_results["test_accuracy"]
    cv_f1   = cv_results["test_f1_macro"]

    if verbose:
        print(f"  CV Accuracy : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
        print(f"  CV F1-macro : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    # ── final model on all data ─────────────────────────────────────────────
    calibrated = CalibratedClassifierCV(
        SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=False,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        method="isotonic",
        cv=CV_FOLDS,
    )
    calibrated.fit(X_scaled, y, sample_weight=weights)

    # ── raga prototype centroids ────────────────────────────────────────────
    prototypes: dict[str, np.ndarray] = {}
    for raga in labels:
        mask = y == raga
        prototypes[raga] = X_scaled[mask].mean(axis=0)

    # final classification report on training data (sanity check)
    if verbose:
        y_pred = calibrated.predict(X_scaled)
        print("\nTraining-set classification report:")
        print(classification_report(y, y_pred, target_names=labels))

    # ── persist artefacts ──────────────────────────────────────────────────
    joblib.dump(scaler,     SCALER_PATH)
    joblib.dump(calibrated, CLASSIFIER_PATH)
    joblib.dump(prototypes, PROTOTYPES_PATH)

    if verbose:
        print(f"\nArtefacts saved:")
        print(f"  Scaler      -> {SCALER_PATH}")
        print(f"  Classifier  -> {CLASSIFIER_PATH}")
        print(f"  Prototypes  -> {PROTOTYPES_PATH}")

    return {
        "cv_accuracy_mean": float(cv_acc.mean()),
        "cv_accuracy_std" : float(cv_acc.std()),
        "cv_f1_mean"      : float(cv_f1.mean()),
        "cv_f1_std"       : float(cv_f1.std()),
        "n_samples"       : int(len(y)),
        "ragas"           : labels,
    }


# ── loading ────────────────────────────────────────────────────────────────────

def load_artefacts() -> tuple:
    """
    Load and return (scaler, classifier, prototypes) from disk.
    Raises FileNotFoundError if the model has not been trained yet.
    """
    for path in (SCALER_PATH, CLASSIFIER_PATH, PROTOTYPES_PATH):
        if not __import__("os").path.isfile(path):
            raise FileNotFoundError(
                f"Model artefact not found: {path}\n"
                "Run  python train.py  first to train the model."
            )
    scaler     = joblib.load(SCALER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    prototypes = joblib.load(PROTOTYPES_PATH)
    return scaler, classifier, prototypes
