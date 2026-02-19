"""
Evaluator: score a user's recorded performance against a target raga.

Scoring strategy
────────────────────────────────────────────────────────────────────────────
Score = w_svm × P(target_raga | features)
      + w_dtw × (1 − normalised_dtw_distance_to_prototype)

• SVM component   – how confidently the model identifies the correct raga.
• DTW component   – how close the feature vector is to the raga's "centre"
                    relative to the farthest raga centre (normalised 0-1).

The combination tolerates individual variability because:
  – SVM is trained on diverse samples → robust to personal style.
  – DTW compares to a *prototype* (average), not a single reference file.
  – Neither requires a perfect match.

Score is returned as a percentage (0 – 100).
"""

import numpy as np
from scipy.spatial.distance import euclidean

from pipeline.config import SCORE_WEIGHT_SVM, SCORE_WEIGHT_DTW
from pipeline.feature_extractor import extract_features
from pipeline.model import load_artefacts


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1-D DTW between two 1-D vectors (treated as sequences of scalars)."""
    n, m = len(a), len(b)
    # initialise cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _normalised_dtw(fv: np.ndarray, prototypes: dict[str, np.ndarray],
                    target_raga: str) -> float:
    """
    Return a similarity score in [0, 1]:
      1.0  → feature vector is identical to the target prototype
      0.0  → feature vector is as far from target as the farthest raga
    """
    distances = {raga: _dtw_distance(fv, proto)
                 for raga, proto in prototypes.items()}
    d_target = distances[target_raga]
    d_max    = max(distances.values()) + 1e-9  # avoid /0
    # closer → higher score
    return float(max(0.0, 1.0 - d_target / d_max))


def evaluate(audio_path: str, target_raga: str) -> dict:
    """
    Evaluate how accurately the audio file matches the target raga.

    Parameters
    ----------
    audio_path  : path to the user's WAV recording
    target_raga : the raga the user intended to sing (must match a label
                  seen during training, case-insensitive)

    Returns
    -------
    dict with keys:
        predicted_raga  – raga the model thinks was sung
        target_raga     – raga the user intended
        score_pct       – overall accuracy 0–100
        svm_confidence  – SVM probability for target raga (0–1)
        dtw_similarity  – DTW similarity to target prototype (0–1)
        class_probs     – {raga: probability} for all ragas
        correct         – True if predicted_raga == target_raga
        feedback        – human-readable feedback string
    """
    # ── load model ──────────────────────────────────────────────────────────
    scaler, classifier, prototypes = load_artefacts()

    # normalise target raga name (case-insensitive prefix match)
    known_ragas = sorted(prototypes.keys())
    target_raga = _resolve_raga(target_raga, known_ragas)

    # ── extract features ─────────────────────────────────────────────────────
    fv = extract_features(audio_path)
    if fv is None:
        raise ValueError(f"Could not extract features from: {audio_path}")

    fv_scaled = scaler.transform(fv.reshape(1, -1))[0]

    # ── SVM probability ──────────────────────────────────────────────────────
    class_order  = list(classifier.classes_)
    proba        = classifier.predict_proba(fv_scaled.reshape(1, -1))[0]
    class_probs  = {raga: float(p) for raga, p in zip(class_order, proba)}
    predicted    = class_order[int(np.argmax(proba))]
    svm_conf     = class_probs.get(target_raga, 0.0)

    # ── DTW similarity ───────────────────────────────────────────────────────
    dtw_sim = _normalised_dtw(fv_scaled, prototypes, target_raga)

    # ── combined score ───────────────────────────────────────────────────────
    raw_score = SCORE_WEIGHT_SVM * svm_conf + SCORE_WEIGHT_DTW * dtw_sim
    score_pct = round(raw_score * 100, 1)

    # ── feedback ─────────────────────────────────────────────────────────────
    feedback = _generate_feedback(score_pct, predicted, target_raga,
                                  svm_conf, dtw_sim)

    return {
        "predicted_raga" : predicted,
        "target_raga"    : target_raga,
        "score_pct"      : score_pct,
        "svm_confidence" : round(svm_conf, 4),
        "dtw_similarity" : round(dtw_sim, 4),
        "class_probs"    : {k: round(v, 4) for k, v in class_probs.items()},
        "correct"        : predicted == target_raga,
        "feedback"       : feedback,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_raga(name: str, known: list[str]) -> str:
    name_l = name.strip().lower()
    # exact (case-insensitive)
    for k in known:
        if k.lower() == name_l:
            return k
    # prefix
    matches = [k for k in known if k.lower().startswith(name_l)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Raga '{name}' not recognised. Available ragas: {known}\n"
        "Use the exact name or an unambiguous prefix."
    )


def _generate_feedback(score: float, predicted: str, target: str,
                       svm_conf: float, dtw_sim: float) -> str:
    lines = []
    if predicted == target:
        lines.append(f"Great job! The model correctly identified your singing as {target}.")
    else:
        lines.append(
            f"The model classified your singing as '{predicted}' instead of '{target}'."
        )

    if score >= 85:
        lines.append("Your raga accuracy is excellent. The pitch and melodic structure closely follow the raga.")
    elif score >= 70:
        lines.append("Good performance. The core melodic phrases are recognisable, with minor deviations.")
    elif score >= 55:
        lines.append("Moderate accuracy. Focus on staying within the raga's characteristic notes (vadi/samvadi).")
    elif score >= 40:
        lines.append("The raga is partially recognisable, but several phrases deviate from the expected pattern.")
    else:
        lines.append("The raga is difficult to identify. Consider practising the characteristic phrases more carefully.")

    # sub-component hints
    if svm_conf < 0.5:
        lines.append("Tip: The overall tonal colour may overlap with another raga — mind the characteristic notes.")
    if dtw_sim < 0.5:
        lines.append("Tip: The melodic contour differs from the reference profile — focus on the correct swar sequence.")

    lines.append(f"\nScore breakdown — SVM confidence: {svm_conf*100:.1f}%  |  DTW similarity: {dtw_sim*100:.1f}%")
    return "\n".join(lines)
