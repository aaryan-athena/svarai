"""
Parameter-based raga identification engine.

Algorithm (no ML):
1. Extract rich features from the audio (via rich_features.py).
2. Load all raga profiles from Firebase.
3. For every raga profile compute four sub-scores:
   a. Chroma score      (45%) — cosine similarity after tonic-aligned circular shift.
   b. Pitch params score(30%) — fraction of acoustic params inside expected range,
                                weighted by how far out-of-range the outliers are.
   c. Note prominence   (25%) — are vadi & samvadi the strongest pitch classes?
   d. Forbidden penalty       — up to 20 pts deducted for presence of forbidden notes.
4. Return a ranked list of ragas with per-parameter breakdowns for the AI tutor.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Chromatic pitch-class names (C=0 … B=11)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_INDEX = {n: i for i, n in enumerate(_NOTE_NAMES)}

# Positive weights MUST sum to 1.0 so a perfect performance can score 100.
_W_CHROMA = 0.45
_W_PITCH  = 0.30
_W_PROM   = 0.25

# Forbidden-note penalty: up to this many points are deducted from the final score.
_MAX_FORB_PENALTY = 20.0

# Swara → pitch-class index (relative to C tonic, octave-independent).
# Handles both shuddha/komal spellings and common aliases.
_SWARA_INDEX = {
    "S":  0,   # Shadja
    "r":  1,   # komal Re
    "R":  2,   # shuddha Re
    "g":  3,   # komal Ga
    "G":  4,   # shuddha Ga
    "m":  5,   # shuddha Ma
    "M":  6,   # tivra Ma
    "M#": 6,   # tivra Ma (alternate spelling)
    "P":  7,   # Panchama
    "d":  8,   # komal Dha
    "D":  9,   # shuddha Dha
    "n":  10,  # komal Ni
    "N":  11,  # shuddha Ni
}

# ── public API ────────────────────────────────────────────────────────────────

def match_raga(features: dict, raga_profiles: list[dict]) -> list[dict]:
    """
    Score every raga profile against the extracted features.

    Returns a list of match dicts sorted by overall_score descending.
    Each dict: id, name, overall_score, chroma_score, pitch_params_score,
               note_prominence_score, forbidden_penalty, param_details, chroma_details.
    """
    chroma_vec = _build_chroma_vec(features)
    results = []

    for raga in raga_profiles:
        try:
            results.append(_score_raga(raga, chroma_vec, features))
        except Exception as exc:
            logger.warning("Skipping raga '%s': %s", raga.get("name"), exc)

    results.sort(key=lambda r: r["overall_score"], reverse=True)
    return results


def get_best_match(features: dict, raga_profiles: list[dict]) -> Optional[dict]:
    """Return only the top-ranked raga match."""
    ranked = match_raga(features, raga_profiles)
    return ranked[0] if ranked else None


def get_identification_confidence(ranked: list[dict]) -> str:
    """
    Return a human-readable confidence level for the top-ranked match.
    Based on the gap between 1st and 2nd place scores.
    """
    if not ranked:
        return "no_match"
    top = ranked[0]["overall_score"]
    if len(ranked) < 2:
        return "high" if top >= 60 else "low"
    gap = top - ranked[1]["overall_score"]
    if top >= 70 and gap >= 15:
        return "high"
    if top >= 55 and gap >= 8:
        return "medium"
    return "low"


# ── scoring helpers ───────────────────────────────────────────────────────────

def _build_chroma_vec(features: dict) -> np.ndarray:
    """Extract normalised 12-element chroma vector from features dict."""
    chroma_energy = features.get("chroma_energy", {})
    vec = np.zeros(12)
    for note, idx in _NOTE_INDEX.items():
        vec[idx] = float(chroma_energy.get(note, 0.0))
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def _profile_to_vec(chroma_profile: dict) -> np.ndarray:
    """Convert a raga's chroma_profile dict to a normalised 12-element numpy vector."""
    vec = np.zeros(12)
    for note, val in chroma_profile.items():
        idx = _NOTE_INDEX.get(note)
        if idx is not None:
            vec[idx] = float(val)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _chroma_score(user_vec: np.ndarray, profile_vec: np.ndarray) -> tuple[float, int, dict]:
    """
    Tonic-agnostic chroma similarity.

    Tries all 12 circular shifts of the user's chroma vector and picks the
    best cosine similarity against the raga's profile.  Returns:
      (score 0-100, best_shift, per-note chroma_details dict)
    """
    best_sim = -1.0
    best_shift = 0
    for shift in range(12):
        sim = _cosine_similarity(np.roll(user_vec, -shift), profile_vec)
        if sim > best_sim:
            best_sim = sim
            best_shift = shift

    score = max(0.0, best_sim) * 100.0

    shifted_user = np.roll(user_vec, -best_shift)
    chroma_details: dict[str, dict] = {}
    for note, idx in _NOTE_INDEX.items():
        expected = float(profile_vec[idx])
        actual   = float(shifted_user[idx])
        issue = None
        if expected < 0.05 and actual > 0.15:
            issue = (
                f"Note absent/forbidden in this raga but appears prominent "
                f"(actual {actual:.2f}, expected ≤0.05)."
            )
        elif expected > 0.40 and actual < 0.10:
            issue = (
                f"Important note (expected {expected:.2f}) is very weak "
                f"in your performance (actual {actual:.2f})."
            )
        chroma_details[note] = {
            "expected": round(expected, 3),
            "actual":   round(actual, 3),
            "issue":    issue,
        }

    return score, best_shift, chroma_details


def _pitch_params_score(features: dict, raga: dict) -> tuple[float, dict]:
    """
    Score how many acoustic parameter values fall within the raga's expected ranges.
    Returns (0-100 score, per-param details dict).
    Missing pitch params (no Firebase data) returns 50.0 as a neutral score.
    """
    pitch_params: dict = raga.get("pitch_params", {})
    if not pitch_params:
        return 50.0, {}

    total_weight  = 0.0
    earned_weight = 0.0
    details: dict[str, dict] = {}

    for param, spec in pitch_params.items():
        lo    = spec.get("min")
        hi    = spec.get("max")
        value = features.get(param)

        if value is None or lo is None or hi is None:
            continue

        range_size = max(hi - lo, 1e-6)
        in_range   = (lo <= value <= hi)
        total_weight += 1.0

        if in_range:
            earned_weight += 1.0
            deviation = 0.0
        else:
            dist      = max(lo - value, value - hi, 0.0)
            deviation = dist / range_size
            partial   = max(0.0, 1.0 - deviation)
            earned_weight += partial * 0.5   # partial credit capped at 50 %

        details[param] = {
            "value":     round(float(value), 4),
            "min":       lo,
            "max":       hi,
            "in_range":  in_range,
            "deviation": round(deviation, 4) if not in_range else 0.0,
            "label":     spec.get("label", param),
            "unit":      spec.get("unit", ""),
        }

    if total_weight == 0:
        return 50.0, details

    return (earned_weight / total_weight) * 100.0, details


def _note_prominence_score(shifted_chroma: np.ndarray, raga: dict) -> float:
    """
    Check that vadi and samvadi are among the most prominent pitch classes.
    Takes the already-shifted chroma vector (shift applied by caller).
    Returns 0-100.
    """
    vadi    = raga.get("vadi", "")
    samvadi = raga.get("samvadi", "")

    ranked_idx = np.argsort(shifted_chroma)[::-1].tolist()

    def rank_of(swara: str) -> int:
        idx = _SWARA_INDEX.get(swara)
        if idx is None:
            return 6  # unknown swara → mid-rank penalty
        return ranked_idx.index(idx) if idx in ranked_idx else 6

    def rank_score(r: int) -> float:
        if r <= 1:  return 100.0
        if r <= 3:  return 75.0
        if r <= 5:  return 40.0
        return 0.0

    return (rank_score(rank_of(vadi)) + rank_score(rank_of(samvadi))) / 2.0


def _forbidden_penalty(shifted_chroma: np.ndarray, raga: dict) -> float:
    """
    Compute penalty (0-100 raw, capped at 50) based on energy in forbidden positions.
    The caller scales this to the final point deduction.
    Takes the already-shifted chroma vector.
    """
    forbidden = raga.get("forbidden_notes", [])
    if not forbidden:
        return 0.0

    total_energy = shifted_chroma.sum()
    if total_energy < 1e-9:
        return 0.0

    forbidden_energy = sum(
        shifted_chroma[_SWARA_INDEX[s]]
        for s in forbidden
        if s in _SWARA_INDEX
    )

    ratio = forbidden_energy / total_energy
    return min(ratio * 200.0, 100.0)   # raw 0-100


def _score_raga(raga: dict, user_chroma: np.ndarray, features: dict) -> dict:
    """Compute all sub-scores for one raga and combine into overall_score."""

    profile_vec = _profile_to_vec(raga.get("chroma_profile", {}))

    # Compute chroma score once — also gives us best_shift for downstream use
    c_score, best_shift, chroma_details = _chroma_score(user_chroma, profile_vec)

    shifted_chroma = np.roll(user_chroma, -best_shift)

    p_score,   param_details = _pitch_params_score(features, raga)
    prom_score               = _note_prominence_score(shifted_chroma, raga)
    forb_raw                 = _forbidden_penalty(shifted_chroma, raga)

    # Positive weights sum to 1.0 → max positive score = 100
    positive = _W_CHROMA * c_score + _W_PITCH * p_score + _W_PROM * prom_score

    # Scale raw forbidden penalty (0-100) to a point deduction (0-_MAX_FORB_PENALTY)
    penalty = (forb_raw / 100.0) * _MAX_FORB_PENALTY

    overall = max(0.0, min(100.0, positive - penalty))

    return {
        "id":                    raga.get("id", raga.get("name", "").lower()),
        "name":                  raga.get("name", ""),
        "overall_score":         round(overall, 1),
        "chroma_score":          round(c_score, 1),
        "pitch_params_score":    round(p_score, 1),
        "note_prominence_score": round(prom_score, 1),
        "forbidden_penalty":     round(forb_raw, 1),
        "param_details":         param_details,
        "chroma_details":        chroma_details,
        "tonic_shift":           best_shift,
    }
