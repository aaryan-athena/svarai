"""
Firebase Firestore client for raga parameter CRUD operations.
Initialises once via a service-account JSON file whose path is read from
FIREBASE_SERVICE_ACCOUNT_PATH, or falls back to FIREBASE_CREDENTIALS_JSON
(the raw JSON string, useful for hosting environments).
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

# ── singleton initialisation ──────────────────────────────────────────────────
_db: Optional[firestore.Client] = None


def _init_firebase() -> firestore.Client:
    global _db
    if _db is not None:
        return _db

    if firebase_admin._apps:
        _db = firestore.client()
        return _db

    sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    sa_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

    if sa_path and os.path.exists(sa_path):
        cred = credentials.Certificate(sa_path)
    elif sa_json:
        cred = credentials.Certificate(json.loads(sa_json))
    else:
        raise RuntimeError(
            "Firebase credentials not found. Set FIREBASE_SERVICE_ACCOUNT_PATH "
            "or FIREBASE_CREDENTIALS_JSON environment variable."
        )

    firebase_admin.initialize_app(cred)
    _db = firestore.client()
    logger.info("Firebase initialised successfully.")
    return _db


def get_db() -> firestore.Client:
    return _init_firebase()


# ── collection helpers ────────────────────────────────────────────────────────
RAGAS_COLLECTION = "ragas"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── READ ──────────────────────────────────────────────────────────────────────

def get_all_ragas() -> list[dict]:
    """Return all raga documents as a list."""
    db = get_db()
    docs = db.collection(RAGAS_COLLECTION).stream()
    result = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        result.append(data)
    return result


def get_raga(raga_id: str) -> Optional[dict]:
    """Return a single raga by its document ID (lowercase raga name)."""
    db = get_db()
    doc = db.collection(RAGAS_COLLECTION).document(raga_id).get()
    if doc.exists:
        data = doc.to_dict()
        data["id"] = doc.id
        return data
    return None


def get_raga_by_name(name: str) -> Optional[dict]:
    """Case-insensitive lookup by the `name` field."""
    db = get_db()
    docs = (
        db.collection(RAGAS_COLLECTION)
        .where("name_lower", "==", name.lower())
        .limit(1)
        .stream()
    )
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        return data
    return None


# ── CREATE ────────────────────────────────────────────────────────────────────

def create_raga(data: dict) -> dict:
    """
    Create a new raga document.  The document ID is derived from the name.
    Returns the saved document with its `id` field populated.
    """
    db = get_db()
    name = data.get("name", "").strip()
    if not name:
        raise ValueError("Raga `name` is required.")

    doc_id = name.lower().replace(" ", "_")

    if db.collection(RAGAS_COLLECTION).document(doc_id).get().exists:
        raise ValueError(f"Raga '{name}' already exists (id='{doc_id}').")

    payload = {
        **data,
        "name_lower": name.lower(),
        "created_at": _now(),
        "updated_at": _now(),
    }
    db.collection(RAGAS_COLLECTION).document(doc_id).set(payload)
    payload["id"] = doc_id
    return payload


# ── UPDATE ────────────────────────────────────────────────────────────────────

def update_raga(raga_id: str, data: dict) -> dict:
    """
    Partial-update a raga document.  Timestamps and name_lower are managed
    automatically; pass only the fields you want to change.
    Returns the full updated document.
    """
    db = get_db()
    ref = db.collection(RAGAS_COLLECTION).document(raga_id)
    if not ref.get().exists:
        raise ValueError(f"Raga '{raga_id}' not found.")

    # Keep name_lower in sync if name was changed
    payload = {k: v for k, v in data.items() if k not in ("id", "created_at")}
    if "name" in payload:
        payload["name_lower"] = payload["name"].lower()
    payload["updated_at"] = _now()

    ref.update(payload)
    updated = ref.get().to_dict()
    updated["id"] = raga_id
    return updated


# ── DELETE ────────────────────────────────────────────────────────────────────

def delete_raga(raga_id: str) -> None:
    """Delete a raga document by ID."""
    db = get_db()
    ref = db.collection(RAGAS_COLLECTION).document(raga_id)
    if not ref.get().exists:
        raise ValueError(f"Raga '{raga_id}' not found.")
    ref.delete()


# ── SEED ─────────────────────────────────────────────────────────────────────

def seed_default_ragas() -> list[str]:
    """
    Insert the three default ragas (Yaman, Asavari, Sarang) if they don't
    already exist.  Returns a list of IDs that were actually created.
    """
    created = []
    for raga in _DEFAULT_RAGAS:
        doc_id = raga["name"].lower().replace(" ", "_")
        db = get_db()
        if not db.collection(RAGAS_COLLECTION).document(doc_id).get().exists:
            create_raga(raga)
            created.append(doc_id)
            logger.info("Seeded raga: %s", doc_id)
    return created


# ── default raga definitions ──────────────────────────────────────────────────
# chroma_profile: expected relative prominence of each pitch class (0-1)
# when the raga is transposed to C as Sa.
# Index 0=C(Sa), 1=C#(r/komal Re), 2=D(R), 3=D#(g), 4=E(G),
#       5=F(m), 6=F#(M#/tivra Ma), 7=G(Pa), 8=G#(d), 9=A(D), 10=A#(n), 11=B(N)

_DEFAULT_RAGAS = [
    {
        "name": "Yaman",
        "description": "Evening raga of the Kalyan thaat. Serene and majestic with a characteristic tivra (sharp) Madhyam.",
        "time": "First quarter of night (6 PM – 9 PM)",
        "season": "All seasons",
        "aroha": ["S", "R", "G", "M#", "D", "N", "S'"],
        "avaroha": ["S'", "N", "D", "P", "M#", "G", "R", "S"],
        "vadi": "G",
        "samvadi": "N",
        "pakad": "N R G M# D N S'",
        "forbidden_notes": ["m", "r", "g", "d", "n"],
        "gamaka_notes": ["G", "N"],
        "characteristic_phrases": ["N R G M# D", "G M# D N S'", "R G M# P D N S'"],
        "chroma_profile": {
            "C": 0.9,   # Sa
            "C#": 0.0,  # komal Re — absent
            "D": 0.7,   # Re
            "D#": 0.0,  # komal Ga — absent
            "E": 0.85,  # Ga (vadi)
            "F": 0.0,   # shuddha Ma — absent
            "F#": 0.75, # tivra Ma
            "G": 0.65,  # Pa
            "G#": 0.0,  # komal Dha — absent
            "A": 0.7,   # Dha
            "A#": 0.0,  # komal Ni — absent
            "B": 0.8    # Ni (samvadi)
        },
        "pitch_params": {
            "mean_pitch": {"min": 180, "max": 520, "label": "Mean Pitch", "unit": "Hz"},
            "std_pitch": {"min": 30, "max": 220, "label": "Pitch Range (std)", "unit": "Hz"},
            "oscillation_depth": {"min": 40, "max": 450, "label": "Gamaka Depth", "unit": "Hz"},
            "oscillation_rate": {"min": 1.0, "max": 7.0, "label": "Ornament Rate", "unit": "Hz"},
            "pitch_continuity": {"min": 0.55, "max": 1.0, "label": "Pitch Continuity", "unit": "ratio"},
            "pitch_drift": {"min": -60, "max": 60, "label": "Pitch Drift", "unit": "Hz/s"},
            "spectral_centroid": {"min": 800, "max": 3500, "label": "Spectral Centroid", "unit": "Hz"}
        },
        "tips": "Emphasise the tivra Madhyam (F#). The raga has a characteristic 'N R G M#' ascent — avoid the komal varieties of all swaras. Ga and Ni both benefit from gentle meend (glide).",
        "difficulty": "beginner"
    },
    {
        "name": "Asavari",
        "description": "Morning raga with a serious, pensive character. Uses three komal (flat) swaras: Ga, Dha, and Ni.",
        "time": "Late morning (9 AM – noon)",
        "season": "All seasons",
        "aroha": ["S", "R", "m", "P", "d", "S'"],
        "avaroha": ["S'", "n", "d", "P", "m", "g", "R", "S"],
        "vadi": "d",
        "samvadi": "g",
        "pakad": "S R m P d m P g R S",
        "forbidden_notes": ["G", "D", "N"],
        "gamaka_notes": ["g", "d", "n"],
        "characteristic_phrases": ["S R m P d m P", "g R S", "d m P g R"],
        "chroma_profile": {
            "C": 0.9,   # Sa
            "C#": 0.0,  # komal Re — absent
            "D": 0.7,   # Re
            "D#": 0.8,  # komal Ga (samvadi)
            "E": 0.0,   # shuddha Ga — absent
            "F": 0.75,  # shuddha Ma
            "F#": 0.0,  # tivra Ma — absent
            "G": 0.7,   # Pa
            "G#": 0.85, # komal Dha (vadi)
            "A": 0.0,   # shuddha Dha — absent
            "A#": 0.75, # komal Ni
            "B": 0.0    # shuddha Ni — absent
        },
        "pitch_params": {
            "mean_pitch": {"min": 160, "max": 500, "label": "Mean Pitch", "unit": "Hz"},
            "std_pitch": {"min": 30, "max": 200, "label": "Pitch Range (std)", "unit": "Hz"},
            "oscillation_depth": {"min": 30, "max": 380, "label": "Gamaka Depth", "unit": "Hz"},
            "oscillation_rate": {"min": 0.5, "max": 5.0, "label": "Ornament Rate", "unit": "Hz"},
            "pitch_continuity": {"min": 0.50, "max": 1.0, "label": "Pitch Continuity", "unit": "ratio"},
            "pitch_drift": {"min": -55, "max": 55, "label": "Pitch Drift", "unit": "Hz/s"},
            "spectral_centroid": {"min": 700, "max": 3200, "label": "Spectral Centroid", "unit": "Hz"}
        },
        "tips": "The three komal swaras (Ga, Dha, Ni) define this raga. Komal Dha is the vadi — give it weight and time. Avoid all shuddha versions of Ga, Dha, and Ni.",
        "difficulty": "intermediate"
    },
    {
        "name": "Sarang",
        "description": "Midday raga of the Kafi thaat. Bright and playful with a characteristic movement around Pa and komal Ni.",
        "time": "Midday (noon – 3 PM)",
        "season": "Rainy season (Varsha)",
        "aroha": ["S", "R", "m", "P", "n", "S'"],
        "avaroha": ["S'", "n", "P", "m", "R", "S"],
        "vadi": "R",
        "samvadi": "P",
        "pakad": "R m P n P m R S",
        "forbidden_notes": ["G", "g", "D", "d", "N", "M#"],
        "gamaka_notes": ["R", "m", "n"],
        "characteristic_phrases": ["R m P n P", "m P n S'", "P m R S"],
        "chroma_profile": {
            "C": 0.85,  # Sa
            "C#": 0.0,  # komal Re — absent
            "D": 0.9,   # Re (vadi)
            "D#": 0.0,  # komal Ga — absent
            "E": 0.0,   # shuddha Ga — absent
            "F": 0.75,  # shuddha Ma
            "F#": 0.0,  # tivra Ma — absent
            "G": 0.85,  # Pa (samvadi)
            "G#": 0.0,  # komal Dha — absent
            "A": 0.0,   # shuddha Dha — absent
            "A#": 0.8,  # komal Ni
            "B": 0.0    # shuddha Ni — absent
        },
        "pitch_params": {
            "mean_pitch": {"min": 170, "max": 510, "label": "Mean Pitch", "unit": "Hz"},
            "std_pitch": {"min": 25, "max": 190, "label": "Pitch Range (std)", "unit": "Hz"},
            "oscillation_depth": {"min": 30, "max": 360, "label": "Gamaka Depth", "unit": "Hz"},
            "oscillation_rate": {"min": 1.0, "max": 6.0, "label": "Ornament Rate", "unit": "Hz"},
            "pitch_continuity": {"min": 0.55, "max": 1.0, "label": "Pitch Continuity", "unit": "ratio"},
            "pitch_drift": {"min": -50, "max": 50, "label": "Pitch Drift", "unit": "Hz/s"},
            "spectral_centroid": {"min": 750, "max": 3300, "label": "Spectral Centroid", "unit": "Hz"}
        },
        "tips": "The movement 'R m P n P' is the heart of Sarang. Komal Ni should be sung lightly and quickly — don't dwell on it. Re (vadi) should be emphasised with meend from Sa.",
        "difficulty": "intermediate"
    }
]
