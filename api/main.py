"""
SvarAI FastAPI backend — parameter-based raga identification + AI tutor.

Endpoints
─────────
Public:
  POST  /api/analyze          Upload audio → match ragas + AI coaching
  GET   /api/ragas            List all raga profiles (names + metadata)
  GET   /api/ragas/{id}       Get full raga profile by ID

Admin (x-admin-key header required):
  GET   /api/admin/ragas               List all ragas (full)
  POST  /api/admin/ragas               Create a new raga
  PUT   /api/admin/ragas/{id}          Update a raga (partial)
  DELETE/api/admin/ragas/{id}          Delete a raga
  POST  /api/admin/seed                Seed default ragas
  POST  /api/admin/login               Verify admin key
  PUT   /api/admin/change-key          Change admin key
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from pipeline.rich_features import extract_rich_features
from pipeline.raga_matcher import match_raga, get_identification_confidence
from ai.tutor import get_tutor_feedback
import db.firebase as firebase_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
CONFIG_DIR   = BASE_DIR / "config"
ADMIN_KEY_FILE = CONFIG_DIR / "admin.json"

app = FastAPI(title="SvarAI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ── Admin key helpers ─────────────────────────────────────────────────────────

def _get_admin_key() -> str:
    key = os.getenv("SVARAI_ADMIN_KEY")
    if key:
        return key
    if ADMIN_KEY_FILE.exists():
        return json.loads(ADMIN_KEY_FILE.read_text())["admin_key"]
    return "svarai-admin"


def _verify_admin(x_admin_key: Optional[str]) -> None:
    if x_admin_key != _get_admin_key():
        raise HTTPException(status_code=401, detail="Invalid admin key.")


# ── Static routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    return (FRONTEND_DIR / "admin.html").read_text(encoding="utf-8")


# ── Public: raga listing ──────────────────────────────────────────────────────

@app.get("/api/ragas")
async def list_ragas():
    """Return lightweight list (id, name, description, time, difficulty)."""
    ragas = firebase_db.get_all_ragas()
    return [
        {
            "id": r.get("id"),
            "name": r.get("name"),
            "description": r.get("description", ""),
            "time": r.get("time", ""),
            "difficulty": r.get("difficulty", ""),
            "vadi": r.get("vadi", ""),
            "samvadi": r.get("samvadi", ""),
            "aroha": r.get("aroha", []),
            "avaroha": r.get("avaroha", []),
        }
        for r in ragas
    ]


@app.get("/api/ragas/{raga_id}")
async def get_raga(raga_id: str):
    raga = firebase_db.get_raga(raga_id)
    if not raga:
        raise HTTPException(status_code=404, detail=f"Raga '{raga_id}' not found.")
    return raga


# ── Public: analyse ───────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    target_raga: Optional[str] = Form(default=None),
    include_ai_feedback: bool = Form(default=True),
):
    """
    Analyse an uploaded audio recording.

    - Extracts acoustic features.
    - Matches against all raga profiles in Firebase (parameter-based, no ML).
    - Optionally calls the Claude AI tutor for detailed coaching.
    """
    allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Accepted: {', '.join(allowed)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 1. Extract features
        features = extract_rich_features(tmp_path)
        if features is None:
            raise HTTPException(
                status_code=422,
                detail="Could not process the audio file. Ensure it is a valid audio file.",
            )

        # 2. Load raga profiles from Firebase
        raga_profiles = firebase_db.get_all_ragas()
        if not raga_profiles:
            raise HTTPException(
                status_code=503,
                detail="No raga profiles found in database. Seed the database first via /api/admin/seed.",
            )

        # 3. Parameter-based matching
        ranked = match_raga(features, raga_profiles)
        confidence = get_identification_confidence(ranked)

        # 4. AI tutor feedback
        ai_feedback = None
        feedback_raga_name = None

        if include_ai_feedback and ranked:
            if target_raga:
                feedback_raga_data = firebase_db.get_raga(
                    target_raga.lower().replace(" ", "_")
                ) or firebase_db.get_raga_by_name(target_raga)
            else:
                feedback_raga_data = next(
                    (r for r in raga_profiles if r.get("id") == ranked[0]["id"]), None
                )

            if feedback_raga_data:
                match_for_raga = next(
                    (r for r in ranked if r["id"] == feedback_raga_data.get("id")),
                    ranked[0],
                )
                try:
                    tutor = get_tutor_feedback(
                        raga_data=feedback_raga_data,
                        features=features,
                        match_result=match_for_raga,
                        target_raga=target_raga,
                    )
                    ai_feedback = {
                        "overall_assessment": tutor.overall_assessment,
                        "score": tutor.score,
                        "deviations": [
                            {
                                "parameter": d.parameter,
                                "issue": d.issue,
                                "suggestion": d.suggestion,
                                "severity": d.severity,
                            }
                            for d in tutor.deviations
                        ],
                        "positive_aspects": tutor.positive_aspects,
                        "practice_tips": tutor.practice_tips,
                        "raga_context": tutor.raga_context,
                    }
                    feedback_raga_name = feedback_raga_data.get("name")
                except Exception as exc:
                    logger.warning("AI tutor failed: %s", exc)
                    ai_feedback = {"error": str(exc)}

        return {
            "status": "success",
            "features": features,
            "ranked_matches": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "overall_score": r["overall_score"],
                    "chroma_score": r["chroma_score"],
                    "pitch_params_score": r["pitch_params_score"],
                    "note_prominence_score": r["note_prominence_score"],
                    "forbidden_penalty": r["forbidden_penalty"],
                    "param_details": r["param_details"],
                }
                for r in ranked
            ],
            "best_match": ranked[0]["name"] if ranked else None,
            "identification_confidence": confidence,
            "target_raga": target_raga,
            "ai_feedback": ai_feedback,
            "ai_feedback_raga": feedback_raga_name,
        }

    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Admin: auth ───────────────────────────────────────────────────────────────

@app.post("/api/admin/login")
async def admin_login(body: dict):
    key = body.get("key", "")
    if key != _get_admin_key():
        raise HTTPException(status_code=401, detail="Invalid admin key.")
    return {"status": "ok", "message": "Authenticated successfully."}


@app.put("/api/admin/change-key")
async def change_admin_key(
    body: dict,
    x_admin_key: Optional[str] = Header(default=None),
):
    _verify_admin(x_admin_key)
    new_key = (body.get("new_key") or "").strip()
    if len(new_key) < 8:
        raise HTTPException(status_code=400, detail="Key must be at least 8 characters.")
    ADMIN_KEY_FILE.parent.mkdir(exist_ok=True)
    ADMIN_KEY_FILE.write_text(json.dumps({"admin_key": new_key}))
    return {"status": "ok", "message": "Admin key updated."}


# ── Admin: raga CRUD ──────────────────────────────────────────────────────────

@app.get("/api/admin/ragas")
async def admin_list_ragas(x_admin_key: Optional[str] = Header(default=None)):
    _verify_admin(x_admin_key)
    return firebase_db.get_all_ragas()


@app.post("/api/admin/ragas")
async def admin_create_raga(
    body: dict,
    x_admin_key: Optional[str] = Header(default=None),
):
    _verify_admin(x_admin_key)
    try:
        created = firebase_db.create_raga(body)
        return {"status": "created", "raga": created}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/admin/ragas/{raga_id}")
async def admin_update_raga(
    raga_id: str,
    body: dict,
    x_admin_key: Optional[str] = Header(default=None),
):
    _verify_admin(x_admin_key)
    try:
        updated = firebase_db.update_raga(raga_id, body)
        return {"status": "updated", "raga": updated}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/admin/ragas/{raga_id}")
async def admin_delete_raga(
    raga_id: str,
    x_admin_key: Optional[str] = Header(default=None),
):
    _verify_admin(x_admin_key)
    try:
        firebase_db.delete_raga(raga_id)
        return {"status": "deleted", "id": raga_id}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/admin/seed")
async def admin_seed(x_admin_key: Optional[str] = Header(default=None)):
    _verify_admin(x_admin_key)
    created = firebase_db.seed_default_ragas()
    return {
        "status": "ok",
        "seeded": created,
        "message": f"Seeded {len(created)} raga(s). Existing ragas were not modified.",
    }
