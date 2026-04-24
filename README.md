# SvarAI — AI Raga Tutor & Pitch-Accuracy Coach

A web-based platform for Indian classical music practice. Upload or record a performance and SvarAI identifies the raga using parameter-based acoustic analysis, then delivers real-time coaching feedback from an AI guru powered by Gemini 2.5 Flash Lite.

**No ML model training required.** Raga profiles are stored in Firebase and matched against your audio using hand-crafted acoustic scoring — meaning the system works immediately with any raga profile you add through the admin panel.

---

## Features

- **Parameter-based raga identification** — scores audio against chroma profile, pitch parameters, note prominence, and forbidden-note penalties
- **AI tutor feedback** — Gemini 2.5 Flash Lite acts as a guru, giving per-deviation suggestions, practice tips, and raga context
- **Firebase-backed raga profiles** — full CRUD from an admin panel; add or tune any raga without touching code
- **Live recording** — record directly in the browser with real-time waveform visualisation
- **Pitch coaching** — oscillation depth, gamaka rate, pitch continuity, and drift shown alongside expected ranges

---

## Project Structure

```
svarai/
├── pipeline/
│   ├── config.py             # audio & feature-extraction constants
│   ├── rich_features.py      # audio → labeled feature dict (pitch, chroma, spectral)
│   └── raga_matcher.py       # parameter-based scoring engine (no ML)
├── ai/
│   └── tutor.py              # Gemini 2.5 Flash Lite coaching, with retry backoff
├── api/
│   └── main.py               # FastAPI backend — analyze + raga CRUD endpoints
├── db/
│   └── firebase.py           # Firestore client (get / create / update / delete ragas)
├── frontend/
│   ├── index.html            # Practice page
│   ├── admin.html            # Admin panel
│   └── static/
│       ├── css/style.css
│       └── js/
│           ├── app.js
│           └── admin.js
├── seed_firebase.py          # One-shot script to seed default raga profiles
├── run.py                    # Uvicorn launcher
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- A Firebase project with Firestore enabled
- A Google Gemini API key (free tier works with Gemini 2.5 Flash Lite)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```env
# Firebase — either paste the JSON content or point to the file
FIREBASE_CREDENTIALS={"type":"service_account", ...}
# or
FIREBASE_CREDENTIALS_PATH=path/to/serviceAccountKey.json

# Gemini API
GEMINI_API_KEY=AIza...

# Optional: override the default admin key (default: svarai-admin)
SVARAI_ADMIN_KEY=your-secret-key
```

### 3. Seed default raga profiles

```bash
python seed_firebase.py
```

This creates Yaman, Asavari, and Sarang in Firestore. Existing ragas are left untouched. You can add more ragas (or edit these) from the admin panel at `/admin`.

### 4. Start the server

```bash
python run.py
```

Open `http://127.0.0.1:8000` in a browser.

Optional flags:

```bash
python run.py --host 0.0.0.0 --port 8080 --reload
```

---

## How Raga Identification Works

No trained model. Each raga profile stored in Firebase defines:

| Field | Purpose |
|---|---|
| `chroma_profile` | Expected prominence (0–1) of each of the 12 pitch classes |
| `pitch_params` | Expected min/max ranges for acoustic parameters |
| `vadi` / `samvadi` | Dominant and second-dominant swaras |
| `forbidden_notes` | Notes that must not appear |

For each uploaded recording, four sub-scores are computed and combined:

```
Overall = 0.45 × chroma_score
        + 0.30 × pitch_params_score
        + 0.25 × note_prominence_score
        − forbidden_penalty   (up to 20 pts)
```

| Component | How it works |
|---|---|
| **Chroma score** (45%) | Cosine similarity between the audio's 12-bin chroma vector and the raga's profile; all 12 tonic shifts are tried so recordings in any key work |
| **Pitch params score** (30%) | Fraction of acoustic parameters (mean pitch, pitch std, gamaka depth, ornament rate, continuity, drift, spectral centroid) that fall within the raga's expected ranges; partial credit for near-misses |
| **Note prominence** (25%) | Are the vadi and samvadi among the most energetic pitch classes? |
| **Forbidden penalty** (−0 to 20 pts) | Energy present in forbidden-note positions is penalised |

The raga with the highest overall score is the best match.

---

## AI Tutor

After identification, the top match (or the raga you specified) is passed to Gemini 2.5 Flash Lite along with:

- The full raga reference (aroha, avaroha, vadi, samvadi, pakad, gamaka notes, tips)
- The extracted acoustic features
- Per-parameter deviation details from the matcher

Gemini returns structured JSON with:

- **Overall assessment** — 2–3 sentence summary
- **Score** — 0–100
- **Deviations** — per-parameter issues with concrete fix suggestions and severity (high / medium / low)
- **Positive aspects** — what you did well
- **Practice tips** — specific exercises
- **Raga context** — what makes this raga distinctive

The tutor uses automatic retry with exponential backoff (up to 4 attempts) to handle transient 503/429 errors from the API.

---

## Admin Panel

Navigate to `/admin` and log in with your admin key.

From here you can:

- **Add / Edit / Delete** raga profiles
- Set the chroma profile (12 pitch-class weights)
- Define pitch parameter ranges (min/max for any acoustic feature)
- **Seed defaults** to restore Yaman, Asavari, and Sarang

All changes take effect immediately — no server restart needed.

---

## API Endpoints

### Public

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/analyze` | Upload audio, get ranked matches + AI feedback |
| `GET` | `/api/ragas` | List all raga profiles (lightweight) |
| `GET` | `/api/ragas/{id}` | Get full raga profile |

`POST /api/analyze` accepts `multipart/form-data`:

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | audio file | ✓ | WAV, MP3, OGG, FLAC, or M4A |
| `target_raga` | string | — | Raga you intended to sing (for focused AI feedback) |
| `include_ai_feedback` | bool | — | Default `true`; set `false` to skip Gemini call |

### Admin (requires `x-admin-key` header)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/admin/login` | Verify admin key |
| `PUT` | `/api/admin/change-key` | Update admin key |
| `GET` | `/api/admin/ragas` | List all ragas (full) |
| `POST` | `/api/admin/ragas` | Create a raga |
| `PUT` | `/api/admin/ragas/{id}` | Update a raga |
| `DELETE` | `/api/admin/ragas/{id}` | Delete a raga |
| `POST` | `/api/admin/seed` | Seed default ragas |

---

## Configuration

`pipeline/config.py` holds audio and feature-extraction constants:

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `22050` | Hz — audio resampling target |
| `SEGMENT_SEC` | `30` | Analysis window length in seconds |
| `SEGMENT_HOP_SEC` | `15` | Step between windows (50% overlap) |
| `N_MFCC` | `20` | MFCC coefficients extracted |
| `N_CHROMA` | `12` | Chroma bins (one per semitone) |
| `PITCH_FMIN` | `60.0` | Hz — lower pitch bound for pyin |
| `PITCH_FMAX` | `1000.0` | Hz — upper pitch bound for pyin |

Scoring weights and the forbidden-note penalty ceiling live in `pipeline/raga_matcher.py`:

```python
_W_CHROMA = 0.45
_W_PITCH  = 0.30
_W_PROM   = 0.25
_MAX_FORB_PENALTY = 20.0
```

---

## Requirements

- Python 3.10+
- librosa, numpy, scipy, soundfile
- fastapi, uvicorn, python-multipart, aiofiles
- firebase-admin
- google-genai
- python-dotenv

Install all with:

```bash
pip install -r requirements.txt
```
