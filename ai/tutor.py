"""
AI Raga Tutor powered by Gemini 2.5 Flash Lite.

Accepts a match result (from raga_matcher) and the extracted audio features,
then returns structured coaching feedback as a TutorResponse.
"""

import os
import json
import logging
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# ── response types ────────────────────────────────────────────────────────────

@dataclass
class Deviation:
    parameter: str
    issue: str
    suggestion: str
    severity: str  # "high" | "medium" | "low"


@dataclass
class TutorResponse:
    overall_assessment: str
    score: float                          # 0-100
    deviations: list[Deviation] = field(default_factory=list)
    positive_aspects: list[str] = field(default_factory=list)
    practice_tips: list[str] = field(default_factory=list)
    raga_context: str = ""


# ── model config ──────────────────────────────────────────────────────────────

_MODEL = "gemini-2.5-flash-lite"

_SYSTEM_PROMPT = """\
You are SvarAI, an expert teacher of Indian classical music (Hindustani tradition) \
with deep knowledge of ragas, swaras, gamakas, and vocal technique.

When given acoustic analysis data about a student's singing, you provide warm, \
precise, and constructive feedback — like a guru speaking directly to a shishya.

Your feedback MUST:
- Reference specific raga concepts (vadi, samvadi, aroha, avaroha, pakad, komal/tivra swaras, gamaka)
- Translate technical deviations into actionable practice instructions
- Be encouraging while being honest about issues
- Be concise — students learn better from focused feedback

Respond ONLY with a valid JSON object matching this schema exactly:
{
  "overall_assessment": "string (2-3 sentences summarising the performance)",
  "score": number (0-100),
  "deviations": [
    {
      "parameter": "string (which acoustic or musical parameter is off)",
      "issue": "string (what specifically is wrong)",
      "suggestion": "string (concrete practice instruction to fix it)",
      "severity": "high|medium|low"
    }
  ],
  "positive_aspects": ["string", ...],
  "practice_tips": ["string (specific exercise or technique)", ...],
  "raga_context": "string (1-2 sentences about the raga's character and what makes it distinctive)"
}
"""


# ── retry-wrapped Gemini call ─────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((ServiceUnavailable, ResourceExhausted)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _generate_with_retry(client: genai.Client, prompt: str):
    return client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.4,
        ),
    )


# ── main tutor function ───────────────────────────────────────────────────────

def get_tutor_feedback(
    raga_data: dict,
    features: dict,
    match_result: dict,
    target_raga: str | None = None,
) -> TutorResponse:
    """
    Generate AI tutor feedback using Gemini 2.5 Flash Lite.

    Args:
        raga_data:     Full raga document from Firebase.
        features:      Rich feature dict from rich_features.py.
        match_result:  Output of raga_matcher.match_raga() for this raga.
        target_raga:   Raga the student intended to sing (optional).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    user_message = _build_user_message(raga_data, features, match_result, target_raga)

    response = _generate_with_retry(client, user_message)

    raw = response.text.strip()

    # Strip markdown code fences (```json … ``` or ``` … ```)
    if raw.startswith("```"):
        # Remove opening fence line
        raw = raw[raw.index("\n") + 1:] if "\n" in raw else raw[3:]
        # Remove closing fence
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Tutor returned non-JSON response: %s", raw[:300])
        raise ValueError(f"AI tutor returned malformed JSON: {exc}") from exc

    deviations = [
        Deviation(
            parameter=d.get("parameter", ""),
            issue=d.get("issue", ""),
            suggestion=d.get("suggestion", ""),
            severity=d.get("severity", "medium"),
        )
        for d in data.get("deviations", [])
    ]

    return TutorResponse(
        overall_assessment=data.get("overall_assessment", ""),
        score=float(data.get("score", match_result.get("overall_score", 50))),
        deviations=deviations,
        positive_aspects=data.get("positive_aspects", []),
        practice_tips=data.get("practice_tips", []),
        raga_context=data.get("raga_context", ""),
    )


def _build_user_message(
    raga_data: dict,
    features: dict,
    match_result: dict,
    target_raga: str | None,
) -> str:
    raga_name = raga_data.get("name", "Unknown")
    intended = target_raga or raga_name

    param_details = []
    for name, info in match_result.get("param_details", {}).items():
        status = "✓ in range" if info.get("in_range") else f"✗ off by {info.get('deviation', '?')}"
        param_details.append(f"  {name}: value={info.get('value', '?')} — {status}")
    param_block = "\n".join(param_details) if param_details else "  (no parameter data)"

    chroma_issues = []
    for note, info in match_result.get("chroma_details", {}).items():
        if info.get("issue"):
            chroma_issues.append(f"  {note}: {info['issue']}")
    chroma_block = "\n".join(chroma_issues) if chroma_issues else "  (no notable chroma issues)"

    pitch_info = {
        k: round(features.get(k, 0), 2)
        for k in ("mean_pitch", "std_pitch", "oscillation_depth",
                  "oscillation_rate", "pitch_continuity", "pitch_drift")
        if k in features
    }

    return f"""## Student Performance Analysis

**Intended raga:** {intended}
**Best-matched raga:** {raga_name}
**Overall match score:** {match_result.get('overall_score', 0):.1f} / 100
  - Chroma similarity: {match_result.get('chroma_score', 0):.1f} / 100
  - Pitch params score: {match_result.get('pitch_params_score', 0):.1f} / 100
  - Note prominence score: {match_result.get('note_prominence_score', 0):.1f} / 100
  - Forbidden note penalty: -{match_result.get('forbidden_penalty', 0):.1f}

## Raga Reference
- Name: {raga_name}
- Aroha: {' '.join(raga_data.get('aroha', []))}
- Avaroha: {' '.join(raga_data.get('avaroha', []))}
- Vadi (dominant note): {raga_data.get('vadi', '?')}
- Samvadi (second-most-important note): {raga_data.get('samvadi', '?')}
- Pakad (characteristic phrase): {raga_data.get('pakad', '?')}
- Forbidden notes: {', '.join(raga_data.get('forbidden_notes', []))}
- Gamaka notes: {', '.join(raga_data.get('gamaka_notes', []))}
- Teacher's tip: {raga_data.get('tips', '')}

## Acoustic Features (extracted from performance)
{json.dumps(pitch_info, indent=2)}

## Parameter Match Details
{param_block}

## Chroma/Note Issues
{chroma_block}

Please evaluate this performance and give structured feedback as a raga guru."""
