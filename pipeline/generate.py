"""Step 1: transcript -> SOAPNote via a single LLM call."""

import json
from typing import Protocol

from pydantic import ValidationError

from config import settings
from data.schemas import Encounter
from pipeline.schemas import SOAPNote

PROMPT_VERSION = "gen-prompt-v1"

SYSTEM_PROMPT = """\
You are a medical scribe. Convert the doctor-patient consultation transcript
into a structured SOAP note.

Use ONLY information explicitly present in the transcript.
The transcript is the sole source of truth.
Do not use external medical knowledge or reference notes.

SOAP SECTIONS:

- subjective:
  Information reported by the PATIENT, including symptoms, onset, duration,
  severity, frequency, relevant medical history, medications, allergies,
  and relevant social or family history.
  Attribute information to the patient (e.g. "Patient reports ...").
  Preserve the patient's timeframes and quantities exactly.

- objective:
  ONLY observations and measurements documented in the conversation,
  including vital signs, examination findings and test results stated aloud.
  Patient-reported symptoms are NOT objective findings.
  Do not infer examination findings from symptoms.

- assessment:
  ONLY the clinician's explicitly stated impression, diagnosis or
  clinical assessment.
  Do not independently infer or introduce a diagnosis.

- plan:
  ONLY treatment, investigations, referrals, safety-netting and
  follow-up instructions that the clinician actually states.

DOCUMENTATION QUALITY:

1. Prioritize clinically relevant information over exhaustive transcription.
2. Consolidate related symptoms and negative findings where appropriate.
3. Avoid repeating the same information across multiple statements.
4. Preserve clinically meaningful details, especially symptom duration,
   severity, frequency, negations, medication details and relevant history.
5. Do not add medication names, dosages, diagnoses or treatment
   instructions unless explicitly supported by the transcript.
6. Do not use the reference note as a source of information.
7. Use concise, professional clinical documentation language.
8. Ignore conversational filler, greetings and disfluencies without
   removing clinically meaningful information.

STRICT ACCURACY RULES:

1. Never invent or infer diagnoses, findings, medications, doses,
   results, medical history or instructions.

2. Preserve negations exactly.
   Example: "I haven't had a fever" becomes "Patient denies fever".
   Never reverse the meaning of a statement.

3. Preserve uncertainty.
   If the clinician says "could be X", document
   "Clinician considers possible X".
   Never convert a possibility into a confirmed diagnosis.

4. Absence of discussion is NOT absence of a condition.
   If allergies, medications, vitals or another topic were never discussed,
   do not document them as negative findings.
   For example, do not write "No known allergies" unless explicitly stated.

5. Only write "denies X" when the patient explicitly denies X.

6. Respect speaker attribution.
   Do not attribute a statement to the patient merely because it appears
   in the transcript.
   A clinician's question is not evidence that the patient has a symptom.
   Identify who made the statement before documenting it.

7. Do not convert patient-reported symptoms into objective findings.
   Only include examination findings, measurements or test results
   explicitly documented by a speaker.

8. If the transcript contains conflicting statements, do not silently
   resolve the conflict. Preserve the discrepancy clearly.

9. Correct obvious transcription errors only when the intended clinical
   term is unambiguous. Otherwise, preserve the original wording.

10. If a SOAP section has no documented content, return exactly:
    ["Not documented"]
    
11. Do not elaborate on or medically interpret documented information.
    Do not add possible causes, classifications, explanations or qualifiers
    unless the speaker explicitly states them.
    Preserve the level of specificity used in the conversation.

OUTPUT FORMAT:

Return ONLY a valid JSON object with exactly these four keys.
Each value must be a list of concise clinical statements.
Each statement should represent one coherent clinical fact.
Related details may be consolidated into one statement when appropriate.

{
  "subjective": ["..."],
  "objective": ["..."],
  "assessment": ["..."],
  "plan": ["..."]
}

Do not include Markdown, explanations, introductory text or conclusions.

CONSULTATION TRANSCRIPT:
{transcript}
"""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from groq import Groq

        key = api_key or settings.groq_api_key
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (add it to .env)")
        self._client = Groq(api_key=key)
        self.model = model or settings.groq_model

    def _call(self, system: str, user: str, json_mode: bool) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return r.choices[0].message.content or ""

    def complete(self, system: str, user: str) -> str:
        from groq import BadRequestError

        # Groq's strict JSON mode sometimes rejects/empties a generation (json_validate_failed).
        # Retry once in JSON mode, then fall back to plain mode (prompt still demands JSON).
        for json_mode in (True, True, False):
            try:
                return self._call(system, user, json_mode)
            except BadRequestError as e:
                if "json_validate_failed" not in str(e):
                    raise
        raise RuntimeError("unreachable")


def extract_json(raw: str) -> dict:
    """Parse a JSON object from model text, tolerating code fences / surrounding prose."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def build_user_prompt(enc: Encounter) -> str:
    return f"TRANSCRIPT:\n{enc.transcript()}\n\nReturn the SOAP note JSON."


def generate_soap(enc: Encounter, client: LLMClient | None = None, retries: int = 1) -> SOAPNote:
    """One LLM call; re-asks at most `retries` times only if the output fails validation."""
    client = client or GroqClient()
    user = build_user_prompt(enc)
    last: Exception | None = None
    for _ in range(retries + 1):
        raw = client.complete(SYSTEM_PROMPT, user)
        try:
            return SOAPNote.model_validate(extract_json(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            last = e
    raise ValueError(f"Generator output invalid for {enc.id}: {last}")
