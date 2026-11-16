"""ClinoTrust API + web UI.

GET /                      landing page
GET /app                   claim review dashboard
GET /api/encounters        encounters that have verifier verdicts
GET /api/encounters/{id}   transcript + verdicts (+ reviewer labels when labelled)
"""

import csv
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PROCESSED = ROOT / "data" / "processed"
VERDICTS = ROOT / "outputs" / "verdicts" / "verifier-v1.1-batched"
LABELS = ROOT / "labels" / "claims.csv"
ID_RE = re.compile(r"^day\d+_consultation\d+$")

app = FastAPI(title="Clinotrust")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def landing():
    return FileResponse(WEB / "index.html")


@app.get("/app", include_in_schema=False)
def dashboard():
    return FileResponse(WEB / "app.html")


def _labels(encounter_id: str) -> dict[str, dict]:
    if not LABELS.exists():
        return {}
    with LABELS.open(newline="") as f:
        return {
            r["claim_id"]: r
            for r in csv.DictReader(f)
            if r["encounter_id"] == encounter_id and r["status"]
        }


@app.get("/api/encounters")
def encounters():
    out = []
    for f in sorted(VERDICTS.glob("*.json")):
        recs = json.loads(f.read_text())["records"]
        flagged = sum(1 for r in recs if r["needs_review"])
        out.append({"id": f.stem, "claims": len(recs), "needs_review": flagged})
    return out


@app.get("/api/encounters/{encounter_id}")
def encounter(encounter_id: str):
    vf, tf = VERDICTS / f"{encounter_id}.json", PROCESSED / f"{encounter_id}.json"
    if not ID_RE.match(encounter_id) or not vf.exists() or not tf.exists():
        raise HTTPException(404, "unknown encounter")
    enc = json.loads(tf.read_text())
    labels = _labels(encounter_id)
    claims = []
    for r in json.loads(vf.read_text())["records"]:
        v = r["verdict"] or {}
        lab = labels.get(r["claim_id"])
        claims.append(
            {
                "id": r["claim_id"],
                "section": r["section"],
                "text": r["claim_text"],
                "status": v.get("status", "uncertain"),
                "analysis": v.get("analysis", ""),
                "evidence_turns": v.get("evidence_turns", []),
                "quote": v.get("evidence_quote", ""),
                "quote_valid": r["quote_valid"],
                "needs_review": r["needs_review"],
                "error": r["error"],
                "reviewer": {"status": lab["status"], "error_type": lab["error_type"]}
                if lab
                else None,
            }
        )
    # reference_note is deliberately not exposed: it is only a hint for omissions.
    turns = [{"id": t["id"], "speaker": t["speaker"], "text": t["text"]} for t in enc["turns"]]
    return {"id": encounter_id, "turns": turns, "claims": claims}
