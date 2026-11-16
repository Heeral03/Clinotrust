"""ClinoTrust API + web UI.

GET /                      landing page
GET /app                   claim review dashboard
GET /api/encounters        encounters that have verifier verdicts
GET /api/encounters/{id}   transcript + verdicts (+ reviewer labels when labelled)
Data is read from web/demo/ (committed); rebuild it with `uv run python -m pipeline.export_demo`.
"""

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DEMO = WEB / "demo"  # built by `python -m pipeline.export_demo`
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


@app.get("/api/encounters")
def encounters():
    out = []
    for f in sorted(DEMO.glob("*.json")):
        claims = json.loads(f.read_text())["claims"]
        out.append(
            {
                "id": f.stem,
                "claims": len(claims),
                "needs_review": sum(c["needs_review"] for c in claims),
            }
        )
    return out


@app.get("/api/encounters/{encounter_id}")
def encounter(encounter_id: str):
    f = DEMO / f"{encounter_id}.json"
    if not ID_RE.match(encounter_id) or not f.exists():
        raise HTTPException(404, "unknown encounter")
    return json.loads(f.read_text())
