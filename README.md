# Clinotrust
AI scribe that shows its evidence: generate SOAP note -> split into claims -> verify each claim
against the transcript -> clinician review. Includes eval comparing baseline vs verified notes.
Rule: anything not discussed is "not documented", never assumed absent.

## Setup
    uv sync
    cp .env.example .env   # add GROQ_API_KEY
    uv run pytest
    uv run uvicorn api.main:app --reload

## Layout
- `data/` loaders + schemas (`data/raw/` holds PriMock57 and ACI-Bench, gitignored)
- `pipeline/` generate -> decompose -> verify -> review (LangGraph)
- `eval/` metrics and `eval/labeled/` labeled claim sets
- `api/` FastAPI, `tests/`

## Data
- PriMock57: github.com/babylonhealth/primock57 (57 consultations)
- ACI-Bench: github.com/wyim/aci-bench + figshare doi 10.6084/m9.figshare.22494601
