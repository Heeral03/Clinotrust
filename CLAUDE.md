# ClinoTrust
Goal: claim-level verification of AI-written SOAP notes against consultation transcripts.
Data: PriMock57 TextGrids (data/raw is read-only). Cleaned output goes to data/processed.
Rules:
- API keys live in .env only. Never print or hard-code them.
- Every LLM output is validated with Pydantic, with one retry.
- Save every generated note with model, prompt version and loader version.
- Evidence is cited as turn IDs (T12) plus a quoted span, and the quote must appear in the transcript.
- Reference notes are hints for omissions only, not ground truth.
- Run ruff and pytest before finishing a task. One task per session.