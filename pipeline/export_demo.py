"""Export the review-dashboard data to web/demo/ so the app can be deployed without the
gitignored outputs/ and data/processed/ folders.

Usage: uv run python -m pipeline.export_demo [--verdicts DIR]

Writes one JSON per encounter: numbered transcript turns (no reference note), verifier verdicts,
and the reviewer label when one exists. Transcripts come from PriMock57 (CC BY 4.0, Babylon Health).
"""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_VERDICTS = Path("outputs/verdicts/verifier-v1.1-batched")
PROCESSED = Path("data/processed")
LABELS = Path("labels/claims.csv")
OUT = Path("web/demo")


def read_labels(path: Path, encounter_id: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {
            r["claim_id"]: r
            for r in csv.DictReader(f)
            if r["encounter_id"] == encounter_id and r["status"]
        }


def build(
    encounter_id: str, verdict_file: Path, transcript_file: Path, labels: dict[str, dict]
) -> dict:
    enc = json.loads(transcript_file.read_text())
    claims = []
    for r in json.loads(verdict_file.read_text())["records"]:
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
    # reference_note is deliberately left out: it is only a hint for omissions.
    turns = [{"id": t["id"], "speaker": t["speaker"], "text": t["text"]} for t in enc["turns"]]
    return {"id": encounter_id, "turns": turns, "claims": claims}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default=str(DEFAULT_VERDICTS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for vf in sorted(Path(a.verdicts).glob("*.json")):
        data = build(vf.stem, vf, PROCESSED / f"{vf.stem}.json", read_labels(LABELS, vf.stem))
        (OUT / f"{vf.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(f"{vf.stem}: {len(data['turns'])} turns, {len(data['claims'])} claims")


if __name__ == "__main__":
    main()
