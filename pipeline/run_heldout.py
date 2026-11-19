"""Prepare the HELD-OUT set: transcripts, SOAP notes and unlabeled claims. No verifier, no labels.

Usage: uv run python -m pipeline.run_heldout [--ids day3_consultation03 ...]

Writes only under heldout/. Resumable: existing notes / claims JSON are reused. Never reads or
writes labels/ (only the header constant) or outputs/verdicts/ (read-only ledger summary).
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from config import settings
from data.loaders import load_primock57
from data.schemas import LOADER_VERSION
from pipeline import run_generate
from pipeline.decompose import (
    DECOMPOSE_PROMPT_VERSION,
    MAX_TOKENS,
    REASONING_EFFORT,
    DecomposeResult,
    decompose,
    load_generated,
)
from pipeline.generate import PROMPT_VERSION
from pipeline.labels import CLAIMS_HEADER, LABEL_COLUMNS

IDS = [f"day3_consultation{n}" for n in ("03", "04", "05", "08", "10")]
OUT = Path("heldout")
usage = {"prompt": 0, "completion": 0, "calls": 0}


def count_tokens() -> None:
    """Sum Groq usage over every chat call (GroqClient does not track it)."""
    from groq.resources.chat.completions import Completions

    orig = Completions.create

    def create(self, *a, **kw):
        r = orig(self, *a, **kw)
        u = getattr(r, "usage", None)
        usage["calls"] += 1
        if u:
            usage["prompt"] += u.prompt_tokens or 0
            usage["completion"] += u.completion_tokens or 0
        return r

    Completions.create = create


def write_transcripts(ids: list[str]) -> None:
    (OUT / "transcripts").mkdir(parents=True, exist_ok=True)
    for enc in load_primock57():
        if enc.id in ids:  # same loader + same line format the verifier reads
            (OUT / "transcripts" / f"{enc.id}.txt").write_text(
                enc.transcript(with_times=True) + "\n", encoding="utf-8"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=IDS)
    a = ap.parse_args()
    from groq import RateLimitError

    count_tokens()
    write_transcripts(a.ids)
    soap_dir, claims_dir = OUT / "soap", OUT / "claims" / DECOMPOSE_PROMPT_VERSION
    claims_dir.mkdir(parents=True, exist_ok=True)
    skipped, fallbacks = [], []
    try:
        sys.argv = ["run_generate", "--ids", *a.ids, "--out", str(soap_dir)]
        run_generate.main()
        results: list[DecomposeResult] = []
        for i in a.ids:
            note, out = load_generated(soap_dir / f"{i}.json"), claims_dir / f"{i}.json"
            if out.exists():
                res = DecomposeResult.model_validate_json(out.read_text(encoding="utf-8"))
                skipped.append(f"{i}: claims JSON reused")
            else:
                res = decompose(note)
                out.write_text(res.model_dump_json(indent=2), encoding="utf-8")
            fallbacks += [f"{i} [{f.section}#{f.index}] {f.reason}" for f in res.fallbacks]
            results.append(res)
    except RateLimitError as e:
        print(f"\nSTOPPED by Groq rate limit: {e}\nRe-run the same command to resume.")
        raise SystemExit(1) from e

    rows = [
        {"encounter_id": c.encounter_id, "claim_id": c.id, "section": c.section,
         "source_statement_index": c.source_statement_index, "claim_text": c.text,
         **dict.fromkeys(LABEL_COLUMNS, "")}
        for r in results for c in r.claims
    ]  # fmt: skip
    with (OUT / "claims_unlabeled.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLAIMS_HEADER)
        w.writeheader()
        w.writerows(rows)
    gen = {i: json.loads((soap_dir / f"{i}.json").read_text()) for i in a.ids}
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "encounters": a.ids,
        "soap": {"model": settings.groq_model, "prompt_version": PROMPT_VERSION,
                 "temperature": 0, "loader_version": LOADER_VERSION,
                 "models_in_saved_notes": sorted({g["model"] for g in gen.values()})},
        "decomposer": {"prompt_version": DECOMPOSE_PROMPT_VERSION, "model": settings.groq_model,
                       "temperature": 0, "reasoning_effort": REASONING_EFFORT, "max_tokens": MAX_TOKENS},
        "groq_tokens_this_invocation": usage | {"note": "counts only calls made in this run"},
        "claims_per_encounter": {r.encounter_id: len(r.claims) for r in results},
        "fallbacks": fallbacks,
        "verifier_run": False, "labels_touched": False,
    }  # fmt: skip
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"\nwrote {len(rows)} claims; tokens {usage}; skipped/reused: {skipped}; fallbacks: {fallbacks}"
    )


if __name__ == "__main__":
    main()
