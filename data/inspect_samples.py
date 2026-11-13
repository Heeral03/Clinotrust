"""Inspect loaded datasets: where things live, speaker separation, note structure, gaps.

Usage: uv run python -m data.inspect_samples [--full primock57|aci-bench] [--n 8]
"""

import argparse
import json
import re
from collections import Counter

from data.loaders import ACI_CHALLENGE, PRIMOCK, load_aci_bench, load_primock57
from data.schemas import Encounter

HEADING = re.compile(r"^([A-Z][A-Z &/,\-]{3,})$")  # ACI-style all-caps section lines
SOAP_HINTS = {
    "S": ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS", "SUBJECTIVE"],
    "O": ["PHYSICAL EXAM", "RESULTS", "OBJECTIVE", "VITALS"],
    "A/P": ["ASSESSMENT", "PLAN"],
}


def headings(note: str) -> list[str]:
    return [l.strip() for l in note.splitlines() if HEADING.match(l.strip())]


def soap_coverage(note: str) -> dict[str, bool]:
    up = note.upper()
    return {k: any(h in up for h in hs) for k, hs in SOAP_HINTS.items()}


def where(e: Encounter) -> str:
    if e.source == "primock57":
        return (
            f"dialogue: {PRIMOCK}/transcripts/{e.id}_{{doctor,patient}}.TextGrid (+audio/)\n"
            f"           note: {PRIMOCK}/notes/{e.id}.json  (key 'note')"
        )
    return (
        f"dialogue+note: {ACI_CHALLENGE}/{e.meta.get('split')}.csv  "
        f"(columns dialogue, note; row encounter_id={e.id.split('-')[-1]})"
    )


def summarize(e: Encounter) -> None:
    speakers = Counter(t.speaker for t in e.turns)
    cov = soap_coverage(e.reference_note)
    hs = headings(e.reference_note)
    print(f"\n=== {e.id}  [{e.source}] ===")
    print(f"  where    : {where(e)}")
    print(
        f"  speakers : {dict(speakers)}  -> speaker-separated: {len(speakers) >= 2}"
        f"{'  (timestamps)' if e.turns[0].start is not None else '  (no timestamps)'}"
    )
    print(
        f"  turns    : {len(e.turns)}   words(dialogue): {sum(len(t.text.split()) for t in e.turns)}"
        f"   words(note): {len(e.reference_note.split())}"
    )
    print(
        f"  note fmt : {'sectioned: ' + '; '.join(hs) if hs else 'free text, no section headings'}"
    )
    print(f"  SOAP hit : {cov}")
    issues = []
    if not e.reference_note.strip():
        issues.append("empty note")
    if any(not t.text for t in e.turns):
        issues.append("empty turn")
    if set(Counter(t.speaker for t in e.turns)) - {"doctor", "patient"}:
        issues.append(f"odd speakers {set(Counter(t.speaker for t in e.turns))}")
    print(f"  issues   : {issues or 'none'}")
    print(f"  meta     : {{{', '.join(f'{k}={str(v)[:40]!r}' for k, v in e.meta.items())}}}")


def full(e: Encounter) -> None:
    print("\n" + "#" * 78 + f"\n# FULL EXAMPLE: {e.id} ({e.source})\n" + "#" * 78)
    print("\n--- META ---\n" + json.dumps(e.meta, indent=2, default=str)[:1500])
    print("\n--- DIALOGUE (structured turns) ---")
    for i, t in enumerate(e.turns):
        ts = f"{t.start:7.1f}s " if t.start is not None else ""
        print(f"{i:3d} {ts}{t.speaker:8s}| {t.text}")
    print("\n--- REFERENCE NOTE ---\n" + e.reference_note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--full", choices=["primock57", "aci-bench"], default=None)
    a = ap.parse_args()
    pm, aci = load_primock57(), load_aci_bench(("train", "valid"))
    half = a.n // 2
    samples = pm[:: max(1, len(pm) // half)][:half] + aci[:: max(1, len(aci) // half)][:half]
    for e in samples:
        summarize(e)

    print("\n\n##### DATASET-WIDE CHECKS #####")
    for name, ds in (("primock57", pm), ("aci-bench", aci)):
        with_h = sum(bool(headings(e.reference_note)) for e in ds)
        full_soap = sum(all(soap_coverage(e.reference_note).values()) for e in ds)
        print(f"{name}: n={len(ds)}  sectioned-notes={with_h}  has S+O+A/P cues={full_soap}")
        print(f"   speakers seen: {dict(Counter(t.speaker for e in ds for t in e.turns))}")
        hc = Counter(h for e in ds for h in headings(e.reference_note))
        if hc:
            print(f"   top headings: {hc.most_common(12)}")
    for e in (pm[0], aci[0]):
        if a.full in (None, e.source):
            full(e)


if __name__ == "__main__":
    main()
