"""Decompose generated notes into claims, then update labels/claims.csv.

Usage: uv run python -m pipeline.run_decompose [--generated outputs/generated] [--force]

- Claims JSON goes to outputs/claims/{decomposer_prompt_version}/{encounter_id}.json, so a new
  prompt version never overwrites an old one.
- Encounters that already have rows in labels/claims.csv are NOT re-decomposed (ids are frozen).
"""

import argparse
from pathlib import Path

from pipeline.decompose import DECOMPOSE_PROMPT_VERSION, DecomposeResult, decompose, load_generated
from pipeline.labels import LABELS_DIR, ensure_omissions, labeled_encounters, merge_claims

CLAIMS = Path("outputs/claims")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default="outputs/generated")
    ap.add_argument("--force", action="store_true", help="re-decompose even if the JSON exists")
    ap.add_argument("--only", nargs="+", metavar="ID", help="only these encounter ids")
    ap.add_argument(
        "--allow-labeled",
        action="store_true",
        help="decompose --only encounters even though they already have labeled rows",
    )
    ap.add_argument(
        "--no-labels",
        action="store_true",
        help="write claims JSON only; do not read or touch labels/ at all",
    )
    a = ap.parse_args()
    if a.allow_labeled and not (a.only and a.no_labels):
        ap.error("--allow-labeled requires --only and --no-labels (labeled ids stay frozen)")
    out_dir = CLAIMS / DECOMPOSE_PROMPT_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = set() if a.no_labels else labeled_encounters(LABELS_DIR / "claims.csv")
    results: list[DecomposeResult] = []
    for f in sorted(Path(a.generated).glob("*.json")):
        note = load_generated(f)
        if a.only and note["id"] not in a.only:
            continue
        out = out_dir / f"{note['id']}.json"
        if note["id"] in frozen:
            print(f"{note['id']}: already in labels/claims.csv - skipped (ids frozen)")
            continue
        if out.exists() and not a.force:
            res = DecomposeResult.model_validate_json(out.read_text(encoding="utf-8"))
            print(f"{note['id']}: reusing {out}")
        else:
            res = decompose(note)
            out.write_text(res.model_dump_json(indent=2), encoding="utf-8")
            print(
                f"{note['id']}: {res.n_statements} statements "
                f"({res.n_passthrough} pass-through, {res.n_llm} via LLM) -> {len(res.claims)} claims"
                f", {len(res.fallbacks)} fallbacks"
            )
            for fb in res.fallbacks:
                print(f"    fallback [{fb.section}#{fb.index}] {fb.reason}: {fb.statement}")
        results.append(res)
    if a.no_labels:
        print("\n--no-labels: labels/ was not read or modified.")
        return
    added, kept, warnings = merge_claims(results, LABELS_DIR / "claims.csv")
    ensure_omissions(LABELS_DIR / "omissions.csv")
    print(f"\nlabels/claims.csv: +{added} rows ({kept} existing kept)")
    for w in warnings:
        print("  WARNING:", w)


if __name__ == "__main__":
    main()
