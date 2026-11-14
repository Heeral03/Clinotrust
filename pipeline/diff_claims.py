"""Compare two decomposer versions for one encounter, statement by statement (read-only).

Usage: uv run python -m pipeline.diff_claims day1_consultation05 --old decomp-prompt-v2 --new decomp-prompt-v3

Old claims show the label you gave them in labels/claims.csv (matched on section + text).
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.decompose import DecomposeResult

CLAIMS = Path("outputs/claims")


def _by_statement(res: DecomposeResult) -> dict[tuple[str, int], list[str]]:
    out: dict[tuple[str, int], list[str]] = defaultdict(list)
    for c in res.claims:
        out[(c.section, c.source_statement_index)].append(c.text)
    return out


def diff(
    enc: str,
    old: str,
    new: str,
    labels: Path = Path("labels/claims.csv"),
    generated: Path = Path("outputs/generated"),
) -> str:
    load = lambda v: DecomposeResult.model_validate_json((CLAIMS / v / f"{enc}.json").read_text())
    o, n = load(old), load(new)
    soap = json.loads((generated / f"{enc}.json").read_text())["soap"]
    lab = {}
    if labels.exists():
        with open(labels, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["encounter_id"] == enc:
                    lab[(r["section"], r["claim_text"])] = r
    ob, nb = _by_statement(o), _by_statement(n)
    lines, same, changed = [], 0, 0
    for key in sorted(
        set(ob) | set(nb),
        key=lambda k: (["subjective", "objective", "assessment", "plan"].index(k[0]), k[1]),
    ):
        a, b = ob.get(key, []), nb.get(key, [])
        if Counter(a) == Counter(b):
            same += 1
            continue
        changed += 1
        lines.append(f"\n[{key[0]}#{key[1]}] SOURCE: {soap[key[0]][key[1]]}")
        for t in a:
            if t not in b:
                r = lab.get((key[0], t))
                tag = f"  (labeled: {r['status'] or '-'}/{r['error_type'] or '-'})" if r else ""
                lines.append(f"   - {t}{tag}")
        for t in b:
            if t not in a:
                lines.append(f"   + {t}")
    head = (
        f"{enc}: {old} ({len(o.claims)} claims) -> {new} ({len(n.claims)} claims)\n"
        f"{same} of {same + changed} statements unchanged; {changed} changed."
    )
    fb = "".join(f"\n   fallback [{f.section}#{f.index}] {f.reason}" for f in n.fallbacks)
    return (
        head
        + "".join(lines)
        + (f"\n\nfallbacks in {new}:{fb}" if fb else f"\n\nfallbacks in {new}: none")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("encounter")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    print(diff(a.encounter, a.old, a.new))


if __name__ == "__main__":
    main()
