"""Compare two verifier runs on the same claims.

Usage: uv run python -m eval.compare_runs DIR_A DIR_B

Reads only the two verdict folders (no labels, no Groq). Claims with no verdict in either folder
(missing file, missing claim, or an ERROR record) are skipped and counted.
"""

import argparse
from pathlib import Path

from eval.score import POSITIVE, load_verdicts


def compare(a: dict[tuple[str, str], dict], b: dict[tuple[str, str], dict]) -> dict:
    keys = sorted(set(a) | set(b))
    skipped = changed = worse = better = text_differs = both = 0
    rows = []
    for k in keys:
        ra, rb = a.get(k), b.get(k)
        if not ra or not rb or ra["verdict"] is None or rb["verdict"] is None:
            skipped += 1
            continue
        if ra["claim_text"] != rb["claim_text"]:
            text_differs += 1
            continue
        both += 1
        va, vb = ra["verdict"], rb["verdict"]
        if va["status"] == vb["status"]:
            continue
        changed += 1
        pa, pb = va["status"] in POSITIVE, vb["status"] in POSITIVE
        worse += (not pa) and pb  # supported -> not supported
        better += pa and (not pb)  # not supported -> supported
        rows.append((k, ra["claim_text"], va, vb))
    return {"compared": both, "skipped": skipped, "text_differs": text_differs,
            "changed": changed, "sup_to_not": worse, "not_to_sup": better, "rows": rows}  # fmt: skip


def render(r: dict, name_a: str, name_b: str) -> str:
    out = [
        f"A = {name_a}\nB = {name_b}",
        f"claims compared: {r['compared']}   skipped (no verdict in A or B): {r['skipped']}"
        + (f"   skipped (claim text differs): {r['text_differs']}" if r["text_differs"] else ""),
        f"status changed: {r['changed']}",
        f"  supported -> not supported: {r['sup_to_not']}",
        f"  not supported -> supported: {r['not_to_sup']}",
        "  other changes (both not supported, different status): "
        + str(r["changed"] - r["sup_to_not"] - r["not_to_sup"]),
    ]
    for (enc, cid), text, va, vb in r["rows"]:
        out.append(f"\n[{enc[-2:]} {cid}] {text}")
        for tag, v in (("A", va), ("B", vb)):
            out.append(
                f"  {tag}: {v['status']:<12} turns={v['evidence_turns'] or '-'} "
                f"quote={v['evidence_quote']!r}\n     {v.get('analysis') or v.get('reason', '')}"
            )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    a = ap.parse_args(argv)
    dirs = [Path(a.run_a), Path(a.run_b)]
    for d in dirs:
        if not d.is_dir() or not any(d.glob("*.json")):
            raise SystemExit(f"error: no verdict JSON files in {d}")
    print(render(compare(load_verdicts(dirs[0]), load_verdicts(dirs[1])), a.run_a, a.run_b))


if __name__ == "__main__":
    main()
