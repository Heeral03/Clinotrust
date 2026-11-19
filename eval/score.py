"""Score verifier verdicts against the hand labels in labels/claims.csv.

Usage: uv run python -m eval.score [--verdicts DIR] [--no-disagreements]

Positive class = "not supported" = contradicted + unsupported + uncertain (all three together).
Reported twice: all rows, and excluding rows whose error_type is decomposition_error.
Two detection rules are scored side by side:
  status        flag a claim when status != supported
  needs_review  flag it when status != supported OR quote_valid is False
Also: recall per error_type, single-vs-batched agreement, and a v0-vs-v1 prompt comparison.
"""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.labels import CLAIMS_HEADER, check_header
from pipeline.verify import (
    BATCH_VERDICTS_DIR,
    V1_BATCH_VERDICTS_DIR,
    VERDICTS_DIR,
    VerdictFile,
    needs_review_of,
)

STATUSES = ["supported", "contradicted", "unsupported", "uncertain"]
POSITIVE = {"contradicted", "unsupported", "uncertain"}  # "not supported"
ERROR_TYPES_SHOWN = ["negation", "overstated", "over_interpretation", "wrong_section", "other"]


def load_gold(
    path: Path = Path("labels/claims.csv"), encounters: set[str] | None = None
) -> list[dict]:
    check_header(path, CLAIMS_HEADER)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["status"] and (not encounters or r["encounter_id"] in encounters)]


def load_verdicts(directory: Path = VERDICTS_DIR) -> dict[tuple[str, str], dict]:
    out = {}
    for f in sorted(directory.glob("*.json")):
        vf = VerdictFile.model_validate_json(f.read_text(encoding="utf-8"))
        for r in vf.records:
            out[(r.encounter_id, r.claim_id)] = r.model_dump()
    return out


def ratio(a: int, b: int) -> float | None:
    return a / b if b else None


@dataclass
class Score:
    n_gold: int = 0
    missing: list[tuple[str, str]] = field(default_factory=list)  # no usable verdict
    text_mismatch: list[tuple[str, str]] = field(default_factory=list)
    matrix: dict[tuple[str, str], int] = field(default_factory=dict)  # (gold, verifier) -> n
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    disagreements: list[dict] = field(default_factory=list)
    quotes_checked: int = 0
    quotes_invalid: int = 0
    turns_corrected: int = 0
    # same binary table, but flagging on needs_review (status != supported OR quote_valid False)
    tp_r: int = 0
    fp_r: int = 0
    fn_r: int = 0
    tn_r: int = 0
    # error_type -> {"n", "gold_not_supported", "flagged_status", "flagged_review"}
    per_error: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def precision_r(self) -> float | None:
        return ratio(self.tp_r, self.tp_r + self.fp_r)

    @property
    def recall_r(self) -> float | None:
        return ratio(self.tp_r, self.tp_r + self.fn_r)

    def error_recall(self, error_type: str, basis: str = "status") -> float | None:
        """Share of this error_type's claims the verifier flagged (basis: status | review)."""
        e = self.per_error.get(error_type)
        return ratio(e[f"flagged_{basis}"], e["n"]) if e else None

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float | None:
        return ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)

    @property
    def accuracy4(self) -> float | None:
        return ratio(sum(v for (g, c), v in self.matrix.items() if g == c), self.n)


def score(
    gold: list[dict],
    verdicts: dict[tuple[str, str], dict],
    exclude_decomposition_error: bool = False,
) -> Score:
    s = Score()
    for g in gold:
        if exclude_decomposition_error and g["error_type"] == "decomposition_error":
            continue
        s.n_gold += 1
        key = (g["encounter_id"], g["claim_id"])
        rec = verdicts.get(key)
        if rec is None or rec["verdict"] is None:
            s.missing.append(key)
            continue
        if rec["claim_text"] != g["claim_text"]:
            s.text_mismatch.append(key)
            continue
        v = rec["verdict"]
        pair = (g["status"], v["status"])
        s.matrix[pair] = s.matrix.get(pair, 0) + 1
        gold_pos, pred_pos = g["status"] in POSITIVE, v["status"] in POSITIVE
        if gold_pos and pred_pos:
            s.tp += 1
        elif pred_pos:
            s.fp += 1
        elif gold_pos:
            s.fn += 1
        else:
            s.tn += 1
        review = needs_review_of(v["status"], rec["quote_valid"])
        if gold_pos and review:
            s.tp_r += 1
        elif review:
            s.fp_r += 1
        elif gold_pos:
            s.fn_r += 1
        else:
            s.tn_r += 1
        e = s.per_error.setdefault(
            g["error_type"] or "none",
            {"n": 0, "gold_not_supported": 0, "flagged_status": 0, "flagged_review": 0},
        )
        e["n"] += 1
        e["gold_not_supported"] += gold_pos
        e["flagged_status"] += pred_pos
        e["flagged_review"] += review
        s.turns_corrected += rec.get("turn_corrected") is True
        if v["evidence_quote"]:
            s.quotes_checked += 1
            s.quotes_invalid += rec["quote_valid"] is False
        if g["status"] != v["status"]:
            kind = (
                "MISSED (gold not-supported, verifier says supported)"
                if gold_pos and not pred_pos
                else (
                    "FALSE ALARM (gold supported, verifier flags it)"
                    if pred_pos and not gold_pos
                    else "same side, different label"
                )
            )
            s.disagreements.append({"key": key, "gold": g, "verdict": v, "quote_valid": rec["quote_valid"], "kind": kind,
                                    "claim_text": g["claim_text"]})  # fmt: skip
    return s


@dataclass
class Comparison:
    """Single-claim vs batched verdicts, on claims that have a usable verdict in both."""

    n_both: int = 0
    same_status: int = 0
    same_side: int = 0  # both "supported", or both "not supported"
    differ: list[dict] = field(default_factory=list)
    gold_n: int = 0  # of the shared claims, how many have a gold label
    single_correct: int = 0
    batched_correct: int = 0
    batched_via: dict[str, int] = field(default_factory=dict)  # how batched verdicts were obtained

    @property
    def agree(self) -> float | None:
        return ratio(self.same_status, self.n_both)

    @property
    def agree_side(self) -> float | None:
        return ratio(self.same_side, self.n_both)


def compare_modes(
    single: dict[tuple[str, str], dict],
    batched: dict[tuple[str, str], dict],
    gold: list[dict] | None = None,
) -> Comparison:
    gold_by = {(g["encounter_id"], g["claim_id"]): g for g in gold or []}
    c = Comparison()
    for key in sorted(set(single) & set(batched)):
        a, b = single[key], batched[key]
        if a["verdict"] is None or b["verdict"] is None or a["claim_text"] != b["claim_text"]:
            continue
        sa, sb = a["verdict"]["status"], b["verdict"]["status"]
        c.n_both += 1
        c.batched_via[b.get("via", "batch")] = c.batched_via.get(b.get("via", "batch"), 0) + 1
        c.same_status += sa == sb
        c.same_side += (sa in POSITIVE) == (sb in POSITIVE)
        if key in gold_by:
            c.gold_n += 1
            c.single_correct += sa == gold_by[key]["status"]
            c.batched_correct += sb == gold_by[key]["status"]
        if sa != sb:
            c.differ.append({"key": key, "claim_text": a["claim_text"], "single": sa, "batched": sb,
                             "gold": gold_by.get(key, {}).get("status", "-")})  # fmt: skip
    return c


def render_comparison(c: Comparison) -> str:
    out = [f"\n{'=' * 78}\nSINGLE vs BATCHED (claims with a usable verdict in both)\n{'=' * 78}"]
    if not c.n_both:
        return "\n".join(out + ["no claims have both a single and a batched verdict yet."])
    out.append(f"claims in both: {c.n_both}")
    out.append(f"  same 4-class status: {c.same_status}/{c.n_both} = {fmt(c.agree)}")
    out.append(
        f"  same side (supported vs not supported): {c.same_side}/{c.n_both} = {fmt(c.agree_side)}"
    )
    out.append(f"  batched verdicts came via: {c.batched_via}  (single-retry = re-asked alone)")
    if c.gold_n:
        out.append(
            f"  vs your labels on the {c.gold_n} shared labeled claims: "
            f"single {c.single_correct}/{c.gold_n}, batched {c.batched_correct}/{c.gold_n} exact-status matches"
        )
    for d in c.differ:
        out.append(
            f"  [{d['key'][0][-2:]} {d['key'][1]}] single={d['single']:<12} batched={d['batched']:<12} "
            f"you={d['gold']:<12} | {d['claim_text'][:60]}"
        )
    return "\n".join(out)


@dataclass
class VersionComparison:
    """Two prompt versions scored on the SAME claims against the SAME labels."""

    v0: Score = field(default_factory=Score)
    v1: Score = field(default_factory=Score)
    n_common: int = 0
    changed: list[dict] = field(default_factory=list)  # claims whose status differs v0 -> v1
    fixed: int = 0  # v1 matches your label, v0 did not
    broken: int = 0  # v0 matched your label, v1 does not


def compare_versions(
    gold: list[dict],
    v0: dict[tuple[str, str], dict],
    v1: dict[tuple[str, str], dict],
    exclude_decomposition_error: bool = False,
) -> VersionComparison:
    def usable(d: dict, k: tuple[str, str]) -> bool:
        return k in d and d[k]["verdict"] is not None

    common = {
        k
        for k in set(v0) & set(v1)
        if usable(v0, k) and usable(v1, k) and v0[k]["claim_text"] == v1[k]["claim_text"]
    }
    sub = [g for g in gold if (g["encounter_id"], g["claim_id"]) in common]
    c = VersionComparison(v0=score(sub, v0, exclude_decomposition_error),
                          v1=score(sub, v1, exclude_decomposition_error))  # fmt: skip
    for g in sub:
        if exclude_decomposition_error and g["error_type"] == "decomposition_error":
            continue
        k = (g["encounter_id"], g["claim_id"])
        c.n_common += 1
        a, b = v0[k]["verdict"]["status"], v1[k]["verdict"]["status"]
        if a == b:
            continue
        fixed, broken = b == g["status"], a == g["status"]
        c.fixed += fixed
        c.broken += broken
        c.changed.append({"key": k, "claim_text": g["claim_text"], "gold": g["status"],
                          "error_type": g["error_type"], "v0": a, "v1": b,
                          "outcome": "FIXED" if fixed else ("BROKEN" if broken else "changed, both wrong")})  # fmt: skip
    return c


def render_version_comparison(c: VersionComparison, title: str) -> str:
    out = [f"\n{'=' * 78}\nv0 vs v1 PROMPT, same claims, same labels  [{title}]\n{'=' * 78}"]
    if not c.n_common:
        return "\n".join(out + ["no claims have a usable verdict in both folders yet."])
    a, b = c.v0, c.v1
    rows = [
        ("claims compared", c.n_common, c.n_common),
        ("4-class accuracy", fmt(a.accuracy4), fmt(b.accuracy4)),
        ("precision (status)", fmt(a.precision), fmt(b.precision)),
        ("recall (status)", fmt(a.recall), fmt(b.recall)),
        ("precision (needs_review)", fmt(a.precision_r), fmt(b.precision_r)),
        ("recall (needs_review)", fmt(a.recall_r), fmt(b.recall_r)),
        ("missed errors FN (status)", a.fn, b.fn),
        ("false alarms FP (status)", a.fp, b.fp),
        ("quotes not found", a.quotes_invalid, b.quotes_invalid),
        ("turn ids corrected", a.turns_corrected, b.turns_corrected),
    ]
    out.append(f"  {'':28}{'v0':>10}{'v1':>10}")
    out += [f"  {name:<28}{x!s:>10}{y!s:>10}" for name, x, y in rows]
    out.append("\n  recall per error_type, flagged/n  (status | needs_review):")
    for et in ERROR_TYPES_SHOWN + ["decomposition_error"]:
        if et not in a.per_error and et not in b.per_error:
            continue
        cell = lambda sc, k=et: (
            f"{sc.per_error[k]['flagged_status']}/{sc.per_error[k]['n']} | "
            f"{sc.per_error[k]['flagged_review']}/{sc.per_error[k]['n']}"
            if k in sc.per_error
            else "-"
        )
        out.append(f"    {et:<20}{cell(a):>16}{cell(b):>16}")
    out.append(f"\n  status changed on {len(c.changed)} claims: {c.fixed} FIXED, {c.broken} BROKEN")
    for d in c.changed:
        out.append(
            f"    [{d['key'][0][-2:]} {d['key'][1]}] {d['outcome']:<19} v0={d['v0']:<12} "
            f"v1={d['v1']:<12} you={d['gold']:<12} ({d['error_type'] or '-'}) | {d['claim_text'][:55]}"
        )
    return "\n".join(out)


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render(s: Score, title: str, show_disagreements: bool = True) -> str:
    out = [f"\n{'=' * 78}\n{title}\n{'=' * 78}"]
    out.append(
        f"gold-labeled claims: {s.n_gold}   scored: {s.n}   no verdict: {len(s.missing)}   text mismatch: {len(s.text_mismatch)}"
    )
    if s.missing:
        out.append(
            "  no verdict for: "
            + ", ".join(f"{e[-2:]}:{c}" for e, c in s.missing[:20])
            + (" ..." if len(s.missing) > 20 else "")
        )
    w = 14
    out.append("\n4-class confusion matrix (rows = YOUR label, columns = VERIFIER)")
    out.append(" " * w + "".join(f"{c:>{w}}" for c in STATUSES) + f"{'total':>{w}}")
    for g in STATUSES:
        row = [s.matrix.get((g, c), 0) for c in STATUSES]
        out.append(f"{g:<{w}}" + "".join(f"{x:>{w}}" for x in row) + f"{sum(row):>{w}}")
    out.append("\nbinary: positive = NOT SUPPORTED (contradicted + unsupported + uncertain)")
    out.append("                      verifier: not supported   verifier: supported")
    out.append(f"  you: not supported  TP={s.tp:<18} FN={s.fn}   (errors the verifier missed)")
    out.append(
        f"  you: supported      FP={s.fp:<18} TN={s.tn}   (correct claims it wrongly flagged)"
    )
    out.append(
        f"  precision={fmt(s.precision)}  recall={fmt(s.recall)}  F1={fmt(s.f1)}  4-class accuracy={fmt(s.accuracy4)}"
    )
    out.append(
        f"  quotes: {s.quotes_checked} given, {s.quotes_invalid} with quote_valid=False, "
        f"{s.turns_corrected} turn ids corrected"
    )
    out.append("\nflagging on needs_review = (status != supported) OR (quote_valid is False):")
    out.append(f"  you: not supported  TP={s.tp_r:<18} FN={s.fn_r}")
    out.append(f"  you: supported      FP={s.fp_r:<18} TN={s.tn_r}")
    out.append(f"  precision={fmt(s.precision_r)}  recall={fmt(s.recall_r)}")
    out.append(
        "\nrecall per error_type (share of that error_type's claims flagged; status | needs_review)"
    )
    shown = ERROR_TYPES_SHOWN + (
        ["decomposition_error"] if "decomposition_error" in s.per_error else []
    )
    for et in shown:
        e = s.per_error.get(et)
        if not e:
            out.append(f"  {et:<20} n=0")
            continue
        out.append(
            f"  {et:<20} n={e['n']:<3} (you: not-supported {e['gold_not_supported']:<2})  "
            f"status {e['flagged_status']}/{e['n']} = {fmt(s.error_recall(et))}   "
            f"needs_review {e['flagged_review']}/{e['n']} = {fmt(s.error_recall(et, 'review'))}"
        )
    if show_disagreements:
        out.append(f"\ndisagreements (4-class): {len(s.disagreements)}")
        for d in s.disagreements:
            g, v = d["gold"], d["verdict"]
            out.append(f"\n[{d['key'][0][-2:]} {d['key'][1]}] {d['kind']}")
            out.append(f"  claim   : {d['claim_text']}")
            out.append(
                f"  YOU     : {g['status']} / {g['error_type'] or '-'}   turns={g['evidence_turns'] or '-'}   quote={g['evidence_quote']!r}"
            )
            out.append(
                f"  VERIFIER: {v['status']}   turns={v['evidence_turns'] or '-'}   quote={v['evidence_quote']!r}   quote_valid={d['quote_valid']}"
            )
            out.append(f"  analysis: {v.get('analysis') or v.get('reason', '')}")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="labels/claims.csv")
    ap.add_argument(
        "--verdicts",
        default=str(VERDICTS_DIR),
        metavar="DIR",
        help=f"folder of verdict JSONs to SCORE (default: {VERDICTS_DIR}, the single-claim run; "
        f"use {BATCH_VERDICTS_DIR} for the batched run)",
    )
    ap.add_argument("--single-verdicts", default=str(VERDICTS_DIR), metavar="DIR",
                    help="single-claim folder, for the single-vs-batched comparison")  # fmt: skip
    ap.add_argument("--batched-verdicts", default=str(BATCH_VERDICTS_DIR), metavar="DIR",
                    help="batched folder, for the single-vs-batched comparison")  # fmt: skip
    ap.add_argument("--v0-verdicts", default=str(BATCH_VERDICTS_DIR), metavar="DIR",
                    help="v0 folder for the v0-vs-v1 comparison (default: the batched v0 run)")  # fmt: skip
    ap.add_argument("--v1-verdicts", default=str(V1_BATCH_VERDICTS_DIR), metavar="DIR",
                    help="v1 folder for the v0-vs-v1 comparison (default: the batched v1 run)")  # fmt: skip
    ap.add_argument("--no-disagreements", action="store_true")
    ap.add_argument(
        "--only",
        nargs="+",
        help="score only these encounters (default: dev 01-05, or all in a custom --labels file)",
    )
    return ap


def _verdict_dir(path: str) -> Path:
    d = Path(path)
    if not d.is_dir() or not any(d.glob("*.json")):
        raise SystemExit(f"error: no verdict JSON files in {d}")
    return d


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.only:
        enc = set(a.only)
    elif a.labels == "labels/claims.csv":
        enc = {f"day1_consultation0{i}" for i in range(1, 6)}  # dev set (unchanged default)
    else:  # a custom label file: score every encounter in it
        with open(a.labels, newline="", encoding="utf-8") as f:
            enc = {r["encounter_id"] for r in csv.DictReader(f)}
    vdir = _verdict_dir(a.verdicts)
    gold = load_gold(Path(a.labels), enc)
    verdicts = load_verdicts(vdir)
    n_files = len(list(vdir.glob("*.json")))
    in_scope = sum(1 for k in verdicts if k[0] in enc)
    print(f"SCORING VERDICTS FROM: {vdir}")
    print(f"  {n_files} verdict files, {in_scope} verdicts for the selected encounters")
    print(f"  labels: {a.labels}   encounters: {', '.join(sorted(e[-2:] for e in enc))}")
    for excl, title in ((False, "ALL ROWS"), (True, "EXCLUDING error_type = decomposition_error")):
        report = score(gold, verdicts, excl)
        print(
            render(
                report,
                f"{title}   [{vdir}]",
                show_disagreements=not excl and not a.no_disagreements,
            )
        )
    sdir, bdir = Path(a.single_verdicts), Path(a.batched_verdicts)
    if all(d.is_dir() and any(d.glob("*.json")) for d in (sdir, bdir)):
        keep = lambda d: {k: v for k, v in d.items() if k[0] in enc}
        print(f"\ncomparison folders:  single={sdir}   batched={bdir}")
        print(
            render_comparison(
                compare_modes(keep(load_verdicts(sdir)), keep(load_verdicts(bdir)), gold)
            )
        )
    else:
        print("\n(single-vs-batched comparison skipped: needs verdict files in both folders)")
    d0, d1 = Path(a.v0_verdicts), Path(a.v1_verdicts)
    if all(d.is_dir() and any(d.glob("*.json")) for d in (d0, d1)):
        g0, g1 = load_verdicts(d0), load_verdicts(d1)
        keep = lambda d: {k: v for k, v in d.items() if k[0] in enc}
        for excl, title in ((False, "ALL ROWS"), (True, "EXCLUDING decomposition_error")):
            print(render_version_comparison(compare_versions(gold, keep(g0), keep(g1), excl),
                                            f"{title}; v0={d0} v1={d1}"))  # fmt: skip
    else:
        print(
            "\n(v0-vs-v1 comparison skipped: needs verdict files in both --v0-verdicts and --v1-verdicts)"
        )


if __name__ == "__main__":
    main()
