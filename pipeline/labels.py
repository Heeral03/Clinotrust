"""Labeling sheets. labels/claims.csv is the id authority once labeling starts.

Rules (so manual labels are never lost):
- Rows are matched to claims by (encounter_id, section, claim_text); matched rows are NEVER touched.
- Ids are never renumbered; a new claim gets an id above the encounter's current maximum.
- Rows whose claim no longer exists are kept and reported as orphans.
Hand-split ids like C14a / C14b are respected (their numeric part counts toward the maximum).
"""

import csv
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pipeline.decompose import DecomposeResult

LABELS_DIR = Path("labels")
CLAIMS_HEADER = [
    "encounter_id", "claim_id", "section", "source_statement_index", "claim_text",
    "status", "evidence_turns", "evidence_quote", "error_type", "severity", "notes",
]  # fmt: skip
LABEL_COLUMNS = ["status", "evidence_turns", "evidence_quote", "error_type", "severity", "notes"]
OMISSIONS_HEADER = [
    "encounter_id", "omission_id", "section", "missing_information",
    "evidence_turns", "evidence_quote", "severity", "notes",
]  # fmt: skip


STATUSES = {"supported", "contradicted", "unsupported", "uncertain"}
# decomposition_error: the DECOMPOSER mangled a correct note statement (e.g. dropped a "no").
# It is not a scribe error and must be excluded from the generator's error rates.
ERROR_TYPES = {
    "none", "negation", "overstated", "over_interpretation", "wrong_section", "other",
    "decomposition_error",
}  # fmt: skip
SEVERITIES = {"none", "minor", "major"}


class LabelsFileError(RuntimeError):
    """A labels/*.csv file is not shaped like the sheet we were asked to read or write."""


def check_header(path: Path, expected: list[str]) -> None:
    """Raise unless `path` exists with EXACTLY the expected header (guards against a sheet being
    overwritten by another sheet, as happened once with claims.csv <- omissions.csv)."""
    with open(path, newline="", encoding="utf-8") as f:
        got = next(csv.reader(f), [])
    if got != expected:
        raise LabelsFileError(
            f"{path}: header is {got}, expected {expected}. Refusing to touch it."
        )


def read_rows(path: Path, header: list[str] | None = None) -> list[dict]:
    if not path.exists():
        return []
    if header is not None:
        check_header(path, header)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def backup(path: Path) -> Path | None:
    """Timestamped copy next to the file, so a bad write can never destroy the last good sheet."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    dest = path.with_name(f"{path.name}.{datetime.now():%Y%m%d-%H%M%S-%f}.bak")
    shutil.copy2(path, dest)
    return dest


def validate_label_rows(rows: list[dict]) -> list[str]:
    """Return problems with label values (blank = not labeled yet, always allowed)."""
    problems = []
    for r in rows:
        for col, allowed in (
            ("status", STATUSES),
            ("error_type", ERROR_TYPES),
            ("severity", SEVERITIES),
        ):
            if r.get(col) and r[col] not in allowed:
                problems.append(
                    f"{r['encounter_id']} {r['claim_id']}: {col}={r[col]!r} not in {sorted(allowed)}"
                )
    return problems


def write_rows(path: Path, header: list[str], rows: list[dict]) -> None:
    if path.exists():
        check_header(path, header)  # never overwrite a file of a different shape
        backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def labeled_encounters(path: Path = LABELS_DIR / "claims.csv") -> set[str]:
    """Encounters that already have rows in the sheet: their ids are frozen."""
    return {r["encounter_id"] for r in read_rows(path, CLAIMS_HEADER)}


def _num(claim_id: str) -> int:
    m = re.match(r"C(\d+)", claim_id)
    return int(m[1]) if m else 0


def merge_claims(
    results: list[DecomposeResult], path: Path = LABELS_DIR / "claims.csv"
) -> tuple[int, int, list[str]]:
    """Add new claims to the sheet. Returns (rows_added, rows_kept, warnings)."""
    rows = read_rows(path, CLAIMS_HEADER)
    warnings: list[str] = []
    for res in results:
        mine = [r for r in rows if r["encounter_id"] == res.encounter_id]
        unused: dict[tuple, list[dict]] = defaultdict(list)
        for r in mine:
            unused[(r["section"], r["claim_text"])].append(r)
        next_id = max((_num(r["claim_id"]) for r in mine), default=0) + 1
        for c in res.claims:
            if unused[(c.section, c.text)]:
                unused[(c.section, c.text)].pop(0)  # already in the sheet: leave it alone
                continue
            rows.append({
                "encounter_id": c.encounter_id, "claim_id": f"C{next_id}", "section": c.section,
                "source_statement_index": c.source_statement_index, "claim_text": c.text,
                **dict.fromkeys(LABEL_COLUMNS, ""),
            })  # fmt: skip
            next_id += 1
            if mine:  # sheet already existed for this encounter => text changed or new claim
                warnings.append(
                    f"{c.encounter_id}: new claim appended as C{next_id - 1}: {c.text!r}"
                )
        for left in unused.values():
            for r in left:
                warnings.append(
                    f"{r['encounter_id']} {r['claim_id']}: no longer produced by decomposer (kept): "
                    f"{r['claim_text']!r}"
                )
    kept = len(read_rows(path, CLAIMS_HEADER))
    if rows and len(rows) > kept:  # nothing new -> do not rewrite (or back up) an untouched sheet
        write_rows(path, CLAIMS_HEADER, rows)
    return len(rows) - kept, kept, warnings


def ensure_omissions(path: Path = LABELS_DIR / "omissions.csv") -> bool:
    """Create the header-only omissions sheet. Never overwrites an existing one."""
    if path.exists():
        check_header(path, OMISSIONS_HEADER)
        return False
    write_rows(path, OMISSIONS_HEADER, [])
    return True
