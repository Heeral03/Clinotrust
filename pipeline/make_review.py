"""Write outputs/review/{encounter_id}.txt: the transcript with timestamps, to read beside the CSV."""

import argparse
import json
from pathlib import Path

from data.loaders import load_primock57
from data.schemas import Encounter

PROCESSED = Path("data/processed")
REVIEW = Path("outputs/review")


def load_encounter(encounter_id: str, processed: Path = PROCESSED) -> Encounter:
    p = processed / f"{encounter_id}.json"
    if p.exists():
        return Encounter.model_validate_json(p.read_text(encoding="utf-8"))
    return next(e for e in load_primock57() if e.id == encounter_id)


def render_review(enc: Encounter) -> str:
    head = (
        f"ENCOUNTER {enc.id}  |  {enc.source}  |  {enc.loader_version}"
        f"  |  merged_same_speaker={enc.merged_same_speaker}\n"
        "Turn ids (T12) are what you cite as evidence_turns. '(131.6s)' = turn start time.\n"
        + "-" * 78
        + "\n"
    )
    return head + enc.transcript(with_times=True) + "\n"


def write_review(
    note: dict, out_dir: Path = REVIEW, processed: Path = PROCESSED
) -> tuple[Path, str]:
    enc = load_encounter(note["id"], processed)
    warn = ""
    if note.get("loader_version") and note["loader_version"] != enc.loader_version:
        warn = f"{enc.id}: note made with {note['loader_version']} but transcript is {enc.loader_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{enc.id}.txt"
    path.write_text(render_review(enc), encoding="utf-8")
    return path, warn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default="outputs/generated")
    a = ap.parse_args()
    for f in sorted(Path(a.generated).glob("*.json")):
        path, warn = write_review(json.loads(f.read_text(encoding="utf-8")))
        print(path, f"  WARNING: {warn}" if warn else "")


if __name__ == "__main__":
    main()
