"""Loaders for PriMock57 and ACI-Bench -> common Encounter schema."""

import csv
import json
import re
from pathlib import Path

from data.cleaning import clean_text
from data.schemas import Encounter, Turn

RAW = Path(__file__).parent / "raw"
PRIMOCK = RAW / "primock57"
ACI_CHALLENGE = RAW / "aci-bench-figshare" / "aci-bench-corpus" / "challenge_data"

_INTERVAL = re.compile(
    r"xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = \"(.*?)\"\s*\n", re.DOTALL
)
_FILE_SPEAKER = re.compile(r"_(doctor|patient)\.TextGrid$")


def _speaker_from_filename(path: Path) -> str:
    # Tier names are unreliable (56/57 files call the tier "Speaker"); the filename is not.
    m = _FILE_SPEAKER.search(path.name)
    if not m:
        raise ValueError(f"Cannot infer speaker from filename: {path.name}")
    return m[1]


def _parse_textgrid(path: Path) -> list[dict]:
    speaker = _speaker_from_filename(path)
    out = []
    for xmin, xmax, raw in _INTERVAL.findall(path.read_text(encoding="utf-8")):
        if not raw.strip():
            continue
        text, unsure, inaudible = clean_text(raw.replace('""', '"'))
        if text:  # drop turns that were only tags
            out.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "start": float(xmin),
                    "end": float(xmax),
                    "had_unsure": unsure,
                    "had_inaudible": inaudible,
                }
            )
    return out


def _merge_same_speaker(rows: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for r in rows:
        if merged and merged[-1]["speaker"] == r["speaker"]:
            m = merged[-1]
            m["text"] += " " + r["text"]
            m["end"] = max(m["end"], r["end"])
            m["had_unsure"] |= r["had_unsure"]
            m["had_inaudible"] |= r["had_inaudible"]
        else:
            merged.append(dict(r))
    return merged


def _to_turns(rows: list[dict]) -> list[Turn]:
    turns, prev_end = [], None
    for i, r in enumerate(rows, 1):
        overlaps = prev_end is not None and r["start"] < prev_end
        turns.append(Turn(id=f"T{i}", overlaps_previous=overlaps, **r))
        prev_end = r["end"]
    return turns


def load_primock57(merge_same_speaker: bool = False) -> list[Encounter]:
    out = []
    for note_path in sorted((PRIMOCK / "notes").glob("day*_consultation*.json")):
        cid = note_path.stem
        rows = []
        for who in ("doctor", "patient"):
            rows += _parse_textgrid(PRIMOCK / "transcripts" / f"{cid}_{who}.TextGrid")
        # Strict start-time order; ties broken deterministically so ids are stable.
        rows.sort(key=lambda r: (r["start"], r["end"], r["speaker"]))
        if merge_same_speaker:
            rows = _merge_same_speaker(rows)
        note = json.loads(note_path.read_text(encoding="utf-8"))
        out.append(
            Encounter(
                id=cid,
                source="primock57",
                turns=_to_turns(rows),
                reference_note=note["note"],
                merged_same_speaker=merge_same_speaker,
                meta={"presenting_complaint": note.get("presenting_complaint")},
            )
        )
    return out


_SPEAKER = re.compile(r"^\[(\w+)\]\s*(.*)$")


def _parse_aci_dialogue(text: str) -> list[Turn]:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _SPEAKER.match(line.strip())
        if m:
            rows.append((m[1], m[2]))
        elif rows and line.strip():
            rows[-1] = (rows[-1][0], rows[-1][1] + " " + line.strip())
    return [Turn(id=f"T{i}", speaker=s, text=t) for i, (s, t) in enumerate(rows, 1)]


def load_aci_bench(splits: tuple[str, ...] = ("train", "valid")) -> list[Encounter]:
    csv.field_size_limit(10**9)
    out = []
    for split in splits:
        with open(ACI_CHALLENGE / f"{split}.csv", encoding="utf-8") as f:
            meta = {}
            mpath = ACI_CHALLENGE / f"{split}_metadata.csv"
            if mpath.exists():
                with open(mpath, encoding="utf-8") as mf:
                    meta = {r["encounter_id"]: r for r in csv.DictReader(mf)}
            for row in csv.DictReader(f):
                eid = row["encounter_id"]
                out.append(
                    Encounter(
                        id=f"aci-{split}-{eid}",
                        source="aci-bench",
                        turns=_parse_aci_dialogue(row["dialogue"]),
                        reference_note=row["note"],
                        meta={"split": split, "dataset": row.get("dataset"), **meta.get(eid, {})},
                    )
                )
    return out


def load_all() -> list[Encounter]:
    return load_primock57() + load_aci_bench(("train", "valid"))
