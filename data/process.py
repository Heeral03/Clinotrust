"""Write cleaned, numbered encounters to data/processed/ (derived; data/raw/ is never modified).

Usage: uv run python -m data.process [--merge-same-speaker]
"""

import argparse
from pathlib import Path

from data.loaders import load_primock57
from data.schemas import Encounter

PROCESSED = Path(__file__).parent / "processed"


def save_processed(enc: Encounter, out_dir: Path = PROCESSED) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{enc.id}.json"
    path.write_text(enc.model_dump_json(indent=2), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-same-speaker", action="store_true")
    a = ap.parse_args()
    encs = load_primock57(merge_same_speaker=a.merge_same_speaker)
    for e in encs:
        save_processed(e)
    print(f"wrote {len(encs)} encounters to {PROCESSED}")


if __name__ == "__main__":
    main()
