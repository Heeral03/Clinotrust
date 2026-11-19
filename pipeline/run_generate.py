"""Generate SOAP notes for the first N PriMock57 transcripts.

Writes to outputs/generated/ - never touches data/raw/.
Usage: uv run python -m pipeline.run_generate [--n 5 | --ids ID ...] [--out DIR]
"""

import argparse
import json
from pathlib import Path

from data.loaders import load_primock57
from data.schemas import LOADER_VERSION
from pipeline.generate import PROMPT_VERSION, GroqClient, generate_soap

OUT = Path("outputs/generated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ids", nargs="+", metavar="ID", help="only these encounters (skips existing)")
    ap.add_argument("--out", default=str(OUT), help="output folder")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = GroqClient()
    encs = load_primock57()
    encs = [e for e in encs if e.id in a.ids] if a.ids else encs[: a.n]
    if a.ids and {e.id for e in encs} != set(a.ids):
        raise SystemExit(f"unknown ids: {set(a.ids) - {e.id for e in encs}}")
    for enc in encs:
        if a.ids and (out_dir / f"{enc.id}.json").exists():
            print(f"{enc.id}: already generated - skipped")
            continue
        note = generate_soap(enc, client)
        (out_dir / f"{enc.id}.json").write_text(
            json.dumps(
                {
                    "id": enc.id,
                    "model": client.model,
                    "prompt_version": PROMPT_VERSION,
                    "loader_version": enc.loader_version or LOADER_VERSION,
                    "merged_same_speaker": enc.merged_same_speaker,
                    "soap": note.model_dump(),
                },
                indent=2,
            )
        )
        print(f"\n=== {enc.id} ===")
        for sec, items in note.model_dump().items():
            print(f"{sec.upper()}:" + "".join(f"\n  - {i}" for i in items))
        print(f"\nREFERENCE: {enc.reference_note.strip()}")


if __name__ == "__main__":
    main()
