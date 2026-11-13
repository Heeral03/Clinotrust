"""Generate SOAP notes for the first N PriMock57 transcripts.

Writes to outputs/generated/ - never touches data/raw/.
Usage: uv run python -m pipeline.run_generate [--n 5]
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
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = GroqClient()
    for enc in load_primock57()[: a.n]:
        note = generate_soap(enc, client)
        (OUT / f"{enc.id}.json").write_text(
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
