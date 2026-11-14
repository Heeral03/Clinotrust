"""Run the verifier (prompt v0 or v1) over labels/claims.csv (reads ONLY encounter_id, claim_id, section, claim_text).

Usage: uv run python -m pipeline.run_verify [--prompt v0|v1] [--batched] [--only ID ...] [--limit N]

Resumable: claims already verified are skipped, and the verdict file is saved after every claim.
Writes to outputs/verdicts/verifier-v0[-batched]/ (v0) or verifier-v1[-batched]/ (v1). A v1 run can
never write into a v0 folder. Never writes to labels/.
"""

import argparse
import json
from pathlib import Path

from pipeline.make_review import load_encounter
from pipeline.verify import (
    BATCH_SIZE,
    BATCH_VERDICTS_DIR,
    DAILY_TOKEN_CAP,
    LEDGER,
    SPECS,
    V1_BATCH_VERDICTS_DIR,
    V1_VERDICTS_DIR,
    VERDICTS_DIR,
    VerifierClient,
    assert_independent,
    load_claims_for_verification,
    tokens_last_24h,
    usage_line,
    verify_all,
    verify_all_batched,
)


def generator_models(generated: Path = Path("outputs/generated")) -> set[str]:
    return {json.loads(f.read_text())["model"] for f in generated.glob("*.json")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="labels/claims.csv")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--limit", type=int, help="stop after N new verdicts (smoke test)")
    ap.add_argument("--batched", action="store_true", help="one call per group of claims")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--prompt", choices=sorted(SPECS), default="v0", help="verifier prompt version")
    ap.add_argument("--out", help="default: outputs/verdicts/verifier-<prompt>[-batched]")
    a = ap.parse_args()
    spec = SPECS[a.prompt]
    defaults = {
        ("v0", False): VERDICTS_DIR, ("v0", True): BATCH_VERDICTS_DIR,
        ("v1", False): V1_VERDICTS_DIR, ("v1", True): V1_BATCH_VERDICTS_DIR,
    }  # fmt: skip
    out = Path(a.out or defaults[(a.prompt, a.batched)])
    v0_dirs = {VERDICTS_DIR.resolve(), BATCH_VERDICTS_DIR.resolve()}
    if a.batched and out.resolve() == VERDICTS_DIR.resolve():
        ap.error("batched results must not go into the single-claim directory")
    if a.prompt != "v0" and out.resolve() in v0_dirs:
        ap.error("a v1 run must not write into a v0 verdicts folder")

    client = VerifierClient()
    assert_independent(client.model, generator_models())
    claims = load_claims_for_verification(Path(a.claims), set(a.only) if a.only else None)
    encounters = {eid: load_encounter(eid) for eid in {c.encounter_id for c in claims}}
    mode = f"{a.prompt} " + (f"batched (<= {a.batch_size}/call)" if a.batched else "single-claim")
    used = tokens_last_24h(LEDGER, client.model)
    print(
        f"verifier={client.model}  mode={mode}  claims={len(claims)}  encounters={len(encounters)}"
    )
    print(
        f"ledger: {used:,}/{DAILY_TOKEN_CAP:,} tokens in the last 24h (only counts calls logged since the ledger began)"
    )
    try:
        if a.batched:
            n = verify_all_batched(
                claims, encounters, client, out, a.batch_size, a.limit, spec=spec
            )
        else:
            n = verify_all(claims, encounters, client, out, a.limit, spec=spec)
    except RuntimeError as e:  # e.g. daily cap: progress is already saved
        print(f"\nSTOPPED: {e}\n{usage_line(client)}")
        raise SystemExit(1) from e
    print(f"\nverified {n} new claims -> {out}\n{usage_line(client)}")


if __name__ == "__main__":
    main()
