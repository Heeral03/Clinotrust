import json
import re
from datetime import UTC, datetime, timedelta

import pytest

from data.schemas import Encounter, Turn
from pipeline.verify import (
    BATCH_PROMPT_VERSION,
    BATCH_SYSTEM_PROMPT,
    ClaimInput,
    load_verdict_file,
    tokens_last_24h,
    usage_line,
    verify_all_batched,
    verify_batch,
)

TURNS = [
    Turn(id="T1", speaker="doctor", text="Any palpitations?", start=1.0, end=2.0),
    Turn(id="T2", speaker="patient", text="Not that it, no.", start=2.5, end=3.0),
    Turn(id="T3", speaker="doctor", text="Any headaches?", start=3.5, end=4.0),
    Turn(id="T4", speaker="patient", text="No.", start=4.2, end=4.5),
]
ENC = Encounter(id="e1", source="primock57", reference_note="", turns=TURNS)


def claims(n):
    return [
        ClaimInput(
            encounter_id="e1",
            claim_id=f"C{i}",
            section="subjective",
            claim_text=f"Patient reports no thing{i}",
        )
        for i in range(1, n + 1)
    ]


def item(cid, status="supported", turns=("T2",), quote="Not that it, no", reason="r"):
    return {
        "claim_id": cid,
        "status": status,
        "evidence_turns": list(turns),
        "evidence_quote": quote,
        "reason": reason,
    }


def batch_reply(*items):
    return json.dumps({"verdicts": list(items)})


class Fake:
    """Records every call. Replies are popped in order; max_out is recorded too."""

    model = "fake-verifier"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.last_usage, self.total_prompt, self.total_completion, self.n_calls = (0, 0), 0, 0, 0
        self.ledger = None

    def complete(self, system, user, max_out=None):
        self.calls.append((system, user, max_out))
        self.total_prompt += 100
        self.total_completion += 20
        self.n_calls += 1
        self.last_usage = (100, 20)
        return self.replies.pop(0)


def single_reply(status="supported", turns=("T2",), quote="Not that it, no"):
    return json.dumps(
        {"status": status, "evidence_turns": list(turns), "evidence_quote": quote, "reason": "r"}
    )


# ------------------------------------------------------------------ one call per group
def test_one_call_for_ten_claims_transcript_sent_once():
    cs = claims(10)
    fake = Fake(batch_reply(*[item(c.claim_id) for c in cs]))
    recs = verify_batch(ENC, cs, fake)
    assert len(fake.calls) == 1 and len(recs) == 10
    system, user, max_out = fake.calls[0]
    assert (
        user.count("TRANSCRIPT:") == 1 and user.count("T2 [Patient] (2.5s): Not that it, no.") == 1
    )
    for c in cs:
        assert f"{c.claim_id}: [Subjective] {c.claim_text}" in user
    assert system == BATCH_SYSTEM_PROMPT and max_out > 600
    assert all(r.via == "batch" and r.verdict.status == "supported" for r in recs)


def test_batch_prompt_keeps_the_single_claim_rules():
    for needle in (
        "ONLY source of truth",
        "contradicted",
        "unsupported",
        "uncertain",
        "Not documented",
        "verbatim",
    ):
        assert needle in BATCH_SYSTEM_PROMPT
    assert '"verdicts"' in BATCH_SYSTEM_PROMPT and "audit several claims" in BATCH_SYSTEM_PROMPT


def test_results_are_keyed_by_claim_id_not_position():
    cs = claims(2)
    recs = verify_batch(
        ENC, cs, Fake(batch_reply(item("C2", "uncertain", turns=(), quote=""), item("C1")))
    )
    assert [(r.claim_id, r.verdict.status) for r in recs] == [
        ("C1", "supported"),
        ("C2", "uncertain"),
    ]


# ------------------------------------------------------------------ same per-claim checks
def test_quote_check_runs_per_claim_and_keeps_status():
    cs = claims(2)
    fake = Fake(batch_reply(item("C1"), item("C2", quote="invented words", turns=("T2",))))
    r1, r2 = verify_batch(ENC, cs, fake)
    assert r1.quote_valid is True
    assert r2.quote_valid is False and r2.verdict.status == "supported"


def test_quote_must_be_in_the_cited_turn_of_that_claim():
    cs = claims(1)
    [r] = verify_batch(
        ENC, cs, Fake(batch_reply(item("C1", turns=("T1",), quote="Not that it, no")))
    )
    assert r.quote_valid is False  # quote exists, but in T2 which was not cited


# ------------------------------------------------------------------ missing claims -> single retry
def test_only_missing_claims_are_retried_individually():
    cs = claims(3)
    fake = Fake(batch_reply(item("C1"), item("C3")), single_reply())
    recs = verify_batch(ENC, cs, fake)
    assert len(fake.calls) == 2  # 1 batch + 1 single for C2 only
    _, single_user, _ = fake.calls[1]
    assert (
        "C2" not in single_user and "Patient reports no thing2" in single_user
    )  # single-claim prompt
    assert (
        "Patient reports no thing1" not in single_user
        and "Patient reports no thing3" not in single_user
    )
    assert [r.via for r in recs] == ["batch", "single-retry", "batch"]


def test_invalid_entry_counts_as_missing_and_is_retried():
    cs = claims(2)
    bad = item("C2", "contradicted", turns=(), quote="")  # contradicted without evidence -> invalid
    fake = Fake(batch_reply(item("C1"), bad), single_reply())
    recs = verify_batch(ENC, cs, fake)
    assert (
        len(fake.calls) == 2
        and recs[1].via == "single-retry"
        and recs[1].verdict.status == "supported"
    )


def test_supported_without_evidence_is_retried_unless_not_documented():
    cs = [ClaimInput(encounter_id="e1", claim_id="C1", section="plan", claim_text="Advise rest")]
    bare = item("C1", "supported", turns=(), quote="")
    fake = Fake(batch_reply(bare), single_reply())
    assert verify_batch(ENC, cs, fake)[0].via == "single-retry"
    nd = [
        ClaimInput(
            encounter_id="e1", claim_id="C1", section="objective", claim_text="Not documented"
        )
    ]
    fake = Fake(batch_reply(bare))
    assert verify_batch(ENC, nd, fake)[0].via == "batch" and len(fake.calls) == 1


def test_unknown_and_duplicate_ids_are_ignored():
    cs = claims(1)
    fake = Fake(batch_reply(item("C9"), item("C1"), item("C1", "uncertain", turns=(), quote="")))
    [r] = verify_batch(ENC, cs, fake)
    assert r.verdict.status == "supported" and len(fake.calls) == 1  # first C1 wins


def test_unusable_batch_is_retried_once_then_claims_go_singly():
    cs = claims(2)
    fake = Fake("garbage", "still garbage", single_reply(), single_reply())
    recs = verify_batch(ENC, cs, fake)
    assert len(fake.calls) == 4 and all(r.via == "single-retry" for r in recs)


def test_batch_retry_succeeds_without_singles():
    cs = claims(2)
    fake = Fake("garbage", batch_reply(item("C1"), item("C2")))
    recs = verify_batch(ENC, cs, fake)
    assert len(fake.calls) == 2 and all(r.via == "batch" for r in recs)


def test_single_that_also_fails_becomes_an_error_record_not_a_crash():
    fake = Fake(batch_reply(), "garbage", "garbage")
    [r] = verify_batch(ENC, claims(1), fake)
    assert r.verdict is None and "invalid verifier output" in r.error


# ------------------------------------------------------------------ grouping, resume, isolation
def claim_lines(prompt):
    return len(re.findall(r"^C\d+: ", prompt, flags=re.MULTILINE))


def ok_batch(cs):
    return batch_reply(*[item(c.claim_id) for c in cs])


def test_groups_of_ten_per_encounter_and_resume(tmp_path):
    cs = claims(25)
    fake = Fake(ok_batch(cs[:10]), ok_batch(cs[10:20]), ok_batch(cs[20:]))
    logs = []
    assert verify_all_batched(cs, {"e1": ENC}, fake, tmp_path, 10, log=logs.append) == 25
    assert len(fake.calls) == 3
    sizes = [claim_lines(c[1]) for c in fake.calls]
    assert sizes == [10, 10, 5]
    vf = load_verdict_file("e1", tmp_path)
    assert (
        vf.mode == "batched" and vf.batch_size == 10 and vf.prompt_version == BATCH_PROMPT_VERSION
    )
    assert vf.model == "fake-verifier" and len(vf.records) == 25
    assert vf.usage == {"prompt": 300, "completion": 60, "calls": 3}
    # resume: nothing pending -> zero calls
    again = Fake()
    assert (
        verify_all_batched(cs, {"e1": ENC}, again, tmp_path, 10, log=logs.append) == 0
        and not again.calls
    )


def test_resume_only_asks_for_pending_claims(tmp_path):
    cs = claims(12)
    verify_all_batched(
        cs, {"e1": ENC}, Fake(ok_batch(cs[:10])), tmp_path, 10, limit=10, log=lambda _: None
    )
    assert len(load_verdict_file("e1", tmp_path).records) == 10
    fake = Fake(ok_batch(cs[10:]))
    assert verify_all_batched(cs, {"e1": ENC}, fake, tmp_path, 10, log=lambda _: None) == 2
    assert claim_lines(fake.calls[0][1]) == 2  # only C11, C12 were sent


def test_limit_counts_claims(tmp_path):
    cs = claims(25)
    fake = Fake(ok_batch(cs[:10]), ok_batch(cs[10:15]))
    assert (
        verify_all_batched(cs, {"e1": ENC}, fake, tmp_path, 10, limit=15, log=lambda _: None) == 15
    )
    assert len(fake.calls) == 2


def test_batched_run_refuses_a_single_mode_file(tmp_path):
    from pipeline.verify import verify_all

    cs = claims(1)
    verify_all(cs, {"e1": ENC}, Fake(single_reply()), tmp_path, log=lambda _: None)
    with pytest.raises(RuntimeError):
        verify_all_batched(cs, {"e1": ENC}, Fake(ok_batch(cs)), tmp_path, log=lambda _: None)


def test_batches_never_mix_encounters(tmp_path):
    enc2 = ENC.model_copy(update={"id": "e2"})
    cs = claims(2) + [
        ClaimInput(encounter_id="e2", claim_id="C1", section="plan", claim_text="Advise rest")
    ]
    fake = Fake(ok_batch(cs[:2]), batch_reply(item("C1")))
    verify_all_batched(cs, {"e1": ENC, "e2": enc2}, fake, tmp_path, 10, log=lambda _: None)
    assert (
        len(fake.calls) == 2
        and "Advise rest" not in fake.calls[0][1]
        and "no thing" not in fake.calls[1][1]
    )


def test_run_verify_has_separate_default_dirs():
    from pipeline.verify import BATCH_VERDICTS_DIR, VERDICTS_DIR

    assert BATCH_VERDICTS_DIR != VERDICTS_DIR and BATCH_VERDICTS_DIR.name == "verifier-v0-batched"


# ------------------------------------------------------------------ token ledger / running total
def test_tokens_last_24h_counts_only_recent_entries_of_that_model(tmp_path):
    now = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
    rows = [
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "model": "m",
            "prompt": 4000,
            "completion": 100,
        },
        {
            "ts": (now - timedelta(hours=23)).isoformat(),
            "model": "m",
            "prompt": 1000,
            "completion": 0,
        },
        {
            "ts": (now - timedelta(hours=25)).isoformat(),
            "model": "m",
            "prompt": 9999,
            "completion": 1,
        },
        {
            "ts": (now - timedelta(hours=2)).isoformat(),
            "model": "other",
            "prompt": 5000,
            "completion": 0,
        },
    ]
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    assert tokens_last_24h(p, "m", now) == 5100
    assert tokens_last_24h(p, None, now) == 10100
    assert tokens_last_24h(tmp_path / "missing.jsonl", "m", now) == 0


def test_usage_line_shows_last_call_run_total_and_cap(tmp_path):
    fake = Fake()
    fake.last_usage, fake.total_prompt, fake.total_completion = (4100, 900), 8200, 1800
    line = usage_line(fake)
    assert "4,100+900" in line and "this run 10,000" in line and "/200,000" in line


def test_every_batch_logs_tokens():
    logs = []
    verify_all_batched(
        claims(2),
        {"e1": ENC},
        Fake(batch_reply(item("C1"), item("C2"))),
        Path_tmp(),
        10,
        log=logs.append,
    )
    assert any("batch of 2" in line and "tokens last call 100+20" in line for line in logs)


def Path_tmp():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())
