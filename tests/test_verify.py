import json

import pytest

from data.schemas import Encounter, Turn
from pipeline.verify import (
    SYSTEM_PROMPT,
    ClaimInput,
    Verdict,
    VerdictFile,
    VerdictRecord,
    assert_independent,
    check_quote,
    load_claims_for_verification,
    load_verdict_file,
    parse_duration,
    save_verdict_file,
    verify_all,
    verify_claim,
)

ENC = Encounter(
    id="e1", source="primock57", reference_note="",
    turns=[
        Turn(id="T1", speaker="doctor", text="Any   palpitations?", start=1.0, end=2.0),
        Turn(id="T2", speaker="patient", text="Not that it, no.", start=2.5, end=3.0),
        Turn(id="T3", speaker="doctor", text="Any headaches? Any problems with your vision?", start=3.5, end=5.0),
        Turn(id="T4", speaker="patient", text="No.", start=5.2, end=5.5),
    ],
)  # fmt: skip
CLAIM = ClaimInput(
    encounter_id="e1",
    claim_id="C1",
    section="subjective",
    claim_text="Patient reports no palpitations",
)


def v(**kw):
    base = {
        "status": "supported",
        "evidence_turns": ["T1", "T2"],
        "evidence_quote": "Not that it, no",
        "reason": "r",
    }
    return Verdict(**{**base, **kw})


# ------------------------------------------------------------------ quote check
def test_quote_found_in_cited_turn():
    assert check_quote(ENC, v()) == (True, True)


def test_quote_whitespace_is_normalised():
    assert check_quote(ENC, v(evidence_quote="Any palpitations?"))[0] is True  # turn has 3 spaces


def test_quote_fragments_joined_with_ellipsis():
    assert check_quote(ENC, v(evidence_quote="Any palpitations? ... Not that it, no."))[0] is True
    assert check_quote(ENC, v(evidence_quote="Any palpitations? … no."))[0] is True


def test_quote_not_in_any_cited_turn_is_flagged_but_status_kept():
    r = verify_claim(ENC, CLAIM, Fake(json.dumps({"status": "supported", "evidence_turns": ["T1"],
                                                   "evidence_quote": "Not that it, no", "reason": "r"})))  # fmt: skip
    # quote exists in T2, but only T1 was cited -> invalid; the model's status is kept
    assert r.quote_valid is False and r.verdict.status == "supported" and r.error is None


def test_altered_quote_is_invalid():
    assert check_quote(ENC, v(evidence_quote="Not that, no"))[0] is False
    assert check_quote(ENC, v(evidence_quote="not that it, no"))[0] is False  # case must match


def test_one_bad_fragment_makes_quote_invalid():
    assert (
        check_quote(ENC, v(evidence_quote="Any palpitations? ... something invented"))[0] is False
    )


def test_unknown_turn_id_flagged():
    quote_valid, turns_valid = check_quote(ENC, v(evidence_turns=["T2", "T99"]))
    assert turns_valid is False and quote_valid is True


def test_no_quote_means_nothing_to_check():
    assert check_quote(ENC, v(status="uncertain", evidence_turns=[], evidence_quote=""))[0] is None


# ------------------------------------------------------------------ verdict model
def test_unsupported_evidence_is_forced_empty():
    x = v(status="unsupported", evidence_turns=["T1"], evidence_quote="Any palpitations?")
    assert x.evidence_turns == [] and x.evidence_quote == ""


def test_contradicted_needs_evidence_and_ids_must_look_like_turns():
    with pytest.raises(ValueError):
        v(status="contradicted", evidence_turns=[], evidence_quote="")
    with pytest.raises(ValueError):
        v(evidence_turns=["12"])
    with pytest.raises(ValueError):
        v(status="maybe")


# ------------------------------------------------------------------ verify_claim
class Fake:
    model = "fake-verifier"

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, system, user, max_out=None):
        self.prompts.append((system, user))
        return self.replies.pop(0)


GOOD = json.dumps(
    {
        "status": "supported",
        "evidence_turns": ["T1", "T2"],
        "evidence_quote": "Not that it, no",
        "reason": "denied",
    }
)


def test_retry_once_on_invalid_json_then_ok():
    fake = Fake("garbage", GOOD)
    r = verify_claim(ENC, CLAIM, fake)
    assert len(fake.prompts) == 2 and r.verdict.status == "supported" and r.error is None


def test_two_invalid_answers_record_an_error_not_a_crash():
    r = verify_claim(ENC, CLAIM, Fake("garbage", '{"status": "maybe"}'))
    assert r.verdict is None and "invalid verifier output" in r.error


def test_supported_without_evidence_is_rejected_except_for_not_documented():
    bare = json.dumps(
        {"status": "supported", "evidence_turns": [], "evidence_quote": "", "reason": "r"}
    )
    assert verify_claim(ENC, CLAIM, Fake(bare, bare)).verdict is None
    nd = ClaimInput(
        encounter_id="e1", claim_id="C2", section="objective", claim_text="Not documented"
    )
    assert verify_claim(ENC, nd, Fake(bare)).verdict.status == "supported"


def test_prompt_has_numbered_transcript_with_times_then_tagged_claim():
    fake = Fake(GOOD)
    verify_claim(ENC, CLAIM, fake)
    system, user = fake.prompts[0]
    assert "T2 [Patient] (2.5s): Not that it, no." in user
    assert user.rstrip().endswith("CLAIM:\n[Subjective] Patient reports no palpitations")
    assert "ONLY source of truth" in system


# ------------------------------------------------------------------ gold-label blindness
def test_only_four_columns_are_read_and_prompt_never_sees_labels(tmp_path):
    p = tmp_path / "claims.csv"
    p.write_text(
        "encounter_id,claim_id,section,source_statement_index,claim_text,status,evidence_turns,"
        "evidence_quote,error_type,severity,notes\n"
        "e1,C1,subjective,0,Patient reports no palpitations,contradicted,T99,SECRETQUOTE,decomposition_error,major,SECRETNOTE\n"
    )
    [c] = load_claims_for_verification(p)
    assert set(c.model_dump()) == {"encounter_id", "claim_id", "section", "claim_text"}
    fake = Fake(GOOD)
    verify_claim(ENC, c, fake)
    sent = " ".join(fake.prompts[0])
    for secret in (
        "SECRETQUOTE",
        "SECRETNOTE",
        "decomposition_error",
        "T99",
    ):  # not "contradicted": the prompt defines it
        assert secret not in sent


def test_claims_filter_by_encounter(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("encounter_id,claim_id,section,claim_text\na,C1,plan,x\nb,C1,plan,y\n")
    assert [c.encounter_id for c in load_claims_for_verification(p, {"b"})] == ["b"]


# ------------------------------------------------------------------ independence, persistence, pacing
def test_verifier_must_differ_from_generator():
    assert_independent("qwen/qwen3.8-27b", {"openai/gpt-oss-20b"})
    with pytest.raises(RuntimeError):
        assert_independent("openai/gpt-oss-20b", {"openai/gpt-oss-20b"})


def test_verify_all_saves_resumes_and_records_model(tmp_path):
    claims = [
        CLAIM,
        ClaimInput(encounter_id="e1", claim_id="C2", section="plan", claim_text="Advise rest"),
    ]
    fake = Fake(GOOD, GOOD)
    logs = []
    assert verify_all(claims, {"e1": ENC}, fake, tmp_path, limit=1, log=logs.append) == 1
    vf = load_verdict_file("e1", tmp_path)
    assert (
        vf.model == "fake-verifier"
        and vf.prompt_version == "verify-prompt-v0"
        and len(vf.records) == 1
    )
    assert verify_all(claims, {"e1": ENC}, fake, tmp_path, log=logs.append) == 1  # only C2 is new
    assert len(load_verdict_file("e1", tmp_path).records) == 2
    assert (
        verify_all(claims, {"e1": ENC}, Fake(), tmp_path, log=logs.append) == 0
    )  # nothing left, no calls


def test_verify_all_refuses_to_mix_models_in_one_file(tmp_path):
    save_verdict_file(VerdictFile(encounter_id="e1", model="other-model", records=[
        VerdictRecord(encounter_id="e1", claim_id="C9", section="plan", claim_text="x", error="e")]), tmp_path)  # fmt: skip
    with pytest.raises(RuntimeError):
        verify_all([CLAIM], {"e1": ENC}, Fake(GOOD), tmp_path, log=lambda _: None)


def test_parse_duration():
    assert parse_duration("31.409s") == pytest.approx(31.409)
    assert parse_duration("1m26.4s") == pytest.approx(86.4)
    assert parse_duration("922ms") == pytest.approx(0.922)
    assert parse_duration("garbage") == 30.0


def test_system_prompt_defines_all_statuses():
    for word in ("supported", "contradicted", "unsupported", "uncertain", "Not documented"):
        assert word in SYSTEM_PROMPT
