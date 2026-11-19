import hashlib
import json
import sys

import pytest

from data.schemas import Encounter, Turn
from pipeline.verify import (
    ANALYSIS_MAX_WORDS,
    BATCH_SYSTEM_PROMPT,
    BATCH_SYSTEM_PROMPT_V1,
    SPECS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V1,
    V0,
    V1,
    ClaimInput,
    Verdict,
    check_quote_v1,
    load_verdict_file,
    needs_review_of,
    verify_all_batched,
    verify_batch,
    verify_claim,
)

ENC = Encounter(
    id="e1", source="primock57", reference_note="",
    turns=[
        Turn(id="T1", speaker="doctor", text="Any palpitations?", start=1.0, end=2.0),
        Turn(id="T2", speaker="patient", text="No.", start=2.5, end=3.0),
        Turn(id="T3", speaker="doctor", text="Any headaches? Any problems with your vision?", start=3.5, end=5.0),
        Turn(id="T4", speaker="patient", text="No.", start=5.2, end=5.5),
        Turn(id="T5", speaker="doctor", text="You're passing urine OK.", start=6.0, end=7.0),
        Turn(id="T6", speaker="patient", text="I I think so.", start=7.2, end=8.0),
    ],
)  # fmt: skip
CLAIM = ClaimInput(
    encounter_id="e1",
    claim_id="C1",
    section="subjective",
    claim_text="Patient reports no palpitations",
)


def v(**kw):
    base = {"analysis": "Doctor asked, patient denied.", "status": "supported",
            "evidence_turns": ["T1", "T2"], "evidence_quote": "No."}  # fmt: skip
    return Verdict(**{**base, **kw})


class Fake:
    model = "fake-verifier"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, user, max_out=None):
        self.calls.append((system, user, max_out))
        return self.replies.pop(0)


def reply(**kw):
    base = {"analysis": "Doctor asked, patient denied.", "status": "supported",
            "evidence_turns": ["T1", "T2"], "evidence_quote": "No."}  # fmt: skip
    return json.dumps({**base, **kw})


def entry(cid, **kw):
    return {"claim_id": cid, **json.loads(reply(**kw))}


# ------------------------------------------------------------------ v0 is untouched
def test_v0_prompts_are_byte_identical_to_the_version_that_produced_the_v0_outputs():
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()
    assert h(SYSTEM_PROMPT) == "20b9968a2104d631e188093fa1c0c07fd9c9eff4428230b6e1b4cddf71853e30"
    assert (
        h(BATCH_SYSTEM_PROMPT) == "0ed90adde0c9bc5814b4c14e9aac5394b3c5ef588e4c4662de120ea69a939b22"
    )
    assert V0.single_system == SYSTEM_PROMPT and V0.batch_system == BATCH_SYSTEM_PROMPT
    assert (V0.single_version, V0.batch_version) == ("verify-prompt-v0", "verify-prompt-v0-batch")
    assert not V0.search_all_turns and V1.search_all_turns and {"v0", "v1"} <= set(SPECS)


# ------------------------------------------------------------------ v1 prompt content
def test_v1_prompt_defines_statuses_and_rules():
    p = SYSTEM_PROMPT_V1
    for needle in (
        "including every qualifier", "directly state the opposite", "never mentioned",
        "hedged, ambiguous, partial, or only implied", "prescribe vs advise", '"only"', "as needed",
        "not painful", "not very", "drug allergies vs allergies", "I think so", "uncertain",
        "question turn", "answers a different question", "Objective is supported ONLY",
        "patient-reported answers do not support an", "no examination findings or measurements",
        "at most 50 words", "No step-by-step thinking dump", "MUST follow",
    ):  # fmt: skip
        assert needle in p, needle


def test_v1_output_key_order_is_analysis_first():
    single = SYSTEM_PROMPT_V1.rstrip().splitlines()[-1]
    assert (
        single.index('"analysis"')
        < single.index('"status"')
        < single.index('"evidence_turns"')
        < single.index('"evidence_quote"')
    )
    batch = BATCH_SYSTEM_PROMPT_V1.rstrip().splitlines()[-1]
    assert batch.index('"claim_id"') < batch.index('"analysis"') < batch.index('"status"')
    assert "audit several claims" in BATCH_SYSTEM_PROMPT_V1


def test_verdict_serialises_analysis_before_status():
    keys = list(json.loads(v().model_dump_json()))
    assert (
        keys.index("analysis")
        < keys.index("status")
        < keys.index("evidence_turns")
        < keys.index("evidence_quote")
    )


# ------------------------------------------------------------------ analysis limit
def test_truncation_sets_flag_and_leaves_verdict_and_quote_check_alone():
    long_ = " ".join(["word"] * (ANALYSIS_MAX_WORDS + 30))
    rec = verify_claim(ENC, CLAIM, Fake(reply(analysis=long_)), spec=V1)
    assert rec.analysis_truncated is True
    ref = verify_claim(ENC, CLAIM, Fake(reply(analysis="short")), spec=V1)
    assert ref.analysis_truncated is False
    for f in ("status", "evidence_turns", "evidence_quote"):
        assert getattr(rec.verdict, f) == getattr(ref.verdict, f)
    assert (rec.quote_valid, rec.turns_valid) == (ref.quote_valid, ref.turns_valid)
    # a model cannot set the flag itself
    sneaky = reply(analysis="short", analysis_truncated=True)
    assert verify_claim(ENC, CLAIM, Fake(sneaky), spec=V1).analysis_truncated is False


def test_analysis_is_required_and_truncated_to_50_words_for_v1():
    ok = " ".join(["word"] * ANALYSIS_MAX_WORDS)
    assert verify_claim(ENC, CLAIM, Fake(reply(analysis=ok)), spec=V1).verdict is not None
    too_long = " ".join(["word"] * (ANALYSIS_MAX_WORDS + 1))
    fake = Fake(reply(analysis=too_long))
    rec = verify_claim(ENC, CLAIM, fake, spec=V1)  # too long: truncated, verdict kept, no retry
    assert rec.verdict is not None and len(rec.verdict.analysis.split()) == ANALYSIS_MAX_WORDS
    assert len(fake.calls) == 1
    assert (
        verify_claim(ENC, CLAIM, Fake(reply(analysis=""), reply(analysis="")), spec=V1).verdict
        is None
    )


def test_overlong_analysis_is_truncated_without_a_retry():
    fake = Fake(reply(analysis=" ".join(["w"] * 80)), reply())
    rec = verify_claim(ENC, CLAIM, fake, spec=V1)
    assert rec.verdict.status == "supported" and len(fake.calls) == 1
    assert rec.analysis_truncated is True
    assert fake.calls[0][0] == SYSTEM_PROMPT_V1


def test_v0_still_requires_a_reason_and_ignores_analysis():
    only_analysis = json.dumps(
        {"analysis": "x", "status": "supported", "evidence_turns": ["T2"], "evidence_quote": "No."}
    )
    assert verify_claim(ENC, CLAIM, Fake(only_analysis, only_analysis), spec=V0).verdict is None


# ------------------------------------------------------------------ quote check v1
def test_quote_in_cited_turn_changes_nothing():
    qc = check_quote_v1(
        ENC, v(evidence_turns=["T1", "T2"], evidence_quote="Any palpitations? ... No.")
    )
    assert (qc.quote_valid, qc.turn_corrected, qc.turns) == (True, False, ["T1", "T2"])


def test_quote_found_in_a_different_turn_corrects_the_turn_id():
    qc = check_quote_v1(ENC, v(evidence_turns=["T3"], evidence_quote="You're passing urine OK."))
    assert qc.quote_valid is True and qc.turn_corrected is True and qc.turns == ["T5"]


def test_quote_not_found_anywhere_is_invalid_and_not_corrected():
    qc = check_quote_v1(ENC, v(evidence_turns=["T2"], evidence_quote="completely invented words"))
    assert qc.quote_valid is False and qc.turn_corrected is False and qc.turns == ["T2"]


def test_one_fragment_missing_everywhere_makes_it_invalid():
    qc = check_quote_v1(
        ENC, v(evidence_turns=["T5"], evidence_quote="You're passing urine OK. ... invented")
    )
    assert qc.quote_valid is False and qc.turn_corrected is False


def test_duplicate_quote_text_picks_the_nearest_turn_to_the_cited_one():
    # "No." is in T2 and T4. Cited T3 (the doctor's question): T2 and T4 are equally near, earlier wins.
    assert check_quote_v1(ENC, v(evidence_turns=["T3"], evidence_quote="No.")).turns == ["T2"]
    # cited T5: T4 is the nearest "No."
    assert check_quote_v1(ENC, v(evidence_turns=["T5"], evidence_quote="No.")).turns == ["T4"]


def test_mixed_fragments_keep_good_cited_turns_and_fix_the_wrong_one():
    qc = check_quote_v1(
        ENC,
        v(evidence_turns=["T5", "T1"], evidence_quote="You're passing urine OK. ... I I think so."),
    )
    assert qc.turn_corrected is True and qc.turns == ["T5", "T6"]  # T1 held no part of the quote


def test_unknown_cited_turn_flagged_but_quote_can_still_be_found_elsewhere():
    qc = check_quote_v1(ENC, v(evidence_turns=["T99"], evidence_quote="I I think so."))
    assert (
        qc.turns_valid is False
        and qc.quote_valid is True
        and qc.turn_corrected is True
        and qc.turns == ["T6"]
    )


def test_no_quote_means_nothing_to_check():
    qc = check_quote_v1(ENC, v(status="uncertain", evidence_turns=[], evidence_quote=""))
    assert qc.quote_valid is None and qc.turn_corrected is False


def test_quote_check_is_whitespace_tolerant_but_case_exact():
    assert (
        check_quote_v1(ENC, v(evidence_turns=["T6"], evidence_quote="I  I think so.")).quote_valid
        is True
    )
    assert (
        check_quote_v1(ENC, v(evidence_turns=["T6"], evidence_quote="i i think so.")).quote_valid
        is False
    )


# ------------------------------------------------------------------ needs_review
@pytest.mark.parametrize(
    ("status", "quote_valid", "expected"),
    [("supported", True, False), ("supported", None, False), ("supported", False, True),
     ("uncertain", True, True), ("contradicted", None, True), ("unsupported", None, True),
     ("uncertain", False, True)],
)  # fmt: skip
def test_needs_review_definition(status, quote_valid, expected):
    assert needs_review_of(status, quote_valid) is expected


# ------------------------------------------------------------------ records
def test_v1_record_corrects_turns_and_flags_review_when_quote_is_unfindable():
    rec = verify_claim(
        ENC,
        CLAIM,
        Fake(reply(evidence_turns=["T3"], evidence_quote="You're passing urine OK.")),
        spec=V1,
    )
    assert rec.turn_corrected is True and rec.cited_turns_original == ["T3"]
    assert rec.verdict.evidence_turns == ["T5"] and rec.quote_valid is True
    assert rec.needs_review is False  # supported + valid quote
    bad = verify_claim(ENC, CLAIM, Fake(reply(evidence_quote="invented words here")), spec=V1)
    assert bad.quote_valid is False and bad.verdict.status == "supported"  # status kept
    assert bad.needs_review is True  # but it goes to the clinician


def test_v0_records_get_needs_review_but_no_turn_correction():
    rec = verify_claim(
        ENC,
        CLAIM,
        Fake(
            json.dumps(
                {
                    "status": "supported",
                    "evidence_turns": ["T2"],
                    "evidence_quote": "No.",
                    "reason": "r",
                }
            )
        ),
        spec=V0,
    )
    assert (
        rec.turn_corrected is None and rec.needs_review is False and rec.cited_turns_original == []
    )


def test_v1_batch_uses_the_v1_prompt_and_falls_back_with_the_v1_single_prompt():
    cs = [
        CLAIM,
        ClaimInput(
            encounter_id="e1",
            claim_id="C2",
            section="subjective",
            claim_text="Patient reports no headaches",
        ),
    ]
    batch = json.dumps(
        {"verdicts": [entry("C1"), {**entry("C2"), "analysis": ""}]}
    )  # C2 has no analysis
    fake = Fake(batch, reply())
    recs = verify_batch(ENC, cs, fake, spec=V1)
    assert fake.calls[0][0] == BATCH_SYSTEM_PROMPT_V1 and fake.calls[1][0] == SYSTEM_PROMPT_V1
    assert [r.via for r in recs] == ["batch", "single-retry"] and all(
        r.needs_review is False for r in recs
    )


# ------------------------------------------------------------------ files and isolation
def test_v1_batched_run_saves_version_and_refuses_to_resume_a_v0_file(tmp_path):
    fake = Fake(json.dumps({"verdicts": [entry("C1")]}))
    verify_all_batched([CLAIM], {"e1": ENC}, fake, tmp_path, log=lambda _: None, spec=V1)
    vf = load_verdict_file("e1", tmp_path)
    assert vf.prompt_version == "verify-prompt-v1-batch" and vf.mode == "batched"
    raw = json.loads((tmp_path / "e1.json").read_text())
    assert "needs_review" in raw["records"][0] and "turn_corrected" in raw["records"][0]
    assert list(raw["records"][0]["verdict"]).index("analysis") < list(
        raw["records"][0]["verdict"]
    ).index("status")
    v0_fake = Fake(json.dumps({"verdicts": [{**entry("C1"), "reason": "r"}]}))
    with pytest.raises(RuntimeError):
        verify_all_batched([CLAIM], {"e1": ENC}, v0_fake, tmp_path, log=lambda _: None, spec=V0)


@pytest.mark.parametrize("extra", [[], ["--batched"]])
def test_run_verify_refuses_v1_into_a_v0_folder(monkeypatch, extra):
    from pipeline import run_verify
    from pipeline.verify import BATCH_VERDICTS_DIR, VERDICTS_DIR

    target = BATCH_VERDICTS_DIR if extra else VERDICTS_DIR
    monkeypatch.setattr(sys, "argv", ["run_verify", "--prompt", "v1", "--out", str(target), *extra])
    with pytest.raises(SystemExit) as e:  # argparse error happens before any client/API use
        run_verify.main()
    assert e.value.code == 2


def test_v1_default_folders_are_separate_from_v0():
    from pipeline.verify import (
        BATCH_VERDICTS_DIR,
        V1_BATCH_VERDICTS_DIR,
        V1_VERDICTS_DIR,
        VERDICTS_DIR,
    )

    assert V1_BATCH_VERDICTS_DIR.as_posix() == "outputs/verdicts/verifier-v1-batched"
    assert len({VERDICTS_DIR, BATCH_VERDICTS_DIR, V1_VERDICTS_DIR, V1_BATCH_VERDICTS_DIR}) == 4


def test_failed_claim_keeps_last_raw_reply_and_error_and_old_files_still_load():
    from pipeline.verify import RAW_OUTPUT_MAX, VerdictRecord

    junk = "not json " + "x" * 5000
    rec = verify_claim(ENC, CLAIM, Fake(reply(analysis=""), junk), spec=V1)
    assert rec.verdict is None and rec.error.startswith("invalid verifier output")
    assert rec.raw_output == junk[:RAW_OUTPUT_MAX] and len(rec.raw_output) == RAW_OUTPUT_MAX
    ok = verify_claim(ENC, CLAIM, Fake(reply()), spec=V1)
    assert ok.raw_output is None  # nothing stored on success
    old = {
        "encounter_id": "e",
        "claim_id": "C1",
        "section": "plan",
        "claim_text": "t",
        "error": "x",
    }
    assert VerdictRecord.model_validate(old).raw_output is None  # files written before still load
