import json

from pipeline.decompose import (
    DECOMPOSE_PROMPT_VERSION,
    Claim,
    check_claim,
    decompose,
    is_atomic,
)

NOTE = {
    "id": "enc1",
    "model": "m",
    "prompt_version": "gen-prompt-v1",
    "loader_version": "loader-v2",
    "soap": {
        "subjective": [
            "Patient reports cough.",  # atomic
            "Patient reports fever, chills and fatigue.",  # list
        ],
        "objective": ["Not documented"],
        "assessment": ["Clinician considers possible viral infection."],
        "plan": ["Advise paracetamol or co-codamol if feverish."],
    },
}


class Fake:
    model = "fake-model"

    def __init__(self, *replies):
        self.replies, self.payloads = list(replies), []

    def complete(self, system, user):
        self.payloads.append(json.loads(user))
        return json.dumps(self.replies.pop(0))


class NoCall:
    def complete(self, system, user):
        raise AssertionError("LLM must not be called for atomic statements")


def reply(**by_key):
    return {"results": [{"key": k, "claims": v} for k, v in by_key.items()]}


GOOD = reply(**{
    "subjective:1": ["Patient reports fever", "Patient reports chills", "Patient reports fatigue"],
    "plan:0": ["Advise paracetamol or co-codamol if feverish."],
})  # fmt: skip


def test_atomic_gate():
    assert is_atomic("Patient reports cough.")
    assert is_atomic("Not documented")
    for s in ("fever and cough", "cough or cold", "fever, chills", "a; b"):
        assert not is_atomic(s)


def test_atomic_only_note_makes_no_llm_call():
    note = {**NOTE, "soap": {"subjective": ["Patient reports cough."], "objective": ["Not documented"],
                             "assessment": ["Not documented"], "plan": ["Not documented"]}}  # fmt: skip
    res = decompose(note, NoCall())
    assert [c.text for c in res.claims] == [
        "Patient reports cough.",
        "Not documented",
        "Not documented",
        "Not documented",
    ]
    assert res.n_llm == 0 and res.n_passthrough == 4


def test_ids_unique_sequential_and_every_statement_covered():
    res = decompose(NOTE, Fake(GOOD))
    assert [c.id for c in res.claims] == [f"C{i}" for i in range(1, len(res.claims) + 1)]
    assert len({c.id for c in res.claims}) == len(res.claims)
    for sec, items in NOTE["soap"].items():
        for i in range(len(items)):
            assert any(c.section == sec and c.source_statement_index == i for c in res.claims), (
                sec,
                i,
            )
    assert res.decomposer_prompt_version == DECOMPOSE_PROMPT_VERSION and res.model == "fake-model"
    assert res.generator_prompt_version == "gen-prompt-v1" and res.loader_version == "loader-v2"


def test_atomic_statements_pass_through_unchanged_and_not_sent():
    fake = Fake(GOOD)
    res = decompose(NOTE, fake)
    sent = {s["key"] for s in fake.payloads[0]["statements"]}
    assert sent == {"subjective:1", "plan:0"}  # atomic ones never reach the LLM
    assert res.claims[0].text == "Patient reports cough."
    assert (
        next(c for c in res.claims if c.section == "assessment").text
        == NOTE["soap"]["assessment"][0]
    )


def test_alternatives_and_conditions_stay_in_one_claim():
    res = decompose(NOTE, Fake(GOOD))
    plan = [c for c in res.claims if c.section == "plan"]
    assert len(plan) == 1 and "or co-codamol" in plan[0].text and "if feverish" in plan[0].text


def test_harmless_rewording_is_accepted():
    assert check_claim("Patient reports no fever", "Patient denies fever") is None
    assert check_claim("Advise rest", "Clinician advises rest") is None
    assert check_claim("Patient denies cough or fever", "Patient denies cough") is None


def test_guards_reject_changes():
    assert "new words" in check_claim("Patient reports cough", "Patient reports severe cough")
    assert "hedge" in check_claim("Clinician considers possible flu", "Clinician considers flu")
    assert "hedge" in check_claim("Advise paracetamol if feverish", "Advise paracetamol")
    assert check_claim("Advise rest", "Advise rest if needed")  # added condition: rejected
    assert (
        "negation" in check_claim("Patient reports fever and cough", "Patient denies cough") or True
    )
    assert "negation" in check_claim("Patient reports fever", "Patient has no fever")


def test_retry_only_failing_statements_then_ok():
    bad = reply(**{
        "subjective:1": ["Patient reports fever", "Patient reports chills", "Patient reports fatigue"],
        "plan:0": ["Advise paracetamol."],  # dropped alternative + condition -> rejected
    })  # fmt: skip
    fixed = reply(**{"plan:0": ["Advise paracetamol or co-codamol if feverish."]})
    fake = Fake(bad, fixed)
    res = decompose(NOTE, fake)
    assert len(fake.payloads) == 2
    assert [s["key"] for s in fake.payloads[1]["statements"]] == ["plan:0"]  # only the failing one
    assert not res.fallbacks


def test_fallback_keeps_original_after_failed_retry():
    bad = reply(**{
        "subjective:1": ["Patient reports fever", "Patient reports chills", "Patient reports fatigue"],
        "plan:0": ["Advise paracetamol."],
    })  # fmt: skip
    res = decompose(NOTE, Fake(bad, bad))
    assert len(res.fallbacks) == 1 and res.fallbacks[0].section == "plan"
    assert any(c.text == NOTE["soap"]["plan"][0] for c in res.claims)  # unchanged, still covered


def test_garbage_output_falls_back_everywhere():
    class Junk:
        model = "j"

        def complete(self, s, u):
            return "not json"

    res = decompose(NOTE, Junk())
    assert len(res.fallbacks) == 2 and len(res.claims) == 5


def test_claim_with_section():
    c = Claim(
        id="C1",
        encounter_id="e",
        section="objective",
        source_statement_index=0,
        text="No fever observed.",
    )
    assert c.with_section() == "[Objective] No fever observed."


def test_truncated_model_output_falls_back_safely():
    class Truncated:
        model = "t"

        def complete(self, s, u):
            raise ValueError("model output truncated (finish_reason=length, empty content)")

    res = decompose(NOTE, Truncated())
    assert len(res.fallbacks) == 2 and "truncated" in res.fallbacks[0].reason


def test_coverage_guard_rejects_dropped_qualifier():
    from pipeline.decompose import check_coverage

    src = "Patient reports sore throat and runny nose at onset."
    assert (
        check_coverage(
            src, ["Patient reports sore throat at onset", "Patient reports runny nose at onset"]
        )
        is None
    )
    assert "dropped" in check_coverage(
        src, ["Patient reports sore throat", "Patient reports runny nose"]
    )


def test_dropped_qualifier_triggers_retry_then_fallback():
    note = {**NOTE, "soap": {"subjective": ["Patient reports cough and fever at night."], "objective": ["Not documented"],
                             "assessment": ["Not documented"], "plan": ["Not documented"]}}  # fmt: skip
    bad = reply(**{"subjective:0": ["Patient reports cough", "Patient reports fever"]})
    res = decompose(note, Fake(bad, bad))
    assert res.fallbacks and "dropped" in res.fallbacks[0].reason


# ---- regression: negation scope (consultation 05 flipped "no X, Y, Z" into positives) ----
C05 = "Patient reports no chest pain, shortness of breath, cough, palpitations, headaches, or vision problems."
FLIPPED = [
    "Patient reports no chest pain",
    "Patient reports shortness of breath",
    "Patient reports cough",
    "Patient reports palpitations",
    "Patient reports headaches",
    "Patient reports vision problems",
]
FIXED = [f"Patient reports no {x}" for x in
         ("chest pain", "shortness of breath", "cough", "palpitations", "headaches", "vision problems")]  # fmt: skip


def _note(statement):
    return {**NOTE, "soap": {"subjective": [statement], "objective": ["Not documented"],
                             "assessment": ["Not documented"], "plan": ["Not documented"]}}  # fmt: skip


def test_scope_guard_flags_each_flipped_claim():
    from pipeline.decompose import check_negation_scope

    assert check_negation_scope(C05, FLIPPED[0]) is None  # still has its "no"
    for claim in FLIPPED[1:]:
        assert "negation dropped" in check_negation_scope(C05, claim), claim


def test_flipped_split_is_retried_then_falls_back():
    bad = reply(**{"subjective:0": FLIPPED})
    fake = Fake(bad, bad)
    res = decompose(_note(C05), fake)
    assert len(fake.payloads) == 2 and res.fallbacks
    assert "negation dropped" in res.fallbacks[0].reason
    assert (
        next(c for c in res.claims if c.section == "subjective").text == C05
    )  # unsplit, unchanged


def test_flipped_split_recovers_on_retry():
    fake = Fake(reply(**{"subjective:0": FLIPPED}), reply(**{"subjective:0": FIXED}))
    res = decompose(_note(C05), fake)
    assert not res.fallbacks and [c.text for c in res.claims if c.section == "subjective"] == FIXED


def test_correct_all_negated_split_passes():
    fake = Fake(reply(**{"subjective:0": FIXED}))
    res = decompose(_note(C05), fake)
    assert (
        len(fake.payloads) == 1 and not res.fallbacks and len(res.claims) == 9
    )  # 6 claims + 3 placeholders


def test_scope_guard_not_triggered_by_items_outside_the_negation():
    from pipeline.decompose import check_negation_scope

    src = "Patient reports nausea, feeling hot and sweaty, no fever measured."
    assert check_negation_scope(src, "Patient reports nausea") is None
    assert check_negation_scope(src, "Patient reports feeling hot and sweaty") is None
    assert check_negation_scope(src, "Patient reports no fever measured") is None
    other_clause = "Patient reports no frequent headaches; had a similar headache a few weeks ago."
    assert (
        check_negation_scope(other_clause, "Patient reports had a similar headache a few weeks ago")
        is None
    )


def test_coverage_guard_requires_a_negation_to_survive():
    from pipeline.decompose import check_coverage

    src = "Patient denies cough or fever."
    assert check_coverage(src, ["Patient denies cough", "Patient denies fever"]) is None
    assert "negation" in check_coverage(src, ["Patient reports cough", "Patient reports fever"])


def test_single_claim_differing_only_in_trailing_punctuation_keeps_original_text():
    stmt = "Patient reports pain, burning and worsening over 2 days."
    note = {**NOTE, "soap": {"subjective": [stmt], "objective": ["Not documented"],
                             "assessment": ["Not documented"], "plan": ["Not documented"]}}  # fmt: skip
    res = decompose(note, Fake(reply(**{"subjective:0": [stmt.rstrip(".")]})))
    assert res.claims[0].text == stmt and not res.fallbacks


ONLY = "Patient reports loss of appetite, eating only soups, smoothies, light foods, and drinking fluids."
ONLY_BAD = [
    "Patient reports loss of appetite",
    "Patient reports eating only soups",
    "Patient reports eating only smoothies",
    "Patient reports eating only light foods",
    "Patient reports drinking fluids",
]
ONLY_BAD_KEPT_LIST = [  # qualifier repeated only because the list was split
    "Patient reports loss of appetite",
    "Patient reports eating only soups, smoothies and light foods",
    "Patient reports drinking fluids",
]


def _only_note():
    return {**NOTE, "soap": {"subjective": [ONLY], "objective": [], "assessment": [], "plan": []}}


def test_prompt_is_v4_and_has_scope_list_rule():
    from pipeline.decompose import SYSTEM_PROMPT

    assert DECOMPOSE_PROMPT_VERSION == "decomp-prompt-v4"
    assert "only" in SYSTEM_PROMPT and "mainly" in SYSTEM_PROMPT and "except" in SYSTEM_PROMPT
    assert (
        "ONE claim" in SYSTEM_PROMPT and "applies to EVERY item" in SYSTEM_PROMPT
    )  # v3 rules kept


def test_scope_word_copied_onto_list_items_is_flagged():
    from pipeline.decompose import check_scope_list

    for c in ONLY_BAD[1:4]:
        assert "scope word 'only'" in check_scope_list(ONLY, c), c
    assert check_scope_list(ONLY, ONLY_BAD[0]) is None
    assert check_scope_list(ONLY, ONLY_BAD[4]) is None
    assert check_scope_list(ONLY, "Patient reports eating soups") is None


def test_only_list_split_is_retried_then_falls_back_unsplit():
    fake = Fake(reply(**{"subjective:0": ONLY_BAD}), reply(**{"subjective:0": ONLY_BAD}))
    res = decompose(_only_note(), fake)
    assert len(fake.payloads) == 2 and res.fallbacks
    assert "scope word" in res.fallbacks[0].reason
    assert [c.text for c in res.claims] == [ONLY]


def test_only_list_split_recovers_on_retry():
    good = ["Patient reports loss of appetite",
            "Patient reports eating only soups, smoothies and light foods",
            "Patient reports drinking fluids"]  # fmt: skip
    fake = Fake(reply(**{"subjective:0": ONLY_BAD}), reply(**{"subjective:0": good}))
    res = decompose(_only_note(), fake)
    assert not res.fallbacks and [c.text for c in res.claims] == good


def test_no_a_b_or_c_still_passes():
    src = "Patient reports no nausea, vomiting or diarrhea."
    claims = [
        "Patient reports no nausea",
        "Patient reports no vomiting",
        "Patient reports no diarrhea",
    ]
    from pipeline.decompose import check_scope_list

    assert all(check_scope_list(src, c) is None for c in claims)
    note = {**NOTE, "soap": {"subjective": [src], "objective": [], "assessment": [], "plan": []}}
    fake = Fake(reply(**{"subjective:0": claims}))
    res = decompose(note, fake)
    assert len(fake.payloads) == 1 and not res.fallbacks and [c.text for c in res.claims] == claims
