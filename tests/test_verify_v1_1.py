from pipeline.verify import (
    BATCH_SYSTEM_PROMPT_V1,
    BATCH_SYSTEM_PROMPT_V1_1,
    SPECS,
    SYSTEM_PROMPT_V1,
    SYSTEM_PROMPT_V1_1,
    V1_1,
    V1_1_BATCH_VERDICTS_DIR,
    V1_1_VERDICTS_DIR,
)


def test_specs_has_v1_1():
    assert SPECS["v1.1"] is V1_1
    assert V1_1.single_version == "verify-prompt-v1.1"
    assert V1_1.batch_version == "verify-prompt-v1.1-batch"
    assert str(V1_1_VERDICTS_DIR) == "outputs/verdicts/verifier-v1.1"
    assert str(V1_1_BATCH_VERDICTS_DIR) == "outputs/verdicts/verifier-v1.1-batched"


def test_v1_1_has_three_changes():
    p = SYSTEM_PROMPT_V1_1
    assert "Hedging: a clear yes or no followed by a softener" in p
    assert "Hedged answers" not in p
    assert "Scope test: could the claim be false" in p
    assert 'that "No." answers the LAST question' in p


def test_v1_prompt_unchanged():
    assert "Hedged answers" in SYSTEM_PROMPT_V1
    assert "Scope test" not in SYSTEM_PROMPT_V1
    assert "answers the LAST question" not in SYSTEM_PROMPT_V1
    assert "Scope test" not in BATCH_SYSTEM_PROMPT_V1


def test_v1_1_batch_format_and_changes():
    assert BATCH_SYSTEM_PROMPT_V1_1.rstrip().endswith("}]}")
    assert '{"verdicts": [{"claim_id": "C1"' in BATCH_SYSTEM_PROMPT_V1_1.splitlines()[-1]
    assert "audit several claims" in BATCH_SYSTEM_PROMPT_V1_1
    assert "Scope test" in BATCH_SYSTEM_PROMPT_V1_1
    assert BATCH_SYSTEM_PROMPT_V1_1.endswith(BATCH_SYSTEM_PROMPT_V1[-300:])
