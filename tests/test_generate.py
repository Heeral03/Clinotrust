import json

import pytest

from data.loaders import load_primock57
from pipeline.generate import SYSTEM_PROMPT, generate_soap
from pipeline.schemas import SOAPNote


class Fake:
    def __init__(self, *outs):
        self.outs, self.calls = list(outs), 0

    def complete(self, system, user):
        self.calls += 1
        return self.outs.pop(0)


ENC = load_primock57()[0]
GOOD = json.dumps(
    {
        "subjective": ["Patient reports diarrhoea for three days"],
        "objective": [],
        "assessment": [],
        "plan": ["  "],
    }
)


def test_empty_sections_become_not_documented():
    n = generate_soap(ENC, Fake(GOOD))
    assert isinstance(n, SOAPNote)
    assert n.objective == n.assessment == n.plan == ["Not documented"]
    assert n.subjective == ["Patient reports diarrhoea for three days"]


def test_retry_on_bad_json_then_ok():
    c = Fake("not json", GOOD)
    generate_soap(ENC, c)
    assert c.calls == 2


def test_gives_up_after_retry():
    with pytest.raises(ValueError):
        generate_soap(ENC, Fake("{}", '{"subjective": "x"}'))


def test_prompt_has_key_rules():
    assert "Not documented" in SYSTEM_PROMPT and "negations" in SYSTEM_PROMPT


def test_extract_json_tolerates_fences():
    from pipeline.generate import extract_json

    assert extract_json('```json\n{"a": [1]}\n```') == {"a": [1]}
