"""Step 2: SOAP statements -> atomic claims.

Statements with no 'and' / 'or' / comma / semicolon are already atomic and pass through
untouched (no LLM call). The rest go to the LLM in ONE batched call per note, then every
returned claim is checked by code guards (new words, hedges, negation). Anything that still
fails after one retry falls back to the original statement, unchanged, and is recorded.
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from pipeline.generate import GroqClient, LLMClient, extract_json

DECOMPOSE_PROMPT_VERSION = "decomp-prompt-v4"
REASONING_EFFORT = (
    "low"  # gpt-oss at default effort exhausts 2048 tokens on reasoning -> empty output
)
MAX_TOKENS = 8192
SECTIONS = ("subjective", "objective", "assessment", "plan")

SYSTEM_PROMPT = """\
You split clinical-note statements into atomic claims. You do NOT summarise, correct or add anything.

Input: JSON {"statements": [{"key": "...", "section": "...", "text": "..."}]}.
For EACH statement return the claims it contains.

Rules:
1. Split only DIFFERENT findings joined by "and", commas or lists: different symptoms, exam findings,
   medications, tests or actions. "Patient reports fever, cough and sore throat" ->
   ["Patient reports fever", "Patient reports cough", "Patient reports sore throat"].
2. Do NOT split the description of ONE finding. Adjectives, severity, frequency, duration, onset,
   location and timing stay in the same claim as the finding they describe.
   "Patient reports loose, watery diarrhea for three days, six to seven times a day" is ONE claim.
   Never output a fragment ("approximately six times a day") that is not understandable alone.
3. Never drop a qualifier when splitting. Repeat time words, locations and conditions in every claim
   they apply to: "sore throat and runny nose at onset" ->
   ["Patient reports sore throat at onset", "Patient reports runny nose at onset"].
4. A negation applies to EVERY item in its list. "no A, B or C" means none of A, B, C.
   Repeat the negation in each split claim: "Patient reports no A", "Patient reports no B",
   "Patient reports no C". Never output a positive claim for an item that sits under a negation.
   (Example: "Patient denies rash, itching or swelling" -> "Patient denies rash",
   "Patient denies itching", "Patient denies swelling".)
5. Each claim states its subject when the statement does ("Patient denies cough or fever" ->
   "Patient denies cough", "Patient denies fever"). If the statement has no subject, do not add one.
6. Keep ALTERNATIVES in one claim: "Advise paracetamol or co-codamol" stays ONE claim.
   Keep CONDITIONS in one claim: "Advise paracetamol if feverish" stays ONE claim.
   If a condition applies to several split facts, repeat it in each claim.
7. Never strengthen or weaken wording. Keep every hedge, verb and negation exactly:
   "considers", "possible", "may", "might", "advises", "denies", "no", "if needed", "appears".
   Never turn a possibility into a fact, or a negation into an affirmation.
8. Use only words already in the statement. Do not add facts, causes, doses or explanations.
9. If a statement is already a single fact, return it unchanged as one claim.
10. A list governed by "only", "just", "mainly", "mostly", "at least", "up to" or "except" stays in
   ONE claim. Never split it by repeating the qualifier on each item.
   "Patient reports eating only soups, smoothies and light foods" is ONE claim, not
   "eating only soups" + "eating only smoothies" + "eating only light foods".

Return ONLY JSON: {"results": [{"key": "...", "claims": ["...", "..."]}]} with one entry per key.
"""

_NON_ATOMIC = re.compile(r"\b(and|or)\b|[,;]", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+")


def is_atomic(statement: str) -> bool:
    return not _NON_ATOMIC.search(statement)


def _same_but_punctuation(a: str, b: str) -> bool:
    strip = lambda t: t.strip().rstrip(".;,: ").strip()
    return strip(a) == strip(b)


def stem(word: str) -> str:
    w = word.lower()
    for suf, rep in (
        ("ies", "y"),
        ("ied", "y"),
        ("ing", ""),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
        ("ly", ""),
    ):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + rep
            break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def _stems(text: str) -> set[str]:
    return {stem(t) for t in _TOKEN.findall(text.lower()) if len(t) > 1 or t.isdigit()}


# Words a rewrite may legitimately introduce or swap (carriers, grammar), compared by stem.
_ALLOWED = _stems(
    "patient clinician doctor report state say deny endorse note has have had is are was were be "
    "been there it its this that the an of to for in on at by from with as also currently no not "
    "nil none without never any"
)
_NEGATION = _stems("no not nil none without never deny denies denied absent negative")
# Hedge groups: a hedge in the source must appear in EVERY claim; a claim may not add one.
_HEDGE_GROUPS = [
    _stems("possible possibly potential potentially"),
    _stems("likely probable probably"),
    _stems("may might could"),
    _stems("suspect suspected"),
    _stems("consider considers considered considering"),
    _stems("suggest suggests suggested"),
    _stems("appear appears appeared seem seems seemed"),
    _stems("if unless"),
]


def _hedges(text: str) -> set[int]:
    s = _stems(text)
    return {i for i, g in enumerate(_HEDGE_GROUPS) if s & g}


_CUE = re.compile(
    r"\b(no|not|nil|none|never|without|denies|denied|deny|negative|absent|cannot|can't|doesn't"
    r"|does not|didn't|did not|hasn't|has not|haven't|have not)\b",
    re.IGNORECASE,
)
_CLAUSE = re.compile(r";|\.|\bbut\b|\bwhile\b|\bwhereas\b", re.IGNORECASE)


def check_negation_scope(source: str, claim: str) -> str | None:
    """Flag a claim that lost a negation it sat under in the source.

    In each clause of the source, the words after the first negation cue are that cue's scope
    ("no A, B or C": A, B, C). A claim with no negation word whose content words ALL fall inside
    such a scope is a dropped negation ("Patient reports cough" from "...no chest pain, cough...").
    Words before the cue ("nausea, hot, no fever") and other clauses are outside its scope.
    """
    if _CUE.search(claim):
        return None
    content = _stems(claim) - _ALLOWED - _NEGATION
    if not content:
        return None
    for clause in _CLAUSE.split(source):
        m = _CUE.search(clause)
        if m and content <= _stems(clause[m.end() :]) - _ALLOWED - _NEGATION:
            return f"negation dropped: '{m[0]}' covers this claim in the source"
    return None


_SCOPE_WORDS = ("only", "just", "mainly", "mostly", "at least", "up to", "except")
_LIST_MARK = r",|\b(and|or)\b"
_SCOPE_RE = re.compile(r"\b(" + "|".join(_SCOPE_WORDS) + r")\b", re.IGNORECASE)


def check_scope_list(source: str, claim: str) -> str | None:
    """Flag a SPLIT claim that carries a scope word (only, just, mainly...) the source puts in
    front of a list. "eating only soups, smoothies, light foods" must not become three claims
    each saying "only". Only call this when the statement was split into several claims."""
    for m in _SCOPE_RE.finditer(source):
        word = m[1].lower()
        tail = _CLAUSE.split(source[m.end() :], maxsplit=1)[0]
        if not re.search(_LIST_MARK, tail, re.IGNORECASE):
            continue  # the source word does not govern a list
        c = re.search(rf"\b{re.escape(word)}\b", claim, re.IGNORECASE)
        if c and not re.search(_LIST_MARK, claim[c.end() :], re.IGNORECASE):
            return f"scope word '{word}' copied onto one item of a list"
    return None


def check_claim(source: str, claim: str) -> str | None:
    """Return a failure reason, or None if the claim is a faithful rewrite of the source."""
    src, cl = _stems(source), _stems(claim)
    if new := cl - src - _ALLOWED - _NEGATION:
        return f"new words not in source: {sorted(new)}"
    src_h, cl_h = _hedges(source), _hedges(claim)
    if src_h != cl_h:
        return "hedge/condition changed (dropped or added)"
    if (cl & _NEGATION) and not (src & _NEGATION):
        return "negation introduced"
    return check_negation_scope(source, claim)


_CONNECTIVES = _stems("and or but while with also well")


def check_coverage(source: str, claims: list[str]) -> str | None:
    """Return a failure reason if splitting dropped any content word of the source."""
    kept: set[str] = set().union(*(_stems(c) for c in claims)) if claims else set()
    if lost := _stems(source) - kept - _ALLOWED - _CONNECTIVES:
        return f"words dropped by split: {sorted(lost)}"
    # _ALLOWED hides negation words from the check above, so require them explicitly.
    if (_stems(source) & _NEGATION) and not (kept & _NEGATION):
        return "negation dropped by split"
    return None


class Claim(BaseModel):
    id: str  # "C1".. per encounter; key is (encounter_id, id)
    encounter_id: str
    section: str
    source_statement_index: int
    text: str

    def with_section(self) -> str:
        """What the verifier should see: the section travels with the claim."""
        return f"[{self.section.title()}] {self.text}"


class Fallback(BaseModel):
    section: str
    index: int
    statement: str
    reason: str


class DecomposeResult(BaseModel):
    encounter_id: str
    claims: list[Claim]
    decomposer_prompt_version: str = DECOMPOSE_PROMPT_VERSION
    model: str = ""
    reasoning_effort: str | None = None
    generator_prompt_version: str = ""
    loader_version: str = ""
    n_statements: int = 0
    n_passthrough: int = 0
    n_llm: int = 0
    fallbacks: list[Fallback] = Field(default_factory=list)


class _StatementClaims(BaseModel):
    key: str
    claims: list[str] = Field(min_length=1)


class _LLMOut(BaseModel):
    results: list[_StatementClaims]


def _ask(
    client: LLMClient, pending: dict[str, tuple[str, str]]
) -> tuple[dict[str, list[str]], str]:
    """One LLM call for the pending {key: (section, text)}. Returns ({key: claims}, error)."""
    payload = {"statements": [{"key": k, "section": s, "text": t} for k, (s, t) in pending.items()]}
    try:
        raw = client.complete(SYSTEM_PROMPT, json.dumps(payload))
        out = _LLMOut.model_validate(extract_json(raw))
    except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
        return {}, f"invalid LLM output: {str(e)[:120]}"
    return {r.key: r.claims for r in out.results}, ""


def decompose_statements(
    soap: dict[str, list[str]], client: LLMClient | None = None
) -> tuple[dict[str, list[str]], list[Fallback], int, int]:
    """Return ({'section:i': [claim texts]}, fallbacks, n_passthrough, n_llm)."""
    texts = {f"{sec}:{i}": (sec, s) for sec in SECTIONS for i, s in enumerate(soap.get(sec, []))}
    done: dict[str, list[str]] = {}
    pending: dict[str, tuple[str, str]] = {}
    for key, (sec, s) in texts.items():
        if is_atomic(s):
            done[key] = [s]
        else:
            pending[key] = (sec, s)
    n_pass, n_llm = len(done), len(pending)

    reasons: dict[str, str] = {}
    for _attempt in range(2):  # first try + ONE retry, for failing statements only
        if not pending:
            break
        client = client or GroqClient(
            reasoning_effort=REASONING_EFFORT, max_completion_tokens=MAX_TOKENS
        )
        got, err = _ask(client, pending)
        failed: dict[str, tuple[str, str]] = {}
        for key, (sec, s) in pending.items():
            claims = got.get(key)
            if not claims:
                reasons[key] = err or "statement missing from LLM output"
                failed[key] = (sec, s)
                continue
            claims = [c.strip() for c in dict.fromkeys(claims) if c.strip()]
            if len(claims) == 1 and _same_but_punctuation(claims[0], s):
                claims = [s]  # "unchanged" means unchanged: keep the generator's exact text
            bad = next((r for c in claims if (r := check_claim(s, c))), None) or check_coverage(
                s, claims
            )
            if not bad and len(claims) > 1:
                bad = next((r for c in claims if (r := check_scope_list(s, c))), None)
            if bad:
                reasons[key] = bad
                failed[key] = (sec, s)
            else:
                done[key] = claims
        pending = failed

    fallbacks = []
    for key, (sec, s) in pending.items():  # still failing: keep the generator's text unchanged
        done[key] = [s]
        fallbacks.append(
            Fallback(section=sec, index=int(key.split(":")[1]), statement=s, reason=reasons[key])
        )
    return done, fallbacks, n_pass, n_llm


def decompose(note: dict, client: LLMClient | None = None) -> DecomposeResult:
    """note = a saved generated-note JSON ({'id', 'model', 'prompt_version', 'loader_version', 'soap'})."""
    soap = note["soap"]
    if client is None and any(not is_atomic(t) for sec in SECTIONS for t in soap.get(sec, [])):
        client = GroqClient(reasoning_effort=REASONING_EFFORT, max_completion_tokens=MAX_TOKENS)
    done, fallbacks, n_pass, n_llm = decompose_statements(soap, client)
    claims, n = [], 0
    for sec in SECTIONS:
        for i in range(len(soap.get(sec, []))):
            for text in done[f"{sec}:{i}"]:
                n += 1
                claims.append(
                    Claim(
                        id=f"C{n}",
                        encounter_id=note["id"],
                        section=sec,
                        source_statement_index=i,
                        text=text,
                    )
                )
    return DecomposeResult(
        encounter_id=note["id"],
        claims=claims,
        model=getattr(client, "model", "") or note.get("model", ""),
        reasoning_effort=getattr(client, "reasoning_effort", None),
        generator_prompt_version=note.get("prompt_version", ""),
        loader_version=note.get("loader_version", ""),
        n_statements=n_pass + n_llm,
        n_passthrough=n_pass,
        n_llm=n_llm,
        fallbacks=fallbacks,
    )


def load_generated(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
