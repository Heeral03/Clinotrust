"""Step 3 (v0): verify ONE claim against the numbered transcript.

The verifier is blind to gold labels: claims are read through `load_claims_for_verification`,
which keeps only encounter_id, claim_id, section and claim_text and drops every other column.
"""

import csv
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from config import settings
from data.schemas import Encounter
from pipeline.generate import extract_json

VERIFY_PROMPT_VERSION = "verify-prompt-v0"
VERDICTS_DIR = Path("outputs/verdicts/verifier-v0")
BATCH_VERDICTS_DIR = Path("outputs/verdicts/verifier-v0-batched")
BATCH_PROMPT_VERSION = "verify-prompt-v0-batch"
V1_PROMPT_VERSION = "verify-prompt-v1"
V1_BATCH_PROMPT_VERSION = "verify-prompt-v1-batch"
V1_VERDICTS_DIR = Path("outputs/verdicts/verifier-v1")
V1_BATCH_VERDICTS_DIR = Path("outputs/verdicts/verifier-v1-batched")
V1_1_PROMPT_VERSION = "verify-prompt-v1.1"
V1_1_BATCH_PROMPT_VERSION = "verify-prompt-v1.1-batch"
V1_1_VERDICTS_DIR = Path("outputs/verdicts/verifier-v1.1")
V1_1_BATCH_VERDICTS_DIR = Path("outputs/verdicts/verifier-v1.1-batched")
ANALYSIS_MAX_WORDS = 50
BATCH_SIZE = 10
LEDGER = Path("outputs/verdicts/token_ledger.jsonl")
DAILY_TOKEN_CAP = 200_000  # Groq tokens-per-day limit for the verifier model
Status = Literal["supported", "contradicted", "unsupported", "uncertain"]
NOT_DOCUMENTED = "not documented"

SYSTEM_PROMPT = """\
You audit one claim from an AI-written clinical SOAP note against the doctor-patient transcript it
was supposedly written from. The transcript is the ONLY source of truth. Use no outside medical
knowledge to fill gaps. The transcript is numbered: "T12 [Patient] (131.6s): text".

Judge the claim exactly as worded: check the subject (patient or clinician), negation, numbers,
durations, timing, severity and hedging. The section tag tells you who should be the source:
Subjective = what the patient reports; Objective = observations/measurements stated aloud;
Assessment = the clinician's stated impression; Plan = what the clinician says to do.

Statuses:
- supported: the transcript clearly states it, with the same subject, negation, numbers and hedging.
- contradicted: the transcript states the opposite or a different value (a denied symptom written as
  present, a different duration, number, side or drug, patient and clinician swapped).
- unsupported: the transcript never says it. This includes details the note adds (a number, a
  cause, a severity, an instruction) and a possibility or assumption written as confirmed fact.
- uncertain: the transcript only partly supports it, is ambiguous, the speaker is unsure, or you
  cannot decide. Prefer uncertain over guessing.
A claim "Not documented" is supported if the transcript has nothing for that section, and
contradicted if the transcript does contain such information (cite it).

Evidence rules:
- evidence_turns: the turn ids you relied on, exactly as written (["T80", "T81"]). Empty list if
  unsupported. For contradicted, cite the turn showing the opposite.
- evidence_quote: a SHORT verbatim quote (at most about 25 words) copied character for character
  from the cited turns. If you need two places, join them with " ... ". Empty string if unsupported.
- reason: ONE sentence.

Return ONLY a JSON object with exactly these keys:
{"status": "...", "evidence_turns": [...], "evidence_quote": "...", "reason": "..."}
"""

_RULES = SYSTEM_PROMPT.split("\nReturn ONLY")[0].replace("audit one claim", "audit several claims")
BATCH_SYSTEM_PROMPT = (
    _RULES
    + """
You will receive several claims, each with an id. Judge every claim separately; one claim's verdict
must not influence another's.

Return ONLY a JSON object with one entry per claim id given, using the ids exactly as given:
{"verdicts": [{"claim_id": "C1", "status": "...", "evidence_turns": [...], "evidence_quote": "...", "reason": "..."}]}
"""
)

SYSTEM_PROMPT_V1 = """\
You audit one claim from an AI-written clinical SOAP note against the doctor-patient transcript it
was supposedly written from. The transcript is the ONLY source of truth. Use no outside medical
knowledge to fill gaps. The transcript is numbered: "T12 [Patient] (131.6s): text".

Write "analysis" FIRST: at most 50 words saying what the transcript says, how that compares with
the claim, and why. No step-by-step thinking dump. Then give the status. The status MUST follow
from your analysis.

Statuses:
- supported: the cited turns directly state the claim, including every qualifier.
- contradicted: the cited turns directly state the opposite.
- unsupported: the claim is never mentioned in the transcript.
- uncertain: hedged, ambiguous, partial, or only implied.

Qualifier check: compare every verb and qualifier in the claim with what was actually said:
"only", prescribe vs advise, "as needed", observed vs reported, "not painful" vs "not very
painful", scope (drug allergies vs allergies), numbers, durations, sides, and who said it. If the
claim is stronger or different than the transcript, it is NOT supported.

Hedged answers ("I think so", "maybe", "I guess", "not sure") are uncertain.

Question-answer pairing: for a claim about what the patient said, cite the doctor's question turn
AND the patient's answer turn. The answer must respond to THAT question. A bare "No." or "Yes."
that answers a different question (for example the doctor asked two things in a row and the
patient answered only the last) does not count.

Sections: Subjective = what the patient reports; Assessment = the clinician's stated impression;
Plan = what the clinician says to do. Objective is supported ONLY if the clinician states an
examination finding, measurement or observation; patient-reported answers do not support an
Objective claim (use uncertain).
A claim "Not documented" means no examination findings or measurements appear in the transcript
(for sections other than Objective: nothing relevant to that section appears). It is supported if
none do, and contradicted if some do (cite them).

Evidence rules:
- evidence_turns: the turn ids you relied on, exactly as written (["T80", "T81"]). Empty list if
  unsupported.
- evidence_quote: a SHORT verbatim quote (at most about 25 words) copied character for character
  from the cited turns. If you need two places, join them with " ... ". Empty string if unsupported.

Return ONLY a JSON object with exactly these keys, in this order:
{"analysis": "...", "status": "...", "evidence_turns": [...], "evidence_quote": "..."}
"""

BATCH_SYSTEM_PROMPT_V1 = (
    SYSTEM_PROMPT_V1.split("\nReturn ONLY")[0].replace("audit one claim", "audit several claims")
    + """
You will receive several claims, each with an id. Judge every claim separately; one claim's verdict
must not influence another's.

Return ONLY a JSON object with one entry per claim id given, using the ids exactly as given; keys in
this order:
{"verdicts": [{"claim_id": "C1", "analysis": "...", "status": "...", "evidence_turns": [...], "evidence_quote": "..."}]}
"""
)


class Verdict(BaseModel):
    # Field order is the JSON order. v1 puts the analysis FIRST so the status follows from it.
    analysis: str = ""  # v1: at most 50 words, written before the status
    status: Status
    evidence_turns: list[str] = Field(default_factory=list)
    evidence_quote: str = ""
    reason: str = ""  # v0: one sentence

    @field_validator("evidence_turns")
    @classmethod
    def _turn_ids(cls, v: list[str]) -> list[str]:
        v = [t.strip() for t in v]
        if bad := [t for t in v if not re.fullmatch(r"T\d+", t)]:
            raise ValueError(f"bad turn ids: {bad}")
        return v

    @model_validator(mode="after")
    def _evidence_rules(self) -> "Verdict":
        if self.status == "unsupported":  # spec: no evidence for an unsupported claim
            self.evidence_turns, self.evidence_quote = [], ""
        if self.status == "contradicted" and not (
            self.evidence_turns and self.evidence_quote.strip()
        ):
            raise ValueError("contradicted needs evidence_turns and evidence_quote")
        return self


class ClaimInput(BaseModel):
    """All the verifier is allowed to know about a claim."""

    encounter_id: str
    claim_id: str
    section: str
    claim_text: str

    def render(self) -> str:
        return f"[{self.section.title()}] {self.claim_text}"


class VerdictRecord(BaseModel):
    encounter_id: str
    claim_id: str
    section: str
    claim_text: str
    verdict: Verdict | None = None
    quote_valid: bool | None = None  # None = nothing to check (no quote)
    turns_valid: bool | None = None  # every cited turn id exists in the transcript
    error: str | None = None  # set when no valid verdict could be obtained
    via: str = "single"  # "single" | "batch" | "single-retry" (a batched claim re-asked on its own)
    turn_corrected: bool | None = None  # v1: quote lived in another turn, so the turn id was fixed
    cited_turns_original: list[str] = Field(default_factory=list)  # model's ids when corrected
    needs_review: bool | None = None  # status != supported, or quote_valid is False


class VerdictFile(BaseModel):
    encounter_id: str
    model: str
    prompt_version: str = VERIFY_PROMPT_VERSION
    loader_version: str = ""
    updated_at: str = ""
    mode: str = "single"
    batch_size: int | None = None
    usage: dict[str, int] = Field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "calls": 0}
    )
    records: list[VerdictRecord] = Field(default_factory=list)


_NEEDED = ("encounter_id", "claim_id", "section", "claim_text")


def load_claims_for_verification(path: Path, only: set[str] | None = None) -> list[ClaimInput]:
    """Read ONLY the four allowed columns. Gold-label columns are never kept or returned."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = [header.index(c) for c in _NEEDED]
        out = [
            ClaimInput(**dict(zip(_NEEDED, (row[i] for i in idx), strict=True)))
            for row in reader
            if row
        ]
    return [c for c in out if not only or c.encounter_id in only]


def _squash(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def check_quote(enc: Encounter, v: Verdict) -> tuple[bool | None, bool | None]:
    """(quote_valid, turns_valid). The quote must appear in one of the CITED turns.

    Whitespace is normalised; a quote joined with ' ... ' / '…' is checked fragment by fragment,
    each fragment against a cited turn. Case and wording must match exactly.
    """
    by_id = {t.id: t.text for t in enc.turns}
    turns_valid = all(t in by_id for t in v.evidence_turns) if v.evidence_turns else None
    if not v.evidence_quote.strip():
        return None, turns_valid
    cited = [_squash(by_id[t]) for t in v.evidence_turns if t in by_id]
    fragments = [_squash(x) for x in re.split(r"\s*(?:\.\.\.|…)\s*", v.evidence_quote) if x.strip()]
    ok = bool(cited) and all(any(fr in c for c in cited) for fr in fragments)
    return ok, turns_valid


@dataclass
class QuoteCheck:
    quote_valid: bool | None  # False only if some part of the quote is in NO turn at all
    turns_valid: bool | None  # every originally cited id exists
    turn_corrected: bool
    turns: list[str]  # evidence turns after any correction


def check_quote_v1(enc: Encounter, v: Verdict) -> QuoteCheck:
    """Search the quote across ALL turns (whitespace-normalised, case-exact, ' ... ' fragments).

    - every fragment found in a cited turn  -> nothing changes;
    - a fragment missing from the cited turns but present in another turn -> the cited ids were
      wrong: evidence_turns becomes the turns that actually contain the quote and
      turn_corrected=True (the nearest matching turn to the cited ones is used);
    - a fragment present in no turn -> quote_valid=False.
    """
    texts = {t.id: _squash(t.text) for t in enc.turns}
    order = {t.id: i for i, t in enumerate(enc.turns)}
    cited = list(v.evidence_turns)
    turns_valid = all(t in texts for t in cited) if cited else None
    if not v.evidence_quote.strip():
        return QuoteCheck(None, turns_valid, False, cited)
    known = [t for t in cited if t in texts]
    anchors = [order[t] for t in known] or [0]
    fragments = [_squash(x) for x in re.split(r"\s*(?:\.\.\.|…)\s*", v.evidence_quote) if x.strip()]
    used: set[str] = set()
    corrected = False
    for fr in fragments:
        in_cited = [t for t in known if fr in texts[t]]
        if in_cited:
            used.update(in_cited)
            continue
        elsewhere = [t for t, tx in texts.items() if fr in tx]
        if not elsewhere:
            return QuoteCheck(False, turns_valid, False, cited)
        used.add(min(elsewhere, key=lambda t: (min(abs(order[t] - a) for a in anchors), order[t])))
        corrected = True
    turns = sorted(used, key=order.__getitem__) if corrected else cited
    return QuoteCheck(True, turns_valid, corrected, turns)


def needs_review_of(status: str, quote_valid: bool | None) -> bool:
    """A verdict goes to the clinician unless it is `supported` with a valid (or absent) quote."""
    return status != "supported" or quote_valid is False


def _check_v0(v: Verdict) -> None:
    if not v.reason.strip():
        raise ValueError("missing reason")


_HEDGE_V1 = 'Hedged answers ("I think so", "maybe", "I guess", "not sure") are uncertain.\n'
_HEDGE_V1_1 = (
    'Hedging: a clear yes or no followed by a softener ("Yeah, I think so", "No, I don\'t think '
    'so") counts as that yes or no. Approximate words about time or amount ("maybe yesterday", '
    '"about three days", "a bit") do not make a claim uncertain. Use uncertain for hedging only '
    'when the patient gives no clear direction (a bare "not sure", "I don\'t know", "maybe", or a '
    "changed answer).\n"
)
_QUAL_END_V1 = "If the\nclaim is stronger or different than the transcript, it is NOT supported.\n"
_SCOPE_V1_1 = (
    "Scope test: could the claim be false while the patient's answer is still true? Example: the "
    'doctor asked about drug allergies and the claim says "no allergies"; or the doctor asked about '
    'regular medicines and the claim says "no medications". If yes, the claim is uncertain.\n'
)
_PAIR_END_V1 = "patient answered only the last) does not count.\n"
_PAIR_V1_1 = (
    "Find the doctor's question about the claim's topic. The patient's answer counts only if it "
    "comes before the doctor asks something else. If the doctor asks a second question straight "
    'after (for example "Bowels OK?" then "Chest OK? Any cough?") and the next patient turn is a '
    'bare "No.", that "No." answers the LAST question. Then nothing answers the first topic, so a '
    "claim about it is uncertain.\n"
)


def _derive_v1_1(v1: str) -> str:
    """SYSTEM_PROMPT_V1 with exactly three changes (hedging, scope test, pairing)."""
    for anchor in (_HEDGE_V1, _QUAL_END_V1, _PAIR_END_V1):
        if v1.count(anchor) != 1:
            raise ValueError(f"v1 prompt anchor not found exactly once: {anchor!r}")
    return (
        v1.replace(_HEDGE_V1, _HEDGE_V1_1)
        .replace(_QUAL_END_V1, _QUAL_END_V1 + _SCOPE_V1_1)
        .replace(_PAIR_END_V1, _PAIR_END_V1 + _PAIR_V1_1)
    )


SYSTEM_PROMPT_V1_1 = _derive_v1_1(SYSTEM_PROMPT_V1)

_BATCH_TAIL_V1 = BATCH_SYSTEM_PROMPT_V1[
    len(
        SYSTEM_PROMPT_V1.split("\nReturn ONLY")[0].replace(
            "audit one claim", "audit several claims"
        )
    ) :
]
BATCH_SYSTEM_PROMPT_V1_1 = (
    SYSTEM_PROMPT_V1_1.split("\nReturn ONLY")[0].replace("audit one claim", "audit several claims")
    + _BATCH_TAIL_V1
)


def _check_v1(v: Verdict) -> None:
    n = len(v.analysis.split())
    if n == 0:
        raise ValueError("missing analysis")
    if n > ANALYSIS_MAX_WORDS:
        raise ValueError(f"analysis has {n} words (max {ANALYSIS_MAX_WORDS})")


@dataclass(frozen=True)
class PromptSpec:
    name: str
    single_system: str
    batch_system: str
    single_version: str
    batch_version: str
    check: Callable[[Verdict], None]
    search_all_turns: bool  # v1 quote check (see check_quote_v1)


V0 = PromptSpec("v0", SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT, VERIFY_PROMPT_VERSION,
                BATCH_PROMPT_VERSION, _check_v0, False)  # fmt: skip
V1 = PromptSpec("v1", SYSTEM_PROMPT_V1, BATCH_SYSTEM_PROMPT_V1, V1_PROMPT_VERSION,
                V1_BATCH_PROMPT_VERSION, _check_v1, True)  # fmt: skip
V1_1 = PromptSpec("v1.1", SYSTEM_PROMPT_V1_1, BATCH_SYSTEM_PROMPT_V1_1, V1_1_PROMPT_VERSION,
                  V1_1_BATCH_PROMPT_VERSION, _check_v1, True)  # fmt: skip
SPECS = {"v0": V0, "v1": V1, "v1.1": V1_1}


def build_user_prompt(enc: Encounter, claim: ClaimInput) -> str:
    return f"TRANSCRIPT:\n{enc.transcript(with_times=True)}\n\nCLAIM:\n{claim.render()}"


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, max_out: int | None = None) -> str: ...


def assert_independent(verifier_model: str, generator_models: set[str]) -> None:
    if verifier_model in generator_models:
        raise RuntimeError(
            f"verifier model {verifier_model!r} is the generator's model; use a different one"
        )


def _check_supported_evidence(claim: ClaimInput, v: Verdict) -> None:
    if (
        v.status == "supported"
        and not v.evidence_turns
        and claim.claim_text.strip().lower() != NOT_DOCUMENTED
    ):
        raise ValueError("supported needs evidence_turns")


def _finish(
    enc: Encounter, claim: ClaimInput, v: Verdict, spec: PromptSpec, via: str
) -> VerdictRecord:
    """Quote check (per spec), then build the record incl. needs_review."""
    corrected: bool | None = None
    original: list[str] = []
    if spec.search_all_turns:
        qc = check_quote_v1(enc, v)
        quote_valid, turns_valid, corrected = qc.quote_valid, qc.turns_valid, qc.turn_corrected
        if corrected:
            original, v.evidence_turns = list(v.evidence_turns), qc.turns
    else:
        quote_valid, turns_valid = check_quote(enc, v)
    return VerdictRecord(
        **{k: getattr(claim, k) for k in _NEEDED},
        verdict=v,
        quote_valid=quote_valid,
        turns_valid=turns_valid,
        via=via,
        turn_corrected=corrected,
        cited_turns_original=original,
        needs_review=needs_review_of(v.status, quote_valid),
    )


def verify_claim(
    enc: Encounter, claim: ClaimInput, client: LLMClient, retries: int = 1, spec: PromptSpec = V0
) -> VerdictRecord:
    """One LLM call (+ at most `retries` more if the answer is invalid), then the code quote check."""
    user = build_user_prompt(enc, claim)
    base = {k: getattr(claim, k) for k in _NEEDED}
    last = ""
    for _ in range(retries + 1):
        try:
            v = Verdict.model_validate(extract_json(client.complete(spec.single_system, user)))
            spec.check(v)
            _check_supported_evidence(claim, v)
        except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
            last = str(e)[:200]
            continue
        return _finish(enc, claim, v, spec, "single")
    return VerdictRecord(**base, error=f"invalid verifier output: {last}")


# ---------------------------------------------------------------- Groq client with pacing
_DUR = re.compile(r"(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def parse_duration(text: str) -> float:
    """Groq reset headers look like '1m26.4s', '31.409s', '922ms'. Returns seconds."""
    m = _DUR.match(text.strip())
    if not m or not any(m.groups()):
        return 30.0
    mins, secs, ms = (float(g) if g else 0.0 for g in m.groups())
    return mins * 60 + secs + ms / 1000


class VerifierClient:
    """Groq chat client for the verifier: JSON mode, temperature 0, paced to the token-per-minute
    limit using Groq's rate-limit headers, resilient to 429s. Key comes from .env only.
    Counts prompt/completion tokens per response and appends each to a ledger file."""

    MAX_OUT = 600
    LONG_WAIT = 900  # a 429 asking for longer than this is a daily cap: stop, progress is saved

    def __init__(
        self,
        model: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        ledger: Path | None = LEDGER,
    ):
        from groq import Groq

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set (add it to .env)")
        self._client = Groq(api_key=settings.groq_api_key, max_retries=0)
        self.model = model or settings.groq_verifier_model
        self._sleep = sleep
        self._remaining: int | None = None
        self._reset = 0.0
        self._limit = 8000
        self.ledger = ledger
        self.last_usage = (0, 0)  # (prompt, completion) of the latest response
        self.total_prompt = self.total_completion = self.n_calls = 0

    def _pace(self, need: int) -> None:
        if self._remaining is not None and need > self._remaining:
            self._sleep(self._reset + 1)
            self._remaining = self._limit

    def _record_usage(self, prompt: int, completion: int) -> None:
        self.last_usage = (prompt, completion)
        self.total_prompt += prompt
        self.total_completion += completion
        self.n_calls += 1
        if self.ledger:
            try:
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                line = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "model": self.model,
                        "prompt": prompt, "completion": completion}  # fmt: skip
                with open(self.ledger, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")
            except OSError:
                pass  # logging must never break a run

    def complete(self, system: str, user: str, max_out: int | None = None) -> str:
        from groq import BadRequestError, RateLimitError

        max_out = max_out or self.MAX_OUT
        self._pace((len(system) + len(user)) // 3 + max_out // 2)
        for _ in range(8):
            try:
                raw = self._client.chat.completions.with_raw_response.create(
                    model=self.model,
                    temperature=0,
                    max_completion_tokens=max_out,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except RateLimitError as e:
                wait = float(e.response.headers.get("retry-after", 30))
                if wait > self.LONG_WAIT:
                    raise RuntimeError(
                        f"Groq rate limit asks for {wait:.0f}s (daily cap?). "
                        "Re-run later; progress is saved."
                    ) from e
                self._sleep(wait + 1)
                continue
            except BadRequestError as e:
                if "json_validate_failed" in str(e):
                    raise ValueError("Groq JSON mode rejected the generation") from e
                raise
            h = raw.headers
            self._limit = int(h.get("x-ratelimit-limit-tokens", self._limit))
            self._remaining = int(h.get("x-ratelimit-remaining-tokens", self._limit))
            self._reset = parse_duration(h.get("x-ratelimit-reset-tokens", "30s"))
            resp = raw.parse()
            if resp.usage:
                self._record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            choice = resp.choices[0]
            if choice.finish_reason == "length" and not (choice.message.content or "").strip():
                raise ValueError("verifier output truncated")
            return choice.message.content or ""
        raise RuntimeError("Groq rate limit: gave up after 8 attempts")


def tokens_last_24h(ledger: Path, model: str | None = None, now: datetime | None = None) -> int:
    """Prompt+completion tokens recorded in the ledger over the last 24 hours (rolling window).
    Only counts calls made since the ledger existed, so it can under-count the true daily total."""
    if not ledger or not ledger.exists():
        return 0
    cutoff = (now or datetime.now(UTC)) - timedelta(hours=24)
    total = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            if (model is None or e["model"] == model) and datetime.fromisoformat(e["ts"]) >= cutoff:
                total += e["prompt"] + e["completion"]
        except (ValueError, KeyError):
            continue
    return total


def usage_snapshot(client: object) -> tuple[int, int, int]:
    return (
        getattr(client, "total_prompt", 0),
        getattr(client, "total_completion", 0),
        getattr(client, "n_calls", 0),
    )


def usage_line(client: object) -> str:
    p, c = getattr(client, "last_usage", (0, 0))
    run = sum(usage_snapshot(client)[:2])
    day = tokens_last_24h(getattr(client, "ledger", None), getattr(client, "model", None))
    return (
        f"tokens last call {p:,}+{c:,} | this run {run:,} | "
        f"last 24h {day:,}/{DAILY_TOKEN_CAP:,} ({day / DAILY_TOKEN_CAP:.0%})"
    )


# ---------------------------------------------------------------- batched mode
def build_batch_prompt(enc: Encounter, claims: list[ClaimInput]) -> str:
    lines = "\n".join(f"{c.claim_id}: {c.render()}" for c in claims)
    return (
        f"TRANSCRIPT:\n{enc.transcript(with_times=True)}\n\n"
        f"CLAIMS (judge each separately):\n{lines}"
    )


def parse_batch(raw: str, wanted: set[str], spec: PromptSpec = V0) -> dict[str, Verdict]:
    """claim_id -> Verdict for every well-formed entry. Unknown ids, duplicates and entries that
    fail Pydantic validation are dropped, so those claims count as missing."""
    data = extract_json(raw)
    items = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("batch output has no 'verdicts' list")  # noqa: TRY004 (caller catches ValueError)
    out: dict[str, Verdict] = {}
    for it in items:
        cid = it.get("claim_id") if isinstance(it, dict) else None
        if cid not in wanted or cid in out:
            continue
        try:
            verdict = Verdict.model_validate({k: v for k, v in it.items() if k != "claim_id"})
            spec.check(verdict)
        except ValueError:  # includes pydantic ValidationError
            continue
        out[cid] = verdict
    return out


def verify_batch(
    enc: Encounter,
    claims: list[ClaimInput],
    client: LLMClient,
    retries: int = 1,
    spec: PromptSpec = V0,
) -> list[VerdictRecord]:
    """One call for a group of claims from one encounter (transcript sent once). Same Pydantic
    validation and quote check per claim. A claim missing or invalid in the answer is re-asked
    ON ITS OWN (single-claim prompt); the others are never re-sent."""
    wanted = {c.claim_id for c in claims}
    user = build_batch_prompt(enc, claims)
    max_out = min(4000, 170 * len(claims) + 200)
    got: dict[str, Verdict] = {}
    for _ in range(retries + 1):  # retry the batch only if the whole answer was unusable
        try:
            got = parse_batch(
                client.complete(spec.batch_system, user, max_out=max_out), wanted, spec
            )
            break
        except ValueError:
            got = {}
    records = []
    for c in claims:
        v = got.get(c.claim_id)
        if v is not None:
            try:
                _check_supported_evidence(c, v)
            except ValueError:
                v = None
        if v is None:
            rec = verify_claim(enc, c, client, spec=spec)
            rec.via = "single-retry"
        else:
            rec = _finish(enc, c, v, spec, "batch")
        records.append(rec)
    return records


# ---------------------------------------------------------------- persistence
def verdict_path(encounter_id: str, out_dir: Path = VERDICTS_DIR) -> Path:
    return out_dir / f"{encounter_id}.json"


def load_verdict_file(encounter_id: str, out_dir: Path = VERDICTS_DIR) -> VerdictFile | None:
    p = verdict_path(encounter_id, out_dir)
    return VerdictFile.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def save_verdict_file(vf: VerdictFile, out_dir: Path = VERDICTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    vf.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
    p = verdict_path(vf.encounter_id, out_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(vf.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)  # atomic: an interrupted run never leaves a half-written file
    return p


def _open_file(
    eid: str, enc: Encounter, client: LLMClient, out_dir: Path, prompt_version: str,
    mode: str, batch_size: int | None,
) -> VerdictFile:  # fmt: skip
    """Load this encounter's verdict file (resume) or start one. Never mixes models/prompts/modes."""
    vf = load_verdict_file(eid, out_dir)
    if vf is None:
        return VerdictFile(
            encounter_id=eid, model=client.model, loader_version=enc.loader_version,
            prompt_version=prompt_version, mode=mode, batch_size=batch_size,
        )  # fmt: skip
    if (vf.model, vf.prompt_version, vf.mode) != (client.model, prompt_version, mode):
        raise RuntimeError(
            f"{verdict_path(eid, out_dir)} was made with {vf.model}/{vf.prompt_version}/{vf.mode}; "
            "use a different out dir"
        )
    return vf


def _add_usage(vf: VerdictFile, before: tuple[int, int, int], client: object) -> None:
    for key, b, a in zip(
        ("prompt", "completion", "calls"), before, usage_snapshot(client), strict=True
    ):
        vf.usage[key] = vf.usage.get(key, 0) + (a - b)


def _log_claim(log: Callable[[str], None], rec: VerdictRecord) -> None:
    v = rec.verdict
    log(
        f"{rec.encounter_id[-2:]} {rec.claim_id:>4} {v.status if v else 'ERROR':12} "
        f"quote_valid={rec.quote_valid}{' turn_corrected' if rec.turn_corrected else ''} "
        f"via={rec.via} | {rec.claim_text[:55]}"
    )


def verify_all(
    claims: list[ClaimInput],
    encounters: dict[str, Encounter],
    client: LLMClient,
    out_dir: Path = VERDICTS_DIR,
    limit: int | None = None,
    log: Callable[[str], None] = print,
    spec: PromptSpec = V0,
) -> int:
    """Single-claim mode. Resumable; saves after every claim. Returns claims processed."""
    done_n = 0
    files: dict[str, VerdictFile] = {}
    for claim in claims:
        eid = claim.encounter_id
        enc = encounters[eid]
        if eid not in files:
            files[eid] = _open_file(eid, enc, client, out_dir, spec.single_version, "single", None)
        vf = files[eid]
        if claim.claim_id in {r.claim_id for r in vf.records if r.error is None}:
            continue
        if limit is not None and done_n >= limit:
            break
        before = usage_snapshot(client)
        rec = verify_claim(enc, claim, client, spec=spec)
        vf.records = [r for r in vf.records if r.claim_id != claim.claim_id] + [rec]
        _add_usage(vf, before, client)
        save_verdict_file(vf, out_dir)
        done_n += 1
        _log_claim(log, rec)
        log("      " + usage_line(client))
    return done_n


def verify_all_batched(
    claims: list[ClaimInput],
    encounters: dict[str, Encounter],
    client: LLMClient,
    out_dir: Path = BATCH_VERDICTS_DIR,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    log: Callable[[str], None] = print,
    spec: PromptSpec = V0,
) -> int:
    """Batched mode: one call per group of <= batch_size pending claims from the SAME encounter.
    Resumable (only pending claims are grouped); saves after every batch. Returns claims processed."""
    done_n = 0
    by_enc: dict[str, list[ClaimInput]] = {}
    for c in claims:
        by_enc.setdefault(c.encounter_id, []).append(c)
    for eid, group in by_enc.items():
        enc = encounters[eid]
        vf = _open_file(eid, enc, client, out_dir, spec.batch_version, "batched", batch_size)
        have = {r.claim_id for r in vf.records if r.error is None}
        pending = [c for c in group if c.claim_id not in have]
        if limit is not None:
            pending = pending[: max(0, limit - done_n)]
        for k in range(0, len(pending), batch_size):
            chunk = pending[k : k + batch_size]
            before = usage_snapshot(client)
            recs = verify_batch(enc, chunk, client, spec=spec)
            ids = {r.claim_id for r in recs}
            vf.records = [r for r in vf.records if r.claim_id not in ids] + recs
            _add_usage(vf, before, client)
            save_verdict_file(vf, out_dir)
            done_n += len(chunk)
            for r in recs:
                _log_claim(log, r)
            n_retry = sum(r.via == "single-retry" for r in recs)
            log(
                f"  batch of {len(chunk)} for {eid[-2:]}: {n_retry} re-asked singly | "
                + usage_line(client)
            )
    return done_n
