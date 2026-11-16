# ClinoTrust

## The problem
AI scribes write SOAP notes that read well but can quietly state things the patient never said:
a softened "no" turned into a flat denial, a patient's "dry skin" reported as a clinician's exam
finding, "advise paracetamol" turned into "prescribe". ClinoTrust checks a generated note
**claim by claim** against the consultation transcript. Each claim gets a verdict (supported,
contradicted, unsupported, uncertain) with evidence cited as turn IDs (T12) plus a quote that must
appear in the transcript, so a clinician can review only the flagged claims. Anything not
discussed is "not documented", never assumed absent.

## Pipeline
```
PriMock57 TextGrids (data/raw, read-only)
        |  data/loaders.py + cleaning.py
        v
numbered transcript  (data/processed/*.json, turns T1..Tn)
        |  pipeline/generate.py   gpt-oss-20b, saves model + prompt + loader version
        v
SOAP note  (outputs/generated/)
        |  pipeline/decompose.py  one note statement -> atomic claims (decomposer v3)
        v
claims  (labels/claims.csv: hand-labelled gold for the dev set)
        |  pipeline/verify.py     qwen verifier (a different model from the generator)
        v
verdict {analysis, status, evidence_turns, evidence_quote}
        |  Pydantic check, one retry, quote must be found in the transcript
        v
needs_review flag  ->  clinician review  (reference notes only hint at omissions)
```
The verifier never sees the gold labels. Every LLM output is validated with Pydantic.

## Results
**Dev set:** 5 encounters (day1_consultation01-05), 211 labelled claims (184 supported, 27 not
supported, of which 8 are decomposer errors). Positive class = "not supported".

| verifier prompt | 4-class acc | precision | recall | errors missed (FN) | false alarms (FP) |
|---|---|---|---|---|---|
| v0 (baseline) | 0.890 | 0.750 | 0.333 | 18 | 3 |
| v1 | 0.928 | 0.808 | 0.778 | 6 | 5 |
| v1.1 | **0.953** | **0.880** | **0.815** | 5 | 3 |

v0 and v1 are the 209 claims both runs scored. v1.1 is all 211 claims, so the rows are close but
not strictly identical sets. Excluding the 8 decomposer errors, v1.1 gets precision 0.833, recall
0.789 and accuracy 0.956.

**Held-out test set: not run yet.** The dev numbers above must not be read as expected
performance. v1.1's three rules (hedging, scope test, question-answer pairing) were written after
reading v1's mistakes on this same dev set. Four labels (02 C15, 02 C42, 04 C23, 04 C29) were also
changed after seeing v1 output, and the hedge/scope rules behind them were my decision; they are
logged in `labels/label_changes.csv`. The honest test is to run v1.1 once, unchanged, on
encounters not used so far.

## Failure taxonomy
Labels use these error types (see `docs/labels.md`). Examples are from the dev set.

| type | example | what went wrong |
|---|---|---|
| decomposition_error | 01 C11 "eating only light foods" | the decomposer changed the note's meaning, so the claim is not the scribe's fault. Reported separately from the generator's error rate |
| wrong_section | 02 C33 "No fever observed." (Objective) | patient answers filed as clinician examination findings. v1 catches 7/8, v1.1 8/8 |
| overstated | 01 C31 "Prescribe paracetamol 2 tablets..." | the doctor only advised it. 04 C30 "no allergies" when the doctor asked about drug allergies |
| over_interpretation | 01 C17 "no other symptoms" | the patient had mentioned loss of appetite |
| other | 04 C44 "...as needed" | the doctor said "continue the regular paracetamol, ibuprofen" with no "as needed" |

Remaining verifier mistakes in v1.1, from `score_v1.1.txt`:
- **Missed errors (5):** 04 C29 "no medications" (the patient was taking paracetamol and
  ibuprofen), 01 C31 (it flagged "Prescribe" as stronger, then still said supported), 01 C11,
  04 C3 and 04 C6. In 04 C6 the patient's "clearing up" answered a "clear or coloured?" question.
- **False alarms (3):** hedging and approximate wording, such as 04 C7 "maybe yesterday", 04 C35
  "a bit of a stressful period", and 02 C15 "I think so".

## Limitations
- **Small dev set.** 5 encounters and 27 not-supported claims, so one claim moves recall by about
  4 points. The v1 to v1.1 gain (+1 caught error, -2 false alarms) is within noise.
- **Tuned on the dev set.** Prompt rules and some labels were chosen after seeing verifier output
  on the same claims (see above).
- **Labels are drafts.** Claude drafted them and I reviewed them. There is no second human
  annotator and no agreement score. Borderline hedged answers are decided by one rule of mine.
- **Noisy source.** PriMock57 transcripts are ASR-style, with overlapping turns, fillers, and
  speaker slips (02 T56/T57 are both doctor turns, so the bowels question is never answered).
  Quote and turn ID matching has to tolerate this (3 to 4 quotes still fail the check).
- One scribe model, one verifier model, role-played consultations. No claim about real clinics.
- Quality checks ran on Groq free-tier limits; runs are resumable but not repeated for variance.

## Next steps
1. Freeze v1.1 and run it once on a held-out set of encounters, labelled before the run.
2. Add a second annotator on a subset and report agreement.
3. Fix the 5 missed errors by looking for general rules (scope of "no medications", answer
   matches the question asked), with examples that are not from the dev transcripts.
4. Check omissions with `labels/omissions.csv`, since reference notes are only hints.
5. Connect the `needs_review` flag to the review UI/API and measure clinician time saved.

## Setup
    uv sync
    cp .env.example .env   # add GROQ_API_KEY
    uv run pytest
    uv run uvicorn api.main:app --reload   # landing page at /, claim review dashboard at /app

## Layout
- `data/` loaders + schemas (`data/raw/` holds PriMock57 and ACI-Bench, gitignored)
- `pipeline/` generate -> decompose -> verify -> review (LangGraph)
- `eval/` metrics and `eval/labeled/` labeled claim sets
- `api/` FastAPI (JSON API + serves `web/`), `web/` landing page and review dashboard, `tests/`

## Data
- PriMock57: github.com/babylonhealth/primock57 (57 consultations)
- ACI-Bench: github.com/wyim/aci-bench + figshare doi 10.6084/m9.figshare.22494601
