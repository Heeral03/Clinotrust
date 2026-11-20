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
**Dev set:** 5 encounters (day1_consultation01-05), 211 labelled claims (185 supported, 26 not
supported, of which 7 are decomposer errors). Positive class = "not supported". Dev was used to
write v1.1, so read it as a ceiling, not as expected performance.

**Held-out set:** 5 encounters (day3_consultation03, 04, 05, 08, 10), 152 labelled claims (128
supported, 24 not supported). v1.1 was frozen and run once. Labels were frozen first
(`heldout/LABELS_SHA256.txt`).

| set | run | claims scored | 4-class acc | precision | recall | missed (FN) | false alarms (FP) |
|---|---|---|---|---|---|---|---|
| dev | v1.1, two runs (range) | 211 | 0.953 to 0.957 | 0.840 to 0.880 | 0.808 to 0.846 | 4 to 5 | 3 to 4 |
| held-out | v1.1, first run | 150 | 0.893 | 0.824 | 0.636 | 8 (TP=14) | 3 |
| held-out | v1.1, after truncation fix | 152 | 0.888 | **0.842** | **0.667** | 8 (TP=16) | 3 |

- **Dev range:** the two dev runs use the same prompt, model and temperature 0. Their verdicts
  differ on 3 of 211 claims. There is no seed, so runs are not exactly repeatable.
- **Held-out change:** the only change between the two held-out rows is the truncation fix. In the
  first run, claims 03:C30 and 10:C2 got no verdict because the analysis was over 50 words. The
  fix cuts the analysis to 50 words instead of rejecting the verdict. Status, evidence and quote
  are not changed. The other 150 verdicts are identical in both rows. Those 2 verdicts
  (03:C30, 10:C2) have `analysis_truncated=True`. The rule was changed after seeing the first
  held-out output (logged in `heldout/manifest.json`).
- **No-verdict claims as misses:** if the 2 no-verdict claims count as misses, first-run recall
  is 0.583 (14 of 24).
- **Without decomposer errors:** held-out precision is 0.842, recall 0.727, accuracy 0.900.
- **Weak spot:** recall is 2/6 on `overstated` and 2/4 on `over_interpretation` claims. The
  verifier accepts soft paraphrases. Details are in `heldout/score_v1.1_final.txt`.
- **Earlier prompts (dev only):** v0 (baseline) had accuracy 0.890, precision 0.750, recall 0.333
  (18 missed, 3 false alarms). v1 had 0.928, 0.808, 0.778 (6 missed, 5 false alarms). Those rows
  were scored before the 01 C11 relabel, on 209 claims, so they are not exactly comparable to
  the table above.
- **Dev run 2** ran after the truncation fix, and 11 of its 211 verdicts have
  `analysis_truncated=True`.

Four dev labels (02 C15, 02 C42, 04 C23, 04 C29) were changed after seeing v1 output, and 01 C11
after seeing v1.1 output. The hedge and scope rules behind them were my decision. All are logged in
`labels/label_changes.csv`.

## Failure taxonomy
Labels use these error types (see `docs/labels.md`). Examples are from the dev set. The last two
columns show how many of that type v1.1 flagged on dev, run 1 and run 2 (labels after the
01 C11 relabel).

| type | example | what went wrong | run 1 | run 2 |
|---|---|---|---|---|
| decomposition_error | 01 C9 "eating only soups" | the decomposer changed the note's meaning, so the claim is not the scribe's fault. Reported separately from the generator's error rate | 7/7 | 7/7 |
| wrong_section | 02 C33 "No fever observed." (Objective) | patient answers filed as clinician examination findings. v1 caught 7/8 | 8/8 | 8/8 |
| overstated | 01 C31 "Prescribe paracetamol 2 tablets..." | the doctor only advised it. 04 C30 "no allergies" when the doctor asked about drug allergies | 3/5 | 3/5 |
| over_interpretation | 01 C17 "no other symptoms" | the patient had mentioned loss of appetite | 2/2 | 1/2 |
| other | 04 C6 "runny nose clearing up" | the patient's "clearing up" answered a "clear or coloured?" question | 2/4 | 2/4 |

01 C11 was a decomposition error until I relabelled it `supported` (logged in
`labels/label_changes.csv`). Dev has 26 not-supported claims and 185 supported.

Remaining verifier mistakes in v1.1 on dev, from the two dev score runs:
- **Missed errors, run 1 (4):** 01 C31 (it flagged "Prescribe" as stronger, then still said
  supported), 04 C3, 04 C6 and 04 C29 "no medications" (the patient was taking paracetamol and
  ibuprofen).
- **Missed errors, run 2 (5):** the same 4, plus 05 C22 "no chest pain" (the patient said "I'm
  not having any problems").
- **False alarms, run 1 (3):** hedging and approximate wording, such as 04 C7 "maybe yesterday",
  04 C35 "a bit of a stressful period", and 02 C15 "I think so".
- **False alarms, run 2 (4):** the same 3, plus 02 C6 "skin involvement on arms". Run 2 marked it
  `contradicted`, and its quote failed the quote check.
- **Run to run:** the two runs differ on 3 of 211 claims: 02 C6 (supported, then contradicted),
  05 C22 (uncertain, then supported) and 02 C30 (unsupported, then uncertain).
- **Recall by section (run 1 / run 2):** subjective 0.800 / 0.733, objective 1.000 / 1.000,
  plan 0.667 / 0.667. Assessment has no gold errors.

## Known limits
- **Small sample.** The held-out set has only 24 gold errors, so one claim moves recall by about
  4 points. The dev set has 26.
- **Acted consultations.** PriMock57 is role-played, not real clinics. No claim about real care.
- **Labels are drafts.** Claude drafted them and I reviewed them. There is no clinician review,
  no second labeler and no agreement score.
- **One known label inconsistency (held-out set).** Held-out 08 C21 ("no palpable glands",
  patient report) is labelled `supported`. Held-out 05 C21 and 05 C22 ("no weakness", patient report plus doctor repeat) are labelled
  `contradicted` / `wrong_section`. The pattern is the same. Held-out labels are frozen, so I
  did not change them.
- **Tuned on dev.** Prompt rules and some labels were chosen after seeing verifier output on the
  dev claims.
- **Noisy source.** PriMock57 transcripts are ASR-style, with overlapping turns, fillers, and
  speaker slips (02 T56/T57 are both doctor turns, so the bowels question is never answered).
  Quote and turn ID matching has to tolerate this (2 to 4 quotes still fail the check).
- One scribe model, one verifier model. Runs used Groq free-tier limits, temperature 0, no seed.
  Runs are resumable, but variance is only measured on two dev runs.

## Next steps
1. Run on a fresh set of encounters after any prompt change. Do not tune on the held-out five.
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

## Deploy (Render)
Web service, Python 3, build `uv sync --frozen && uv cache prune --ci`, start
`uv run -- uvicorn api.main:app --host 0.0.0.0 --port $PORT`. No API keys needed: the dashboard
serves the committed `web/demo/` data. Refresh it with `uv run python -m pipeline.export_demo`.
