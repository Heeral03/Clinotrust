# Label sheets

`labels/claims.csv` (one row per claim) and `labels/omissions.csv` are hand-labeled. Code never
rewrites a labeled row; see `pipeline/labels.py`.

## claims.csv columns
`encounter_id, claim_id, section, source_statement_index, claim_text, status, evidence_turns,
evidence_quote, error_type, severity, notes` - the last six are filled by hand.

| column | allowed values |
|---|---|
| `status` | `supported`, `contradicted`, `unsupported`, `uncertain` |
| `error_type` | `none`, `negation`, `overstated`, `over_interpretation`, `wrong_section`, `other`, **`decomposition_error`** |
| `severity` | `none`, `minor`, `major` |
| `evidence_turns` | turn ids, `;`-separated (`T26;T27`) |
| `evidence_quote` | text that must appear in the cited turns |

Blank = not labeled yet. Constants live in `pipeline/labels.py` (`STATUSES`, `ERROR_TYPES`,
`SEVERITIES`); `validate_label_rows()` reports values outside these sets.

### decomposition_error
The claim is wrong because the **decomposer** mangled a correct note statement (for example it
dropped the "no" from "no cough, headaches or palpitations"), not because the scribe was wrong.
Label it as you would any claim (usually `contradicted` or `unsupported`) but set
`error_type = decomposition_error`. **Evaluation must exclude these rows from the generator's
unsupported / contradiction rates** and report them separately as a decomposer defect rate.
The fix for such a row is a new decomposer version, never an edit to the generated note.

## Safeguards
- Header check: a labels sheet is only read or written if its header matches exactly.
- Backup: every write to an existing sheet first saves `<file>.<timestamp>.bak` beside it.
- Ids are frozen: rows match on `(encounter_id, section, claim_text)`, ids are never renumbered,
  new claims get ids above the current maximum, and hand-split ids (`C14a`) are respected.
- `run_decompose` skips any encounter already in `claims.csv`. Re-decomposing a labeled encounter
  needs `--only ID --allow-labeled --no-labels` and writes JSON only, never to `labels/`.
