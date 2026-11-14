from eval.score import POSITIVE, load_gold, ratio, render, score
from pipeline.labels import CLAIMS_HEADER


def g(cid, status, error_type="none", text=None, enc="e1", turns="", quote=""):
    return {"encounter_id": enc, "claim_id": cid, "section": "subjective", "source_statement_index": "0",
            "claim_text": text or f"claim {cid}", "status": status, "evidence_turns": turns,
            "evidence_quote": quote, "error_type": error_type, "severity": "none", "notes": ""}  # fmt: skip


def vd(cid, status, text=None, enc="e1", quote="", quote_valid=None, turns=None):
    return (enc, cid), {"encounter_id": enc, "claim_id": cid, "section": "subjective", "claim_text": text or f"claim {cid}",
                        "verdict": {"status": status, "evidence_turns": turns or [], "evidence_quote": quote, "reason": "why"},
                        "quote_valid": quote_valid, "turns_valid": None, "error": None}  # fmt: skip


GOLD = [
    g("C1", "supported"), g("C2", "supported"), g("C3", "supported"), g("C4", "supported"),
    g("C5", "contradicted"), g("C6", "contradicted", "decomposition_error"),
    g("C7", "unsupported"), g("C8", "uncertain"),
]  # fmt: skip
VERDICTS = dict([
    vd("C1", "supported"), vd("C2", "supported"),
    vd("C3", "uncertain", quote="q3", quote_valid=False),   # false alarm
    vd("C4", "contradicted"),                                 # false alarm
    vd("C5", "contradicted"),                                 # TP
    vd("C6", "supported"),                                    # missed (decomposition_error row)
    vd("C7", "uncertain"),                                    # TP, different label
    vd("C8", "supported"),                                    # missed
])  # fmt: skip


def test_positive_class_is_all_three_not_supported_labels():
    assert POSITIVE == {"contradicted", "unsupported", "uncertain"}


def test_binary_counts_precision_recall_all_rows():
    s = score(GOLD, VERDICTS)
    assert (s.tp, s.fp, s.fn, s.tn) == (2, 2, 2, 2)
    assert s.precision == 0.5 and s.recall == 0.5 and s.f1 == 0.5 and s.n == 8


def test_confusion_matrix_cells():
    s = score(GOLD, VERDICTS)
    assert s.matrix[("supported", "supported")] == 2
    assert (
        s.matrix[("supported", "uncertain")] == 1 and s.matrix[("supported", "contradicted")] == 1
    )
    assert s.matrix[("unsupported", "uncertain")] == 1 and s.matrix[("uncertain", "supported")] == 1
    assert sum(s.matrix.values()) == 8


def test_excluding_decomposition_error_rows():
    s = score(GOLD, VERDICTS, exclude_decomposition_error=True)
    assert s.n_gold == 7 and (s.tp, s.fp, s.fn, s.tn) == (2, 2, 1, 2)
    assert s.precision == 0.5 and s.recall == 2 / 3
    assert ("contradicted", "supported") not in s.matrix  # the excluded row is gone entirely


def test_disagreements_list_both_sides_with_quotes():
    gold = [g("C3", "supported", quote="gold q", turns="T4")]
    s = score(
        gold, dict([vd("C3", "uncertain", quote="verifier q", quote_valid=False, turns=["T9"])])
    )
    [d] = s.disagreements
    assert d["kind"].startswith("FALSE ALARM")
    text = render(s, "t")
    for needle in (
        "claim C3",
        "gold q",
        "verifier q",
        "supported",
        "uncertain",
        "quote_valid=False",
        "T9",
        "T4",
    ):
        assert needle in text
    assert s.quotes_checked == 1 and s.quotes_invalid == 1


def test_disagreement_kinds():
    s = score(GOLD, VERDICTS)
    kinds = {d["key"][1]: d["kind"].split(" (")[0] for d in s.disagreements}
    assert (
        kinds["C8"] == "MISSED"
        and kinds["C3"] == "FALSE ALARM"
        and kinds["C7"] == "same side, different label"
    )
    assert "C1" not in kinds and len(s.disagreements) == 5  # C3, C4, C6, C7, C8


def test_missing_verdicts_and_text_mismatch_are_reported_not_scored():
    gold = [g("C1", "supported"), g("C2", "supported"), g("C3", "supported", text="new text")]
    s = score(gold, dict([vd("C1", "supported"), vd("C3", "supported", text="old text")]))
    assert s.missing == [("e1", "C2")] and s.text_mismatch == [("e1", "C3")] and s.n == 1


def test_error_verdict_counts_as_missing():
    key, rec = vd("C1", "supported")
    rec["verdict"], rec["error"] = None, "invalid"
    assert score([g("C1", "supported")], {key: rec}).missing == [key]


def test_empty_and_zero_division_safe():
    s = score([], {})
    assert s.precision is None and s.recall is None and s.f1 is None and s.accuracy4 is None
    assert ratio(1, 0) is None
    assert "n/a" in render(s, "empty")


def test_load_gold_filters_encounters_and_unlabeled(tmp_path):
    p = tmp_path / "claims.csv"
    rows = [",".join(CLAIMS_HEADER)]
    rows += [
        "e1,C1,subjective,0,a,supported,,,none,none,",
        "e1,C2,subjective,0,b,,,,,,",
        "e2,C1,plan,0,c,supported,,,none,none,",
    ]
    p.write_text("\n".join(rows) + "\n")
    assert [r["claim_id"] for r in load_gold(p, {"e1"})] == ["C1"]  # unlabeled row and e2 dropped


# ------------------------------------------------------------------ single vs batched comparison
def test_compare_modes_agreement_and_gold_matches():
    from eval.score import compare_modes, render_comparison

    single = dict(
        [vd("C1", "supported"), vd("C2", "supported"), vd("C3", "uncertain"), vd("C4", "supported")]
    )
    batched = dict(
        [
            vd("C1", "supported"),
            vd("C2", "uncertain"),
            vd("C3", "contradicted"),
            vd("C9", "supported"),
        ]
    )
    batched[("e1", "C1")]["via"] = "single-retry"
    gold = [g("C1", "supported"), g("C2", "supported"), g("C3", "contradicted")]
    c = compare_modes(single, batched, gold)
    assert c.n_both == 3  # C4 only single, C9 only batched
    assert c.same_status == 1 and c.agree == 1 / 3
    assert c.same_side == 2  # C1 both supported; C3 both not-supported; C2 flips side
    assert (c.gold_n, c.single_correct, c.batched_correct) == (3, 2, 2)
    assert c.batched_via == {"single-retry": 1, "batch": 2}
    text = render_comparison(c)
    assert "claims in both: 3" in text and "0.333" in text and "C2" in text and "C3" in text


def test_compare_modes_skips_errors_text_changes_and_empty():
    from eval.score import compare_modes, render_comparison

    s = dict([vd("C1", "supported"), vd("C2", "supported", text="old")])
    b = dict([vd("C1", "supported"), vd("C2", "supported", text="new")])
    b[("e1", "C1")]["verdict"] = None
    c = compare_modes(s, b)
    assert c.n_both == 0 and c.agree is None
    assert "no claims have both" in render_comparison(c)


# ------------------------------------------------------------------ CLI: --verdicts folder, header, comparison
def _write_labels(path, rows):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLAIMS_HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _write_verdicts(folder, entries, enc="e1", via="single"):
    from pipeline.verify import Verdict, VerdictFile, VerdictRecord, save_verdict_file

    recs = [
        VerdictRecord(encounter_id=enc, claim_id=cid, section="subjective", claim_text=f"claim {cid}",
                      verdict=Verdict(status=st, evidence_turns=["T1"] if st != "unsupported" else [],
                                      evidence_quote="q" if st != "unsupported" else "", reason="r"),
                      via=via)
        for cid, st in entries
    ]  # fmt: skip
    save_verdict_file(VerdictFile(encounter_id=enc, model="m", records=recs), folder)


def _fixture(tmp_path):
    labels = tmp_path / "claims.csv"
    _write_labels(labels, [g(f"C{i}", "supported") for i in range(1, 5)])
    single, batched = tmp_path / "single", tmp_path / "batched"
    _write_verdicts(single, [("C1", "supported")])  # partial run: 1 claim
    _write_verdicts(
        batched,
        [("C1", "supported"), ("C2", "uncertain"), ("C3", "supported"), ("C4", "supported")],
        via="batch",
    )
    return labels, single, batched


def test_verdicts_option_defaults_to_the_single_claim_folder():
    from eval.score import build_parser
    from pipeline.verify import BATCH_VERDICTS_DIR, VERDICTS_DIR

    a = build_parser().parse_args([])
    assert a.verdicts == str(VERDICTS_DIR) and a.single_verdicts == str(VERDICTS_DIR)
    assert a.batched_verdicts == str(BATCH_VERDICTS_DIR)


def test_verdicts_option_selects_which_folder_is_scored_and_prints_its_path(tmp_path, capsys):
    from eval.score import main

    labels, single, batched = _fixture(tmp_path)
    common = [
        "--labels",
        str(labels),
        "--single-verdicts",
        str(single),
        "--batched-verdicts",
        str(batched),
        "--only",
        "e1",
    ]
    main(["--verdicts", str(single), *common])
    out = capsys.readouterr().out
    assert out.splitlines()[0] == f"SCORING VERDICTS FROM: {single}"
    assert "scored: 1 " in out and "no verdict: 3" in out  # only the single folder's 1 claim

    main(["--verdicts", str(batched), *common])
    out = capsys.readouterr().out
    assert out.splitlines()[0] == f"SCORING VERDICTS FROM: {batched}"
    assert "scored: 4 " in out and "no verdict: 0" in out
    assert out.count(f"[{batched}]") == 2  # the folder is repeated in both report titles


def test_both_reports_and_the_comparison_are_printed(tmp_path, capsys):
    from eval.score import main

    labels, single, batched = _fixture(tmp_path)
    main(["--verdicts", str(batched), "--labels", str(labels), "--single-verdicts", str(single),
          "--batched-verdicts", str(batched), "--only", "e1"])  # fmt: skip
    out = capsys.readouterr().out
    assert "ALL ROWS" in out and "EXCLUDING error_type = decomposition_error" in out
    assert "SINGLE vs BATCHED" in out and f"single={single}" in out and f"batched={batched}" in out
    assert "claims in both: 1" in out  # comparison uses the two folders, not --verdicts twice


def test_comparison_is_independent_of_which_folder_is_scored(tmp_path, capsys):
    from eval.score import main

    labels, single, batched = _fixture(tmp_path)
    args = [
        "--labels",
        str(labels),
        "--single-verdicts",
        str(single),
        "--batched-verdicts",
        str(batched),
        "--only",
        "e1",
    ]
    main(["--verdicts", str(single), *args])
    a = capsys.readouterr().out.split("SINGLE vs BATCHED")[1]
    main(["--verdicts", str(batched), *args])
    b = capsys.readouterr().out.split("SINGLE vs BATCHED")[1]
    assert a == b


def test_empty_or_missing_verdict_folder_is_an_error_not_an_empty_report(tmp_path):
    import pytest

    from eval.score import main

    labels, _, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["--verdicts", str(tmp_path / "nope"), "--labels", str(labels)])
    assert "no verdict JSON files" in str(e.value)


def test_comparison_skipped_when_a_folder_is_missing(tmp_path, capsys):
    from eval.score import main

    labels, single, _ = _fixture(tmp_path)
    main(["--verdicts", str(single), "--labels", str(labels), "--single-verdicts", str(single),
          "--batched-verdicts", str(tmp_path / "none"), "--only", "e1"])  # fmt: skip
    assert "comparison skipped" in capsys.readouterr().out


# ------------------------------------------------------------------ needs_review, per-error recall, v0 vs v1
def test_needs_review_flags_a_supported_verdict_with_an_unfindable_quote():
    gold = [g("C1", "uncertain"), g("C2", "supported"), g("C3", "supported")]
    ver = dict([
        vd("C1", "supported", quote="q", quote_valid=False),  # missed by status, caught by needs_review
        vd("C2", "supported", quote="q", quote_valid=False),  # false alarm only under needs_review
        vd("C3", "supported", quote="q", quote_valid=True),
    ])  # fmt: skip
    s = score(gold, ver)
    assert (s.tp, s.fp, s.fn, s.tn) == (0, 0, 1, 2) and s.recall == 0.0
    assert (s.tp_r, s.fp_r, s.fn_r, s.tn_r) == (1, 1, 0, 1)
    assert s.recall_r == 1.0 and s.precision_r == 0.5


def test_recall_per_error_type_on_status_and_needs_review():
    gold = [
        g("C1", "uncertain", "overstated"), g("C2", "uncertain", "overstated"),
        g("C3", "supported", "wrong_section"), g("C4", "contradicted", "negation"),
        g("C5", "supported", "none"),
    ]  # fmt: skip
    ver = dict([
        vd("C1", "uncertain"), vd("C2", "supported", quote="q", quote_valid=False),
        vd("C3", "supported"), vd("C4", "contradicted"), vd("C5", "supported"),
    ])  # fmt: skip
    s = score(gold, ver)
    assert s.per_error["overstated"] == {
        "n": 2,
        "gold_not_supported": 2,
        "flagged_status": 1,
        "flagged_review": 2,
    }
    assert s.error_recall("overstated") == 0.5 and s.error_recall("overstated", "review") == 1.0
    assert s.error_recall("negation") == 1.0 and s.error_recall("wrong_section") == 0.0
    assert s.error_recall("other") is None
    text = render(s, "t")
    for et in ("negation", "overstated", "over_interpretation", "wrong_section", "other"):
        assert et in text
    assert "status 1/2 = 0.500" in text and "needs_review 2/2 = 1.000" in text


def test_decomposition_error_row_only_shown_when_present_and_excluded_on_request():
    gold = [g("C1", "contradicted", "decomposition_error"), g("C2", "supported")]
    ver = dict([vd("C1", "contradicted"), vd("C2", "supported")])
    assert "decomposition_error" in render(score(gold, ver), "t")
    assert "decomposition_error" not in render(
        score(gold, ver, exclude_decomposition_error=True), "t"
    )


def test_compare_versions_counts_fixed_and_broken_on_common_claims_only():
    from eval.score import compare_versions, render_version_comparison

    gold = [
        g("C1", "uncertain"),
        g("C2", "supported"),
        g("C3", "supported"),
        g("C4", "uncertain"),
        g("C5", "supported"),
    ]
    v0 = dict(
        [vd("C1", "supported"), vd("C2", "supported"), vd("C3", "supported"), vd("C4", "supported")]
    )
    v1 = dict(
        [vd("C1", "uncertain"), vd("C2", "uncertain"), vd("C3", "supported"), vd("C5", "supported")]
    )
    c = compare_versions(gold, v0, v1)
    assert c.n_common == 3  # C4 only in v0, C5 only in v1
    assert (c.fixed, c.broken, len(c.changed)) == (1, 1, 2)
    by = {d["key"][1]: d["outcome"] for d in c.changed}
    assert by == {"C1": "FIXED", "C2": "BROKEN"}
    assert (c.v0.tp, c.v0.fn) == (0, 1) and (c.v1.tp, c.v1.fp) == (1, 1)
    text = render_version_comparison(c, "t")
    assert (
        "claims compared" in text
        and "recall (needs_review)" in text
        and "1 FIXED, 1 BROKEN" in text
    )


def test_compare_versions_can_exclude_decomposition_errors_and_handles_empty():
    from eval.score import compare_versions, render_version_comparison

    gold = [g("C1", "contradicted", "decomposition_error"), g("C2", "supported")]
    v0 = dict([vd("C1", "supported"), vd("C2", "supported")])
    v1 = dict([vd("C1", "contradicted"), vd("C2", "supported")])
    assert compare_versions(gold, v0, v1).fixed == 1
    ex = compare_versions(gold, v0, v1, exclude_decomposition_error=True)
    assert ex.n_common == 1 and ex.fixed == 0 and not ex.changed
    assert "no claims have a usable verdict" in render_version_comparison(
        compare_versions([], {}, {}), "t"
    )


def test_cli_prints_the_v0_vs_v1_section(tmp_path, capsys):
    from eval.score import main

    labels = tmp_path / "claims.csv"
    _write_labels(labels, [g("C1", "uncertain"), g("C2", "supported")])
    v0d, v1d = tmp_path / "v0", tmp_path / "v1"
    _write_verdicts(v0d, [("C1", "supported"), ("C2", "supported")], via="batch")
    _write_verdicts(v1d, [("C1", "uncertain"), ("C2", "supported")], via="batch")
    base = [
        "--labels",
        str(labels),
        "--only",
        "e1",
        "--single-verdicts",
        str(v0d),
        "--batched-verdicts",
        str(v0d),
    ]
    main(["--verdicts", str(v1d), "--v0-verdicts", str(v0d), "--v1-verdicts", str(v1d), *base])
    out = capsys.readouterr().out
    assert "v0 vs v1 PROMPT" in out and "1 FIXED, 0 BROKEN" in out
    assert "ALL ROWS; v0=" in out and "EXCLUDING decomposition_error; v0=" in out
    assert "recall per error_type" in out and "flagging on needs_review" in out
    main(
        [
            "--verdicts",
            str(v1d),
            "--v0-verdicts",
            str(v0d),
            "--v1-verdicts",
            str(tmp_path / "none"),
            *base,
        ]
    )
    assert "v0-vs-v1 comparison skipped" in capsys.readouterr().out
