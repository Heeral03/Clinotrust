from pathlib import Path

from pipeline.decompose import Claim, DecomposeResult
from pipeline.labels import (
    CLAIMS_HEADER,
    LABEL_COLUMNS,
    OMISSIONS_HEADER,
    ensure_omissions,
    labeled_encounters,
    merge_claims,
    read_rows,
    write_rows,
)
from pipeline.make_review import render_review, write_review


def res(texts, enc="e1", section="subjective"):
    return DecomposeResult(
        encounter_id=enc,
        claims=[Claim(id=f"C{i}", encounter_id=enc, section=section, source_statement_index=i - 1, text=t)
                for i, t in enumerate(texts, 1)],
    )  # fmt: skip


def test_header_and_blank_label_columns(tmp_path):
    p = tmp_path / "claims.csv"
    added, kept, warns = merge_claims([res(["a", "b"])], p)
    assert (added, kept, warns) == (2, 0, [])
    assert p.read_text().splitlines()[0] == ",".join(CLAIMS_HEADER)
    assert len(LABEL_COLUMNS) == 6  # five label columns + notes
    for r in read_rows(p):
        assert all(r[c] == "" for c in LABEL_COLUMNS)
    assert [r["claim_id"] for r in read_rows(p)] == ["C1", "C2"]


def test_omissions_header_only_and_never_overwritten(tmp_path):
    p = tmp_path / "omissions.csv"
    assert ensure_omissions(p) and p.read_text().strip() == ",".join(OMISSIONS_HEADER)
    p.write_text(p.read_text() + "e1,O1,plan,x,,,,\n")
    assert not ensure_omissions(p) and "O1" in p.read_text()


def test_rerun_is_idempotent(tmp_path):
    p = tmp_path / "claims.csv"
    merge_claims([res(["a", "b"])], p)
    before = p.read_text()
    added, kept, warns = merge_claims([res(["a", "b"])], p)
    assert added == 0 and kept == 2 and warns == [] and p.read_text() == before


def test_labels_survive_and_ids_are_frozen(tmp_path):
    p = tmp_path / "claims.csv"
    merge_claims([res(["a", "b", "c"])], p)
    rows = read_rows(p)
    rows[1]["status"], rows[1]["evidence_turns"] = "supported", "T3"
    rows[2]["claim_id"] = "C3a"  # hand-split ids are respected
    write_rows(p, CLAIMS_HEADER, rows)
    # decomposer now yields a new claim inserted first; ids in the sheet must not shift
    added, _, warns = merge_claims([res(["new", "a", "b", "c"])], p)
    out = {r["claim_text"]: r for r in read_rows(p)}
    assert added == 1 and out["b"]["status"] == "supported" and out["b"]["claim_id"] == "C2"
    assert out["a"]["claim_id"] == "C1" and out["c"]["claim_id"] == "C3a"
    assert out["new"]["claim_id"] == "C4" and warns  # above current maximum, and reported


def test_orphans_are_kept_and_reported(tmp_path):
    p = tmp_path / "claims.csv"
    merge_claims([res(["a", "b"])], p)
    _, _, warns = merge_claims([res(["a"])], p)
    assert len(read_rows(p)) == 2 and any("no longer produced" in w for w in warns)


def test_labeled_encounters(tmp_path):
    p = tmp_path / "claims.csv"
    assert labeled_encounters(p) == set()
    merge_claims([res(["a"], "e1"), res(["z"], "e2")], p)
    assert labeled_encounters(p) == {"e1", "e2"}


def test_review_file_has_timestamps_and_ids(tmp_path):
    from data.loaders import load_primock57

    enc = load_primock57()[0]
    text = render_review(enc)
    assert "T1 [" in text and "s):" in text and enc.id in text and enc.loader_version in text
    p, warn = write_review(
        {"id": enc.id, "loader_version": enc.loader_version}, tmp_path, Path("data/processed")
    )
    assert p.exists() and warn == ""
    _, warn = write_review(
        {"id": enc.id, "loader_version": "loader-v1"}, tmp_path, Path("data/processed")
    )
    assert "loader-v1" in warn


def test_wrong_shaped_sheet_is_refused_and_untouched(tmp_path):
    import pytest

    from pipeline.labels import LabelsFileError

    p = tmp_path / "claims.csv"
    ensure_omissions(p)  # an omissions-shaped file sitting where claims.csv should be
    before = p.read_text()
    with pytest.raises(LabelsFileError):
        merge_claims([res(["a"])], p)
    with pytest.raises(LabelsFileError):
        labeled_encounters(p)
    assert p.read_text() == before and not list(tmp_path.glob("*.bak"))


def test_write_makes_timestamped_backup_of_good_sheet(tmp_path):
    p = tmp_path / "claims.csv"
    merge_claims([res(["a"])], p)
    assert not list(tmp_path.glob("*.bak"))  # first write: nothing to back up
    merge_claims([res(["a", "b"])], p)
    baks = list(tmp_path.glob("claims.csv.*.bak"))
    assert len(baks) == 1 and baks[0].read_text().count("\n") == 2  # header + 1 row
    merge_claims([res(["a", "b"])], p)  # no change -> no rewrite, no extra backup
    assert len(list(tmp_path.glob("*.bak"))) == 1


def test_error_type_vocabulary_includes_decomposition_error():
    from pipeline.labels import ERROR_TYPES, validate_label_rows

    assert "decomposition_error" in ERROR_TYPES
    ok = {
        "encounter_id": "e",
        "claim_id": "C1",
        "status": "contradicted",
        "error_type": "decomposition_error",
        "severity": "major",
    }
    bad = {**ok, "claim_id": "C2", "error_type": "made_up"}
    assert validate_label_rows([ok, {**ok, "status": ""}]) == []
    assert len(validate_label_rows([bad])) == 1


def test_diff_claims_reports_changes(tmp_path):
    import json

    from pipeline import diff_claims

    claims = tmp_path / "claims"
    for v, texts in (("v-old", ["Patient reports cough"]), ("v-new", ["Patient reports no cough"])):
        (claims / v).mkdir(parents=True)
        (claims / v / "e1.json").write_text(res(texts).model_dump_json())
    gen = tmp_path / "gen"
    gen.mkdir()
    (gen / "e1.json").write_text(json.dumps({"soap": {"subjective": ["Patient reports no cough"]}}))
    diff_claims.CLAIMS = claims
    out = diff_claims.diff("e1", "v-old", "v-new", labels=tmp_path / "none.csv", generated=gen)
    assert "- Patient reports cough" in out and "+ Patient reports no cough" in out
