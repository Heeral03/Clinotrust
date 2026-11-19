from eval.compare_runs import compare, render
from tests.test_score import vd


def test_compare_counts_changes_and_skips_missing():
    a = dict([vd("C1", "supported"), vd("C2", "supported"), vd("C3", "uncertain"),
              vd("C4", "uncertain"), vd("C5", "supported"), vd("C6", "supported")])  # fmt: skip
    b = dict([vd("C1", "supported"), vd("C2", "uncertain"), vd("C3", "supported"),
              vd("C4", "contradicted"), vd("C5", "supported")])  # C6 missing in b  # fmt: skip
    key, rec = vd("C5", "supported")
    rec["verdict"] = None  # C5 errored in b
    b[key] = rec
    r = compare(a, b)
    assert (r["compared"], r["skipped"], r["changed"]) == (4, 2, 3)
    assert (r["sup_to_not"], r["not_to_sup"]) == (1, 1)
    assert [row[0][1] for row in r["rows"]] == ["C2", "C3", "C4"]
    out = render(r, "A", "B")
    assert "skipped (no verdict in A or B): 2" in out and "other changes" in out and ": 1" in out
