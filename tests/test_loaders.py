import re

import pytest

from data.loaders import PRIMOCK, _speaker_from_filename, load_aci_bench, load_primock57
from data.process import save_processed
from data.schemas import LOADER_VERSION


@pytest.fixture(scope="module")
def pm():
    return load_primock57()


def test_counts(pm):
    assert len(pm) == 57
    assert len(load_aci_bench(("train", "valid"))) == 87


def test_no_tags_remain(pm):
    for e in pm:
        for t in e.turns:
            assert not re.search(r"[<>]|INAUDIBLE|UNIN|UNSURE", t.text), (e.id, t.id)
            assert t.text and t.text == " ".join(t.text.split())


def test_sorted_by_start(pm):
    for e in pm:
        starts = [t.start for t in e.turns]
        assert starts == sorted(starts), e.id


def test_ids_sequential(pm):
    for e in pm:
        assert [t.id for t in e.turns] == [f"T{i}" for i in range(1, len(e.turns) + 1)]


def test_speaker_from_filename_both_layouts():
    # day1_consultation01 uses tier names "Doctor"/"Patient"; day1_consultation02 uses "Speaker".
    for cid in ("day1_consultation01", "day1_consultation02"):
        for who in ("doctor", "patient"):
            assert _speaker_from_filename(PRIMOCK / "transcripts" / f"{cid}_{who}.TextGrid") == who
    with pytest.raises(ValueError):
        _speaker_from_filename(PRIMOCK / "transcripts" / "weird.TextGrid")


def test_both_layouts_load_both_speakers(pm):
    by_id = {e.id: e for e in pm}
    for cid in ("day1_consultation01", "day1_consultation02"):
        assert {t.speaker for t in by_id[cid].turns} == {"doctor", "patient"}


def test_short_answer_follows_its_question(pm):
    e = next(x for x in pm if x.id == "day1_consultation01")
    idx = next(
        i
        for i, t in enumerate(e.turns)
        if t.speaker == "patient" and t.text == "Uh, no." and t.start and 131 < t.start < 132
    )
    q, a = e.turns[idx - 1], e.turns[idx]
    assert q.speaker == "doctor" and "night sweats" in q.text
    assert a.overlaps_previous and q.start < a.start < q.end


def test_flags_present(pm):
    assert any(t.had_unsure for e in pm for t in e.turns)
    assert any(t.had_inaudible for e in pm for t in e.turns)
    assert any(t.overlaps_previous for e in pm for t in e.turns)
    assert not pm[0].turns[0].overlaps_previous


def test_transcript_rendering(pm):
    e = pm[0]
    plain, timed = e.transcript().splitlines(), e.transcript(with_times=True).splitlines()
    assert re.fullmatch(r"T1 \[(Doctor|Patient)\]: .+", plain[0])
    assert re.fullmatch(r"T1 \[(Doctor|Patient)\] \(\d+\.\ds\): .+", timed[0])
    assert "s)" not in plain[0].split(":")[0]


def test_merge_same_speaker():
    raw = load_primock57()
    mrg = load_primock57(merge_same_speaker=True)
    assert mrg[0].merged_same_speaker and not raw[0].merged_same_speaker
    for e in mrg:
        assert all(a.speaker != b.speaker for a, b in zip(e.turns, e.turns[1:])), e.id
        assert [t.id for t in e.turns] == [f"T{i}" for i in range(1, len(e.turns) + 1)]
    assert sum(len(e.turns) for e in mrg) < sum(len(e.turns) for e in raw)


def test_save_processed_roundtrip(pm, tmp_path):
    p = save_processed(pm[0], tmp_path)
    assert p.name == f"{pm[0].id}.json"
    from data.schemas import Encounter

    back = Encounter.model_validate_json(p.read_text())
    assert back == pm[0] and back.loader_version == LOADER_VERSION


def test_aci_turns_have_ids():
    e = load_aci_bench(("train",))[0]
    assert e.turns[0].id == "T1" and e.turns[0].start is None
