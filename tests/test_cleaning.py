from data.cleaning import clean_text


def test_unin_and_inaudible_removed_with_flag():
    t, u, i = clean_text("<UNIN/> Hello <INAUDIBLE_SPEECH/> there")
    assert (t, u, i) == ("Hello there", False, True)


def test_unsure_unwrapped_with_flag():
    t, u, i = clean_text("<UNSURE>Hello how</UNSURE> um. Good morning")
    assert (t, u, i) == ("Hello how um. Good morning", True, False)


def test_whitespace_collapsed():
    assert clean_text("  a \t b\n c  ")[0] == "a b c"


def test_only_tags_gives_empty():
    assert clean_text("<UNIN/> <INAUDIBLE_SPEECH/>")[0] == ""
