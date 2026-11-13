"""Text cleaning for PriMock57 TextGrid annotations."""

import re

_UNSURE = re.compile(r"<UNSURE>(.*?)</UNSURE>", re.DOTALL)
_INAUDIBLE = re.compile(r"<INAUDIBLE_SPEECH\s*/>")
_UNIN = re.compile(r"<UNIN\s*/>")
_LEFTOVER = re.compile(r"</?[A-Za-z_]+\s*/?>")  # safety net for any unseen annotation tag


def clean_text(raw: str) -> tuple[str, bool, bool]:
    """Return (clean_text, had_unsure, had_inaudible). Flags are read before tags are removed."""
    had_unsure = bool(_UNSURE.search(raw))
    had_inaudible = bool(_INAUDIBLE.search(raw))
    text = _UNSURE.sub(r"\1", raw)
    text = _INAUDIBLE.sub(" ", _UNIN.sub(" ", text))
    text = _LEFTOVER.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, had_unsure, had_inaudible
