from pydantic import BaseModel

# Bump whenever loader output (turn ids, cleaning, rendering) changes. Stored with every
# processed encounter and every generated note so results are only compared like-for-like.
LOADER_VERSION = "loader-v2"


class Turn(BaseModel):
    id: str  # "T1", "T2", ... sequential within an encounter
    speaker: str  # "doctor" | "patient" (ACI-Bench also has "patient_guest")
    text: str
    start: float | None = None  # seconds; None for datasets without timestamps (ACI-Bench)
    end: float | None = None
    had_unsure: bool = False  # text contained <UNSURE>..</UNSURE> (uncertain transcription)
    had_inaudible: bool = False  # text contained <INAUDIBLE_SPEECH/>
    overlaps_previous: bool = False  # starts before the previous turn ended

    def render(self, with_times: bool = False) -> str:
        who = self.speaker.replace("_", " ").title()
        when = f" ({self.start:.1f}s)" if with_times and self.start is not None else ""
        return f"{self.id} [{who}]{when}: {self.text}"


class Encounter(BaseModel):
    id: str
    source: str  # "primock57" | "aci-bench"
    turns: list[Turn]
    reference_note: str
    meta: dict = {}
    merged_same_speaker: bool = False
    loader_version: str = LOADER_VERSION

    def transcript(self, with_times: bool = False) -> str:
        """One line per turn: 'T12 [Patient]: text'. with_times adds '(131.6s)' for labeling /
        the verifier; the generator's input deliberately uses the default (no times)."""
        return "\n".join(t.render(with_times) for t in self.turns)
