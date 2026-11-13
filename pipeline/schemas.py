from pydantic import BaseModel, field_validator

NOT_DOCUMENTED = "Not documented"


class SOAPNote(BaseModel):
    subjective: list[str]
    objective: list[str]
    assessment: list[str]
    plan: list[str]

    @field_validator("subjective", "objective", "assessment", "plan")
    @classmethod
    def _clean(cls, items: list[str]) -> list[str]:
        items = [s.strip() for s in items if s and s.strip()]
        # An empty section must say so explicitly; absence is never implied.
        return items or [NOT_DOCUMENTED]
