from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)