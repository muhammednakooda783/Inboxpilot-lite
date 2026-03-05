from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["question", "complaint", "sales", "spam", "other"]


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def text_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ClassifyResponse(BaseModel):
    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_reply: str = Field(..., min_length=1, max_length=500)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"

