from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["question", "complaint", "sales", "spam", "other"]
Channel = Literal["whatsapp", "email", "webchat"]
Priority = Literal["low", "medium", "high"]


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


class ClassifyResponseWithMeta(ClassifyResponse):
    classifier_used: Literal["lmstudio", "rules"]
    latency_ms: int = Field(..., ge=0)
    request_id: str = Field(..., min_length=1)


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)

    @field_validator("texts")
    @classmethod
    def items_cannot_be_blank(cls, values: list[str]) -> list[str]:
        for index, value in enumerate(values):
            if not value.strip():
                raise ValueError(f"texts[{index}] must not be blank")
            if len(value) > 4000:
                raise ValueError(f"texts[{index}] must be at most 4000 characters")
        return values


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class MetricsResponse(BaseModel):
    counters: dict[str, int]


class InfoResponse(BaseModel):
    active_classifier: str
    model: str | None = None
    version: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class CopilotRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    channel: Channel = "webchat"

    @field_validator("text")
    @classmethod
    def copilot_text_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class IntentResult(BaseModel):
    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0)


class CopilotResponse(BaseModel):
    intent: IntentResult
    priority: Priority
    next_actions: list[str] = Field(..., min_length=1)
    draft_reply: str = Field(..., min_length=1, max_length=500)
    classifier_used: Literal["lmstudio", "rules"]
    latency_ms: int = Field(..., ge=0)
    request_id: str = Field(..., min_length=1)
