from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import Settings
from app.models.schemas import Category, ClassifyResponse

logger = logging.getLogger(__name__)

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class MessageClassifier(Protocol):
    async def classify(self, text: str) -> ClassifyResponse:
        ...


class RulesClassifier:
    spam_patterns = [
        r"\bfree money\b",
        r"\bclick here\b",
        r"\bguaranteed profit\b",
        r"\bwin(?:ner)?\b",
        r"\blottery\b",
    ]
    complaint_patterns = [
        r"\brefund\b",
        r"\bcancel(?:lation)?\b",
        r"\bunhappy\b",
        r"\bnot working\b",
        r"\bterrible\b",
        r"\bdamaged\b",
    ]
    sales_patterns = [
        r"\bpricing\b",
        r"\bquote\b",
        r"\bdemo\b",
        r"\bsubscription\b",
        r"\benterprise\b",
        r"\bbuy\b",
    ]
    question_patterns = [
        r"\?$",
        r"^\s*(how|what|when|where|why|can|could|do|does|is|are)\b",
    ]

    async def classify(self, text: str) -> ClassifyResponse:
        normalized = text.strip().lower()
        if self._matches_any(normalized, self.spam_patterns):
            return self._build("spam", 0.96)
        if self._matches_any(normalized, self.complaint_patterns):
            return self._build("complaint", 0.9)
        if self._matches_any(normalized, self.sales_patterns):
            return self._build("sales", 0.89)
        if self._matches_any(normalized, self.question_patterns) or "?" in normalized:
            return self._build("question", 0.84)
        return self._build("other", 0.65)

    def _matches_any(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def _build(self, category: Category, confidence: float) -> ClassifyResponse:
        replies: dict[Category, str] = {
            "question": "Thanks for your question. Share a bit more detail and I can help quickly.",
            "complaint": "I'm sorry you had this experience. Please share details so we can resolve it.",
            "sales": "Thanks for reaching out. Share your goals and budget and we can suggest a plan.",
            "spam": "This message looks like spam. Reply with context if this was sent in error.",
            "other": "Thanks for your message. Could you clarify what you need help with?",
        }
        return ClassifyResponse(
            category=category,
            confidence=confidence,
            suggested_reply=replies[category],
        )


class OpenAIClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_reply: str = Field(..., min_length=1, max_length=500)


class OpenAIClassifier:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        fallback: MessageClassifier,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.fallback = fallback

    async def classify(self, text: str) -> ClassifyResponse:
        if not self.api_key:
            return await self.fallback.classify(text)

        for attempt in range(self.max_retries + 1):
            try:
                body = await self._request_openai(text)
                return self._parse_response(body)
            except Exception as exc:
                if attempt < self.max_retries and self._is_transient_error(exc):
                    wait_seconds = self.retry_backoff_seconds * (2**attempt)
                    logger.warning(
                        "openai_retry attempt=%d wait_seconds=%.2f reason=%s",
                        attempt + 1,
                        wait_seconds,
                        type(exc).__name__,
                        extra={"request_id": "-"},
                    )
                    await asyncio.sleep(wait_seconds)
                    continue
                logger.warning(
                    "openai_classification_failed fallback=rules reason=%s",
                    type(exc).__name__,
                    extra={"request_id": "-"},
                )
                return await self.fallback.classify(text)

        return await self.fallback.classify(text)

    async def _request_openai(self, text: str) -> dict[str, object]:
        payload = self._build_payload(text)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        return response.json()

    def _build_payload(self, text: str) -> dict[str, object]:
        output_schema = OpenAIClassificationOutput.model_json_schema()
        return {
            "model": self.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification_result",
                    "strict": True,
                    "schema": output_schema,
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify incoming support message. Return JSON only with keys: "
                        "category, confidence, suggested_reply."
                    ),
                },
                {"role": "user", "content": text},
            ],
        }

    def _parse_response(self, body: dict[str, object]) -> ClassifyResponse:
        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "{}")
        )
        output = OpenAIClassificationOutput.model_validate_json(
            self._extract_json(content if isinstance(content, str) else "{}")
        )
        return ClassifyResponse.model_validate(output.model_dump())

    def _extract_json(self, content: str) -> str:
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        return match.group(0) if match else "{}"

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in TRANSIENT_STATUS_CODES
        if isinstance(exc, (json.JSONDecodeError, ValidationError)):
            return False
        return False


def get_classifier(settings: Settings) -> MessageClassifier:
    rules_classifier = RulesClassifier()
    if settings.openai_api_key:
        logger.info(
            "classifier_selected type=OpenAIClassifier",
            extra={"request_id": "-"},
        )
        return OpenAIClassifier(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            retry_backoff_seconds=settings.openai_retry_backoff_seconds,
            fallback=rules_classifier,
        )
    logger.info(
        "classifier_selected type=RulesClassifier",
        extra={"request_id": "-"},
    )
    return rules_classifier
