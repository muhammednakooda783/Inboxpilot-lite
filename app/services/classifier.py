from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import httpx

from app.core.config import Settings
from app.models.schemas import Category, ClassifyResponse

logger = logging.getLogger(__name__)

VALID_CATEGORIES: set[str] = {"question", "complaint", "sales", "spam", "other"}


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


class OpenAIClassifier:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        fallback: MessageClassifier,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback

    async def classify(self, text: str) -> ClassifyResponse:
        try:
            payload = self._build_payload(text)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
            return self._parse_response(response.json())
        except Exception as exc:
            logger.warning(
                "openai_classification_failed fallback=rules reason=%s",
                type(exc).__name__,
                extra={"request_id": "-"},
            )
            return await self.fallback.classify(text)

    def _build_payload(self, text: str) -> dict[str, object]:
        return {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify incoming support message. Return JSON only with keys: "
                        "category (question|complaint|sales|spam|other), confidence (0-1), "
                        "suggested_reply (short helpful reply)."
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
        parsed = json.loads(self._extract_json(content if isinstance(content, str) else "{}"))
        raw_category = str(parsed.get("category", "other")).lower().strip()
        category: Category = (
            raw_category if raw_category in VALID_CATEGORIES else "other"
        )  # type: ignore[assignment]
        confidence = self._clamp_confidence(parsed.get("confidence", 0.6))
        suggested_reply = str(parsed.get("suggested_reply", "")).strip()
        if not suggested_reply:
            suggested_reply = (
                "Thanks for your message. Could you share a bit more detail so I can help?"
            )
        return ClassifyResponse(
            category=category,
            confidence=confidence,
            suggested_reply=suggested_reply,
        )

    def _extract_json(self, content: str) -> str:
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        return match.group(0) if match else "{}"

    def _clamp_confidence(self, value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.6
        return max(0.0, min(1.0, number))


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
            fallback=rules_classifier,
        )
    logger.info(
        "classifier_selected type=RulesClassifier",
        extra={"request_id": "-"},
    )
    return rules_classifier
