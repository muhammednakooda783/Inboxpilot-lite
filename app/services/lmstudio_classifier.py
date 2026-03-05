from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from app.models.schemas import ClassifyResponse
from app.services.classifier import MessageClassifier, RulesClassifier

logger = logging.getLogger(__name__)

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = "openai/gpt-oss-20b"
ALLOWED_CATEGORIES: set[str] = {"question", "complaint", "sales", "spam", "other"}


class LMStudioClassifier:
    def __init__(
        self,
        fallback: MessageClassifier | None = None,
        model: str = LMSTUDIO_MODEL,
        base_url: str = LMSTUDIO_BASE_URL,
        api_key: str = LMSTUDIO_API_KEY,
        timeout_seconds: float = 20.0,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.fallback = fallback or RulesClassifier()
        self.client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

    async def classify(self, text: str) -> ClassifyResponse:
        result, _, _, _ = await self.classify_with_details(text)
        return result

    async def classify_with_details(
        self, text: str
    ) -> tuple[ClassifyResponse, str, bool, str | None]:
        prompt = self._build_prompt(text)
        try:
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = self._extract_content(completion)
            logger.info("lmstudio_raw_output=%s", content[:300])
            payload = extract_first_json_object(content)
            normalized = self._validate_payload(payload)
            return ClassifyResponse.model_validate(normalized), "lmstudio", True, None
        except Exception as exc:
            message = "lmstudio_unreachable fallback=rules reason=%s"
            if not self._is_unreachable(exc):
                message = "lmstudio_classification_failed fallback=rules reason=%s"
                logger.warning(
                    "lmstudio_parse_or_validation_failure reason=%s",
                    str(exc)[:300],
                )
            logger.warning(message, type(exc).__name__)
            fallback_result = await self.fallback.classify(text)
            return fallback_result, "rules", True, str(exc)[:300]

    def _build_prompt(self, user_text: str) -> str:
        return (
            "Classify the message into ONE category:\n"
            "- question: how-to/help/info requests\n"
            "- complaint: broken, refund, unhappy, problem\n"
            "- sales: questions about price, quotes, costs, availability, ordering, buying, "
            "bulk orders, discounts\n"
            "- spam: scams, giveaways, suspicious links\n"
            "- other: greetings, unclear, unrelated\n\n"
            "Examples:\n"
            '"I want to know the price for bulk orders." -> sales\n'
            '"How much does it cost?" -> sales\n'
            '"How do I reset my password?" -> question\n'
            '"I want a refund. Item is broken." -> complaint\n'
            '"Win a free iPhone click here" -> spam\n'
            '"Hello" -> other\n\n'
            "Return exactly ONE JSON object and no extra text.\n"
            "Do not return multiple JSON objects.\n\n"
            "Schema:\n"
            '{\n  "category": "question|complaint|sales|spam|other",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "suggested_reply": "..."\n'
            "}\n\n"
            f"Message:\n{user_text}"
        )

    def _extract_content(self, completion: Any) -> str:
        choices = getattr(completion, "choices", [])
        if not choices:
            return "{}"
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else "{}"

    def _validate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        category_raw = payload.get("category")
        if not isinstance(category_raw, str):
            raise ValueError("category must be a string")
        category = category_raw.strip().lower()
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"invalid category: {category}")

        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        confidence = self._normalize_confidence(confidence)

        suggested_reply_raw = payload.get("suggested_reply")
        if not isinstance(suggested_reply_raw, str) or not suggested_reply_raw.strip():
            raise ValueError("suggested_reply must be a non-empty string")

        return {
            "category": category,
            "confidence": confidence,
            "suggested_reply": suggested_reply_raw.strip(),
        }

    def _normalize_confidence(self, value: float) -> float:
        if 0.0 <= value <= 1.0:
            return value
        if -0.05 <= value <= 1.05:
            return max(0.0, min(1.0, value))
        raise ValueError("confidence is out of accepted range")

    def _is_unreachable(self, exc: Exception) -> bool:
        return isinstance(exc, (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError))


def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    depth = 0
    in_string = False
    escape = False
    end = -1

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end == -1:
        raise ValueError("No complete JSON object found in model output")

    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object")
    return parsed
