from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from app.models.schemas import ClassifyResponse
from app.services.classifier import MessageClassifier, RulesClassifier

logger = logging.getLogger(__name__)

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = "openai/gpt-oss-20b"


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
            payload = json.loads(self._extract_json(content))
            return ClassifyResponse.model_validate(payload)
        except Exception as exc:
            message = "lmstudio_unreachable fallback=rules reason=%s"
            if not self._is_unreachable(exc):
                message = "lmstudio_classification_failed fallback=rules reason=%s"
            logger.warning(message, type(exc).__name__)
            return await self.fallback.classify(text)

    def _build_prompt(self, user_text: str) -> str:
        return (
            "Classify the message into ONE category:\n\n"
            "question\n"
            "complaint\n"
            "sales\n"
            "spam\n"
            "other\n\n"
            "Return ONLY valid JSON.\n\n"
            "Use this exact format:\n"
            '{\n  "category": "",\n  "confidence": 0-1,\n  "suggested_reply": ""\n}\n\n'
            f"Message:\n{user_text}"
        )

    def _extract_content(self, completion: Any) -> str:
        choices = getattr(completion, "choices", [])
        if not choices:
            return "{}"
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else "{}"

    def _extract_json(self, content: str) -> str:
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        return match.group(0) if match else "{}"

    def _is_unreachable(self, exc: Exception) -> bool:
        return isinstance(exc, (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError))
