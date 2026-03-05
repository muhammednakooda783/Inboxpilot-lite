from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from app.models.schemas import Category, Channel, ClassifyResponse, Priority
from app.services.classifier import MessageClassifier
from app.services.lmstudio_classifier import LMStudioClassifier

logger = logging.getLogger(__name__)

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class RoutingDecision:
    priority: Priority
    next_actions: list[str]


@dataclass(frozen=True)
class CopilotResult:
    intent: ClassifyResponse
    priority: Priority
    next_actions: list[str]
    draft_reply: str
    classifier_used: str
    draft_source: str


ROUTING_MAP: dict[Category, RoutingDecision] = {
    "complaint": RoutingDecision(
        priority="high",
        next_actions=[
            "Apologize and acknowledge the issue",
            "Ask for order number / reference",
            "Confirm refund/replacement preference",
        ],
    ),
    "sales": RoutingDecision(
        priority="medium",
        next_actions=[
            "Ask product/service and quantity",
            "Confirm delivery location and timeline",
            "Offer to prepare a quote",
        ],
    ),
    "question": RoutingDecision(
        priority="low",
        next_actions=[
            "Ask clarifying question if needed",
            "Provide steps/instructions",
        ],
    ),
    "spam": RoutingDecision(
        priority="low",
        next_actions=[
            "Do not engage",
            "Mark as spam",
        ],
    ),
    "other": RoutingDecision(
        priority="low",
        next_actions=[
            "Ask for clarification",
        ],
    ),
}


class LMStudioDraftService:
    def __init__(
        self,
        model: str = LMSTUDIO_MODEL,
        base_url: str = LMSTUDIO_BASE_URL,
        api_key: str = LMSTUDIO_API_KEY,
        timeout_seconds: float = 20.0,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

    async def draft_reply(
        self,
        text: str,
        channel: Channel,
        category: Category,
        next_actions: list[str],
    ) -> str:
        prompt = self._build_prompt(
            text=text,
            channel=channel,
            category=category,
            next_actions=next_actions,
        )
        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = self._extract_content(completion).strip()
        if not content:
            raise ValueError("LM Studio returned an empty draft reply")
        return content

    def _build_prompt(
        self,
        text: str,
        channel: Channel,
        category: Category,
        next_actions: list[str],
    ) -> str:
        style = {
            "whatsapp": "Write a short friendly WhatsApp response.",
            "email": "Write a professional, slightly formal email response.",
            "webchat": "Write a clear and neutral live chat response.",
        }[channel]
        actions = "\n".join(f"- {action}" for action in next_actions)
        return (
            "You are a customer support copilot.\n"
            f"{style}\n"
            "Use 1-3 sentences, plain text only, no markdown.\n\n"
            f"Category: {category}\n"
            "Next actions:\n"
            f"{actions}\n\n"
            f"Customer message:\n{text}\n\n"
            "Draft the best next reply now."
        )

    def _extract_content(self, completion: Any) -> str:
        choices = getattr(completion, "choices", [])
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""

    def is_unreachable_error(self, exc: Exception) -> bool:
        return isinstance(exc, (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError))


class SupportCopilotService:
    def __init__(self, classifier: MessageClassifier, draft_service: LMStudioDraftService) -> None:
        self.classifier = classifier
        self.draft_service = draft_service

    async def run(self, text: str, channel: Channel) -> CopilotResult:
        intent, classifier_used = await self._classify(text)
        decision = ROUTING_MAP[intent.category]
        logger.info(
            "copilot_intent category=%s confidence=%.2f classifier=%s",
            intent.category,
            intent.confidence,
            classifier_used,
        )

        draft_source = "lmstudio"
        try:
            draft_reply = await self.draft_service.draft_reply(
                text=text,
                channel=channel,
                category=intent.category,
                next_actions=decision.next_actions,
            )
        except Exception as exc:
            draft_source = "template"
            draft_reply = build_templated_draft_reply(intent.category, channel)
            reason = type(exc).__name__
            if self.draft_service.is_unreachable_error(exc):
                logger.warning("copilot_draft_fallback source=template reason=%s", reason)
            else:
                logger.warning("copilot_draft_generation_failed source=template reason=%s", reason)

        return CopilotResult(
            intent=intent,
            priority=decision.priority,
            next_actions=decision.next_actions,
            draft_reply=draft_reply,
            classifier_used=classifier_used,
            draft_source=draft_source,
        )

    async def _classify(self, text: str) -> tuple[ClassifyResponse, str]:
        if isinstance(self.classifier, LMStudioClassifier):
            result, classifier_used, _, _ = await self.classifier.classify_with_details(text)
            return result, classifier_used
        result = await self.classifier.classify(text)
        return result, "rules"


def build_templated_draft_reply(category: Category, channel: Channel) -> str:
    prefix = {
        "whatsapp": "Thanks for your message.",
        "email": "Thank you for contacting support.",
        "webchat": "Thanks for reaching out.",
    }[channel]
    body = {
        "question": "I can help with that and will share the steps right away.",
        "complaint": "I am sorry about this issue and will help resolve it quickly.",
        "sales": "I can help with pricing and the right option for your needs.",
        "spam": "We cannot proceed with this request.",
        "other": "Could you share a bit more detail so we can assist?",
    }[category]
    return f"{prefix} {body}"
