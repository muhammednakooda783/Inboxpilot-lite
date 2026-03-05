from __future__ import annotations

import re
from typing import Protocol

from app.models.schemas import Category, ClassifyResponse


class MessageClassifier(Protocol):
    async def classify(self, text: str) -> ClassifyResponse: ...


def build_suggested_reply(category: Category) -> str:
    replies: dict[Category, str] = {
        "question": "Thanks for your question. Share a bit more detail and I can help quickly.",
        "complaint": (
            "I'm sorry you had this experience. Please share details so we can resolve it."
        ),
        "sales": "Thanks for reaching out. Share your goals and budget and we can suggest a plan.",
        "spam": "This message looks like spam. Reply with context if this was sent in error.",
        "other": "Thanks for your message. Could you clarify what you need help with?",
    }
    return replies[category]


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
        return ClassifyResponse(
            category=category,
            confidence=confidence,
            suggested_reply=build_suggested_reply(category),
        )
