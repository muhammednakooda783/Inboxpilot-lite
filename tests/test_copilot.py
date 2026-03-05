from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.metrics import InMemoryMetrics
from app.core.rate_limit import InMemoryRateLimiter
from app.db import init_db
from app.main import app
from app.services.classifier import RulesClassifier
from app.services.copilot import build_templated_draft_reply


class StubDraftService:
    def __init__(self, reply: str = "I can help with that.", should_fail: bool = False) -> None:
        self.reply = reply
        self.should_fail = should_fail

    async def draft_reply(self, **_: object) -> str:
        if self.should_fail:
            raise ConnectionError("LM Studio unavailable")
        return self.reply

    def is_unreachable_error(self, exc: Exception) -> bool:
        return isinstance(exc, ConnectionError)


@pytest.fixture(autouse=True)
def setup_app_state(tmp_path) -> None:
    app.state.classifier = RulesClassifier()
    app.state.draft_service = StubDraftService()
    app.state.metrics = InMemoryMetrics()
    app.state.rate_limiter = InMemoryRateLimiter(max_requests=1000, window_seconds=60)
    db_path = str(tmp_path / "test.sqlite3")
    init_db(db_path)
    app.state.db_path = db_path


async def post_copilot(text: str, channel: str = "webchat"):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/copilot", json={"text": text, "channel": channel})


@pytest.mark.asyncio
async def test_copilot_complaint_returns_high_priority_and_expected_actions():
    response = await post_copilot("I want a refund. The item is broken.")
    body = response.json()

    assert response.status_code == 200
    assert body["intent"]["category"] == "complaint"
    assert body["priority"] == "high"
    assert "Apologize and acknowledge the issue" in body["next_actions"]
    assert "Ask for order number / reference" in body["next_actions"]
    assert "Confirm refund/replacement preference" in body["next_actions"]
    assert body["classifier_used"] == "rules"
    assert body["request_id"]


@pytest.mark.asyncio
async def test_copilot_sales_returns_medium_priority_and_non_empty_draft_reply():
    app.state.draft_service = StubDraftService(reply="Great question. I can prepare a quote today.")
    response = await post_copilot("How much does this cost for bulk orders?", channel="email")
    body = response.json()

    assert response.status_code == 200
    assert body["intent"]["category"] == "sales"
    assert body["priority"] == "medium"
    assert body["draft_reply"].strip()
    assert body["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_copilot_uses_template_reply_when_lmstudio_draft_fails():
    app.state.draft_service = StubDraftService(should_fail=True)
    response = await post_copilot("How do I reset my password?", channel="whatsapp")
    body = response.json()

    assert response.status_code == 200
    assert body["intent"]["category"] == "question"
    assert body["draft_reply"] == build_templated_draft_reply("question", "whatsapp")
