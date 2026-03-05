from __future__ import annotations

import pytest
import httpx
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.classifier import OpenAIClassifier, RulesClassifier, get_classifier
from app.core.config import Settings


@pytest.fixture(autouse=True)
def force_rules_classifier() -> None:
    app.state.classifier = RulesClassifier()


async def post_classify(text: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/classify", json={"text": text})


@pytest.mark.asyncio
async def test_health_returns_ok():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_classifies_question():
    response = await post_classify("How do I reset my password?")
    body = response.json()
    assert response.status_code == 200
    assert body["category"] == "question"
    assert 0 <= body["confidence"] <= 1


@pytest.mark.asyncio
async def test_classifies_complaint():
    response = await post_classify("I am unhappy. The product arrived damaged and I want a refund.")
    assert response.status_code == 200
    assert response.json()["category"] == "complaint"


@pytest.mark.asyncio
async def test_classifies_sales():
    response = await post_classify("Can we schedule a demo and get pricing for enterprise?")
    assert response.status_code == 200
    assert response.json()["category"] == "sales"


@pytest.mark.asyncio
async def test_classifies_spam():
    response = await post_classify("You are a winner! Click here for free money now!")
    assert response.status_code == 200
    assert response.json()["category"] == "spam"


@pytest.mark.asyncio
async def test_classifies_other():
    response = await post_classify("Hello team, just sharing an update from our side.")
    assert response.status_code == 200
    assert response.json()["category"] == "other"


@pytest.mark.asyncio
async def test_blank_text_returns_validation_error():
    response = await post_classify("   ")
    assert response.status_code == 422


def _openai_http_response(url: str, content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", url),
        json={"choices": [{"message": {"content": content}}]},
    )


@pytest.mark.asyncio
async def test_openai_classifier_uses_strict_json_and_timeout(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    async def fake_post(self, url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _openai_http_response(
            url,
            '{"category":"sales","confidence":0.91,"suggested_reply":"Thanks. I can share pricing options."}',
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    classifier = OpenAIClassifier(
        api_key="test-key",
        model="gpt-4o-mini",
        timeout_seconds=3.5,
        max_retries=2,
        retry_backoff_seconds=0.01,
        fallback=RulesClassifier(),
    )

    result = await classifier.classify("Please send me pricing details for enterprise.")

    assert result.category == "sales"
    assert result.confidence == 0.91
    assert "pricing" in result.suggested_reply.lower()
    assert captured["timeout"] == 3.5
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    payload = captured["json"]
    assert isinstance(payload, dict)
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_openai_classifier_retries_transient_error(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}
    delays: list[float] = []

    async def fake_sleep(delay: float):
        delays.append(delay)

    async def fake_post(self, url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadTimeout("timed out")
        return _openai_http_response(
            url,
            '{"category":"question","confidence":0.77,"suggested_reply":"Happy to help. Can you share more detail?"}',
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("app.services.classifier.asyncio.sleep", fake_sleep)

    classifier = OpenAIClassifier(
        api_key="test-key",
        model="gpt-4o-mini",
        timeout_seconds=2.0,
        max_retries=2,
        retry_backoff_seconds=0.25,
        fallback=RulesClassifier(),
    )
    result = await classifier.classify("How can I start a trial?")

    assert result.category == "question"
    assert calls["count"] == 2
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_openai_classifier_falls_back_on_invalid_json_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_post(self, url, headers=None, json=None, timeout=None):
        return _openai_http_response(
            url,
            '{"category":"unknown","confidence":1.2,"suggested_reply":""}',
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    classifier = OpenAIClassifier(
        api_key="test-key",
        model="gpt-4o-mini",
        timeout_seconds=2.0,
        max_retries=1,
        retry_backoff_seconds=0.1,
        fallback=RulesClassifier(),
    )
    result = await classifier.classify("I need a refund for my order.")

    assert result.category == "complaint"


def test_get_classifier_selects_rules_without_api_key():
    settings = Settings(
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_timeout_seconds=2.0,
        openai_max_retries=2,
        openai_retry_backoff_seconds=0.1,
    )
    classifier = get_classifier(settings)
    assert isinstance(classifier, RulesClassifier)


def test_get_classifier_selects_openai_with_api_key():
    settings = Settings(
        openai_api_key="abc123",
        openai_model="gpt-4o-mini",
        openai_timeout_seconds=2.0,
        openai_max_retries=2,
        openai_retry_backoff_seconds=0.1,
    )
    classifier = get_classifier(settings)
    assert isinstance(classifier, OpenAIClassifier)
