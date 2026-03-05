from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.metrics import InMemoryMetrics
from app.core.rate_limit import InMemoryRateLimiter
from app.main import app
from app.services.classifier import RulesClassifier
from app.services.lmstudio_classifier import LMStudioClassifier


@pytest.fixture(autouse=True)
def force_rules_classifier() -> None:
    app.state.classifier = RulesClassifier()
    app.state.metrics = InMemoryMetrics()
    app.state.rate_limiter = InMemoryRateLimiter(max_requests=1000, window_seconds=60)


async def post_classify(text: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/classify", json={"text": text})


async def post_classify_batch(texts: list[str]):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/classify/batch", json={"texts": texts})


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


@pytest.mark.asyncio
async def test_request_id_is_generated_when_missing():
    response = await post_classify("How does billing work?")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_request_id_is_preserved_when_provided():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/classify",
            headers={"x-request-id": "req-123"},
            json={"text": "Can I get a demo?"},
        )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_counters():
    await post_classify("Hello there")
    await post_classify_batch(["What is your pricing?", "I need a refund"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert "counters" in body
    counters = body["counters"]
    assert counters["classify_requests_total"] == 1
    assert counters["classify_batch_requests_total"] == 1
    assert counters["classify_messages_total"] == 3


@pytest.mark.asyncio
async def test_rate_limit_blocks_excess_requests():
    app.state.rate_limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    first = await post_classify("Question one?")
    second = await post_classify("Question two?")
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded. Try again later."


@pytest.mark.asyncio
async def test_classify_batch_success():
    response = await post_classify_batch(["How do I upgrade?", "I want a refund"])
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["category"] == "question"
    assert body["results"][1]["category"] == "complaint"


@pytest.mark.asyncio
async def test_classify_batch_rejects_oversized_batch():
    response = await post_classify_batch(["hello"] * 21)
    assert response.status_code == 400
    assert "exceeds max allowed 20" in response.json()["detail"]


@pytest.mark.asyncio
async def test_lmstudio_classifier_returns_model_json():
    captured: dict[str, object] = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"category":"question","confidence":0.92,'
                            '"suggested_reply":"Happy to help. What details can you share?"}'
                        )
                    )
                )
            ]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    classifier = LMStudioClassifier(client=fake_client, fallback=RulesClassifier())  # type: ignore[arg-type]

    result = await classifier.classify("How do I change my billing plan?")

    assert result.category == "question"
    assert result.confidence == 0.92
    assert "help" in result.suggested_reply.lower()
    assert captured["model"] == "openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_lmstudio_classifier_falls_back_when_unreachable():
    def fake_create(**kwargs):  # noqa: ARG001
        raise ConnectionError("LM Studio is down")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    classifier = LMStudioClassifier(client=fake_client, fallback=RulesClassifier())  # type: ignore[arg-type]

    result = await classifier.classify("I need a refund for a damaged order")

    assert result.category == "complaint"
