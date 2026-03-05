from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.classifier import RulesClassifier


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

