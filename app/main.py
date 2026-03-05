from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request

from app.core.config import configure_logging, get_settings
from app.models.schemas import ClassifyRequest, ClassifyResponse, HealthResponse
from app.services.classifier import MessageClassifier, get_classifier

configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.state.classifier = get_classifier(settings)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_complete method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        extra={"request_id": request_id},
    )
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponse:
    request_id = request.state.request_id
    classifier: MessageClassifier = app.state.classifier
    logger.info(
        "classify_started text_length=%d",
        len(payload.text),
        extra={"request_id": request_id},
    )
    result = await classifier.classify(payload.text)
    logger.info(
        "classify_completed category=%s confidence=%.2f",
        result.category,
        result.confidence,
        extra={"request_id": request_id},
    )
    return result

