from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.config import (
    configure_logging,
    get_settings,
    reset_request_id,
    set_request_id,
)
from app.core.metrics import InMemoryMetrics
from app.core.rate_limit import InMemoryRateLimiter
from app.models.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    MetricsResponse,
)
from app.services.classifier import MessageClassifier, RulesClassifier
from app.services.lmstudio_classifier import LMStudioClassifier

configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.state.classifier = LMStudioClassifier(
    fallback=RulesClassifier(),
    model=settings.lmstudio_model,
    base_url=settings.lmstudio_base_url,
    api_key=settings.lmstudio_api_key,
    timeout_seconds=settings.lmstudio_timeout_seconds,
)
app.state.metrics = InMemoryMetrics()
app.state.rate_limiter = InMemoryRateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = set_request_id(request_id)
    request.state.request_id = request_id
    metrics: InMemoryMetrics = app.state.metrics
    rate_limiter: InMemoryRateLimiter = app.state.rate_limiter
    metrics.increment("http_requests_total")
    client_ip = _get_client_ip(request)
    started_at = time.perf_counter()
    try:
        if not rate_limiter.allow(client_ip):
            metrics.increment("rate_limited_total")
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        else:
            response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        metrics.increment(f"http_status_{response.status_code}_total")
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_complete method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
    finally:
        reset_request_id(token)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    store: InMemoryMetrics = app.state.metrics
    return MetricsResponse(counters=store.snapshot())


@app.post("/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest) -> ClassifyResponse:
    classifier: MessageClassifier = app.state.classifier
    metrics_store: InMemoryMetrics = app.state.metrics
    metrics_store.increment("classify_requests_total")
    logger.info(
        "classify_started text_length=%d",
        len(payload.text),
    )
    result = await classifier.classify(payload.text)
    metrics_store.increment("classify_messages_total")
    metrics_store.increment(f"classify_category_{result.category}_total")
    logger.info(
        "classify_completed category=%s confidence=%.2f",
        result.category,
        result.confidence,
    )
    return result


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(payload: BatchClassifyRequest) -> BatchClassifyResponse:
    if len(payload.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch size {len(payload.texts)} exceeds max allowed {settings.max_batch_size}"
            ),
        )

    classifier: MessageClassifier = app.state.classifier
    metrics_store: InMemoryMetrics = app.state.metrics
    metrics_store.increment("classify_batch_requests_total")
    results: list[ClassifyResponse] = []
    for text in payload.texts:
        result = await classifier.classify(text)
        metrics_store.increment("classify_messages_total")
        metrics_store.increment(f"classify_category_{result.category}_total")
        results.append(result)

    logger.info("classify_batch_completed count=%d", len(results))
    return BatchClassifyResponse(results=results)
