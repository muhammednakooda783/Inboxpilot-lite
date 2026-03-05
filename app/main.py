from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import (
    configure_logging,
    get_settings,
    reset_request_id,
    set_request_id,
)
from app.core.metrics import InMemoryMetrics
from app.core.rate_limit import InMemoryRateLimiter
from app.db import get_recent, get_stats, init_db, insert_classification
from app.models.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    ClassifyResponseWithMeta,
    CopilotRequest,
    CopilotResponse,
    HealthResponse,
    InfoResponse,
    IntentResult,
    MetricsResponse,
)
from app.services.classifier import MessageClassifier, RulesClassifier
from app.services.copilot import LMStudioDraftService, SupportCopilotService
from app.services.lmstudio_classifier import LMStudioClassifier

configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.classifier = LMStudioClassifier(
    fallback=RulesClassifier(),
    model=settings.lmstudio_model,
    base_url=settings.lmstudio_base_url,
    api_key=settings.lmstudio_api_key,
    timeout_seconds=settings.lmstudio_timeout_seconds,
)
app.state.draft_service = LMStudioDraftService(
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
init_db(settings.db_path)
app.state.db_path = settings.db_path


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


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    classifier = app.state.classifier
    active_classifier = "lmstudio" if isinstance(classifier, LMStudioClassifier) else "rules"
    model = classifier.model if isinstance(classifier, LMStudioClassifier) else None
    return InfoResponse(
        active_classifier=active_classifier,
        model=model,
        version=settings.app_version,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    store: InMemoryMetrics = app.state.metrics
    return MetricsResponse(counters=store.snapshot())


@app.get("/recent")
async def recent(
    limit: int = Query(default=20, ge=1, le=100),
    category: str | None = None,
    classifier: str | None = None,
    status: str | None = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    if status not in (None, "ok", "error"):
        raise HTTPException(status_code=400, detail="status must be 'ok' or 'error'")
    return get_recent(limit=limit, category=category, classifier=classifier, status=status, q=q)


@app.get("/stats")
async def stats(window_minutes: int = Query(default=60, ge=1, le=1440)) -> dict[str, Any]:
    return get_stats(window_minutes=window_minutes)


@app.post("/classify", response_model=ClassifyResponseWithMeta)
async def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponseWithMeta:
    classifier: MessageClassifier = app.state.classifier
    metrics_store: InMemoryMetrics = app.state.metrics
    request_id = request.state.request_id
    started_at = time.perf_counter()
    metrics_store.increment("classify_requests_total")
    logger.info(
        "classify_started text_length=%d",
        len(payload.text),
    )
    try:
        classifier_used = "rules"
        ok = True
        error_message: str | None = None
        if isinstance(classifier, LMStudioClassifier):
            result, classifier_used, ok, error_message = await classifier.classify_with_details(
                payload.text
            )
        else:
            result = await classifier.classify(payload.text)

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        metrics_store.increment("classify_messages_total")
        metrics_store.increment(f"classify_category_{result.category}_total")
        logger.info(
            "classify_completed category=%s confidence=%.2f classifier=%s latency_ms=%d",
            result.category,
            result.confidence,
            classifier_used,
            latency_ms,
        )
        insert_classification(
            request_id=request_id,
            text=payload.text,
            category=result.category,
            confidence=result.confidence,
            suggested_reply=result.suggested_reply,
            classifier_name=classifier_used,
            latency_ms=latency_ms,
            ok=ok,
            error_message=error_message,
            created_at=datetime.now(UTC).isoformat(),
        )
        return ClassifyResponseWithMeta(
            category=result.category,
            confidence=result.confidence,
            suggested_reply=result.suggested_reply,
            classifier_used=classifier_used,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            request_id=request_id,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        insert_classification(
            request_id=request_id,
            text=payload.text,
            category=None,
            confidence=None,
            suggested_reply=None,
            classifier_name="rules",
            latency_ms=latency_ms,
            ok=False,
            error_message=str(exc)[:300],
            created_at=datetime.now(UTC).isoformat(),
        )
        raise HTTPException(status_code=500, detail="Classification failed") from exc


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


@app.post("/copilot", response_model=CopilotResponse)
async def copilot(payload: CopilotRequest, request: Request) -> CopilotResponse:
    classifier: MessageClassifier = app.state.classifier
    draft_service: LMStudioDraftService = app.state.draft_service
    metrics_store: InMemoryMetrics = app.state.metrics
    request_id = request.state.request_id
    started_at = time.perf_counter()
    metrics_store.increment("copilot_requests_total")

    try:
        service = SupportCopilotService(classifier=classifier, draft_service=draft_service)
        result = await service.run(text=payload.text, channel=payload.channel)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        metrics_store.increment(f"copilot_priority_{result.priority}_total")
        metrics_store.increment(f"copilot_category_{result.intent.category}_total")
        logger.info(
            "copilot_completed category=%s priority=%s classifier=%s draft_source=%s latency_ms=%d",
            result.intent.category,
            result.priority,
            result.classifier_used,
            result.draft_source,
            latency_ms,
        )
        return CopilotResponse(
            intent=IntentResult(
                category=result.intent.category,
                confidence=result.intent.confidence,
            ),
            priority=result.priority,
            next_actions=result.next_actions,
            draft_reply=result.draft_reply,
            classifier_used=result.classifier_used,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            request_id=request_id,
        )
    except Exception as exc:
        logger.exception("copilot_failed reason=%s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Copilot request failed") from exc
