from __future__ import annotations

import logging
import os
from contextvars import ContextVar, Token
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    app_name: str = Field(default="inboxpilot-lite")
    log_level: str = Field(default="INFO")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1")
    lmstudio_api_key: str = Field(default="lm-studio")
    lmstudio_model: str = Field(default="openai/gpt-oss-20b")
    lmstudio_timeout_seconds: float = Field(default=20.0)
    rate_limit_requests: int = Field(default=60)
    rate_limit_window_seconds: int = Field(default=60)
    max_batch_size: int = Field(default=20)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("APP_NAME", "inboxpilot-lite"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            lmstudio_api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            lmstudio_model=os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b"),
            lmstudio_timeout_seconds=float(os.getenv("LMSTUDIO_TIMEOUT_SECONDS", "20")),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "60")),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "20")),
        )


request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return request_id_ctx.get()


def set_request_id(request_id: str) -> Token[str]:
    return request_id_ctx.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    request_id_ctx.reset(token)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def configure_logging(level: str | None = None) -> None:
    desired_level = (level or get_settings().log_level).upper()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s request_id=%(request_id)s %(name)s %(message)s"
        )
    )
    handler.addFilter(RequestIdFilter())
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, desired_level, logging.INFO))
