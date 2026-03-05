from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    app_name: str = Field(default="inboxpilot-lite")
    log_level: str = Field(default="INFO")
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    openai_timeout_seconds: float = Field(default=8.0)
    openai_max_retries: int = Field(default=2)
    openai_retry_backoff_seconds: float = Field(default=0.4)

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY")
        return cls(
            app_name=os.getenv("APP_NAME", "inboxpilot-lite"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            openai_api_key=api_key if api_key else None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "8")),
            openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
            openai_retry_backoff_seconds=float(
                os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "0.4")
            ),
        )


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
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
