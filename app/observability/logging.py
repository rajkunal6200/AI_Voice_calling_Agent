"""Structured logging.

Every log line emitted while handling a call carries the same ``call_id``, so
the full chain — SIP answer, caller lookup, each tool invocation, each HubSpot
write, the hangup — is filterable with one query. The binding is done with
context variables rather than passing a logger around, because LiveKit invokes
tool callbacks for us and there is no argument we control to thread through.

Call ``configure_logging()`` once per process (agent worker and API server each
do it at startup), then ``get_logger(__name__)`` anywhere.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

from app.config import settings


def configure_logging() -> None:
    """Idempotently configure structlog and route stdlib logging through it."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        # Renders exc_info into a `exception` key so tracebacks survive JSON.
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    # LiveKit, uvicorn, httpx and SQLAlchemy all log through stdlib. Point the
    # root handler at stderr at the same level so nothing is silently dropped,
    # and emit our own logs through stdlib too so both land on one stream.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
        force=True,
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        # stdlib BoundLogger (not make_filtering_bound_logger) — `add_logger_name`
        # reads `.name` off the underlying logger, which only stdlib loggers have.
        # Level filtering happens in stdlib, configured just above.
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # httpx logs a line per request at INFO; that is noise next to our own
    # structured integration logs, which record method, URL, status and timing.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


@contextmanager
def call_context(**bindings: Any) -> Iterator[None]:
    """Bind values onto every log line emitted inside the block.

    Used at the top of an agent session with ``call_id`` and ``room``, so that
    tool callbacks fired later by LiveKit inherit them without plumbing.
    """
    tokens = structlog.contextvars.bind_contextvars(**bindings)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


def bind_call_context(**bindings: Any) -> None:
    """Bind for the remainder of the current task, without a scope.

    LiveKit's session entrypoint outlives any single ``with`` block, so the
    entrypoint binds and never unbinds — the task ends with the call.
    """
    structlog.contextvars.bind_contextvars(**bindings)
