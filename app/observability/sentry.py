"""Error reporting.

Sentry is optional — with no DSN set, ``init_error_reporting()`` is a no-op and
errors still reach the structured logs. That keeps local development free of a
required external account while giving production real error tracking.

``UserFacingError`` is filtered out before send: "caller asked about an order
that doesn't exist" is normal call traffic, not an incident, and letting it
through would bury genuine integration failures.
"""

from __future__ import annotations

from typing import Any

import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.types import Event, Hint

from app.config import settings
from app.errors import UserFacingError
from app.observability.logging import get_logger

log = get_logger(__name__)


def _before_send(event: Event, hint: Hint) -> Event | None:
    exc_info = hint.get("exc_info")
    if exc_info and isinstance(exc_info[1], UserFacingError):
        return None
    return event


def init_error_reporting() -> None:
    if not settings.sentry_dsn:
        log.info("sentry_disabled", reason="no SENTRY_DSN configured")
        return

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        integrations=[AsyncioIntegration(), FastApiIntegration()],
        before_send=_before_send,
        # Transcripts and caller details flow through this process; never let
        # request bodies or local variables be shipped off-box by default.
        send_default_pii=False,
        traces_sample_rate=0.1,
    )
    log.info("sentry_enabled", environment=settings.environment)


def capture(exc: BaseException, **context: Any) -> None:
    """Report an exception with extra structured context attached."""
    if not settings.sentry_dsn:
        return
    with sentry_sdk.new_scope() as scope:
        for key, value in context.items():
            scope.set_extra(key, value)
        sentry_sdk.capture_exception(exc)
