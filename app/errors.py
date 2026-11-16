"""Error taxonomy.

The distinction that matters on a live phone call is *what the agent should say
next*, so the hierarchy is organised around that rather than around which
subsystem failed:

- ``UserFacingError``  — the request was understood but cannot be satisfied
  (no such order, no availability). The agent says ``caller_message`` and
  carries on. Not an incident; logged at INFO.
- ``IntegrationError`` — an upstream dependency misbehaved (timeout, 5xx).
  Worth retrying, and worth alerting on. The agent apologises vaguely and
  offers a fallback. Logged at ERROR and reported to Sentry.
- ``ConfigurationError`` — the process is misconfigured (missing token). There
  is no recovery at call time, so we fail fast at startup instead.

Services raise these. The agent's tool layer is the only place that translates
them into LiveKit's ``ToolError``, which keeps the services importable and
testable without the voice runtime.
"""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base class for every error the agent layer knows how to handle."""

    #: Safe to speak aloud to the caller. Never include identifiers, upstream
    #: error text, or anything else we would not want read out on a phone line.
    caller_message: str = "Sorry, something went wrong on my end."
    #: Whether an immediate retry could plausibly succeed.
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        caller_message: str | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        if caller_message is not None:
            self.caller_message = caller_message
        #: Structured fields merged into the log entry for this error.
        self.context = context


class UserFacingError(AgentError):
    """The caller asked for something we understood but cannot provide."""

    caller_message = "I couldn't find that, but I can try something else."


class NotFoundError(UserFacingError):
    caller_message = "I couldn't find a record matching those details."


class ValidationError(UserFacingError):
    caller_message = "That doesn't look quite right — could you repeat it?"


class IntegrationError(AgentError):
    """An upstream system failed in a way that is not the caller's fault."""

    caller_message = (
        "I'm having trouble reaching our system right now. "
        "Let me take your details and have someone follow up."
    )
    retryable = True


class RateLimitedError(IntegrationError):
    """Upstream asked us to slow down. Retryable after a delay."""

    def __init__(self, message: str, *, retry_after: float | None = None, **context: Any):
        super().__init__(message, **context)
        self.retry_after = retry_after


class ConfigurationError(AgentError):
    """Missing or invalid configuration. Raised at startup, never mid-call."""

    retryable = False
