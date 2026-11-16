"""The seam between the voice runtime and the services.

Every ``@function_tool`` body is wrapped by ``instrumented``, which is the one
place that:

1. times the tool and writes ``tool_call`` / ``tool_result`` rows to the call
   timeline, so the dashboard can show what the agent did and how slow it was;
2. translates our error taxonomy into LiveKit's ``ToolError``.

Point 2 is the whole reason the services raise domain errors instead of
returning strings. ``ToolError`` hands its message back to the LLM rather than
crashing the session, so the agent hears "I couldn't find an order with that
reference" and says something sensible. An *unhandled* exception, by contrast,
kills the turn and leaves the caller in silence — which is why the catch-all
here is deliberately broad.

The distinction between error classes is preserved on the way out: user-facing
errors log at INFO and are not reported as incidents; integration and unknown
errors log at ERROR, go to Sentry, and get a vaguer caller message so we never
read upstream error text down a phone line.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar

from livekit.agents import RunContext
from livekit.agents.llm import ToolError

from app.errors import AgentError, UserFacingError
from app.observability.logging import get_logger
from app.observability.sentry import capture
from app.services import calls

log = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Set once per session in the entrypoint. A context variable rather than an
# argument because LiveKit invokes tool callbacks for us — there is no
# parameter we control to thread the call id through.
current_call_id: ContextVar[str] = ContextVar("current_call_id", default="")

GENERIC_APOLOGY = (
    "Sorry, I hit a problem doing that. Let me take your details "
    "and have someone follow up."
)


def instrumented(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Time, record, and translate errors for one tool.

    Applied *under* ``@function_tool()`` so LiveKit still sees the original
    signature and docstring (``functools.wraps`` sets ``__wrapped__``, which
    ``inspect.signature`` follows) — those are what become the tool schema the
    model sees, so they must survive wrapping intact.
    """

    # Computed once at decoration time; `inspect.signature` follows __wrapped__.
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        call_id = current_call_id.get()
        tool_name = fn.__name__
        # LiveKit invokes tools with positional arguments, so reading kwargs
        # alone silently records an empty payload. Bind against the real
        # signature to recover every argument by name, then drop `self` and the
        # RunContext, which are noise in a log line.
        loggable_args = _named_arguments(signature, args, kwargs)

        await calls.record_event(
            call_id=call_id, kind="tool_call", name=tool_name, payload=loggable_args
        )
        log.info("tool_call", tool=tool_name, **loggable_args)
        started = time.monotonic()

        try:
            result = await fn(*args, **kwargs)

        except UserFacingError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            # Expected outcome, not an incident: the caller asked for something
            # that does not exist. Do not page anyone for this.
            log.info(
                "tool_user_error",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                reason=exc.message,
                **exc.context,
            )
            await calls.record_event(
                call_id=call_id,
                kind="tool_result",
                name=tool_name,
                payload={"outcome": "user_error", "reason": exc.message},
                latency_ms=elapsed_ms,
            )
            raise ToolError(exc.caller_message) from exc

        except AgentError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            log.error(
                "tool_integration_error",
                tool=tool_name,
                elapsed_ms=elapsed_ms,
                reason=exc.message,
                retryable=exc.retryable,
                **exc.context,
            )
            await calls.record_event(
                call_id=call_id,
                kind="error",
                name=tool_name,
                # The upstream detail goes on the timeline, not to the caller.
                # "rejected: 400" is not diagnosable; the body says which
                # property was wrong and why.
                payload={
                    "type": type(exc).__name__,
                    "reason": exc.message,
                    **{k: str(v)[:400] for k, v in exc.context.items()},
                },
                latency_ms=elapsed_ms,
            )
            capture(exc, tool=tool_name, call_id=call_id)
            raise ToolError(exc.caller_message) from exc

        except Exception as exc:  # noqa: BLE001 — see module docstring
            elapsed_ms = int((time.monotonic() - started) * 1000)
            log.exception("tool_unhandled_error", tool=tool_name, elapsed_ms=elapsed_ms)
            await calls.record_event(
                call_id=call_id,
                kind="error",
                name=tool_name,
                payload={"type": type(exc).__name__, "reason": str(exc)},
                latency_ms=elapsed_ms,
            )
            capture(exc, tool=tool_name, call_id=call_id)
            raise ToolError(GENERIC_APOLOGY) from exc

        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.info("tool_result", tool=tool_name, elapsed_ms=elapsed_ms)
        await calls.record_event(
            call_id=call_id,
            kind="tool_result",
            name=tool_name,
            payload={"outcome": "ok", "result": _summarise(result)},
            latency_ms=elapsed_ms,
        )
        return result

    return wrapper


#: Arguments that carry no information in a timeline row.
_UNLOGGABLE = frozenset({"self", "cls", "context", "ctx"})


def _named_arguments(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Recover every tool argument by name, however it was passed."""
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        # Never let logging break a tool call; the tool itself will raise a
        # clearer error a moment later if the arguments really are wrong.
        return {}
    bound.apply_defaults()
    return {
        name: value
        for name, value in bound.arguments.items()
        if name not in _UNLOGGABLE and not isinstance(value, RunContext)
    }


def _summarise(result: Any) -> Any:
    """Keep timeline payloads small and JSON-safe."""
    if isinstance(result, str):
        return result[:500]
    if isinstance(result, dict | list):
        return result
    return str(result)[:500]
