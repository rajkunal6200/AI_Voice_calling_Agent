"""Call history — the observable record of what the agent did.

HubSpot records the business outcome (a note on the contact, a booked meeting).
This records the *conversation*: every utterance, every tool invocation and its
latency, every error. That is what the dashboard reads and what makes a failed
call debuggable after the caller has hung up.

Writes here must never break a call in progress. ``record_event`` swallows its
own failures for that reason — losing a timeline row is regrettable, dropping
the call because the timeline write failed is not acceptable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from app.db import execute, fetch_all, fetch_one
from app.errors import NotFoundError
from app.observability.logging import get_logger

log = get_logger(__name__)

EventKind = Literal["user_speech", "agent_speech", "tool_call", "tool_result", "error"]
CallStatus = Literal["in_progress", "completed", "failed"]


@dataclass(frozen=True)
class CallSummary:
    call_id: str
    direction: str
    from_number: str | None
    to_number: str | None
    hubspot_contact_id: str | None
    caller_name: str | None
    status: str
    summary: str | None
    started_at: datetime
    ended_at: datetime | None


async def start_call(
    *,
    call_id: str,
    direction: Literal["inbound", "outbound"],
    from_number: str | None = None,
    to_number: str | None = None,
) -> None:
    # ON CONFLICT because LiveKit can re-dispatch a job onto the same room
    # after a worker restart; that should resume, not fail.
    await execute(
        """
        INSERT INTO calls (call_id, direction, from_number, to_number)
        VALUES (:call_id, :direction, :from_number, :to_number)
        ON CONFLICT (call_id) DO NOTHING
        """,
        call_id=call_id,
        direction=direction,
        from_number=from_number,
        to_number=to_number,
    )
    log.info("call_started", direction=direction, from_number=from_number)


async def attach_contact(
    *, call_id: str, hubspot_contact_id: str, caller_name: str | None = None
) -> None:
    """Record who the caller turned out to be.

    The name is stored, not looked up later. It is a snapshot of who they were
    on this call — resolving it live would cost a CRM round trip per dashboard
    row, and would rewrite history every time somebody edits a contact.
    """
    await execute(
        "UPDATE calls SET hubspot_contact_id = :contact_id, "
        "caller_name = COALESCE(:caller_name, caller_name) WHERE call_id = :call_id",
        call_id=call_id,
        contact_id=hubspot_contact_id,
        caller_name=caller_name,
    )


async def record_event(
    *,
    call_id: str,
    kind: EventKind,
    name: str | None = None,
    payload: dict[str, Any] | None = None,
    latency_ms: int | None = None,
) -> None:
    """Append to the call timeline. Never raises."""
    try:
        await execute(
            """
            INSERT INTO call_events (call_id, kind, name, payload, latency_ms)
            VALUES (:call_id, :kind, :name, CAST(:payload AS jsonb), :latency_ms)
            """,
            call_id=call_id,
            kind=kind,
            name=name,
            payload=json.dumps(payload or {}, default=str),
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 — see module docstring
        log.warning("call_event_write_failed", kind=kind, name=name, error=str(exc))


async def end_call(
    *,
    call_id: str,
    status: CallStatus = "completed",
    summary: str | None = None,
) -> None:
    await execute(
        """
        UPDATE calls
        SET status = :status, summary = :summary, ended_at = :ended_at
        WHERE call_id = :call_id
        """,
        call_id=call_id,
        status=status,
        summary=summary,
        ended_at=datetime.now(UTC),
    )
    log.info("call_ended", status=status)


# --- dashboard reads -------------------------------------------------------


async def list_calls(*, limit: int = 50, offset: int = 0) -> list[CallSummary]:
    rows = await fetch_all(
        """
        SELECT call_id, direction, from_number, to_number, hubspot_contact_id,
               caller_name, status, summary, started_at, ended_at
        FROM calls
        ORDER BY started_at DESC
        LIMIT :limit OFFSET :offset
        """,
        limit=limit,
        offset=offset,
    )
    return [CallSummary(**row) for row in rows]


async def get_call(call_id: str) -> CallSummary:
    row = await fetch_one(
        """
        SELECT call_id, direction, from_number, to_number, hubspot_contact_id,
               caller_name, status, summary, started_at, ended_at
        FROM calls WHERE call_id = :call_id
        """,
        call_id=call_id,
    )
    if row is None:
        raise NotFoundError(f"no call {call_id!r}", call_id=call_id)
    return CallSummary(**row)


async def get_events(call_id: str) -> list[dict[str, Any]]:
    return await fetch_all(
        """
        SELECT id, kind, name, payload, latency_ms, created_at
        FROM call_events
        WHERE call_id = :call_id
        ORDER BY id
        """,
        call_id=call_id,
    )


async def recent_errors(*, limit: int = 50) -> list[dict[str, Any]]:
    """Error feed for the dashboard, newest first."""
    return await fetch_all(
        """
        SELECT id, call_id, name, payload, created_at
        FROM call_events
        WHERE kind = 'error'
        ORDER BY id DESC
        LIMIT :limit
        """,
        limit=limit,
    )


async def stats() -> dict[str, Any]:
    """Headline numbers for the dashboard."""
    row = await fetch_one(
        """
        SELECT
            COUNT(*)                                              AS total_calls,
            COUNT(*) FILTER (WHERE status = 'in_progress')        AS active_calls,
            COUNT(*) FILTER (WHERE status = 'failed')             AS failed_calls,
            COUNT(*) FILTER (WHERE started_at > now() - interval '24 hours')
                                                                  AS calls_24h
        FROM calls
        """
    )
    errors = await fetch_one(
        """
        SELECT COUNT(*) AS errors_24h
        FROM call_events
        WHERE kind = 'error' AND created_at > now() - interval '24 hours'
        """
    )
    # Turn latency is what a caller experiences: how long the silence lasts
    # between them finishing and the agent starting to speak. Tool latency is
    # kept because it is genuinely useful for spotting a slow integration, but
    # it must not be the headline — it runs 4-7ms while turns run in seconds,
    # so leading with it says the system is fast when the caller is waiting.
    turn_latency = await fetch_one(
        """
        SELECT
            COALESCE(ROUND(AVG(latency_ms))::int, 0) AS avg_turn_latency_ms,
            COALESCE(
                percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms), 0
            )::int AS p95_turn_latency_ms
        FROM call_events
        WHERE kind = 'agent_speech' AND latency_ms IS NOT NULL
          AND created_at > now() - interval '24 hours'
        """
    )
    tool_latency = await fetch_one(
        """
        SELECT
            COALESCE(ROUND(AVG(latency_ms))::int, 0) AS avg_tool_latency_ms,
            COALESCE(
                percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms), 0
            )::int AS p95_tool_latency_ms
        FROM call_events
        WHERE kind = 'tool_result' AND latency_ms IS NOT NULL
          AND created_at > now() - interval '24 hours'
        """
    )
    return {
        **(row or {}),
        **(errors or {}),
        **(turn_latency or {}),
        **(tool_latency or {}),
    }


async def outcomes(call_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """What each call produced — orders placed, opportunities opened, bookings.

    The dashboard showed how a call *went* (status, duration) but never what it
    was *for*. A call that ends "completed" having placed a £600 order and one
    that ends "completed" having answered a question about opening hours are
    not the same event, and the timeline already knows the difference.
    """
    if not call_ids:
        return {}
    rows = await fetch_all(
        """
        SELECT call_id, name, payload
        FROM call_events
        WHERE kind = 'tool_result'
          AND name IN ('order_placed', 'purchase_interest', 'appointment_booked')
          AND call_id = ANY(:call_ids)
        ORDER BY id
        """,
        call_ids=call_ids,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        grouped.setdefault(row["call_id"], []).append(
            {"kind": row["name"], **(payload or {})}
        )
    return grouped
