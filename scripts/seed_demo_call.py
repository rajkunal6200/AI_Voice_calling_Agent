"""Write a realistic finished call to the database.

Lets you develop the dashboard and demo the project without a phone line or
LiveKit credentials. Run: ``uv run python -m scripts.seed_demo_call``
"""

from __future__ import annotations

import asyncio
import uuid

from app.observability.logging import configure_logging, get_logger
from app.services import calls

log = get_logger(__name__)


async def seed_successful_call() -> str:
    call_id = f"demo-{uuid.uuid4().hex[:8]}"
    await calls.start_call(
        call_id=call_id, direction="inbound", from_number="+15551234567"
    )
    await calls.attach_contact(call_id=call_id, hubspot_contact_id="101")

    timeline: list[tuple[str, str | None, dict[str, object], int | None]] = [
        ("agent_speech", None, {"text": "Hi Ada, thanks for calling. How can I help?"}, None),
        ("user_speech", None, {"text": "I'm chasing an order, I don't have the number."}, None),
        ("tool_call", "find_my_recent_orders", {}, None),
        (
            "tool_result",
            "find_my_recent_orders",
            {"outcome": "ok", "result": {"orders": [{"order_reference": "ORD-5001"}]}},
            88,
        ),
        (
            "agent_speech",
            None,
            {"text": "I found order ORD-5001. It shipped and should arrive Wednesday."},
            None,
        ),
        ("user_speech", None, {"text": "Great. Can I also book someone to assemble it?"}, None),
        ("tool_call", "offer_appointment_slots", {}, None),
        ("tool_result", "offer_appointment_slots", {"outcome": "ok"}, 41),
        ("tool_call", "book_appointment", {"slot_id": 2}, None),
        ("tool_result", "book_appointment", {"outcome": "ok", "result": {"booked": True}}, 63),
        ("tool_call", "log_call_summary", {"summary": "..."}, None),
        ("tool_result", "log_call_summary", {"outcome": "ok"}, 402),
    ]
    for kind, name, payload, latency in timeline:
        await calls.record_event(
            call_id=call_id,
            kind=kind,  # type: ignore[arg-type]
            name=name,
            payload=payload,
            latency_ms=latency,
        )

    await calls.end_call(
        call_id=call_id,
        status="completed",
        summary=(
            "Ada chased order ORD-5001, which has shipped. "
            "Booked an assembly appointment for Tuesday at 9:30am."
        ),
    )
    return call_id


async def seed_failed_call() -> str:
    """A call where the CRM was down — the error feed needs something in it."""
    call_id = f"demo-{uuid.uuid4().hex[:8]}"
    await calls.start_call(
        call_id=call_id, direction="outbound", to_number="+15559876543"
    )
    await calls.record_event(
        call_id=call_id,
        kind="agent_speech",
        payload={"text": "Hi, I'm calling from Northwind Office about your enquiry."},
    )
    await calls.record_event(
        call_id=call_id,
        kind="tool_call",
        name="record_qualification",
        payload={"qualified": True},
    )
    await calls.record_event(
        call_id=call_id,
        kind="error",
        name="record_qualification",
        payload={"type": "IntegrationError", "reason": "HubSpot returned 503"},
        latency_ms=5031,
    )
    await calls.end_call(
        call_id=call_id,
        status="failed",
        summary="Lead qualified but the CRM write failed; needs manual entry.",
    )
    return call_id


async def main() -> None:
    configure_logging()
    ok = await seed_successful_call()
    failed = await seed_failed_call()
    log.info("demo_calls_seeded", completed=ok, failed=failed)


if __name__ == "__main__":
    asyncio.run(main())
