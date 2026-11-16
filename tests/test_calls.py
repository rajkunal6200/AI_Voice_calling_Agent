from app.services import calls


async def test_start_call_is_idempotent() -> None:
    """LiveKit can re-dispatch a job onto the same room after a worker restart.

    That must resume the existing call, not fail and drop it.
    """
    await calls.start_call(call_id="c1", direction="inbound", from_number="+1555")
    await calls.start_call(call_id="c1", direction="inbound", from_number="+1555")

    assert len(await calls.list_calls()) == 1


async def test_record_event_never_raises_on_a_bad_payload() -> None:
    """A timeline write must never be able to disturb a live call."""
    await calls.start_call(call_id="c1", direction="inbound")

    # Unserialisable value, and an invalid kind that violates the CHECK
    # constraint — both must be swallowed rather than propagated.
    await calls.record_event(call_id="c1", kind="tool_call", payload={"o": object()})
    await calls.record_event(call_id="c1", kind="not_a_valid_kind")  # type: ignore[arg-type]


async def test_events_survive_non_serialisable_payloads() -> None:
    await calls.start_call(call_id="c1", direction="inbound")
    await calls.record_event(call_id="c1", kind="tool_call", payload={"o": object()})

    events = await calls.get_events("c1")
    assert len(events) == 1, "the row should still be written, with the value stringified"


async def test_attach_contact_then_end_call() -> None:
    await calls.start_call(call_id="c1", direction="inbound")
    await calls.attach_contact(call_id="c1", hubspot_contact_id="101")
    await calls.end_call(call_id="c1", status="completed", summary="Did the thing.")

    call = await calls.get_call("c1")
    assert call.hubspot_contact_id == "101"
    assert call.status == "completed"
    assert call.ended_at is not None


async def test_stats_ignore_calls_without_latency() -> None:
    await calls.start_call(call_id="c1", direction="inbound")
    await calls.record_event(call_id="c1", kind="agent_speech", payload={"text": "hi"})

    stats = await calls.stats()
    assert stats["total_calls"] == 1
    assert stats["avg_tool_latency_ms"] == 0
