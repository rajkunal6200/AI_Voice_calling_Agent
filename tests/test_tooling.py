"""The seam between services and the voice runtime.

This is the piece that decides what the caller actually hears when something
goes wrong, so it is tested directly rather than only through the agents.
"""

from __future__ import annotations

import pytest
from livekit.agents.llm import ToolError

from agent.tooling import GENERIC_APOLOGY, current_call_id, instrumented
from app.errors import IntegrationError, NotFoundError
from app.services import calls

CALL_ID = "test-call-1"


@pytest.fixture(autouse=True)
async def a_call_in_progress() -> None:
    current_call_id.set(CALL_ID)
    await calls.start_call(call_id=CALL_ID, direction="inbound", from_number="+1555")


@instrumented
async def succeeds(value: str = "ok") -> dict[str, str]:
    return {"value": value}


@instrumented
async def raises_user_error() -> None:
    raise NotFoundError("no such order", caller_message="I couldn't find that order.")


@instrumented
async def raises_integration_error() -> None:
    raise IntegrationError("HubSpot 503", status=503)


@instrumented
async def raises_unexpected() -> None:
    raise KeyError("some internal bug")


async def test_successful_tool_returns_its_value_unchanged() -> None:
    assert await succeeds(value="hello") == {"value": "hello"}


async def test_success_is_recorded_with_latency() -> None:
    await succeeds()
    events = await calls.get_events(CALL_ID)
    result = next(e for e in events if e["kind"] == "tool_result")
    assert result["payload"]["outcome"] == "ok"
    assert result["latency_ms"] is not None


async def test_user_error_becomes_a_tool_error_the_agent_can_speak() -> None:
    with pytest.raises(ToolError) as excinfo:
        await raises_user_error()
    assert str(excinfo.value) == "I couldn't find that order."


async def test_user_error_is_not_recorded_as_an_incident() -> None:
    """A caller asking for a missing order must not pollute the error feed."""
    with pytest.raises(ToolError):
        await raises_user_error()

    kinds = [e["kind"] for e in await calls.get_events(CALL_ID)]
    assert "error" not in kinds
    assert "tool_result" in kinds


async def test_integration_error_is_recorded_as_an_error() -> None:
    with pytest.raises(ToolError):
        await raises_integration_error()

    errors = [e for e in await calls.get_events(CALL_ID) if e["kind"] == "error"]
    assert len(errors) == 1
    assert errors[0]["payload"]["type"] == "IntegrationError"


async def test_integration_error_never_leaks_upstream_detail_to_the_caller() -> None:
    with pytest.raises(ToolError) as excinfo:
        await raises_integration_error()
    spoken = str(excinfo.value)
    assert "503" not in spoken
    assert "HubSpot" not in spoken


async def test_unexpected_exception_becomes_a_generic_apology() -> None:
    """An unhandled error must not kill the turn and leave dead air."""
    with pytest.raises(ToolError) as excinfo:
        await raises_unexpected()
    assert str(excinfo.value) == GENERIC_APOLOGY


async def test_unexpected_exception_is_recorded_for_debugging() -> None:
    with pytest.raises(ToolError):
        await raises_unexpected()

    errors = [e for e in await calls.get_events(CALL_ID) if e["kind"] == "error"]
    assert errors[0]["payload"]["type"] == "KeyError"


async def test_tool_call_is_recorded_before_the_tool_runs() -> None:
    """Ordering matters: a tool that hangs should still show as attempted."""
    await succeeds(value="x")
    events = await calls.get_events(CALL_ID)
    assert [e["kind"] for e in events][:2] == ["tool_call", "tool_result"]
    assert events[0]["payload"] == {"value": "x"}


async def test_arguments_are_recorded_when_passed_positionally() -> None:
    """LiveKit calls tools positionally, not by keyword.

    Reading kwargs alone recorded an empty payload against every real tool
    call — the timeline showed that check_stock ran but not what was looked
    up, which is most of the point of having a timeline.
    """
    await succeeds("positional-value")

    events = await calls.get_events(CALL_ID)
    assert events[0]["payload"] == {"value": "positional-value"}


async def test_defaulted_arguments_are_recorded_too() -> None:
    """What the tool actually ran with, not just what the model bothered to send."""

    @instrumented
    async def with_default(value: str, limit: int = 5) -> str:
        return value

    await with_default("x")

    events = await calls.get_events(CALL_ID)
    assert events[0]["payload"] == {"value": "x", "limit": 5}


async def test_agent_tools_record_their_arguments(seed_products: None) -> None:
    """End to end through a real agent tool, called positionally."""
    from agent.specialists import SalesAgent
    from agent.state import CallState

    class _Ctx:
        userdata = CallState(call_id=CALL_ID, caller_phone="+1555")

    agent = SalesAgent()
    await agent.check_stock.__wrapped__(agent, _Ctx(), "desk lamp")

    call_row = next(
        e for e in await calls.get_events(CALL_ID) if e["kind"] == "tool_call"
    )
    # `self` and the RunContext must not leak into the timeline.
    assert call_row["payload"] == {"product": "desk lamp"}


async def test_wrapper_preserves_signature_for_schema_generation() -> None:
    """LiveKit builds the tool schema from the signature, so it must survive."""
    import inspect

    assert list(inspect.signature(succeeds).parameters) == ["value"]
    assert succeeds.__name__ == "succeeds"
