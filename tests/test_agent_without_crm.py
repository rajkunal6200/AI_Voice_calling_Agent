"""The agents must be useful before HubSpot exists.

Getting a CRM account is the slowest part of standing this up, so running
without one is a first-class configuration, not a degraded edge case.
"""

from __future__ import annotations

import pytest

from agent.specialists import (
    OrdersAgent,
    PolicyAgent,
    ReceptionAgent,
    SalesAgent,
    SchedulingAgent,
)
from agent.state import CallState
from agent.tooling import current_call_id
from app.services import calls

CALL_ID = "no-crm-call"


@pytest.fixture
async def state() -> CallState:
    current_call_id.set(CALL_ID)
    await calls.start_call(
        call_id=CALL_ID, direction="inbound", from_number="+15551234567"
    )
    return CallState(call_id=CALL_ID, caller_phone="+15551234567", hubspot=None)


class _Ctx:
    """Stands in for RunContext, of which the tools only read userdata."""

    def __init__(self, state: CallState) -> None:
        self.userdata = state


async def test_every_agent_constructs_without_a_crm(state: CallState) -> None:
    assert ReceptionAgent(state=state) is not None
    for cls in (OrdersAgent, SalesAgent, SchedulingAgent, PolicyAgent):
        assert cls() is not None


async def test_stock_lookup_still_works(state: CallState, seed_products: None) -> None:
    agent = SalesAgent()
    result = await agent.check_stock.__wrapped__(agent, _Ctx(state), "desk lamp")
    assert result["in_stock"] is True


async def test_order_lookup_still_works(state: CallState, seed_orders: None) -> None:
    agent = OrdersAgent()
    result = await agent.check_order_status.__wrapped__(agent, _Ctx(state), "ORD-1")
    assert result["status"] == "shipped"


async def test_booking_still_works(state: CallState) -> None:
    agent = SchedulingAgent()
    ctx = _Ctx(state)
    offered = await agent.offer_appointment_slots.__wrapped__(agent, ctx)
    slot_id = offered["slots"][0]["slot_id"]

    booked = await agent.book_appointment.__wrapped__(agent, ctx, slot_id)

    assert booked["booked"] is True


async def test_saving_details_reports_honestly_that_no_crm_was_written(
    state: CallState,
) -> None:
    """The model must not tell the caller they were added to a system."""
    agent = SalesAgent()
    result = await agent.save_caller_details.__wrapped__(
        agent, _Ctx(state), "Ada", "", "ada@example.com"
    )
    assert result["saved_to_crm"] is False
    assert "no CRM" in result["reason"]


async def test_caller_details_are_kept_on_the_call_record(state: CallState) -> None:
    agent = SalesAgent()
    await agent.save_caller_details.__wrapped__(agent, _Ctx(state), "Ada")

    noted = [e for e in await calls.get_events(CALL_ID) if e["name"] == "caller_details"]
    assert noted and noted[0]["payload"]["first_name"] == "Ada"


async def test_summary_is_written_locally_by_any_specialist(state: CallState) -> None:
    """log_call_summary lives on the base class, so whichever agent is
    speaking when the call ends can close it out."""
    agent = PolicyAgent()
    result = await agent.log_call_summary.__wrapped__(
        agent, _Ctx(state), "Checked the returns policy."
    )

    assert result["logged_to_crm"] is False
    call = await calls.get_call(CALL_ID)
    assert call.summary == "Checked the returns policy."
    assert call.status == "completed"
    assert state.summary_logged is True


# --- creating a client the CRM has never seen -------------------------------


async def test_a_new_caller_becomes_a_contact(state: CallState) -> None:
    """The only way a client is added during a call. There is no customer
    table in Postgres — a client *is* a HubSpot contact, and orders hang off
    the phone number rather than off a local record."""
    from unittest.mock import AsyncMock

    from agent.specialists import SalesAgent
    from app.services.hubspot import Contact

    state.hubspot = AsyncMock()
    state.contact = None
    state.hubspot.create_contact.return_value = Contact(
        id="new-1", first_name="Ada", last_name="Okonkwo", email=None,
        phone=state.caller_phone,
    )

    agent = SalesAgent()
    result = await agent.save_caller_details.__wrapped__(
        agent, _Ctx(state), "Ada", "Okonkwo"
    )

    assert result["created_new_record"] is True
    assert state.contact is not None and state.contact.id == "new-1"


async def test_a_new_caller_can_give_a_number_other_than_the_one_that_rang(
    state: CallState,
) -> None:
    """Someone ringing from a withheld or office line who gives their mobile.

    The dictated number used to be dropped on this path — the argument existed
    but only the update branch read it — so the contact was created against
    whatever rang, and the number they actually gave was lost.
    """
    from unittest.mock import AsyncMock

    from agent.specialists import SalesAgent
    from app.services.hubspot import Contact

    state.hubspot = AsyncMock()
    state.contact = None
    state.hubspot.create_contact.return_value = Contact(
        id="new-2", first_name="Ada", last_name=None, email=None,
        phone="+998998886810",
    )

    agent = SalesAgent()
    await agent.save_caller_details.__wrapped__(
        agent, _Ctx(state), "Ada", "", "", "", "+998 99 888 68 10"
    )

    assert state.hubspot.create_contact.await_args.kwargs["phone"] == "+998998886810"
    # Subsequent order lookups in this call must use the number they gave.
    assert state.caller_phone == "+998998886810"


async def test_an_unusable_dictated_number_does_not_replace_the_one_that_rang(
    state: CallState,
) -> None:
    """Better to keep a number that works than to save a misheard one."""
    from unittest.mock import AsyncMock

    from agent.specialists import SalesAgent
    from app.services.hubspot import Contact

    rang = state.caller_phone
    state.hubspot = AsyncMock()
    state.contact = None
    state.hubspot.create_contact.return_value = Contact(
        id="new-3", first_name="Ada", last_name=None, email=None, phone=rang
    )

    agent = SalesAgent()
    result = await agent.save_caller_details.__wrapped__(
        agent, _Ctx(state), "Ada", "", "", "", "double four"
    )

    assert state.hubspot.create_contact.await_args.kwargs["phone"] == rang
    assert "not sound complete" in result["note"]
