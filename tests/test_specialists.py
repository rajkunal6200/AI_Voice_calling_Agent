"""The specialist split and the handoffs between them.

The split exists to make tool selection easier for a small model, so the
property worth pinning is the *distribution* of tools, not just that each one
works. A change that quietly moves every tool back onto one agent should fail
here.
"""

from __future__ import annotations

import pytest
from livekit.agents.llm import utils as llm_utils

from agent.base import BaseAgent
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
from app.services.hubspot import Contact

CALL_ID = "specialist-call"

SPECIALISTS = (OrdersAgent, SalesAgent, SchedulingAgent, PolicyAgent)


@pytest.fixture
async def state() -> CallState:
    current_call_id.set(CALL_ID)
    await calls.start_call(call_id=CALL_ID, direction="inbound", from_number="+1555")
    return CallState(call_id=CALL_ID, caller_phone="+1555")


class _Ctx:
    def __init__(self, state: CallState) -> None:
        self.userdata = state


#: On every agent, so they are not part of the routing decision the split
#: exists to simplify.
SHARED_TOOLS = {
    "return_to_reception",
    "log_call_summary",
    "request_human_callback",
    # The caller's own details belong to whoever is speaking, not to sales.
    "save_caller_details",
}


def tool_names(agent: BaseAgent) -> set[str]:
    return {
        llm_utils.build_legacy_openai_schema(t)["function"]["name"] for t in agent.tools
    }


def domain_tools(agent: BaseAgent) -> set[str]:
    """The tools that make this agent a specialist, excluding shared ones."""
    return tool_names(agent) - SHARED_TOOLS


# --- the split -------------------------------------------------------------


@pytest.mark.parametrize("cls", SPECIALISTS)
def test_no_specialist_carries_more_than_four_domain_tools(cls: type) -> None:
    """The whole point of the split.

    Tool-selection accuracy falls off as the menu grows, fastest on the small
    models this targets. One agent previously carried eight. Shared tools are
    excluded: they are identical everywhere and are not the choice the model
    struggles with.
    """
    assert len(domain_tools(cls())) <= 4


@pytest.mark.parametrize("cls", SPECIALISTS)
def test_total_tool_count_stays_manageable(cls: type) -> None:
    """Shared tools still enlarge the menu, so cap the total too.

    Seven, not six: the shared set grew to four when making a capability
    reachable from only one agent turned out to be worse than a slightly
    longer menu — a caller wanting their name corrected got passed around
    instead of helped. Domain tools are still capped at four above, and those
    are the choice the model actually struggles with.

    Eight, not seven: sales gained place_order, because opening an opportunity
    and committing to a sale are different decisions and the caller makes both
    while talking to sales. Putting it on another agent would have made it
    unreachable from the conversation that leads to it, which is the failure
    this codebase has already had twice.

    Note the domain cap above is now *met* by SalesAgent, not merely passed.
    The next sales tool does not get to raise this number — it forces either a
    genuine split of the sales role or a merge of two tools into one.
    """
    assert len(tool_names(cls())) <= 8


def test_reception_only_routes(state: CallState) -> None:
    """Reception must not answer anything itself — it has no data tools."""
    assert domain_tools(ReceptionAgent(state=state)) == {
        "to_orders",
        "to_sales",
        "to_scheduling",
        "to_policies",
    }


def test_reception_cannot_hand_off_to_itself(state: CallState) -> None:
    """An easy loop for a small model to fall into, so the tool is removed."""
    assert "return_to_reception" not in tool_names(ReceptionAgent(state=state))


@pytest.mark.parametrize("cls", SPECIALISTS)
def test_every_specialist_can_hand_back(cls: type) -> None:
    """Otherwise a caller who changes subject is stuck with the wrong agent."""
    assert "return_to_reception" in tool_names(cls())


@pytest.mark.parametrize("cls", SPECIALISTS)
def test_every_agent_can_close_the_call(cls: type) -> None:
    """Calls end wherever they end, not necessarily back at reception."""
    assert "log_call_summary" in tool_names(cls())


def test_data_tools_are_not_duplicated_across_specialists(state: CallState) -> None:
    """Two agents offering check_stock would put the routing decision back
    inside the tool menu, which is what the split removes."""
    seen: set[str] = set()
    for cls in SPECIALISTS:
        names = domain_tools(cls())
        assert not (names & seen), f"{cls.__name__} duplicates {names & seen}"
        seen |= names


# --- handoff mechanics -----------------------------------------------------


async def test_reception_hands_off_to_the_right_specialist(state: CallState) -> None:
    reception = ReceptionAgent(state=state)
    ctx = _Ctx(state)

    for tool, expected in [
        (reception.to_orders, OrdersAgent),
        (reception.to_sales, SalesAgent),
        (reception.to_scheduling, SchedulingAgent),
        (reception.to_policies, PolicyAgent),
    ]:
        agent, message = await tool.__wrapped__(reception, ctx)
        assert isinstance(agent, expected)
        assert isinstance(message, str) and message


async def test_specialist_hands_back_to_reception(state: CallState) -> None:
    orders = OrdersAgent()
    agent, _ = await orders.return_to_reception.__wrapped__(orders, _Ctx(state))
    assert isinstance(agent, ReceptionAgent)


async def test_handoff_is_recorded_on_the_timeline(state: CallState) -> None:
    """The dashboard should show which specialist handled which part."""
    reception = ReceptionAgent(state=state)
    await reception.to_orders.__wrapped__(reception, _Ctx(state))

    events = [e for e in await calls.get_events(CALL_ID) if e["name"] == "to_orders"]
    assert events, "the routing decision should appear in the timeline"


# --- state survives handoff ------------------------------------------------


async def test_offered_slots_survive_a_handoff(state: CallState) -> None:
    """The reason state lives on the session rather than the agent.

    Scheduling reads out slots, the caller detours to another agent and comes
    back, then picks one. Held on the agent instance, the mapping would be gone
    and the booking would fail or land on an invented time.
    """
    scheduling = SchedulingAgent()
    ctx = _Ctx(state)
    offered = await scheduling.offer_appointment_slots.__wrapped__(scheduling, ctx)
    slot_id = offered["slots"][0]["slot_id"]

    # Caller wanders off to policies and back; a fresh agent instance results.
    fresh = SchedulingAgent()
    booked = await fresh.book_appointment.__wrapped__(fresh, ctx, slot_id)

    assert booked["booked"] is True


async def test_booking_a_slot_that_was_never_offered_is_rejected(
    state: CallState,
) -> None:
    scheduling = SchedulingAgent()
    from livekit.agents.llm import ToolError

    with pytest.raises(ToolError):
        await scheduling.book_appointment.__wrapped__(scheduling, _Ctx(state), 99)


async def test_contact_saved_by_sales_is_visible_to_later_agents(
    state: CallState,
) -> None:
    """State mutations must be on the shared object, not a copy."""
    assert state.contact is None
    state.contact = None  # explicit starting point

    # Simulate what save_caller_details does with a CRM attached.
    state.contact = Contact(
        id="101", first_name="Ada", last_name=None, email=None, phone="+1555"
    )

    scheduling = SchedulingAgent()
    offered = await scheduling.offer_appointment_slots.__wrapped__(
        scheduling, _Ctx(state)
    )
    booked = await scheduling.book_appointment.__wrapped__(
        scheduling, _Ctx(state), offered["slots"][0]["slot_id"]
    )
    assert booked["booked"] is True


# --- CRM actions -----------------------------------------------------------


@pytest.mark.parametrize("cls", SPECIALISTS)
def test_every_agent_can_escalate_to_a_human(cls: type) -> None:
    """A caller needing something no tool covers must not hit a dead end."""
    assert "request_human_callback" in tool_names(cls())


async def test_callback_request_is_recorded_without_a_crm(state: CallState) -> None:
    """The request must survive HubSpot being absent or down."""
    agent = PolicyAgent()
    result = await agent.request_human_callback.__wrapped__(
        agent, _Ctx(state), "Wants a bespoke quote", True
    )

    assert result["created_in_crm"] is False
    events = await calls.get_events(CALL_ID)
    noted = [e for e in events if e["name"] == "callback_requested"]
    assert noted and noted[0]["payload"]["urgent"] is True


async def test_purchase_interest_needs_a_contact_to_attach_to(
    state: CallState, seed_products: None
) -> None:
    """An unattached deal is invisible on anyone's record, so refuse early
    and tell the model to take a name first.

    A CRM is attached here — without one the earlier no-CRM branch answers
    instead, which is a different case.
    """
    state.hubspot = object()  # type: ignore[assignment]  # never called on this path
    state.contact = None

    agent = SalesAgent()
    result = await agent.record_purchase_interest.__wrapped__(
        agent, _Ctx(state), "desk lamp", 2
    )

    assert result["opportunity_created"] is False
    assert "save_caller_details" in result["reason"]


async def test_purchase_interest_prices_from_the_erp_not_the_model(
    state: CallState, seed_products: None
) -> None:
    """The amount must come from the catalogue, not from whatever figure the
    model remembers reading out a moment earlier."""
    agent = SalesAgent()
    await agent.record_purchase_interest.__wrapped__(agent, _Ctx(state), "desk lamp", 3)

    events = await calls.get_events(CALL_ID)
    interest = next(e for e in events if e["name"] == "purchase_interest")
    # seed_products prices the lamp at 12900 cents.
    assert interest["payload"]["amount_cents"] == 12900 * 3
    assert interest["payload"]["quantity"] == 3


async def test_purchase_interest_rejects_an_unknown_product(
    state: CallState, seed_products: None
) -> None:
    from livekit.agents.llm import ToolError

    agent = SalesAgent()
    with pytest.raises(ToolError):
        await agent.record_purchase_interest.__wrapped__(agent, _Ctx(state), "hovercraft")


# --- reachability ----------------------------------------------------------


@pytest.mark.parametrize("cls", SPECIALISTS)
def test_every_agent_can_take_the_callers_details(cls: type) -> None:
    """Regression: this lived only on SalesAgent.

    A caller on orders who said "actually, my name is spelled wrong" could not
    be helped by the agent talking to them. Rather than routing, the model
    reached for request_human_callback — so a name change became a promised
    callback that should never have been needed.
    """
    assert "save_caller_details" in tool_names(cls())


def test_reception_can_take_details_without_routing(state: CallState) -> None:
    """Otherwise a name change costs a handoff before anything happens."""
    assert "save_caller_details" in tool_names(ReceptionAgent(state=state))


@pytest.mark.parametrize("cls", SPECIALISTS)
def test_escalation_is_described_as_a_last_resort(cls: type) -> None:
    """The tool description is the prompt the model reads when choosing.

    It was reached for whenever an agent lacked a capability, which is what
    made simple requests dead-end.
    """
    # Collapse whitespace: the docstring wraps, so a literal substring check
    # fails on the line break rather than on the meaning.
    doc = " ".join((cls().request_human_callback.__wrapped__.__doc__ or "").split()).lower()
    assert "last resort" in doc
    assert "never use it for something another tool does" in doc


# --- loop protection -------------------------------------------------------


async def test_handoffs_are_refused_once_they_start_looping(
    state: CallState,
) -> None:
    """Reproduces a real stuck call.

    The caller asked "what furniture do you sell?". Sales could not answer it,
    handed back; reception correctly routes product questions to sales; round
    it went, forever, with the caller listening to nothing.
    """
    from agent.base import MAX_CONSECUTIVE_HANDOFFS

    reception = ReceptionAgent(state=state)
    ctx = _Ctx(state)

    state.handoffs_since_user_turn = MAX_CONSECUTIVE_HANDOFFS
    result = await reception.to_sales.__wrapped__(reception, ctx)

    assert isinstance(result, dict), "should refuse rather than hand off"
    assert result["handoff_refused"] is True


async def test_specialist_also_refuses_to_bounce_back(state: CallState) -> None:
    from agent.base import MAX_CONSECUTIVE_HANDOFFS

    state.handoffs_since_user_turn = MAX_CONSECUTIVE_HANDOFFS
    orders = OrdersAgent()

    result = await orders.return_to_reception.__wrapped__(orders, _Ctx(state))

    assert isinstance(result, dict)
    assert result["handoff_refused"] is True


async def test_a_single_handoff_is_still_allowed(state: CallState) -> None:
    """The guard must not break ordinary routing."""
    state.handoffs_since_user_turn = 0
    reception = ReceptionAgent(state=state)

    agent, _ = await reception.to_sales.__wrapped__(reception, _Ctx(state))

    assert isinstance(agent, SalesAgent)


def test_refusal_tells_the_model_what_to_do_instead(state: CallState) -> None:
    """A bare refusal would leave it stuck a different way."""
    from agent.base import LOOP_BROKEN

    reason = LOOP_BROKEN["reason"].lower()
    assert "do not hand off again" in reason
    assert "call them back" in reason


# --- browsing --------------------------------------------------------------


async def test_sales_can_answer_what_do_you_sell(
    state: CallState, seed_products: None
) -> None:
    """The capability whose absence caused the loop.

    check_stock needs a named product; a caller asking what we sell has not
    named one, so there was no tool that could answer at all.
    """
    agent = SalesAgent()
    result = await agent.list_products.__wrapped__(agent, _Ctx(state))

    names = [p["name"] for p in result["products"]]
    assert "Aeron Desk Lamp" in names


async def test_catalogue_puts_available_items_first(
    state: CallState, seed_products: None
) -> None:
    """Read aloud, the first few are all most callers will hear."""
    agent = SalesAgent()
    products = (await agent.list_products.__wrapped__(agent, _Ctx(state)))["products"]

    in_stock = [i for i, p in enumerate(products) if p["in_stock"]]
    out = [i for i, p in enumerate(products) if not p["in_stock"]]
    assert not out or not in_stock or max(in_stock) < min(out)


# --- changing the number that links the two systems ------------------------


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("+998 99 888 68 10", "+998998886810"),
        ("plus nine nine eight 998886810", "+998998886810"),
        ("+1 202 555 0100", "+12025550100"),
        ("plus four four two zero seven nine four six zero nine five eight",
         "+442079460958"),
    ],
)
def test_dictated_phone_numbers_are_repaired(spoken: str, expected: str) -> None:
    from app.services.hubspot import normalise_spoken_phone

    assert normalise_spoken_phone(spoken) == expected


@pytest.mark.parametrize(
    "junk",
    [
        "0100",
        "my name is Ada",
        "",
        "12",
        # Nine digits, so any length rule accepts it — but +999 is not an
        # assigned country code. This exact string came off a real call, was
        # written to the CRM, and took the caller's orders with it.
        "+999899999",
        # Right country code, wrong length for that country.
        "+9989988",
        # Reserved US fiction, but too short to be a real number.
        "+15550100",
    ],
)
def test_implausible_phone_numbers_are_not_saved(junk: str) -> None:
    """A misheard number silently orphans the record from its orders.

    Validation is by libphonenumber rather than by a length range: it knows
    which country codes exist and how long a number may be within each, which
    is the part a hand-written rule gets wrong.
    """
    from app.services.hubspot import normalise_spoken_phone

    assert normalise_spoken_phone(junk) is None


def test_lookup_is_lenient_where_saving_is_strict() -> None:
    """Deliberate asymmetry. A number already in the CRM was typed by a human
    and may be malformed, so identification tries several spellings and must
    keep matching it. Writing a *new* number is the moment to refuse, because
    that is when a mishearing becomes permanent.
    """
    from app.services.hubspot import normalise_spoken_phone, phone_variants

    assert normalise_spoken_phone("+15550100") is None
    assert "+15550100" in phone_variants("+15550100")


async def test_changing_the_number_moves_the_orders_with_it(
    state: CallState, seed_orders: None
) -> None:
    """The number is the join key between the CRM and the ERP.

    Updating only the CRM leaves the caller greeted by name and told they have
    no orders — which reads as a broken lookup, not a half-finished change.
    """
    from unittest.mock import AsyncMock

    from app.services import erp

    state.caller_phone = "+15551234567"  # seed_orders uses this number
    state.contact = Contact(
        id="101", first_name="Ada", last_name=None, email=None, phone=state.caller_phone
    )
    state.hubspot = AsyncMock()

    before = await erp.find_orders_for_phone("+15551234567")
    assert before, "fixture should have orders on the old number"

    agent = SalesAgent()
    result = await agent.save_caller_details.__wrapped__(
        agent, _Ctx(state), "", "", "", "", "+998998886810"
    )

    assert result["orders_moved"] == len(before)
    assert await erp.find_orders_for_phone("+998998886810")
    assert not await erp.find_orders_for_phone("+15551234567")
    # State must follow, or later lookups in the same call use the old number.
    assert state.caller_phone == "+998998886810"


async def test_an_unusable_number_is_not_saved_and_says_so(
    state: CallState, seed_orders: None
) -> None:
    from unittest.mock import AsyncMock

    from app.services import erp

    state.caller_phone = "+15551234567"
    state.contact = Contact(
        id="101", first_name="Ada", last_name=None, email=None, phone=state.caller_phone
    )
    state.hubspot = AsyncMock()

    result = await agent_save_phone(state, "double four")

    assert result["phone_changed"] is False
    assert "did not sound complete" in result["note"]
    # Orders must not move on a number we refused to write.
    assert await erp.find_orders_for_phone("+15551234567")


async def agent_save_phone(state: CallState, phone: str) -> dict:
    agent = SalesAgent()
    return await agent.save_caller_details.__wrapped__(
        agent, _Ctx(state), "", "", "", "", phone
    )
