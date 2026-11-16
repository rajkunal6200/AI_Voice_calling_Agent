"""The unified agent, and its equivalence with the specialist split.

Both modes ship, so the property worth pinning is that they can do the same
things. A tool added to a specialist must appear on the unified agent too, or
the two configurations quietly diverge and only one of them gets tested.
"""

from __future__ import annotations

import pytest
from livekit.agents.llm import utils as llm_utils

from agent import entry_agent
from agent.specialists import (
    OrdersAgent,
    PolicyAgent,
    ReceptionAgent,
    SalesAgent,
    SchedulingAgent,
)
from agent.state import CallState
from agent.tooling import current_call_id
from agent.unified import UnifiedAgent
from app.services import calls

CALL_ID = "unified-call"

SPECIALISTS = (OrdersAgent, SalesAgent, SchedulingAgent, PolicyAgent)
#: Only meaningful when there is somewhere to route.
ROUTING_TOOLS = {"to_orders", "to_sales", "to_scheduling", "to_policies"}


@pytest.fixture
async def state() -> CallState:
    current_call_id.set(CALL_ID)
    await calls.start_call(call_id=CALL_ID, direction="inbound", from_number="+1555")
    return CallState(call_id=CALL_ID, caller_phone="+1555")


class _Ctx:
    def __init__(self, state: CallState) -> None:
        self.userdata = state


def tool_names(agent: object) -> set[str]:
    return {
        llm_utils.build_legacy_openai_schema(t)["function"]["name"]
        for t in agent.tools  # type: ignore[attr-defined]
    }


# --- equivalence -----------------------------------------------------------


def test_unified_can_do_everything_the_specialists_can(state: CallState) -> None:
    """Otherwise switching mode silently loses a capability."""
    split: set[str] = set()
    for cls in (ReceptionAgent,):
        split |= tool_names(cls(state=state))
    for cls in SPECIALISTS:
        split |= tool_names(cls())

    unified = tool_names(UnifiedAgent(state=state))
    missing = (split - ROUTING_TOOLS - {"return_to_reception"}) - unified
    assert not missing, f"unified is missing {missing}"


def test_unified_has_no_handoff_tools(state: CallState) -> None:
    """One agent holds everything, so there is nowhere to hand off to — and a
    handoff tool with no target is a way to get stuck."""
    names = tool_names(UnifiedAgent(state=state))
    assert not (names & ROUTING_TOOLS)
    assert "return_to_reception" not in names


def test_unified_cannot_loop_because_it_cannot_hand_off(state: CallState) -> None:
    """The stuck-call failure is structurally impossible in this mode."""
    assert UnifiedAgent(state=state).return_to_reception is None


# --- mode selection --------------------------------------------------------


def test_entry_agent_follows_the_configured_mode(
    state: CallState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "agent_mode", "unified")
    assert isinstance(entry_agent(state), UnifiedAgent)

    monkeypatch.setattr(settings, "agent_mode", "specialists")
    assert isinstance(entry_agent(state), ReceptionAgent)


def test_unknown_mode_falls_back_to_specialists(
    state: CallState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in AGENT_MODE must not leave the worker with no agent."""
    from app.config import settings

    monkeypatch.setattr(settings, "agent_mode", "nonsense")
    assert isinstance(entry_agent(state), ReceptionAgent)


# --- behaviour -------------------------------------------------------------


async def test_unified_answers_a_product_question(
    state: CallState, seed_products: None
) -> None:
    agent = UnifiedAgent(state=state)
    result = await agent.check_stock.__wrapped__(agent, _Ctx(state), "desk lamp")
    assert result["in_stock"] is True


async def test_unified_answers_an_order_question(
    state: CallState, seed_orders: None
) -> None:
    agent = UnifiedAgent(state=state)
    result = await agent.check_order_status.__wrapped__(agent, _Ctx(state), "ORD-1")
    assert result["status"] == "shipped"


async def test_unified_books_an_appointment(state: CallState) -> None:
    agent = UnifiedAgent(state=state)
    ctx = _Ctx(state)
    offered = await agent.offer_appointment_slots.__wrapped__(agent, ctx)
    booked = await agent.book_appointment.__wrapped__(
        agent, ctx, offered["slots"][0]["slot_id"]
    )
    assert booked["booked"] is True


def test_unified_instructions_list_what_it_can_do(state: CallState) -> None:
    """With no specialist prompts, the one prompt has to carry the map — and
    the failure being guarded against is the agent saying it cannot help with
    something it has a tool for."""
    text = UnifiedAgent(state=state).instructions.lower()
    assert "never tell a caller you cannot help" in text
    for capability in ("list_products", "check_stock", "search_policies"):
        assert capability in text
