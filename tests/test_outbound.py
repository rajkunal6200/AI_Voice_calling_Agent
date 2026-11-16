"""The outbound qualification agent.

Outbound carries a risk inbound does not: these calls are unsolicited, so the
behaviour that matters most is what happens when someone says no. Those cases
are first here for that reason.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from livekit.agents.llm import utils as llm_utils

from agent.outbound import STAGE_CLOSED_LOST, STAGE_QUALIFIED, OutboundQualificationAgent
from agent.state import CallState
from agent.tooling import current_call_id
from app.services import calls
from app.services.hubspot import Contact, HubSpotClient

CALL_ID = "outbound-call"
BASE = "https://api.hubapi.com"


@pytest.fixture
async def state() -> CallState:
    current_call_id.set(CALL_ID)
    await calls.start_call(call_id=CALL_ID, direction="outbound", to_number="+15550101")
    return CallState(
        call_id=CALL_ID,
        caller_phone="+15550101",
        contact=Contact(
            id="101", first_name="Grace", last_name="Whitfield",
            email=None, phone="+15550101",
        ),
        deal_id="d-1",
    )


class _Ctx:
    def __init__(self, state: CallState) -> None:
        self.userdata = state


def tool_names(agent: OutboundQualificationAgent) -> set[str]:
    return {
        llm_utils.build_legacy_openai_schema(t)["function"]["name"] for t in agent.tools
    }


# --- consent ---------------------------------------------------------------


def test_instructions_put_stopping_above_qualifying(state: CallState) -> None:
    """An unsolicited call must honour "stop" before it pursues its goal, so
    that instruction appears before the qualification steps."""
    agent = OutboundQualificationAgent(state=state)
    text = agent.instructions
    stop_at = text.index("not interested")
    qualify_at = text.index("What problem they are trying to solve")
    assert stop_at < qualify_at
    assert "do not ask a second time" in text.lower()


def test_agent_identifies_itself_as_ai(state: CallState) -> None:
    assert "ai assistant" in OutboundQualificationAgent(state=state).instructions.lower()


async def test_declining_still_records_an_outcome(state: CallState) -> None:
    """A refusal must be written down, or the same person gets called again."""
    agent = OutboundQualificationAgent(state=state)
    result = await agent.record_qualification.__wrapped__(
        agent, _Ctx(state), False, "Asked not to be contacted again."
    )

    assert result["recorded"] is True
    assert result["qualified"] is False
    call = await calls.get_call(CALL_ID)
    assert call.summary == "Asked not to be contacted again."


# --- structure --------------------------------------------------------------


def test_outbound_has_no_reception_to_return_to(state: CallState) -> None:
    """One agent, one purpose — offering a handoff would just get it stuck."""
    assert "return_to_reception" not in tool_names(OutboundQualificationAgent(state=state))


def test_tool_count_stays_small(state: CallState) -> None:
    """Six: three of its own, plus the universal ones every agent carries.

    save_caller_details is among them deliberately — a lead correcting their
    name mid-call should not dead-end here either.
    """
    assert len(tool_names(OutboundQualificationAgent(state=state))) <= 6


# --- CRM outcomes -----------------------------------------------------------


@respx.mock
async def test_qualified_lead_moves_the_deal_forward(state: CallState) -> None:
    state.hubspot = HubSpotClient(
        client=httpx.AsyncClient(base_url=BASE, headers={"Authorization": "Bearer t"})
    )
    respx.post(f"{BASE}/crm/v3/objects/notes").mock(
        return_value=httpx.Response(201, json={"id": "n-1"})
    )
    patch = respx.patch(f"{BASE}/crm/v3/objects/deals/d-1").mock(
        return_value=httpx.Response(200, json={})
    )

    await agent_qualify(state, qualified=True)

    assert STAGE_QUALIFIED in patch.calls.last.request.read().decode()


@respx.mock
async def test_unqualified_lead_closes_the_deal(state: CallState) -> None:
    state.hubspot = HubSpotClient(
        client=httpx.AsyncClient(base_url=BASE, headers={"Authorization": "Bearer t"})
    )
    respx.post(f"{BASE}/crm/v3/objects/notes").mock(
        return_value=httpx.Response(201, json={"id": "n-1"})
    )
    patch = respx.patch(f"{BASE}/crm/v3/objects/deals/d-1").mock(
        return_value=httpx.Response(200, json={})
    )

    await agent_qualify(state, qualified=False)

    assert STAGE_CLOSED_LOST in patch.calls.last.request.read().decode()


@respx.mock
async def test_outcome_survives_the_crm_being_down(state: CallState) -> None:
    """The local record is the one that must not be lost — it is how we know
    not to call this person again."""
    state.hubspot = HubSpotClient(
        client=httpx.AsyncClient(base_url=BASE, headers={"Authorization": "Bearer t"})
    )
    respx.post(f"{BASE}/crm/v3/objects/notes").mock(return_value=httpx.Response(500))

    from livekit.agents.llm import ToolError

    with pytest.raises(ToolError):
        await agent_qualify(state, qualified=False)

    # end_call runs before the CRM write, so the outcome is still recorded.
    call = await calls.get_call(CALL_ID)
    assert call.status == "completed"
    assert state.summary_logged is True


async def test_no_deal_id_still_records_the_call(state: CallState) -> None:
    """A lead with no deal is still worth a note, not a crash."""
    state.deal_id = None
    state.hubspot = None

    result = await agent_qualify(state, qualified=True)

    assert result["written_to_crm"] is False
    assert (await calls.get_call(CALL_ID)).status == "completed"


async def agent_qualify(state: CallState, *, qualified: bool) -> dict[str, object]:
    agent = OutboundQualificationAgent(state=state)
    return await agent.record_qualification.__wrapped__(
        agent, _Ctx(state), qualified, "Discussed requirements.", "5k", "next quarter"
    )
