"""Outbound agent: qualifies a lead and moves its deal forward.

Outbound differs from inbound in ways that shape the design:

*We know who we called*, so there is no identification step and no reception —
the call has one purpose, so one agent handles it end to end. Routing would add
a handoff with nothing to route to.

*The call is unsolicited.* The agent must say who and what it is immediately,
and must stop the moment the person wants it to. That is stated first in the
instructions, above everything about qualifying, because it outranks the
business goal — it is a legal and courtesy requirement, not a preference.

*Nobody is waiting on the line for us to answer*, so a slow tool matters less
here than on an inbound call.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import RunContext, function_tool

from agent.base import SPEECH_RULES, BaseAgent
from agent.state import CallState
from agent.tooling import instrumented
from app.errors import ValidationError
from app.observability.logging import get_logger
from app.services import calls, scheduling

log = get_logger(__name__)

# Internal stage ids differ per portal. These are HubSpot's defaults; override
# them for a custom pipeline rather than scattering ids through the codebase.
STAGE_QUALIFIED = "qualifiedtobuy"
STAGE_CLOSED_LOST = "closedlost"


def _instructions(state: CallState, company: str) -> str:
    who = state.contact.display_name if state.contact else "the person"
    return f"""You are calling {who} on behalf of {company}. They submitted an
enquiry on the website, which is why you are calling.

Open by saying who you are, that you are an AI assistant calling from
{company}, and that you are following up on their enquiry. Then ask if it is a
good time to talk.

If they say they are busy, not interested, want to be removed from the list, or
ask you to stop: end the conversation politely and immediately. Call
record_qualification with qualified set to false, noting what they said. Do not
attempt to persuade them and do not ask a second time. This outranks everything
below.

If they are willing to talk, find out three things through conversation, not as
a checklist read aloud:
- What problem they are trying to solve.
- Roughly what budget they have in mind.
- When they hope to have it in place.

Then call record_qualification. If they are a good fit and want to speak to a
person, offer times and book one.

Never invent details about pricing, stock or delivery. If you do not know, say
a colleague will follow up with specifics.
{SPEECH_RULES}"""


class OutboundQualificationAgent(BaseAgent):
    def __init__(
        self,
        *,
        state: CallState | None = None,
        company: str = "Northwind Office",
        chat_ctx: Any = None,
    ) -> None:
        # The worker speaks first here — we called them, so we open.
        self.replies_on_enter = state is None
        super().__init__(
            instructions=_instructions(state, company) if state else "",
            chat_ctx=chat_ctx,
        )

    # There is nowhere to hand back to on an outbound call: one agent, one
    # purpose. Offering the tool would just be a way to get stuck.
    return_to_reception = None  # type: ignore[assignment]

    @function_tool()
    @instrumented
    async def record_qualification(
        self,
        context: RunContext[CallState],
        qualified: bool,
        summary: str,
        budget: str = "",
        timeline: str = "",
    ) -> dict[str, Any]:
        """Record the outcome of the call in the CRM.

        Call this once before the call ends, whether or not they qualified —
        including when they ask not to be contacted.

        Args:
            qualified: Whether they are a fit and want to continue.
            summary: Two sentences on what was discussed and what happens next.
            budget: What they said about budget, in their words.
            timeline: What they said about timing, in their words.
        """
        state = context.userdata
        note = "\n".join(
            [
                f"Outbound qualification call — {'qualified' if qualified else 'not qualified'}.",
                "",
                summary,
                f"Budget: {budget or 'not discussed'}",
                f"Timeline: {timeline or 'not discussed'}",
            ]
        )

        # Local record first: it is what the dashboard shows and it must
        # survive the CRM being unreachable.
        await calls.end_call(call_id=state.call_id, status="completed", summary=summary)
        state.summary_logged = True

        if state.hubspot is None or state.contact is None:
            log.info("lead_qualified", qualified=qualified, crm=False)
            return {"recorded": True, "qualified": qualified, "written_to_crm": False}

        await state.hubspot.log_call_note(contact_id=state.contact.id, body=note)
        if state.deal_id:
            await state.hubspot.update_deal_stage(
                deal_id=state.deal_id,
                stage=STAGE_QUALIFIED if qualified else STAGE_CLOSED_LOST,
            )

        log.info("lead_qualified", qualified=qualified, contact_id=state.contact.id)
        return {"recorded": True, "qualified": qualified, "written_to_crm": True}

    @function_tool()
    @instrumented
    async def offer_appointment_slots(
        self, context: RunContext[CallState]
    ) -> dict[str, Any]:
        """Get the next few free slots to offer for a follow-up with a person."""
        slots = await scheduling.available_slots()
        context.userdata.offered_slots = {
            i: s.starts_at for i, s in enumerate(slots, start=1)
        }
        return {
            "slots": [{"slot_id": i, "when": s.spoken} for i, s in enumerate(slots, start=1)]
        }

    @function_tool()
    @instrumented
    async def book_appointment(
        self, context: RunContext[CallState], slot_id: int, notes: str = ""
    ) -> dict[str, Any]:
        """Book one of the slots previously offered.

        Args:
            slot_id: The slot_id from offer_appointment_slots that they chose.
            notes: Anything they asked to be noted against the booking.
        """
        state = context.userdata
        starts_at = state.offered_slots.get(slot_id)
        if starts_at is None:
            raise ValidationError(
                f"slot_id {slot_id} was not offered",
                caller_message="Let me read out those times again.",
                slot_id=slot_id,
            )

        appointment = await scheduling.book(
            starts_at=starts_at,
            call_id=state.call_id,
            hubspot_contact_id=state.contact.id if state.contact else None,
            notes=notes or None,
        )
        return {
            "booked": True,
            "when": appointment.starts_at.strftime("%A %-d %B at %-I:%M %p"),
        }
