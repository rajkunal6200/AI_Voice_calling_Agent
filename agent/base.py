"""Shared behaviour for every agent on a call.

Splitting one agent into specialists is a reliability measure, not tidiness.
A small model picking from three tools is markedly more accurate than the same
model picking from eight, and every specialist here carries two or three. The
tradeoff is that routing can now go wrong, which is why triage exists and why
every specialist can hand back.

Two rules are shared rather than repeated, because they apply on every turn no
matter which agent is speaking: how to talk on a phone, and that facts come
from tools rather than from the model's own knowledge.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from agent.state import CallState
from agent.tooling import instrumented
from app.observability.logging import get_logger
from app.services import calls, erp
from app.services.hubspot import normalise_spoken_phone

log = get_logger(__name__)

#: Consecutive handoffs allowed before one is refused. A caller changing
#: subject needs one; needing three in a row without saying anything means two
#: agents are passing a request neither can serve.
MAX_CONSECUTIVE_HANDOFFS = 3


def handoff_would_loop(state: CallState) -> bool:
    return state.handoffs_since_user_turn >= MAX_CONSECUTIVE_HANDOFFS


#: Returned instead of performing a handoff, so the model has to act.
LOOP_BROKEN = {
    "handoff_refused": True,
    "reason": (
        "This request has already been passed between agents several times. "
        "Do not hand off again. Answer with a tool you have, or tell the "
        "caller plainly that you cannot help with this and offer to have "
        "someone call them back."
    ),
}


SPEECH_RULES = """
How to speak, since this is a phone call and your words are read aloud:
- Keep replies to a sentence or two. Long answers are hard to follow by ear.
- Write plain spoken English. No markdown, no bullet points, no asterisks,
  no emoji, and no symbols the listener cannot hear.
- Say prices and numbers as words, like "one hundred and twenty nine dollars".
- Do not read out SKUs, identifiers or internal references unless the caller
  used one first.
- If a tool says something was not found, say so plainly and offer the next
  best step. Do not apologise repeatedly.
"""

GROUNDING_RULES = """
You do not know anything about this company's products, prices, stock, orders,
policies or availability. That information exists only in the tools. Anything
you say about them must come from a tool result in this conversation. Never
fill a gap from your own knowledge — "I don't have that" is a correct answer.
"""

HANDOFF_RULE = """
You handle one area. If the caller moves to something outside it, hand back
with return_to_reception rather than guessing or answering from memory.

Two things you can always do yourself, whatever your area: take or correct the
caller's own contact details with save_caller_details, and close the call with
log_call_summary. Never hand those on or promise a colleague will do them.
"""


class BaseAgent(Agent):
    """Common tools and speech rules. Not used directly."""

    #: Whether to answer immediately on taking over. True for handoff targets:
    #: the caller has already asked their question, and without this the
    #: specialist arrives silent and they have to repeat themselves. False for
    #: the agent that opens the call, where the worker sends the greeting.
    replies_on_enter = True

    def __init__(self, instructions: str, chat_ctx: Any = None) -> None:
        super().__init__(instructions=instructions, chat_ctx=chat_ctx)

    async def on_enter(self) -> None:
        """Record the handoff, then pick up where the caller left off."""
        state: CallState = self.session.userdata
        name = type(self).__name__
        state.handoffs_since_user_turn += 1
        log.info(
            "agent_entered", agent=name, consecutive=state.handoffs_since_user_turn
        )
        await calls.record_event(
            call_id=state.call_id, kind="tool_result", name="__handoff",
            payload={"agent": name},
        )

        if self.replies_on_enter:
            # The inherited chat context already holds the caller's question,
            # so answer it rather than greeting them a second time.
            self.session.generate_reply(
                instructions=(
                    "Answer the caller's most recent question using your tools. "
                    "Do not greet them again and do not mention being "
                    "transferred — from their side this is one conversation."
                )
            )

    # --- shared tools ------------------------------------------------------

    @function_tool()
    @instrumented
    async def return_to_reception(self, context: RunContext[CallState]) -> Any:
        """Hand the caller back when their question is outside your area.

        Use this rather than answering something you have no tool for.
        """
        state = context.userdata
        if handoff_would_loop(state):
            log.warning(
                "handoff_loop_broken",
                agent=type(self).__name__,
                handoffs=state.handoffs_since_user_turn,
            )
            return LOOP_BROKEN

        # Imported here: reception imports the specialists, so a module-level
        # import would be circular.
        from agent.specialists import ReceptionAgent

        return (
            ReceptionAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True)),
            "Handing back to reception. Do not announce this.",
        )

    @function_tool()
    @instrumented
    async def request_human_callback(
        self, context: RunContext[CallState], reason: str, urgent: bool = False
    ) -> dict[str, Any]:
        """Last resort: ask a human to call the customer back.

        Only when nothing else can help — a complaint, a bespoke quote, a
        decision needing judgement. Never use it for something another tool
        does: check first whether you can act, and hand back to reception if
        the request belongs to a different area. Promising a callback for work
        the agent could have done itself is worse than doing nothing, because
        the caller waits for a call that was never needed.

        Args:
            reason: What the caller needs and any detail they gave.
            urgent: True if they are unhappy or the matter is time-sensitive.
        """
        state = context.userdata
        caller = state.contact.display_name if state.contact else "an unidentified caller"
        subject = f"Call back {caller}" + (" (urgent)" if urgent else "")

        # Always on the local record, so the request survives a CRM outage and
        # shows on the dashboard whether or not HubSpot took it.
        await calls.record_event(
            call_id=state.call_id,
            kind="tool_result",
            name="callback_requested",
            payload={"reason": reason, "urgent": urgent, "phone": state.caller_phone},
        )

        if state.hubspot is None:
            return {
                "created_in_crm": False,
                "reason": "no CRM is connected; noted on the call record only",
            }

        await state.hubspot.create_task(
            subject=subject,
            body=(
                f"Raised by the voice agent during a call.\n\n{reason}\n\n"
                f"Caller number: {state.caller_phone or 'unknown'}"
            ),
            contact_id=state.contact.id if state.contact else None,
            priority="HIGH" if urgent else "MEDIUM",
        )
        return {"created_in_crm": True, "assigned": "follow-up task"}

    @function_tool()
    @instrumented
    async def save_caller_details(
        self,
        context: RunContext[CallState],
        first_name: str = "",
        last_name: str = "",
        email: str = "",
        company: str = "",
        phone: str = "",
    ) -> dict[str, Any]:
        """Record or correct the caller's contact details.

        Works for both an unrecognised caller and a known one who gives
        something we do not hold — one tool rather than two near-identical
        ones, which a model picking under time pressure would confuse.

        Args:
            first_name: The caller's first name.
            last_name: The caller's surname, if given.
            email: The caller's email address, if given or corrected.
            company: The caller's company, if given or corrected.
            phone: A new phone number, if they are changing it. Read it back
                to them before saving — a misheard digit loses the record.
        """
        state = context.userdata
        if state.hubspot is None:
            # No CRM. Keep the details on the call record so they are not lost,
            # and tell the model the truth so it does not promise the caller a
            # record that was never created.
            await calls.record_event(
                call_id=state.call_id,
                kind="tool_result",
                name="caller_details",
                payload={"first_name": first_name, "last_name": last_name, "email": email},
            )
            return {
                "saved_to_crm": False,
                "reason": "no CRM is connected; details noted on the call record only",
            }

        if state.contact is not None:
            new_phone = normalise_spoken_phone(phone) if phone else None
            await state.hubspot.update_contact(
                contact_id=state.contact.id,
                email=email or None,
                company=company or None,
                first_name=first_name or None,
                last_name=last_name or None,
                phone=new_phone,
            )
            moved = 0
            if new_phone and state.caller_phone and new_phone != state.caller_phone:
                # The number joins the two systems. Updating only the CRM
                # leaves the caller greeted by name and told they have no
                # orders, which reads as a broken lookup rather than a
                # half-finished change.
                moved = await erp.relink_orders(
                    old_phone=state.caller_phone, new_phone=new_phone
                )
                state.caller_phone = new_phone
            return {
                "saved_to_crm": True,
                "updated_existing": True,
                "phone_changed": bool(new_phone),
                "orders_moved": moved,
                **(
                    {"note": "That number did not sound complete, so it was not saved."}
                    if phone and not new_phone
                    else {}
                ),
            }

        # A caller with no usable number cannot be found again — identification
        # looks people up by number, and so does order lookup. Prefer the one
        # that rang, but honour a dictated one: a caller ringing from a withheld
        # or office number who gives their mobile is exactly the case where the
        # number that rang is the wrong one to keep.
        dictated = normalise_spoken_phone(phone) if phone else None
        contact = await state.hubspot.create_contact(
            phone=dictated or state.caller_phone or "",
            first_name=first_name or None,
            last_name=last_name or None,
            email=email or None,
        )
        state.contact = contact
        if dictated:
            state.caller_phone = dictated
        await calls.attach_contact(
            call_id=state.call_id,
            hubspot_contact_id=contact.id,
            caller_name=contact.display_name,
        )
        return {
            "saved_to_crm": True,
            "created_new_record": True,
            "name": contact.display_name,
            **(
                {"note": "That number did not sound complete, so it was not saved."}
                if phone and not dictated
                else {}
            ),
        }

    @function_tool()
    @instrumented
    async def log_call_summary(
        self, context: RunContext[CallState], summary: str
    ) -> dict[str, Any]:
        """Write a summary of this call before it ends.

        Call this once, at the end, whichever agent is speaking.

        Args:
            summary: Two sentences on what the caller wanted and what was done.
        """
        state = context.userdata
        # Local record first: it is what the dashboard shows and it must
        # survive the CRM being unreachable or absent.
        await calls.end_call(
            call_id=state.call_id, status="completed", summary=summary
        )
        state.summary_logged = True

        if state.hubspot is None:
            return {"logged_to_crm": False, "reason": "no CRM is connected"}
        if state.contact is None:
            return {"logged_to_crm": False, "reason": "no contact record for this caller"}

        await state.hubspot.log_call_note(
            contact_id=state.contact.id,
            body=f"Voice agent call summary:\n\n{summary}",
        )
        return {"logged_to_crm": True}
