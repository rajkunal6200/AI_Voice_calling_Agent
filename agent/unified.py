"""One agent that does everything, as an alternative to the specialist split.

The split buys tool-selection accuracy by giving each agent a short menu. It
costs a model round trip on the first turn, and it creates failure modes that
do not exist with one agent: a capability reachable from only one specialist,
and two agents passing a request neither can serve. Both happened.

To the caller the difference is whether the conversation feels like one person
or like being passed around — which is a product decision, not only a technical
one, and it deserves to be settled with measurements rather than principle.

The tools are inherited from the specialists rather than redefined, so the two
modes cannot drift apart: a tool added to a specialist appears here too.

Switch with ``AGENT_MODE=unified`` and compare:

    uv run python -m evals.run
    AGENT_MODE=unified uv run python -m evals.run
"""

from __future__ import annotations

from typing import Any

from agent.base import GROUNDING_RULES, SPEECH_RULES, BaseAgent
from agent.specialists import OrdersAgent, PolicyAgent, SalesAgent, SchedulingAgent
from agent.state import CallState
from app.observability.logging import get_logger

log = get_logger(__name__)


def _instructions(state: CallState) -> str:
    if state.contact is not None:
        who = (
            f"The caller is {state.contact.display_name}, an existing customer. "
            "Greet them by name."
        )
    else:
        who = "The caller is not in the CRM. Greet them warmly, without a name."

    return f"""You are the phone agent for an office furniture retailer. You
handle the whole call yourself — there is nobody to transfer to.

{who}

You can do all of this, so never tell a caller you cannot help without
checking whether one of these applies:
- Products: list_products for "what do you sell", check_stock for a named item.
- Orders: check_order_status if they give a reference. If they do not, call
  find_my_recent_orders straight away — you have their number, so do not ask
  them for a reference you can look up yourself.
- Appointments: offer_appointment_slots, then book_appointment with a slot_id
  that tool gave you.
- Policies: search_policies for returns, warranty, delivery, hours, accounts.
- Their details: save_caller_details to take or correct a name, email, company.
- Buying: record_purchase_interest while they are still deciding, then
  place_order once they say yes. Only place_order takes the stock and makes it
  real, so it needs a clear yes.

Only use request_human_callback when nothing above fits — a complaint, a
bespoke quote, a decision needing judgement.
{GROUNDING_RULES}{SPEECH_RULES}
Before the call ends, call log_call_summary once with two sentences on what the
caller wanted and what you did."""


class UnifiedAgent(OrdersAgent, SalesAgent, SchedulingAgent, PolicyAgent):
    """Every tool on one agent. Tools come from the specialists via the MRO."""

    def __init__(self, *, state: CallState | None = None, chat_ctx: Any = None) -> None:
        # The worker greets on an inbound call, so do not also reply on enter.
        self.replies_on_enter = state is None
        # Skip the specialists' __init__, each of which sets its own narrow
        # instructions; go straight to the base with the combined prompt.
        BaseAgent.__init__(
            self,
            instructions=_instructions(state) if state else "",
            chat_ctx=chat_ctx,
        )

    # There is nowhere to hand back to when one agent holds every tool.
    return_to_reception = None  # type: ignore[assignment]
