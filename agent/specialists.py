"""The specialists, and the reception agent that routes to them.

    ReceptionAgent          greets, works out what the caller wants, routes
        ├── OrdersAgent     order status and tracking
        ├── SalesAgent      stock, prices, purchase interest
        ├── SchedulingAgent offering slots and booking
        └── PolicyAgent     returns, warranty, delivery, hours

Each specialist carries two or three tools instead of the eight a single agent
had. That is the point: tool selection accuracy falls off as the menu grows,
and it falls off fastest on the small models this is built to run on.

Handoff returns ``(agent, message)`` from a tool. The chat context is copied
across with ``exclude_instructions=True``, so the caller never repeats
themselves but the new agent is not confused by its predecessor's prompt.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import RunContext, function_tool

from agent.base import (
    GROUNDING_RULES,
    HANDOFF_RULE,
    LOOP_BROKEN,
    SPEECH_RULES,
    BaseAgent,
    handoff_would_loop,
)
from agent.state import CallState
from agent.tooling import instrumented
from app.errors import ValidationError
from app.observability.logging import get_logger
from app.services import calls, erp, knowledge, scheduling

log = get_logger(__name__)


# --- reception -------------------------------------------------------------


def _reception_instructions(state: CallState) -> str:
    if state.contact is not None:
        who = (
            f"The caller is {state.contact.display_name}, an existing customer. "
            "Greet them by name."
        )
    else:
        who = "The caller is not in the CRM. Greet them warmly, without a name."

    return f"""You are the receptionist for an office furniture retailer.

{who}

Your job is to work out what the caller needs and hand them to the right
person. Do not try to answer questions about orders, stock, bookings or
policies yourself — you have no tools for those.

Route as soon as the subject is clear, without interrogating them:
- Orders, deliveries, tracking, "where is my", "has it shipped" -> to_orders
- Products, stock, prices, "do you sell", "how much is" -> to_sales
- Booking, appointments, assembly visits, "can someone come" -> to_scheduling
- Returns, refunds, warranty, delivery terms, opening hours, trade accounts
  -> to_policies

If the caller wants to give or correct their own name, email or company,
do it yourself with save_caller_details. That is not a reason to route.

If it is genuinely ambiguous, ask one short question, then route.

Route silently. Do not say you are transferring them, do not name a team, and
do not say "one moment" — to the caller this is one continuous conversation,
and the announcement only adds a delay before their answer.
{SPEECH_RULES}"""


class ReceptionAgent(BaseAgent):
    def __init__(self, *, state: CallState | None = None, chat_ctx: Any = None) -> None:
        # On handoff back, state comes from the session rather than the caller.
        self._initial_state = state
        # Opening the call: the worker greets. Handed back mid-call: pick up.
        self.replies_on_enter = state is None
        super().__init__(
            instructions=_reception_instructions(state) if state else _RETURN_INSTRUCTIONS,
            chat_ctx=chat_ctx,
        )

    @function_tool()
    @instrumented
    async def to_orders(self, context: RunContext[CallState]) -> Any:
        """Hand to the orders specialist for order status, tracking or delivery."""
        if handoff_would_loop(context.userdata):
            return LOOP_BROKEN
        return OrdersAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True)), (
            "Handing to orders. Do not announce this."
        )

    @function_tool()
    @instrumented
    async def to_sales(self, context: RunContext[CallState]) -> Any:
        """Hand to the sales specialist for products, stock levels or prices."""
        if handoff_would_loop(context.userdata):
            return LOOP_BROKEN
        return SalesAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True)), (
            "Handing to sales. Do not announce this."
        )

    @function_tool()
    @instrumented
    async def to_scheduling(self, context: RunContext[CallState]) -> Any:
        """Hand to the scheduling specialist to book an appointment or visit."""
        if handoff_would_loop(context.userdata):
            return LOOP_BROKEN
        return SchedulingAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True)), (
            "Handing to scheduling. Do not announce this."
        )

    @function_tool()
    @instrumented
    async def to_policies(self, context: RunContext[CallState]) -> Any:
        """Hand to the policy specialist for returns, warranty, delivery or hours."""
        if handoff_would_loop(context.userdata):
            return LOOP_BROKEN
        return PolicyAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True)), (
            "Handing to policies. Do not announce this."
        )

    # Reception is where handoffs land, so it must not offer to hand back to
    # itself — that is an easy loop for a small model to fall into.
    return_to_reception = None  # type: ignore[assignment]


_RETURN_INSTRUCTIONS = f"""You are the receptionist for an office furniture
retailer. The caller has been handed back to you because their question moved
outside the previous specialist's area.

Do not greet them again — the call is already in progress. Work out what they
need now and route them, using the same areas as before: orders, sales,
scheduling, policies.
{SPEECH_RULES}"""


# --- orders ----------------------------------------------------------------


class OrdersAgent(BaseAgent):
    def __init__(self, chat_ctx: Any = None) -> None:
        super().__init__(
            instructions=f"""You handle order enquiries: status, tracking and
expected delivery dates.

If the caller gives an order reference, use check_order_status. If they do
not, call find_my_recent_orders straight away — you have their number, so do
not ask for a reference you can look up yourself.
{GROUNDING_RULES}{HANDOFF_RULE}{SPEECH_RULES}""",
            chat_ctx=chat_ctx,
        )

    @function_tool()
    @instrumented
    async def check_order_status(
        self, context: RunContext[CallState], order_reference: str
    ) -> dict[str, Any]:
        """Look up one order by its reference number.

        Args:
            order_reference: The reference the caller gave, e.g. ORD-5001.
        """
        order = await erp.get_order(order_reference)
        return {
            "order_reference": order.order_ref,
            "status": order.status,
            "tracking_reference": order.tracking_ref,
            "estimated_delivery": order.eta.isoformat() if order.eta else None,
        }

    @function_tool()
    @instrumented
    async def find_my_recent_orders(
        self, context: RunContext[CallState]
    ) -> dict[str, Any]:
        """Find recent orders placed from the caller's phone number.

        Use when the caller asks about an order without a reference to hand.
        """
        phone = context.userdata.caller_phone
        if not phone:
            return {"orders": [], "note": "No caller ID available on this call."}
        orders = await erp.find_orders_for_phone(phone)
        return {
            "orders": [
                {
                    "order_reference": o.order_ref,
                    "status": o.status,
                    "estimated_delivery": o.eta.isoformat() if o.eta else None,
                }
                for o in orders
            ]
        }


# --- sales -----------------------------------------------------------------


class SalesAgent(BaseAgent):
    def __init__(self, chat_ctx: Any = None) -> None:
        super().__init__(
            instructions=f"""You handle product enquiries: what we sell, what is
in stock, and what it costs.

If the caller names a product, call check_stock before saying anything
about it. If they have not named one — "what do you sell?", "tell me about
your furniture" — call list_products and describe the range in a sentence or
two, then offer to check a specific item.

Our catalogue is small, so "we don't stock that" is a normal and correct
answer — much better than describing something we do not sell.

If an unrecognised caller wants to be contacted, take their name and email with
save_caller_details, which every agent has.

Buying happens in two steps. While they are still weighing it up, call
record_purchase_interest so it becomes a real opportunity rather than a note.
Once they commit — "yes, order it", "go ahead" — call place_order, which takes
the stock and creates the order. Check stock first so both record the right
product and price, and read the order reference back slowly.
{GROUNDING_RULES}{HANDOFF_RULE}{SPEECH_RULES}""",
            chat_ctx=chat_ctx,
        )

    @function_tool()
    @instrumented
    async def list_products(self, context: RunContext[CallState]) -> dict[str, Any]:
        """List what we sell, for a caller who has not named a product.

        Use for "what do you sell", "what have you got", "tell me about your
        furniture". Summarise the range in a sentence or two rather than
        reading every item aloud — a list of fourteen products is unlistenable
        on a phone. Offer to check a specific one.
        """
        items = await erp.list_catalogue()
        return {
            "products": [
                {"name": i.name, "price": i.price_display, "in_stock": i.in_stock}
                for i in items
            ]
        }

    @function_tool()
    @instrumented
    async def check_stock(
        self, context: RunContext[CallState], product: str
    ) -> dict[str, Any]:
        """Check whether a product is in stock and what it costs.

        Args:
            product: What the caller called it, e.g. "desk lamp" or a SKU.
        """
        item = await erp.find_product(product)
        return {
            "name": item.name,
            "in_stock": item.in_stock,
            "quantity_available": item.stock_qty,
            "price": item.price_display,
        }

    @function_tool()
    @instrumented
    async def record_purchase_interest(
        self,
        context: RunContext[CallState],
        product: str,
        quantity: int = 1,
        notes: str = "",
    ) -> dict[str, Any]:
        """Open a sales opportunity when the caller wants to buy something.

        Use this once they express real intent — "I'd like to order two of
        those", "can you put me down for one". Check the product with
        check_stock first so the opportunity records the right item and price.

        Args:
            product: The product they want, as check_stock returned it.
            quantity: How many they asked for.
            notes: Anything relevant they said about timing or requirements.
        """
        state = context.userdata
        # Price it from the ERP rather than trusting the model to remember the
        # figure it read out a moment ago.
        item = await erp.find_product(product)
        amount = item.price_cents * max(1, quantity)

        await calls.record_event(
            call_id=state.call_id,
            kind="tool_result",
            name="purchase_interest",
            payload={"sku": item.sku, "quantity": quantity, "amount_cents": amount},
        )

        if state.hubspot is None:
            return {
                "opportunity_created": False,
                "reason": "no CRM is connected; interest noted on the call record only",
            }
        if state.contact is None:
            # An unattached deal is invisible on anyone's record, so ask for a
            # name first rather than creating an orphan.
            return {
                "opportunity_created": False,
                "reason": "no contact record yet — take their name with save_caller_details first",
            }

        # Held on the call state so place_order can close *this* deal rather
        # than opening a second one for the same conversation.
        state.deal_id = await state.hubspot.create_deal(
            name=f"{state.contact.display_name} — {quantity} x {item.name}",
            amount_cents=amount,
            contact_id=state.contact.id,
        )
        if notes:
            await state.hubspot.log_call_note(
                contact_id=state.contact.id,
                body=f"Interested in {quantity} x {item.name}.\n\n{notes}",
            )
        return {
            "opportunity_created": True,
            "product": item.name,
            "quantity": quantity,
            "value": f"${amount / 100:,.2f}",
        }

    @function_tool()
    @instrumented
    async def place_order(
        self,
        context: RunContext[CallState],
        product: str,
        quantity: int = 1,
    ) -> dict[str, Any]:
        """Place a firm order once the caller has committed to buying.

        Use this when they have decided — "yes, order it", "I'll take two",
        "go ahead". Not for interest or a maybe; that is
        record_purchase_interest. This takes the stock and creates a real
        order, so only call it on a clear yes.

        Read the order reference back to them slowly, and give them the
        delivery estimate this returns.

        Args:
            product: The product they are buying, as check_stock returned it.
            quantity: How many. Ask if they have not said.
        """
        state = context.userdata
        item = await erp.find_product(product)

        order = await erp.place_order(
            sku=item.sku,
            quantity=quantity,
            contact_phone=state.caller_phone,
            contact_email=state.contact.email if state.contact else None,
            hubspot_deal_id=state.deal_id,
        )

        # Make the sale visible to whoever works in the CRM. The deal carries a
        # pointer to the order rather than a copy of it — the ERP owns
        # fulfilment, and a mirrored status goes stale the moment it ships. The
        # note covers the caller who rings up and orders outright, where there
        # is no deal and nothing would otherwise reach HubSpot at all.
        #
        # Both writes are best-effort and happen after the order. The ERP is
        # the system of record for the sale, and a caller who has just been
        # read an order reference must not have that order vanish because
        # HubSpot was down.
        if state.hubspot is not None:
            try:
                if state.deal_id is not None:
                    await state.hubspot.close_deal_won(
                        state.deal_id, order_ref=order.order_ref
                    )
                if state.contact is not None:
                    await state.hubspot.log_call_note(
                        contact_id=state.contact.id,
                        body=(
                            f"Order {order.order_ref} placed by phone: "
                            f"{quantity} x {item.name}, "
                            f"${order.total_cents / 100:,.2f}."
                        ),
                    )
            except Exception as exc:  # noqa: BLE001 - never lose a placed order
                log.warning(
                    "crm_update_failed",
                    deal_id=state.deal_id,
                    order_ref=order.order_ref,
                    error=str(exc),
                )

        await calls.record_event(
            call_id=state.call_id,
            kind="tool_result",
            name="order_placed",
            payload={
                "order_ref": order.order_ref,
                "sku": item.sku,
                "quantity": quantity,
                "deal_id": state.deal_id,
            },
        )
        return {
            "order_placed": True,
            "order_ref": order.order_ref,
            "product": item.name,
            "quantity": quantity,
            "total": f"${order.total_cents / 100:,.2f}",
            "estimated_delivery": order.eta.isoformat() if order.eta else None,
        }


# --- scheduling ------------------------------------------------------------


class SchedulingAgent(BaseAgent):
    def __init__(self, chat_ctx: Any = None) -> None:
        super().__init__(
            instructions=f"""You book appointments for deliveries and assembly
visits.

Read out the times from offer_appointment_slots, then book the one the caller
picks with book_appointment, passing the slot_id that tool gave you. Never
invent a time or a slot_id.
{GROUNDING_RULES}{HANDOFF_RULE}{SPEECH_RULES}""",
            chat_ctx=chat_ctx,
        )

    @function_tool()
    @instrumented
    async def offer_appointment_slots(
        self, context: RunContext[CallState]
    ) -> dict[str, Any]:
        """Get the next few free appointment slots to read out to the caller."""
        slots = await scheduling.available_slots()
        # Stored on session state, not on self: the caller may be handed to
        # another agent and back before choosing, and the mapping must survive.
        context.userdata.offered_slots = {
            i: slot.starts_at for i, slot in enumerate(slots, start=1)
        }
        return {
            "slots": [
                {"slot_id": i, "when": slot.spoken}
                for i, slot in enumerate(slots, start=1)
            ]
        }

    @function_tool()
    @instrumented
    async def book_appointment(
        self, context: RunContext[CallState], slot_id: int, notes: str = ""
    ) -> dict[str, Any]:
        """Book one of the slots previously offered.

        Args:
            slot_id: The slot_id from offer_appointment_slots the caller chose.
            notes: Anything the caller asked to be noted against the booking.
        """
        state = context.userdata
        starts_at = state.offered_slots.get(slot_id)
        if starts_at is None:
            # The model picked an id we never offered. Recoverable: re-offer.
            raise ValidationError(
                f"slot_id {slot_id} was not offered",
                caller_message="Let me read out the available times again.",
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


# --- policies --------------------------------------------------------------


class PolicyAgent(BaseAgent):
    def __init__(self, chat_ctx: Any = None) -> None:
        super().__init__(
            instructions=f"""You answer questions about how things work:
returns, refunds, warranty, delivery terms and charges, assembly costs, opening
hours and business accounts.

Always call search_policies first, then summarise the answer in one or two
sentences. Do not read the policy out word for word, and do not answer from
memory — if the search finds nothing, say you do not have that to hand.
{GROUNDING_RULES}{HANDOFF_RULE}{SPEECH_RULES}""",
            chat_ctx=chat_ctx,
        )

    @function_tool()
    @instrumented
    async def search_policies(
        self, context: RunContext[CallState], question: str
    ) -> dict[str, Any]:
        """Search company policy documents.

        Args:
            question: The caller's question, in their own words.
        """
        passages = await knowledge.search(question)
        if not passages:
            return {
                "found": False,
                "note": "Nothing in the policy documents covers this.",
            }
        return {
            "found": True,
            "passages": [{"topic": p.title, "text": p.body} for p in passages],
        }
