"""Placing an order: the step that turns a CRM deal into an ERP order.

Until this existed the agent could open an opportunity and nothing more — no
code path wrote an order, so a deal could never become one and every order in
the database was seeded history with no origin.

Two properties are worth pinning here, and both are enforced by the database
rather than by Python: stock cannot go negative under concurrency, and a
failed order leaves nothing behind.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from agent.specialists import SalesAgent
from agent.state import CallState
from agent.tooling import current_call_id
from app.db import engine, fetch_all, fetch_one
from app.errors import UserFacingError, ValidationError
from app.services import calls, erp
from app.services.hubspot import Contact

PHONE = "+15551234567"


@pytest.fixture
async def one_lamp_left() -> None:
    """A product with exactly one in stock, for the race."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO products (sku, name, price_cents, stock_qty) "
                "VALUES ('SKU-LAST', 'Last Lamp', 12900, 1)"
            )
        )


# --- the happy path --------------------------------------------------------


async def test_placing_an_order_takes_the_stock(seed_products: None) -> None:
    before = await erp.find_product("SKU-1")
    order = await erp.place_order(sku="SKU-1", quantity=2, contact_phone=PHONE)

    after = await erp.find_product("SKU-1")
    assert after.stock_qty == before.stock_qty - 2
    assert order.total_cents == before.price_cents * 2
    assert order.status == "processing"


async def test_the_order_is_priced_from_the_erp_not_the_caller(
    seed_products: None,
) -> None:
    """The model reads a price aloud earlier in the call; the order must not
    trust it to remember the figure."""
    order = await erp.place_order(sku="SKU-1", quantity=3, contact_phone=PHONE)
    assert order.total_cents == 12900 * 3


async def test_the_order_records_its_line_items(seed_products: None) -> None:
    order = await erp.place_order(sku="SKU-1", quantity=2, contact_phone=PHONE)
    items = await fetch_all(
        "SELECT sku, qty FROM order_items WHERE order_ref = :ref",
        ref=order.order_ref,
    )
    assert items == [{"sku": "SKU-1", "qty": 2}]


async def test_a_placed_order_is_findable_the_way_a_caller_asks_for_it(
    seed_products: None,
) -> None:
    """A caller rings back and says "where's my order?" — no reference."""
    placed = await erp.place_order(sku="SKU-1", quantity=1, contact_phone=PHONE)
    found = await erp.find_orders_for_phone(PHONE)
    assert placed.order_ref in [o.order_ref for o in found]


async def test_order_references_do_not_collide_with_seeded_history(
    seed_products: None, seed_orders: None
) -> None:
    order = await erp.place_order(sku="SKU-1", quantity=1, contact_phone=PHONE)
    assert order.order_ref.startswith("ORD-")
    assert order.order_ref not in {"ORD-1", "ORD-2"}


# --- the link back to the CRM ----------------------------------------------


async def test_an_order_remembers_the_deal_it_came_from(seed_products: None) -> None:
    """This is the join that did not exist: a deal traceable to fulfilment."""
    order = await erp.place_order(
        sku="SKU-1", quantity=1, contact_phone=PHONE, hubspot_deal_id="deal-42"
    )
    assert (await erp.find_order_by_deal("deal-42")) is not None
    assert (await erp.find_order_by_deal("deal-42")).order_ref == order.order_ref


async def test_an_order_without_a_deal_is_still_allowed(seed_products: None) -> None:
    """Most orders arrive by some other route than a phone call."""
    order = await erp.place_order(sku="SKU-1", quantity=1, contact_phone=PHONE)
    row = await fetch_one(
        "SELECT hubspot_deal_id FROM orders WHERE order_ref = :ref",
        ref=order.order_ref,
    )
    assert row is not None and row["hubspot_deal_id"] is None


async def test_an_unknown_deal_matches_no_order(seed_products: None) -> None:
    assert await erp.find_order_by_deal("deal-does-not-exist") is None


# --- refusals, and what they leave behind ----------------------------------


async def test_cannot_order_more_than_is_in_stock(seed_products: None) -> None:
    with pytest.raises(UserFacingError) as excinfo:
        await erp.place_order(sku="SKU-1", quantity=999, contact_phone=PHONE)
    # The caller hears a number they can act on, not a constraint violation.
    assert "42" in excinfo.value.caller_message


async def test_cannot_order_something_out_of_stock(seed_products: None) -> None:
    with pytest.raises(UserFacingError) as excinfo:
        await erp.place_order(sku="SKU-2", quantity=1, contact_phone=PHONE)
    assert "out of stock" in excinfo.value.caller_message.lower()


async def test_a_refused_order_leaves_no_row_behind(seed_products: None) -> None:
    """The stock decrement and the order share a transaction, so a refusal
    must not leave a half-written order."""
    before = await fetch_all("SELECT order_ref FROM orders")
    with pytest.raises(UserFacingError):
        await erp.place_order(sku="SKU-2", quantity=1, contact_phone=PHONE)
    assert await fetch_all("SELECT order_ref FROM orders") == before


async def test_a_refused_order_does_not_move_stock(seed_products: None) -> None:
    before = (await erp.find_product("SKU-1")).stock_qty
    with pytest.raises(UserFacingError):
        await erp.place_order(sku="SKU-1", quantity=999, contact_phone=PHONE)
    assert (await erp.find_product("SKU-1")).stock_qty == before


async def test_ordering_nothing_is_rejected(seed_products: None) -> None:
    with pytest.raises(ValidationError):
        await erp.place_order(sku="SKU-1", quantity=0, contact_phone=PHONE)


async def test_ordering_an_unknown_product_is_rejected() -> None:
    from app.errors import NotFoundError

    with pytest.raises(NotFoundError):
        await erp.place_order(sku="SKU-NOPE", quantity=1, contact_phone=PHONE)


# --- concurrency, settled by the database ----------------------------------


async def test_two_callers_racing_for_the_last_item(one_lamp_left: None) -> None:
    """Same shape as double-booking: the guard is in the UPDATE, so one of the
    two matches no row rather than both reading "in stock" and overselling."""
    results = await asyncio.gather(
        erp.place_order(sku="SKU-LAST", quantity=1, contact_phone="+1555000111"),
        erp.place_order(sku="SKU-LAST", quantity=1, contact_phone="+1555000222"),
        return_exceptions=True,
    )
    placed = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, UserFacingError)]

    assert len(placed) == 1, "the last item was sold twice"
    assert len(refused) == 1
    assert (await erp.find_product("SKU-LAST")).stock_qty == 0


# --- the tool, and what it does to the CRM ---------------------------------


class FakeCRM:
    """Enough of HubSpotClient for the sales tool, with a failure switch.

    Signatures must match the real client's. The tool wraps its CRM writes in a
    broad ``except`` so a HubSpot outage never costs the caller their order —
    which also means a wrong keyword argument would be logged as a warning and
    silently do nothing. These tests assert the call actually happened, which
    is what turns that back into a failure someone sees.
    """

    def __init__(self, *, closing_fails: bool = False) -> None:
        self.closing_fails = closing_fails
        self.closed: list[tuple[str, str | None]] = []
        self.notes: list[str] = []

    async def close_deal_won(self, deal_id: str, *, order_ref: str | None = None) -> bool:
        if self.closing_fails:
            raise RuntimeError("hubspot is down")
        self.closed.append((deal_id, order_ref))
        return True

    async def log_call_note(self, *, contact_id: str, body: str) -> str:
        self.notes.append(body)
        return "note-1"


class _Ctx:
    def __init__(self, state: CallState) -> None:
        self.userdata = state


@pytest.fixture
async def sales_state() -> CallState:
    current_call_id.set("order-call")
    await calls.start_call(call_id="order-call", direction="inbound", from_number=PHONE)
    return CallState(call_id="order-call", caller_phone=PHONE)


async def test_the_tool_closes_the_deal_it_created(
    seed_products: None, sales_state: CallState
) -> None:
    """Otherwise the CRM keeps showing an open opportunity for something the
    caller has already bought."""
    crm = FakeCRM()
    sales_state.hubspot = crm  # type: ignore[assignment]
    sales_state.deal_id = "deal-7"

    agent = SalesAgent()
    result = await agent.place_order.__wrapped__(agent, _Ctx(sales_state), "SKU-1", 1)

    assert result["order_placed"] is True
    # The deal carries a pointer to the order, so someone looking at the won
    # deal in the CRM can find what fulfils it.
    assert crm.closed == [("deal-7", result["order_ref"])]
    order = await erp.find_order_by_deal("deal-7")
    assert order is not None and order.order_ref == result["order_ref"]


async def test_an_order_without_a_deal_still_reaches_the_crm(
    seed_products: None, sales_state: CallState
) -> None:
    """A caller who rings up and orders outright has no opportunity to close.
    Without a note, nothing about the sale would appear in HubSpot at all."""
    crm = FakeCRM()
    sales_state.hubspot = crm  # type: ignore[assignment]
    sales_state.contact = Contact(
        id="c-1", first_name="Ada", last_name="Okonkwo", email=None, phone=PHONE
    )
    sales_state.deal_id = None

    agent = SalesAgent()
    result = await agent.place_order.__wrapped__(agent, _Ctx(sales_state), "SKU-1", 1)

    assert crm.closed == []
    assert result["order_ref"] in crm.notes[0]


async def test_stock_is_never_written_to_the_crm(
    seed_products: None, sales_state: CallState
) -> None:
    """HubSpot has no model for inventory. Giving a stock level a second home
    is how the two stop agreeing, and the ERP already owns it."""
    crm = FakeCRM()
    sales_state.hubspot = crm  # type: ignore[assignment]
    sales_state.contact = Contact(
        id="c-1", first_name="Ada", last_name="Okonkwo", email=None, phone=PHONE
    )

    agent = SalesAgent()
    await agent.place_order.__wrapped__(agent, _Ctx(sales_state), "SKU-1", 1)

    remaining = str((await erp.find_product("SKU-1")).stock_qty)
    assert all(remaining not in note for note in crm.notes)


async def test_a_crm_failure_does_not_lose_the_placed_order(
    seed_products: None, sales_state: CallState
) -> None:
    """The ERP is the system of record for the sale. A CRM that is down is a
    stale opportunity to tidy up later, not a reason to drop a real order the
    caller has just been told about."""
    sales_state.hubspot = FakeCRM(closing_fails=True)  # type: ignore[assignment]
    sales_state.deal_id = "deal-8"

    agent = SalesAgent()
    result = await agent.place_order.__wrapped__(agent, _Ctx(sales_state), "SKU-1", 1)

    assert result["order_placed"] is True
    assert await erp.find_order_by_deal("deal-8") is not None


async def test_ordering_without_a_deal_still_works(
    seed_products: None, sales_state: CallState
) -> None:
    """A caller can ring up knowing exactly what they want, with no prior
    opportunity — there is nothing to close, and that is not an error."""
    agent = SalesAgent()
    result = await agent.place_order.__wrapped__(agent, _Ctx(sales_state), "SKU-1", 1)
    assert result["order_placed"] is True


async def test_the_tool_gives_the_caller_something_to_repeat_back(
    seed_products: None, sales_state: CallState
) -> None:
    """Everything here is read aloud, so a reference and a date are the point."""
    agent = SalesAgent()
    result = await agent.place_order.__wrapped__(agent, _Ctx(sales_state), "SKU-1", 2)
    assert result["order_ref"].startswith("ORD-")
    assert result["total"] == "$258.00"
    assert result["estimated_delivery"] is not None
